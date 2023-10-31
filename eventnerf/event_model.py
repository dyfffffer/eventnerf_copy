
from dataclasses import dataclass, field
from typing import List, Type

import nerfacc
import torch
from torch.nn import Parameter
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from typing import Any, Dict, Literal

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.fields.nerfacto_field import NerfactoField
from nerfstudio.model_components.losses import MSELoss
from nerfstudio.model_components.ray_samplers import VolumetricSampler
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    RGBRenderer,
)
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps, colors, misc
from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig  # for subclassing Nerfacto
from nerfstudio.models.vanilla_nerf import NeRFModel, VanillaModelConfig
from nerfstudio.models.instant_ngp import InstantNGPModelConfig, NGPModel
from nerfstudio.models.base_model import Model, ModelConfig  # for custom Model
from nerfstudio.model_components.renderers import RGBRenderer
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.configs.config_utils import to_immutable_dict
from eventnerf.event_field import EventField

@dataclass
class EventModelConfig(InstantNGPModelConfig):

    _target: Type = field(default_factory=lambda: EventModel)

    disable_scene_contraction: bool = True

    #event_threshold = 0.11

    grid_levels: int = 1

    cone_angle: float = 0 #0.004

    alpha_thre: float = 1e-3

    background_color: Literal["random", "black", "white"] = torch.Tensor([0.62, 0.62, 0.62])

    loss_coefficients: Dict[str, float] = to_immutable_dict({
        "rgb_loss": 1.0, "event_loss": 0.0,
    })


class EventModel(NGPModel):

    config: EventModelConfig

    def populate_modules(self):
        super().populate_modules()
        self.event_threshold = torch.Tensor([1]) * 0.25
        
    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param = super().get_param_groups()
        #param["event_threshold"] = list(self.event_threshold)
        return param

    def get_loss_dict(self, outputs, batch, metric_dict=None):
        image = batch["image"][..., :3].to(self.device)
        self.event_threshold = self.event_threshold.to(self.device)
        pred_rgb, image = self.renderer_rgb.blend_background_for_loss_computation(
            pred_image=outputs["rgb"],
            pred_accumulation=outputs["accumulation"],
            gt_image=image,
        )
        rgb_loss = self.rgb_loss(image, pred_rgb)

        ef = batch["event_frame_selected"].to(self.device)
        log_rgb = torch.log(pred_rgb) * 2.2 # ?
        log_rgb = log_rgb.reshape(2, len(log_rgb) // 2, 3)
        diff = (log_rgb[1] - log_rgb[0]) * (ef != 0)
        event_loss  = 0
        #event_loss += self.rgb_loss(ef / ef.max(), diff / diff.max())
        #event_loss += self.rgb_loss(ef / ef.min(), diff / diff.min())
        #event_loss += self.rgb_loss(ef ,(ef.max() - ef.min()) * diff / (diff.max() - diff.min()))
        #event_loss += self.rgb_loss(ef / (ef.max() - ef.min()), diff / (diff.max() - diff.min()))
        #event_loss += self.rgb_loss(ef / (ef.max() - ef.min()), diff / (diff.max() - diff.min()))
        #event_loss /= 3
        #event_loss += self.rgb_loss(ef * 0.35, diff * 2.2)
        event_loss += self.rgb_loss(ef * 0.25, diff)
        #CONSOLE.print (event_loss, (diff.max() - diff.min()) / (ef.max() - ef.min()), diff.sum() / ef.sum())

        loss_dict = {"event_loss:": event_loss, "rgb_loss": rgb_loss}
        loss_dict = misc.scale_dict(loss_dict, self.config.loss_coefficients)
        return loss_dict
    
#@dataclass
#class EventModelConfig(VanillaModelConfig):
#
#    _target: Type = field(default_factory=lambda: EventModel)
#    background_color = torch.Tensor([0.624, 0.624, 0.624])
#    loss_coefficients: Dict[str, float] = to_immutable_dict({
#        "rgb_loss_coarse": 0.0, "rgb_loss_fine": 0.0,
#        "event_loss_coarse": 1.0, "event_loss_fine": 1.0
#    })
#
#class EventModel(NeRFModel):
#
#    config: EventModelConfig
#
#    def populate_modules(self):
#        super().populate_modules()
#
#    def get_loss_dict(self, outputs, batch, metric_dict=None):
#        device = outputs["rgb_coarse"].device
#        image = batch["image"].to(device)
#        coarse_pred, coarse_image = self.renderer_rgb.blend_background_for_loss_computation(
#            pred_image=outputs["rgb_coarse"],
#            pred_accumulation=outputs["accumulation_coarse"],
#            gt_image=image,
#        )
#        fine_pred, fine_image = self.renderer_rgb.blend_background_for_loss_computation(
#            pred_image=outputs["rgb_fine"],
#            pred_accumulation=outputs["accumulation_fine"],
#            gt_image=image,
#        )
#
#        rgb_loss_coarse = self.rgb_loss(coarse_image, coarse_pred)
#        rgb_loss_fine = self.rgb_loss(fine_image, fine_pred)
#
#
#        ef = batch["event_frame_selected"].to(self.device) * 0.2
#
#        pred_rgb_coarse= torch.log(coarse_pred)
#        pred_rgb_coarse= pred_rgb_coarse.reshape(2, len(pred_rgb_coarse) // 2, 3)
#        diff_coarse = (pred_rgb_coarse[1] - pred_rgb_coarse[0]) * (ef != 0)
#        event_loss_coarse = \
#            self.rgb_loss(ef / ef.max(), diff_coarse / diff_coarse.max()) + \
#            self.rgb_loss(ef / ef.min(), diff_coarse / diff_coarse.min()) + \
#            self.rgb_loss(ef / (ef.max() - ef.min()), diff_coarse / (diff_coarse.max() - diff_coarse.min()))
#
#        pred_rgb_fine = torch.log(fine_pred)
#        pred_rgb_fine = pred_rgb_fine.reshape(2, len(pred_rgb_fine) // 2, 3)
#        diff_fine = (pred_rgb_fine[1] - pred_rgb_fine[0]) * (ef != 0)
#        event_loss_fine = \
#            self.rgb_loss(ef / ef.max(), diff_fine / diff_fine.max()) + \
#            self.rgb_loss(ef / ef.min(), diff_fine / diff_fine.min()) + \
#            self.rgb_loss(ef / (ef.max() - ef.min()), diff_fine / (diff_fine.max() - diff_fine.min()))
#
#        loss_dict = {
#            "rgb_loss_coarse": rgb_loss_coarse, "rgb_loss_fine": rgb_loss_fine,
#            "event_loss_coarse:": event_loss_coarse, "event_loss_fine": event_loss_fine
#        }
#        loss_dict = misc.scale_dict(loss_dict, self.config.loss_coefficients)
#        #CONSOLE.log(loss_dict["event_loss"])
#        return loss_dict
#    