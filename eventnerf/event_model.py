
from dataclasses import dataclass, field
from typing import Type

import nerfacc
import torch
from torch.nn import Parameter
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

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
from nerfstudio.utils import colormaps
from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig  # for subclassing Nerfacto
from nerfstudio.models.vanilla_nerf import NeRFModel, VanillaModelConfig
from nerfstudio.models.instant_ngp import InstantNGPModelConfig, NGPModel
from nerfstudio.models.base_model import Model, ModelConfig  # for custom Model
from nerfstudio.model_components.renderers import RGBRenderer
from nerfstudio.utils.rich_utils import CONSOLE

from eventnerf.event_field import EventField


@dataclass
class EventModelConfig(NerfactoModelConfig):

    _target: Type = field(default_factory=lambda: EventModel)
    background_color = "white"
    cone_angle = 0.0
    grid_resolution = 128

    #disable_scene_contraction = True
    #alpha_thre = 0.8


class EventModel(NerfactoModel):

    config: EventModelConfig

    def populate_modules(self):
        super().populate_modules()

    def get_loss_dict(self, outputs, batch, metric_dict=None):
        loss_dict = {}
        acc_map_selected = batch["acc_map_selected"].to(self.device)
        if True:
            pred_rgb = torch.log(outputs["rgb"]**2.2 + 1e-8)
            pred_rgb = pred_rgb.reshape(2, len(pred_rgb) // 2, 3)
            diff = (pred_rgb[1] - pred_rgb[0]) * (acc_map_selected != 0)
            loss_dict["event_loss"] = self.rgb_loss(acc_map_selected, diff)
        else:
            pred_rgb_coarse = torch.log(outputs["rgb_coarse"]**2.2 + 1e-5)
            pred_rgb_coarse = pred_rgb_coarse.reshape(2, len(pred_rgb_coarse) // 2, 3)
            diff_coarse = (pred_rgb_coarse[1] - pred_rgb_coarse[0])
            pred_rgb_fine = torch.log(outputs["rgb_fine"]**2.2 + 1e-5)
            pred_rgb_fine = pred_rgb_fine.reshape(2, len(pred_rgb_fine) // 2, 3)
            diff_fine = (pred_rgb_fine[1] - pred_rgb_fine[0])
            loss_dict["event_loss_coarse"] = self.rgb_loss(acc_map_selected, diff_coarse)
            loss_dict["event_loss_fine"] = self.rgb_loss(acc_map_selected, diff_fine)

        #CONSOLE.print(diff - acc_map_selected)
        #CONSOLE.log(loss_dict["event_loss"])
        return loss_dict
    
    def get_metrics_dict(self, outputs, batch):
        return {}