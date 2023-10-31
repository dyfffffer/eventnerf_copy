
from dataclasses import dataclass, field
from typing import Type

import nerfacc
import torch
from torch.nn import Parameter
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from typing import Any, Dict

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
class EventModel5Config(InstantNGPModelConfig):

    _target: Type = field(default_factory=lambda: EventModel5)
    background_color = torch.Tensor([0.624, 0.624, 0.624])
    loss_coefficients: Dict[str, float] = to_immutable_dict({
        "rgb_loss": 0.0, "event_loss": 1.0,
    })

class EventModel5(NGPModel):

    config: EventModel5Config

    def populate_modules(self):
        super().populate_modules()

    def get_loss_dict(self, outputs, batch, metric_dict=None):
        image = batch["image"][..., :3].to(self.device)
        pred_rgb, image = self.renderer_rgb.blend_background_for_loss_computation(
            pred_image=outputs["rgb"],
            pred_accumulation=outputs["accumulation"],
            gt_image=image,
        )
        rgb_loss = self.rgb_loss(image, pred_rgb)

        ef = batch["event_frame_selected"].to(self.device)
        pred_rgb = pred_rgb.reshape(2, len(pred_rgb) // 2, 3)
        diff = (pred_rgb[1] - pred_rgb[0]) * (ef != 0)
        event_loss = \
            self.rgb_loss(ef / ef.max(), diff / diff.max()) + \
            self.rgb_loss(ef / ef.min(), diff / diff.min()) + \
            self.rgb_loss(ef / (ef.max() - ef.min()), diff / (diff.max() - diff.min()))

        loss_dict = {"event_loss:": event_loss, "rgb_loss": rgb_loss}
        loss_dict = misc.scale_dict(loss_dict, self.config.loss_coefficients)
        return loss_dict
    