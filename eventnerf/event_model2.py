

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Type, Union, Any
from jaxtyping import Float

import nerfacc
import torch
from torch.nn import Parameter
from torch import Tensor
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from nerfstudio.configs.config_utils import to_immutable_dict
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.field_components.temporal_distortions import TemporalDistortionKind
from nerfstudio.field_components.encodings import NeRFEncoding
from nerfstudio.fields.nerfacto_field import NerfactoField
from nerfstudio.fields.vanilla_nerf_field import NeRFField
from nerfstudio.model_components.losses import MSELoss
from nerfstudio.model_components.ray_samplers import VolumetricSampler, UniformSampler, PDFSampler
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    RGBRenderer,
)
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.utils import colormaps, colors, misc

from eventnerf.event_field import EventField

@dataclass
class EventModel2Config(ModelConfig):

    _target: Type = field(default_factory=lambda: EventModel2)
    num_coarse_samples: int = 64
    num_importance_samples: int = 128
    background_color: Union[Literal["random", "last_sample", "black", "white"], Float[Tensor, "3"], Float[Tensor, "*bs 3"]] = Tensor([0.6, 0.6, 0.6])
    loss_coefficients: Dict[str, float] = to_immutable_dict({"event_loss_coarse": 1.0, "event_loss_fine": 1.0})
    event_threshold: float = 0.25

class EventModel2(Model):  # based vanilla NeRF model

    config: EventModel2Config

    def __init__(self, config: EventModel2Config, **kwargs) -> None:
        self.field_coarse = None
        self.field_fine = None
        super().__init__(config=config, **kwargs)

    def populate_modules(self):
        """Set the field and modules."""
        super().populate_modules()

        # fields
        position_encoding = NeRFEncoding(
            in_dim=3, num_frequencies=10, min_freq_exp=0.0, max_freq_exp=8.0, include_input=True
        )
        direction_encoding = NeRFEncoding(
            in_dim=3, num_frequencies=4, min_freq_exp=0.0, max_freq_exp=4.0, include_input=True
        )

        self.field_coarse = NeRFField(
            position_encoding=position_encoding,
            direction_encoding=direction_encoding,
        )
        self.field_fine = NeRFField(
            position_encoding=position_encoding,
            direction_encoding=direction_encoding,
        )

        # samplers
        self.sampler_uniform = UniformSampler(num_samples=self.config.num_coarse_samples)
        self.sampler_pdf = PDFSampler(num_samples=self.config.num_importance_samples)

        # renderers
        self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)
        self.renderer_accmulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer()

        # losses
        self.event_loss = MSELoss()

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        if self.field_coarse is None or self.field_fine is None:
            raise ValueError("populate_fields() must be called before get_param_groups")
        param_groups["fields"] = list(self.field_coarse.parameters()) + list(self.field_fine.parameters())
        return param_groups
    
    def get_outputs(self, ray_bundle: RayBundle):
        if self.field_coarse is None or self.field_fine is None:
            raise ValueError("populate_fields() must be called before get_outputs")
        
        # uniform sampling
        ray_samples_uniform = self.sampler_uniform(ray_bundle)

        # coarse field
        field_outputs_coarse = self.field_coarse.forward(ray_samples_uniform)
        weights_coarse = ray_samples_uniform.get_weights(field_outputs_coarse[FieldHeadNames.DENSITY])
        rgb_coarse = self.renderer_rgb(
            rgb=field_outputs_coarse[FieldHeadNames.RGB],
            weights=weights_coarse,
        )
        accumulation_coarse = self.renderer_accmulation(weights_coarse)
        depth_coarse = self.renderer_depth(weights_coarse, ray_samples_uniform)

        # pdf sampling
        ray_samples_pdf = self.sampler_pdf(ray_bundle, ray_samples_uniform, weights_coarse)

        # fine field
        field_outputs_fine = self.field_fine.forward(ray_samples_pdf)
        weights_fine = ray_samples_pdf.get_weights(field_outputs_fine[FieldHeadNames.DENSITY])
        rgb_fine = self.renderer_rgb(
            rgb=field_outputs_fine[FieldHeadNames.RGB],
            weights=weights_fine,
        )
        accumulation_fine = self.renderer_accmulation(weights_fine)
        depth_fine = self.renderer_depth(weights_fine, ray_samples_pdf)

        outputs = {
            "rgb_coarse": rgb_coarse,
            "rgb_fine": rgb_fine,
            "accumulation_coarse": accumulation_coarse,
            "accumulation_fine": accumulation_fine,
            "depth_coarse": depth_coarse,
            "depth_fine": depth_fine,
        }
        return outputs
    
    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        event_frame_selected = batch["event_frame_selected"].to(self.device) * self.config.event_threshold
        pred_rgb_coarse = torch.log((outputs["rgb_coarse"] + 1e-8) * 1e-5)
        pred_rgb_coarse = pred_rgb_coarse.reshape(2, len(pred_rgb_coarse) // 2, 3)
        diff_coarse = (pred_rgb_coarse[1] - pred_rgb_coarse[0]) * (event_frame_selected != 0)
        pred_rgb_fine = torch.log((outputs["rgb_fine"] + 1e-8) * 1e-5)
        pred_rgb_fine = pred_rgb_fine.reshape(2, len(pred_rgb_fine) // 2, 3)
        diff_fine= (pred_rgb_fine[1] - pred_rgb_fine[0]) * (event_frame_selected != 0)
        #print(diff_fine.mean(), event_frame_selected.mean())

        event_loss_coarse = self.event_loss(event_frame_selected, diff_coarse)
        event_loss_fine = self.event_loss(event_frame_selected, diff_fine)

        loss_dict = {"event_loss_coarse": event_loss_coarse, "event_loss_fine": event_loss_fine}
        loss_dict = misc.scale_dict(loss_dict, self.config.loss_coefficients)
        return loss_dict
    
    def get_image_metrics_and_images(self, outputs: Dict[str, Tensor], batch: Dict[str, Tensor]) -> Tuple[Dict[str, float], Dict[str, Tensor]]:
        return {}