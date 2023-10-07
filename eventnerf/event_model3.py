

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
from nerfstudio.model_components.losses import MSELoss
from nerfstudio.model_components.ray_samplers import VolumetricSampler, UniformSampler, PDFSampler, LogSampler, NeuSSampler, ProposalNetworkSampler
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    RGBRenderer,
)
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.utils import colormaps, colors, misc

from eventnerf.event_field2 import EventField2

@dataclass
class EventModel3Config(ModelConfig):
    """ref: instantNGP"""

    _target: Type = field(default_factory=lambda: EventModel3)
    num_layers: int = 2
    hidden_dim: int = 64
    num_layers_color: int = 3
    hidden_dim_color: int = 64

    num_levels:int = 64 #32 #16
    features_per_level: int = 2
    grid_resolution: int = 128
    grid_levels: int = 1
    max_res: int = 2048
    log2_hashmap_size: int = 19
    alpha_thre: float = 0.0 #1e-5
    cone_angle: float = 0 #0.004

    render_step_size: Optional[float] = None
    near_plane: float = 0.05
    far_plane: float = 1e3
    enable_collider: bool = False
    collider_params: Optional[Dict[str, float]] = None
    background_color: Union[Literal["random", "last_sample", "black", "white"], Float[Tensor, "3"], Float[Tensor, "*bs 3"]] = Tensor([0.624, 0.624, 0.624])
    #background_color: Union[Literal["random", "last_sample", "black", "white"], Float[Tensor, "3"], Float[Tensor, "*bs 3"]] = "random"
    disable_scene_contraction: bool = True 


class EventModel3(Model):  # based vanilla NeRF model

    config: EventModel3Config

    def __init__(self, config: EventModel3Config, **kwargs) -> None:
        super().__init__(config=config, **kwargs)

    def populate_modules(self):
        super().populate_modules()

        if self.config.disable_scene_contraction:
            scene_contraction = None
        else:
            scene_contraction = SceneContraction(order=float("inf"))

        self.field = EventField2(
            aabb=self.scene_box.aabb,
            num_layers=self.config.num_layers,
            hidden_dim=self.config.hidden_dim,
            num_levels=self.config.num_levels,
            num_layers_color=self.config.num_layers_color,
            hidden_dim_color=self.config.hidden_dim_color,
            features_per_level=self.config.features_per_level,
            log2_hashmap_size=self.config.log2_hashmap_size,
            max_res=self.config.max_res,
            spatial_distortion=scene_contraction,
        )

        self.scene_aabb = Parameter(self.scene_box.aabb.flatten(), requires_grad=False)

        if self.config.render_step_size is None:
            # auto step size: ~1000 samples in the base level grid
            self.config.render_step_size = ((self.scene_aabb[3:] - self.scene_aabb[:3]) **2).sum().sqrt().item() / 1000
        # Occupancy Grid.
        self.occupancy_grid = nerfacc.OccGridEstimator(
            roi_aabb=self.scene_aabb,
            resolution=self.config.grid_resolution,
            levels=self.config.grid_levels,
        )

        # sampler
        self.sampler = VolumetricSampler(
            occupancy_grid=self.occupancy_grid,
            density_fn=self.field.density_fn,
        )

        # renderers
        self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer(method="expected")

        # losses
        self.event_loss = MSELoss()

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)

    def get_training_callbacks(self, training_callback_attributes: TrainingCallbackAttributes) -> List[TrainingCallback]:
        def update_occupancy_grid(step: int):
            self.occupancy_grid.update_every_n_steps(
                step=step,
                occ_eval_fn=lambda x: self.field.density_fn(x) * self.config.render_step_size,
            )
        return [
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                update_every_num_iters=1,
                func=update_occupancy_grid,
            ),
        ]
    
    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        if self.field is None:
            raise ValueError("populate_fields() must be called before get_param_groups")
        param_groups["fields"] = list(self.field.parameters())
        return param_groups
    
    def get_outputs(self, ray_bundle: RayBundle) -> Dict[str, Tensor | List]:
        assert self.field is not None
        num_rays = len(ray_bundle)

        with torch.no_grad():
            ray_samples, ray_indices = self.sampler(
                ray_bundle=ray_bundle,
                near_plane=self.config.near_plane,
                far_plane=self.config.far_plane,
                render_step_size=self.config.render_step_size,
                alpha_thre=self.config.alpha_thre,
                cone_angle=self.config.cone_angle,
            )

        field_outputs = self.field(ray_samples)

        # accumulation
        packed_info = nerfacc.pack_info(ray_indices, num_rays)
        weights = nerfacc.render_weight_from_density(
            t_starts=ray_samples.frustums.starts[..., 0],
            t_ends=ray_samples.frustums.ends[..., 0],
            sigmas=field_outputs[FieldHeadNames.DENSITY][..., 0],
            packed_info=packed_info,
        )[0]
        weights=weights[..., None]

        rgb = self.renderer_rgb(
            rgb=field_outputs[FieldHeadNames.RGB],
            weights=weights,
            ray_indices=ray_indices,
            num_rays=num_rays,
        )
        depth = self.renderer_depth(
            weights=weights, ray_samples=ray_samples, ray_indices=ray_indices, num_rays=num_rays,
        )
        accumulation = self.renderer_accumulation(weights=weights, ray_indices=ray_indices, num_rays=num_rays)

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
            #"num_samples_per_ray": packed_info[:, 1],
        }
        return outputs
    
    def get_metrics_dict(self, outputs, batch):
        return {}
        image = batch["image"].to(self.deivce)
        image = self.renderer_rgb.blend_background(image)
        metrics_dict = {}
        metrics_dict["psnr"] = self.psnr(outputs["rgb"], image)
        metrics_dict["num_samples_per_batch"] = outputs["num_samples_per_ray"].sum()
        return metrics_dict
    
    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = {}
        event_frame_selected = batch["event_frame_selected"].to(self.device)
        pred_rgb = torch.log(outputs["rgb"]**2.2 + 1e-8)
        pred_rgb = pred_rgb.reshape(2, len(pred_rgb) // 2, 3)
        diff = (pred_rgb[1] - pred_rgb[0]) * (event_frame_selected != 0)
        loss_dict["event_loss"] = self.event_loss(event_frame_selected, diff)
        return loss_dict
    
    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        image = batch["image"].to(self.device)
        image = self.renderer_rgb.blend_background(image)
        rgb = outputs["rgb"]
        acc = colormaps.apply_colormap(outputs["accumulation"])
        depth = colormaps.apply_depth_colormap(
            outputs["depth"],
            accumulation=outputs["accumulation"],
        )

        combined_rgb = torch.cat([image, rgb], dim=1)
        combined_acc = torch.cat([acc], dim=1)
        combined_depth = torch.cat([depth], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb = torch.moveaxis(rgb, -1, 0)[None, ...]

        psnr = self.psnr(image, rgb)
        ssim = self.ssim(image, rgb)
        lpips = self.lpips(image, rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim), "lpips": float(lpips)}
        # TODO(ethan): return an image dictionary

        image_dict = {
            "img": combined_rgb,
            "accumulation": combined_acc,
            "depth": combined_depth,
        }
        
        return metrics_dict, image_dict
