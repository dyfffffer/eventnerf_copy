

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
    render_step_size: Optional[float] = None
    grid_resolution: int = 128
    grid_levels: int = 1
    near_plane: float = 0.05
    far_plane: float = 2 #1e3
    alpha_thre: float = 1e-3
    cone_angle: float = 0 #0.004

class EventModel2(Model):  # based vanilla NeRF model

    config: EventModel2Config

    def __init__(self, config: EventModel2Config, **kwargs) -> None:
        self.field_coarse = None
        self.field = None
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

        self.field = NeRFField(
            position_encoding=position_encoding,
            direction_encoding=direction_encoding,
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
        self.renderer_accmulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer()

        # losses
        self.event_loss = MSELoss()
        self.rgb_loss = MSELoss()

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
        weights = weights[..., None]

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
        #return {}
        image = batch["image"]#.to(self.deivce)
        image = self.renderer_rgb.blend_background(image)
        metrics_dict = {}
        metrics_dict["psnr"] = self.psnr(outputs["rgb"], image)
        #metrics_dict["num_samples_per_batch"] = outputs["num_samples_per_ray"].sum()
        return metrics_dict
    
    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = {}

        # rgb loss
        loss_dict["rgb_loss"] = self.rgb_loss(outputs["rgb"], batch["image"])

        # event loss
        event_frame_selected = batch["event_frame_selected"].to(self.device) * self.config.event_threshold
        pred_rgb = torch.log((outputs["rgb"] + 1e-8) * 1e-5)
        pred_rgb = pred_rgb.reshape(2, len(pred_rgb) // 2, 3)
        diff = (pred_rgb[1] - pred_rgb[0]) * (event_frame_selected != 0)  # mask color
        #print(diff.mean(), event_frame_selected.mean())
        loss_dict["event_loss"] = self.event_loss(event_frame_selected, diff)

        #linlog_thres = 20 / 255
        #lin_slope = np.log(linlog_thres) / linlog_thres
        #color = outputs["rgb"]# ** 2.2
        #color = color.reshape(2, len(color) // 2, 3)
        #lin_log_rgb = torch.where(color < linlog_thres, lin_slope * color, torch.log(color))
        #diff = (lin_log_rgb[1] - lin_log_rgb[0]) * (event_frame_selected != 0)  # mask color
        #loss_dict["event_loss"] = self.event_loss(event_frame_selected, diff)

        lss_dict = misc.scale_dict(loss_dict, self.config.loss_coefficients)
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
