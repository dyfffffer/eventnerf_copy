
from typing import Literal, Optional, Dict, Tuple, Type
from dataclasses import dataclass, field
from jaxtyping import Float, Shaped

import torch
from torch import Tensor, nn

from nerfstudio.cameras.rays import Frustums, RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.embedding import Embedding
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.fields.nerfacto_field import NerfactoField  # for subclassing NerfactoField
from nerfstudio.field_components.encodings import SHEncoding, NeRFEncoding, HashEncoding
from nerfstudio.field_components.field_heads import (
    FieldHeadNames,
    PredNormalsFieldHead,
    SemanticFieldHead,
    UncertaintyFieldHead, 
    TransientRGBFieldHead, 
    TransientDensityFieldHead,
    RGBFieldHead,
    FieldHead,
    DensityFieldHead,
)
from nerfstudio.field_components.mlp import MLP
from nerfstudio.field_components.encodings import Encoding, Identity
from nerfstudio.field_components.spatial_distortions import SpatialDistortion, SceneContraction
from nerfstudio.fields.base_field import Field, get_normalized_directions  # for custom Field

try:
    import tinycudann as tcnn
except ImportError:
    pass


class EventField(Field):

    def __init__(
        self,
        position_encoding: Encoding = Identity(in_dim=3),
        direction_encoding: Encoding = Identity(in_dim=3),
        base_mlp_num_layers: int = 8,
        base_mlp_layer_width: int = 256,
        head_mlp_num_layers: int = 2,
        head_mlp_layer_width: int = 128,
        skip_connections: Tuple[int] = (4,),
        field_heads: Optional[Tuple[Type[FieldHead]]] = (RGBFieldHead,),
        use_integrated_encoding: bool = False,
        spatial_distortion: Optional[SpatialDistortion] = None,
    ) -> None:
        super().__init__()
        self.position_encoding = position_encoding
        self.direction_encoding = direction_encoding
        self.use_integrated_encoding = use_integrated_encoding
        self.spatial_distortion = spatial_distortion

        self.mlp_base = MLP(
            in_dim=self.position_encoding.get_out_dim(),
            num_layers=base_mlp_num_layers,
            layer_width=base_mlp_layer_width,
            skip_connections=skip_connections,
            out_activation=nn.Tanh(),
        )

        self.mlp_head = MLP(
            in_dim=self.mlp_base.get_out_dim() + self.direction_encoding.get_out_dim(),
            num_layers=head_mlp_num_layers,
            layer_width=head_mlp_layer_width,
            out_activation=nn.Tanh(),
        )

        self.field_output_density = DensityFieldHead(in_dim=self.mlp_base.get_out_dim())
        self.field_heads = nn.ModuleList([field_head() for field_head in field_heads] if field_heads else [])
        for field_head in self.field_heads:
            field_head.set_in_dim(self.mlp_head.get_out_dim())

    def get_density(self, ray_samples: RaySamples) -> Tuple[Tensor, Tensor]:
        if self.use_integrated_encoding:
            gaussian_samples = ray_samples.frustums.get_gaussian_blob()
            if self.spatial_distortion is not None:
                gaussian_samples = self.spatial_distortion(gaussian_samples)
            encoded_xyz = self.position_encoding(gaussian_samples.mean, covs=gaussian_samples.cov)
        else:
            positions = ray_samples.frustums.get_positions()
            if self.spatial_distortion is not None:
                positions = self.spatial_distortion(positions)
            encoded_xyz = self.position_encoding(positions)
        base_mlp_out = self.mlp_base(encoded_xyz)
        density = self.field_output_density(base_mlp_out)
        return density, base_mlp_out
    
    def get_outputs(
        self, ray_samples: RaySamples, density_embedding: Optional[Tensor] = None
    ) -> Dict[FieldHeadNames, Tensor]:
        outputs = {}
        for field_head in self.field_heads:
            encoded_dir = self.direction_encoding(ray_samples.frustums.directions)
            mlp_out = self.mlp_head(torch.cat([encoded_dir, density_embedding], dim=-1))
            outputs[field_head.field_head_name] = field_head(mlp_out)
        return outputs