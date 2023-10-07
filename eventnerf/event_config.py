
from __future__ import annotations

from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.engine.schedulers import (
    ExponentialDecaySchedulerConfig,
)
from nerfstudio.plugins.registry_dataparser import DataParserSpecification
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification
from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig

from eventnerf.event_datamanager import (
    EventDataManagerConfig,
)
from eventnerf.event_model import EventModelConfig
from eventnerf.event_model2 import EventModel2Config
from eventnerf.event_model3 import EventModel3Config
from eventnerf.event_model4 import EventModel4Config
from eventnerf.event_pipeline import EventPipelineConfig
from eventnerf.event_dataparser import EventDataParserConfig

#EventDataparser = DataParserSpecification(config=EventDataParserConfig())

EventConfig = MethodSpecification(
    config=TrainerConfig(
        method_name="eventnerf",  # TODO: rename to your own model
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=30000,
        mixed_precision=True,
        pipeline=EventPipelineConfig(
            datamanager=EventDataManagerConfig(
                dataparser=EventDataParserConfig(),
                #train_num_rays_per_batch=4096,
                train_num_rays_per_batch=2048,
                #eval_num_rays_per_batch=4096,
                eval_num_rays_per_batch=2048,
            ),
            model=EventModelConfig(eval_num_rays_per_chunk=8192),
        ),
        optimizers={
            # TODO: consider changing the optimizers depending on your custom Model
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": None,
            },
            "fields": {
                "optimizer": RAdamOptimizerConfig(lr=4e-2, eps=1e-15),
                #"optimizer": RAdamOptimizerConfig(lr=1e-2, eps=1e-15), #, weight_decay=1e-9),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=200000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 12),
        vis="viewer",
    ),
    description="Event Nerf method.",
)

EventConfig2 = MethodSpecification(
    config=TrainerConfig(
        method_name="eventnerf2",  # TODO: rename to your own model
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=5000,
        mixed_precision=True,
        pipeline=EventPipelineConfig(
            datamanager=EventDataManagerConfig(
                dataparser=EventDataParserConfig(),
                train_num_rays_per_batch=2048,
                #train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=2048,
                camera_optimizer=CameraOptimizerConfig(
                    mode="SO3xR3",
                    #optimizer=RAdamOptimizerConfig(lr=6e-3, eps=1e-8, weight_decay=1e-3),
                    optimizer=RAdamOptimizerConfig(lr=6e-3, eps=1e-8, weight_decay=1e-5),
                    #1optimizer=AdamOptimizerConfig(lr=1e-2, eps=1e-8, weight_decay=1e-5),
                    scheduler=ExponentialDecaySchedulerConfig(lr_final=6e-6, max_steps=200000),
                ),
            ),
            model=EventModel3Config(eval_num_rays_per_chunk=8192),
        ),
        optimizers={
            # TODO: consider changing the optimizers depending on your custom Model
            "fields": {
                #"optimizer": AdamOptimizerConfig(lr=5e-3, eps=1e-15),
                #"optimizer": AdamOptimizerConfig(lr=5e-3, eps=1e-15),
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 12),
        vis="viewer",
    ),
    description="Event Nerf method.",
)

EventConfig3 = MethodSpecification(
    config=TrainerConfig(
        method_name="eventnerf3",  # TODO: rename to your own model
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=5000,
        mixed_precision=True,
        pipeline=EventPipelineConfig(
            datamanager=EventDataManagerConfig(
                dataparser=EventDataParserConfig(),
                train_num_rays_per_batch=2048,
                #train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=2048,
                #camera_optimizer=CameraOptimizerConfig(
                #    mode="SO3xR3",
                #    #optimizer=RAdamOptimizerConfig(lr=6e-3, eps=1e-8, weight_decay=1e-3),
                #    optimizer=RAdamOptimizerConfig(lr=6e-3, eps=1e-8, weight_decay=1e-5),
                #    #1optimizer=AdamOptimizerConfig(lr=1e-2, eps=1e-8, weight_decay=1e-5),
                #    scheduler=ExponentialDecaySchedulerConfig(lr_final=6e-6, max_steps=200000),
                #),
            ),
            model=EventModel3Config(eval_num_rays_per_chunk=8192),
        ),
        optimizers={
            # TODO: consider changing the optimizers depending on your custom Model
            "fields": {
                #"optimizer": AdamOptimizerConfig(lr=5e-3, eps=1e-15),
                "optimizer": AdamOptimizerConfig(lr=3e-3, eps=1e-15),
                #"optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                #"optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-5, max_steps=200000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 12),
        vis="viewer",
    ),
    description="Event Nerf method.",
)

EventConfig4 = MethodSpecification(
    config=TrainerConfig(
        method_name="eventnerf4",  # TODO: rename to your own model
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=5000,
        mixed_precision=True,
        pipeline=EventPipelineConfig(
            datamanager=EventDataManagerConfig(
                dataparser=EventDataParserConfig(),
                train_num_rays_per_batch=2048,
                #train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=2048,
                #camera_optimizer=CameraOptimizerConfig(
                #    mode="SO3xR3",
                #    #optimizer=RAdamOptimizerConfig(lr=6e-3, eps=1e-8, weight_decay=1e-3),
                #    optimizer=RAdamOptimizerConfig(lr=6e-3, eps=1e-8, weight_decay=1e-5),
                #    #1optimizer=AdamOptimizerConfig(lr=1e-2, eps=1e-8, weight_decay=1e-5),
                #    scheduler=ExponentialDecaySchedulerConfig(lr_final=6e-6, max_steps=200000),
                #),
            ),
            model=EventModel4Config(eval_num_rays_per_chunk=8192),
        ),
        optimizers={
            # TODO: consider changing the optimizers depending on your custom Model
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.001, max_steps=200000),
            },
            "fields": {
                #"optimizer": AdamOptimizerConfig(lr=5e-3, eps=1e-15),
                "optimizer": AdamOptimizerConfig(lr=3e-3, eps=1e-15),
                #"optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                #"optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-5, max_steps=200000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 12),
        vis="viewer",
    ),
    description="Event Nerf method.",
)
