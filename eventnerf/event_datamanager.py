
from dataclasses import dataclass, field
from typing import Dict, Literal, Tuple, Type, Union, Optional, Callable, Any, cast, List, Generic
from functools import cached_property
from pathlib import Path
import numpy as np

import torch
import random
from torch.nn import Parameter
from copy import deepcopy

from nerfstudio.data.utils.dataloaders import (
    CacheDataloader,
    FixedIndicesEvalDataloader,
    RandIndicesEvalDataloader,
)
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.utils.nerfstudio_collate import nerfstudio_collate
from nerfstudio.data.pixel_samplers import PixelSampler
from nerfstudio.data.utils.dataloaders import CacheDataloader
from nerfstudio.data.datamanagers.base_datamanager import (
    DataManager,
    DataManagerConfig,
)
from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.utils.misc import get_orig_class
from nerfstudio.model_components.ray_generators import RayGenerator
from nerfstudio.cameras.cameras import Cameras

from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManager,
    VanillaDataManagerConfig,
)

#from eventnerf.event_stream import EventStream
from eventnerf.event_stream2 import EventStream2
from eventnerf.event_dataparser import (
    EventDataParserConfig,
)


@dataclass
class EventDataManagerConfig(VanillaDataManagerConfig):

    _target: Type = field(default_factory=lambda: EventDataManager)
    dataparser: EventDataParserConfig = EventDataParserConfig()
    downscale_factor = 1.0
    is_colored = True
    #neg_ratio=0.05
    neg_ratio = 0.1  # for nerf data
    max_winsize = 50 


class EventDataManager(VanillaDataManager):

    config: EventDataManagerConfig
    event_stream: EventStream2

    def __init__(
        self,
        config: EventDataManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,
    ):
        super().__init__(config=config, device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank)

    def setup_train(self):
        assert self.train_dataset is not None
        CONSOLE.print("Setting up training dataset...")
        self.event_stream = EventStream2(self.train_dataparser_outputs.metadata["event_files"][0], 
                                         downscale_factor=self.config.downscale_factor, max_winsize=self.config.max_winsize)
        self.train_iter = iter(self.event_stream)
        self.train_pixel_sampler = self._get_pixel_sampler(self.train_dataset, self.config.train_num_rays_per_batch)
        self.train_camera_optimizer = self.config.camera_optimizer.setup(
            num_cameras=self.train_dataset.cameras.size, device=self.device
        )
        CONSOLE.print("cameras.size", self.train_dataset.cameras.size)
        self.train_ray_generator = RayGenerator(
            self.train_dataset.cameras.to(self.device),
            self.train_camera_optimizer,
        )

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the train dataloader."""
        #CONSOLE.print("next_train")
        self.train_count += 1
        batch = next(self.train_iter)

        #if self.train_count % 2 == 0:
        #if self.train_count > 1000 and self.train_count % 2 == 0:
        #    self.config.neg_ratio = 1-self.config.neg_ratio

        #if self.train_count % 2 == 1:
        if False:
            ray_bundle, batch = self.sample1(batch)
        else:
            ray_bundle, batch = self.sample0(batch)

        return ray_bundle, batch

    def sample1(self, batch):
        splits = batch["splits"].int()
        event_frames = batch["event_frames"]

        pos_size = int(self.get_train_rays_per_batch() * (1 - self.config.neg_ratio))
        neg_size = int(self.get_train_rays_per_batch() - pos_size)

        nonzero_indices = batch["nonzero_indices"]
        select_inds_raw = np.random.choice(nonzero_indices.shape[0], size=(pos_size,))
        nonzero_indices = nonzero_indices[select_inds_raw][:, :3]
        #print(nonzero_indices.shape)

        zero_indices = batch["zero_indices"]
        select_inds_raw = np.random.choice(zero_indices.shape[0], size=(neg_size,))
        zero_indices = zero_indices[select_inds_raw]
        #print(zero_indices.shape)

        coords = torch.concat((nonzero_indices[:, 0:3], zero_indices), dim=0)
        #print(coords.shape)

        event_frame_selected = event_frames[coords[:, 0], coords[:, 1], coords[:, 2]]
        #CONSOLE.print(event_frame_selected.shape)
        splits = splits[coords[:, 0]]
        #CONSOLE.print(splits[:, 0].shape, coords.shape)
        ray_indices = torch.concat((torch.concat((splits[:, 0][..., None], coords[:, 1:]), dim=-1), 
                                    torch.concat((splits[:, 1][..., None], coords[:, 1:]), dim=-1)), dim=0).int()
        ray_bundle : RayBundle = self.train_ray_generator(ray_indices)
        win_size = torch.Tensor(splits[:, 1] - splits[:, 0])
        win_size = win_size[..., None]
        batch["win_size"] = win_size
        #print(win_size.shape)
        #print(event_frame_selected.shape)
        batch["event_frame_selected"] = event_frame_selected
        return ray_bundle, batch
        
    
    def sample0(self, batch):
        split = batch["split"].int()
        event_frame = batch["event_frame"]
        if True:
            pos_size = int(self.get_train_rays_per_batch() * (1 - self.config.neg_ratio))
            neg_size = int(self.get_train_rays_per_batch() - pos_size)
            nonzero_indices = torch.nonzero(event_frame.sum(2))
            zero_indices = torch.nonzero(event_frame.sum(2) == 0)
            if pos_size > nonzero_indices.shape[0]:
                pos_size = nonzero_indices.shape[0]
            if neg_size > zero_indices.shape[0]:
                neg_size = zero_indices.shape[0]
            
            chosen_indices = random.sample(range(len(nonzero_indices)), k=pos_size)
            zero_chosen_indices = random.sample(range(len(zero_indices)), k=neg_size)

            #coords = nonzero_indices[chosen_indices][:, :2]
            coords = torch.concat((nonzero_indices[chosen_indices], zero_indices[zero_chosen_indices]), dim=0)
        else:
            nonzero_indices = torch.nonzero(event_frame)
            pos_size = nonzero_indices.shape[0]
            neg_size = 0 #int(p_batch_size * self.config.neg_ratio)
            zero_indices = torch.nonzero(event_frame.sum(2) == 0)
            if neg_size > zero_indices.shape[0]:
                neg_size = zero_indices.shape[0]
            zero_chosen_indices = random.sample(range(len(zero_indices)), k=neg_size)

            coords = torch.concat((nonzero_indices[:, :2], zero_indices[zero_chosen_indices]), dim=0)
        ones = torch.ones((coords.shape[0], 1))
        ray_indices = torch.concat((torch.concat((ones * (split[0] % 1000), coords), dim=-1), 
                                    torch.concat((ones * (split[1] % 1000), coords), dim=-1)), dim=0).int()
        ray_bundle : RayBundle = self.train_ray_generator(ray_indices)
        batch["event_frame_selected"] = event_frame[coords[:, 0], coords[:, 1]]

        return ray_bundle, batch


if __name__ == "__main__":
    print("hello")
    #split = torch.tensor([13, 114]).broadcast_to([1024, 2])
    #print("split", split)
    #print("split", split.shape)
    cfg = EventDataManagerConfig()
    data_manager = cfg.setup()
    #data_manager.sample1()
    ray, batch = data_manager.next_train(1)
    #print (ray)
    #print (batch)
    polarity_offset = 0
    #neg_ratio=0.05
    neg_ratio = 0.1  # for nerf data
    max_winsize = 150 


class EventDataManager(VanillaDataManager):

    config: EventDataManagerConfig
    event_stream: EventStream2

    def __init__(
        self,
        config: EventDataManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,
    ):
        super().__init__(config=config, device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank)

    def setup_train(self):
        assert self.train_dataset is not None
        CONSOLE.print("Setting up training dataset...")
        self.event_stream = EventStream2(self.train_dataparser_outputs.metadata["event_files"][0], 
                                         downscale_factor=self.config.downscale_factor, max_winsize=self.config.max_winsize)
        self.train_iter = iter(self.event_stream)
        self.train_pixel_sampler = self._get_pixel_sampler(self.train_dataset, self.config.train_num_rays_per_batch)
        self.train_camera_optimizer = self.config.camera_optimizer.setup(
            num_cameras=self.train_dataset.cameras.size, device=self.device
        )
        CONSOLE.print("cameras.size", self.train_dataset.cameras.size)
        self.train_ray_generator = RayGenerator(
            self.train_dataset.cameras.to(self.device),
            self.train_camera_optimizer,
        )

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the train dataloader."""
        #CONSOLE.print("next_train")
        self.train_count += 1
        batch = next(self.train_iter)

        #if self.train_count % 2 == 0:
        #if self.train_count > 1000 and self.train_count % 2 == 0:
        #    self.config.neg_ratio = 1-self.config.neg_ratio

        #if self.train_count % 2 == 1:
        if True:
            ray_bundle, batch = self.sample1(batch)
        else:
            ray_bundle, batch = self.sample0(batch)

        return ray_bundle, batch

    def sample1(self, batch):
        splits = batch["splits"].int()
        event_frames = batch["event_frames"]

        pos_size = int(self.get_train_rays_per_batch() * (1 - self.config.neg_ratio))
        neg_size = int(self.get_train_rays_per_batch() - pos_size)

        nonzero_indices = batch["nonzero_indices"]
        select_inds_raw = np.random.choice(nonzero_indices.shape[0], size=(pos_size,))
        nonzero_indices = nonzero_indices[select_inds_raw][:, :3]
        #print(nonzero_indices.shape)

        zero_indices = batch["zero_indices"]
        select_inds_raw = np.random.choice(zero_indices.shape[0], size=(neg_size,))
        zero_indices = zero_indices[select_inds_raw]
        #print(zero_indices.shape)

        coords = torch.concat((nonzero_indices[:, 0:3], zero_indices), dim=0)
        #print(coords.shape)

        event_frame_selected = event_frames[coords[:, 0], coords[:, 1], coords[:, 2]]
        #CONSOLE.print(event_frame_selected.shape)
        splits = splits[coords[:, 0]]
        #CONSOLE.print(splits[:, 0].shape, coords.shape)
        ray_indices = torch.concat((torch.concat((splits[:, 0][..., None], coords[:, 1:]), dim=-1), 
                                    torch.concat((splits[:, 1][..., None], coords[:, 1:]), dim=-1)), dim=0).int()
        ray_bundle : RayBundle = self.train_ray_generator(ray_indices)
        win_size = torch.Tensor(splits[:, 1] - splits[:, 0])
        win_size = win_size[..., None]
        batch["win_size"] = win_size
        #print(win_size.shape)
        #print(event_frame_selected.shape)
        batch["event_frame_selected"] = event_frame_selected
        return ray_bundle, batch
        
    
    def sample0(self, batch):
        split = batch["split"].int()
        event_frame = batch["event_frame"]
        if True:
            p_batch_size = int(self.get_train_rays_per_batch() * (1 - self.config.neg_ratio))
            n_batch_size = int(self.get_train_rays_per_batch() - p_batch_size)
            nonzero_indices = torch.nonzero(event_frame)
            zero_indices = torch.nonzero(event_frame.sum(2) == 0)
            if p_batch_size > nonzero_indices.shape[0]:
                p_batch_size = nonzero_indices.shape[0]
            if n_batch_size > zero_indices.shape[0]:
                n_batch_size = zero_indices.shape[0]
            
            chosen_indices = random.sample(range(len(nonzero_indices)), k=p_batch_size)
            zero_chosen_indices = random.sample(range(len(zero_indices)), k=n_batch_size)

            #coords = nonzero_indices[chosen_indices][:, :2]
            coords = torch.concat((nonzero_indices[chosen_indices][:, :2], zero_indices[zero_chosen_indices]), dim=0)
        else:
            nonzero_indices = torch.nonzero(event_frame)
            p_batch_size = nonzero_indices.shape[0]
            n_batch_size = 0 #int(p_batch_size * self.config.neg_ratio)
            zero_indices = torch.nonzero(event_frame.sum(2) == 0)
            if n_batch_size > zero_indices.shape[0]:
                n_batch_size = zero_indices.shape[0]
            zero_chosen_indices = random.sample(range(len(zero_indices)), k=n_batch_size)

            coords = torch.concat((nonzero_indices[:, :2], zero_indices[zero_chosen_indices]), dim=0)
        ones = torch.ones((coords.shape[0], 1))
        ray_indices = torch.concat((torch.concat((ones * (split[0] % 1000), coords), dim=-1), 
                                    torch.concat((ones * (split[1] % 1000), coords), dim=-1)), dim=0).int()
        ray_bundle : RayBundle = self.train_ray_generator(ray_indices)
        batch["event_frame_selected"] = (event_frame[coords[:, 0], coords[:, 1]] + self.config.polarity_offset) * self.config.event_threshod

        return ray_bundle, batch


if __name__ == "__main__":
    print("hello")
    #split = torch.tensor([13, 114]).broadcast_to([1024, 2])
    #print("split", split)
    #print("split", split.shape)
    cfg = EventDataManagerConfig()
    data_manager = cfg.setup()
    #data_manager.sample1()
    ray, batch = data_manager.next_train(1)
    #print (ray)
    #print (batch)