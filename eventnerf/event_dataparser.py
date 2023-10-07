
import os
import glob
import json
from abc import abstractclassmethod
import numpy as np
from torch import Tensor
from torchtyping import TensorType
import torch

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Type, List, Any, Literal
from jaxtyping import Float

from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras import camera_utils
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.configs.config_utils import to_immutable_dict
from nerfstudio.data.dataparsers.base_dataparser import DataParser, DataParserConfig, DataparserOutputs
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.data.utils.dataparsers_utils import (
    get_train_eval_split_all,
    get_train_eval_split_filename,
    get_train_eval_split_fraction,
    get_train_eval_split_interval,
)
from nerfstudio.utils.io import load_from_json


@dataclass
class EventDataParserConfig(DataParserConfig):

    _target: Type = field(default_factory=lambda: EventDataParser)
    data: Path = Path("/DATA/wyj/EventNeRF/data/lego1/test1")
    scale_factor: float = 1.0
    #scene_scale: float = 1.0
    scene_scale: float = 0.4
    orientation_menthod: Literal["pca", "up", "vertical", "none"] = "up"
    center_method: Literal["poses", "focus", "none"] = "poses"
    auto_scale_poses: bool = True
    eval_mode: Literal["fraction", "filename", "interval", "all"] = "all"
    train_split_fraction: float = 0.9
    eval_interval: int = 125
    depth_unit_scale_factor: float = 1e-3


@dataclass
class EventDataParser(DataParser):

    config: EventDataParserConfig

    def _generate_dataparser_outputs(self, split="train", **kwargs: Optional[Dict]):
        assert self.config.data.exists(), f"Data directory {self.config.data} does not exist."
        if split != "train":
            split = "test"
        dir_path = self.config.data / f"{split}"
        print(dir_path)

        Height = 260
        Width = 346

        # event files
        enames = self.__find_files(dir_path / "events", ["*.npz"])
        assert len(enames) == 1, "event files not 1"
        metadata = {"event_files": enames}

        # rgb files
        image_filenames = self.__find_files(dir_path / "rgb", ["*.png"])
        fnames = []
        for filepath in image_filenames:
            fnames.append(Path(filepath).name)
        inds = np.argsort(fnames)
        image_filenames = [image_filenames[ind] for ind in inds]
        #CONSOLE.print(image_filenames)

        # pose filies
        pose_files = self.__find_files(dir_path/"pose", ["*.txt"])
        fnames = []
        for filepath in pose_files:
            fnames.append(Path(filepath).name)
        inds = np.argsort(fnames)
        pose_files = [pose_files[ind] for ind in inds]
        #CONSOLE.print(pose_files)
        cam_cnt = len(pose_files)
        poses = []
        for i in range(0, cam_cnt):
            pose = self.__parse_txt(pose_files[i], (4, 4)).reshape(1, 4, 4)
            poses.append(pose)
        poses = torch.from_numpy(np.array(poses).astype(np.float32)).reshape((-1, 4, 4))
        poses[..., 0:3, 1:3] *= -1
        #CONSOLE.print(poses.shape)

        if self.config.eval_mode == "fraction":
            i_train, i_eval = get_train_eval_split_fraction(image_filenames, self.config.train_split_fraction)
        elif self.config.eval_mode == "filename":
            i_train, i_eval = get_train_eval_split_filename(image_filenames)
        elif self.config.eval_mode == "interval":
            i_train, i_eval = get_train_eval_split_interval(image_filenames, self.config.eval_interval)
        elif self.config.eval_mode == "all":
            CONSOLE.log(
                "[yellow] Be careful with '--eval-mode=all'. If using camera optimization, the cameras may diverge in the current implementation, giving unpredictable results."
            )
            i_train, i_eval = get_train_eval_split_all(image_filenames)
        else:
            raise ValueError(f"Unknown eval mode {self.config.eval_mode}")
        
        if split == "train":
            indices = i_train
        else:
            indices = i_eval

        transform_matrix = torch.eye(4)
        #poses, transform_matrix = camera_utils.auto_orient_and_center_poses(
        #    poses, 
        #    method=self.config.orientation_menthod, 
        #    center_method=self.config.center_method,
        #)

        # Scale poses
        scale_factor = 1.0
        if self.config.auto_scale_poses:
            scale_factor /= float(torch.max(torch.abs(poses[:, :3, 3])))
        scale_factor *= self.config.scale_factor
        #CONSOLE.print(scale_factor)

        poses[:, :3, 3] *= scale_factor

        # Chose image_filenames and poses based on split, but after auto orient and scaling the poses.
        image_filenames = [image_filenames[i] for i in indices]
        idx_tensor = torch.tensor(indices, dtype=torch.long)
        poses = poses[idx_tensor]
        #CONSOLE.print(image_filenames)
        #CONSOLE.print(poses)


        # in x,y,z order
        # assumes that the scene is centered at the origin
        aabb_scale = self.config.scene_scale
        scene_box = SceneBox(
            aabb=torch.tensor([[-aabb_scale, -aabb_scale, -aabb_scale], [aabb_scale, aabb_scale, aabb_scale]], dtype=torch.float32)
        )

        # intrinsic files
        intrinsic_files = self.__find_files(dir_path/"intrinsics", ["*.txt"])
        intrinsic = self.__parse_txt(intrinsic_files[0], (-1, 4))
        fx = torch.tensor(intrinsic[0][0])
        fy = torch.tensor(intrinsic[1][1])
        cx = torch.tensor(intrinsic[0][2])
        cy = torch.tensor(intrinsic[1][2])
        distortion_params = torch.zeros(6)
        if intrinsic.shape[0] == 5:
            distortion_params[0] = intrinsic[4][0]
            distortion_params[1] = intrinsic[4][1]

        cameras = Cameras(
            camera_to_worlds=poses[:, :3, :4], 
            fx=fx, fy=fy, cx=cx, cy=cy, 
            distortion_params=distortion_params, 
            height=Height, width=Width,
        )

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
            dataparser_scale=scale_factor,
            dataparser_transform=transform_matrix,
            metadata=metadata,
        )
        return dataparser_outputs

    def __find_files(self, dir, exts):
        if os.path.isdir(dir):
            files_grabbed = []
            for ext in exts:
                files_grabbed.extend(glob.glob(os.path.join(dir, ext)))
            if len(files_grabbed) > 0:
                files_grabbed = sorted(files_grabbed)
            return files_grabbed
        else:
            return []
        
    def __parse_txt(self, filename, shape):
        assert os.path.isfile(filename), "file not exist"
        nums = open(filename).read().split()
        return np.array([float(x) for x in nums], dtype=np.float32).reshape(shape)


if __name__ == "__main__":
    #print(Path("Hello ")/"s")
    config = EventDataParserConfig()
    parser = config.setup()
    outputs = parser.get_dataparser_outputs()
    #print(outputs)