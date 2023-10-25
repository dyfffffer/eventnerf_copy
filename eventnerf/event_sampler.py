import numpy as np
from pathlib import Path
from typing import List
import torch
from torch import Tensor
from torch.utils.data.dataset import Dataset
from nerfstudio.utils.rich_utils import CONSOLE
import timeit
import numba

@numba.jit()
def accumulate_events(xs, ys, ts, ps, pos_frames, neg_frames):
    assert len(pos_frames) == len(neg_frames)
    ts_max = ts.max() + 1e-8
    t_slice = ts_max / len(pos_frames)
    #tmp = 0
    for i in range(len(xs)):
        x, y, t, p = xs[i], ys[i], ts[i], ps[i]
        #if tmp == int(t // t_slice):
        #    print (t, t / t_slice)
        #    tmp += 1
        if t >= ts_max:
            break
        if p > 0:
            pos_frames[int(t // t_slice), y, x] += 1
        else:
            neg_frames[int(t // t_slice), y, x] -= 1

def event_split(file_path: Path, cam_cnt=1001, h=260, w=346):
    """split event stream and accumulate events"""
    file_path = Path(file_path)
    events = np.load(file_path)
    xs = np.array(events['x'])
    ys = np.array(events['y'])
    ts = np.array(events['t'])
    ps = np.array(events['p'])
    pos_frames = np.zeros((cam_cnt - 1, h, w))
    neg_frames = np.zeros((cam_cnt - 1, h, w))
    accumulate_events(xs, ys, ts, ps, pos_frames, neg_frames)
    return torch.Tensor(pos_frames), torch.Tensor(neg_frames)

def event_fusion(pos_frames: Tensor, neg_frames, pos_thre, neg_thre, max_winsize=50, device="cuda:0"):
    #print(pos_frames.device, neg_frames.device, pos_thre.device, neg_thre.device)
    fusion_frames = pos_frames * pos_thre + neg_frames * neg_thre
    event_frames = torch.Tensor().to(device)
    splits = torch.Tensor()
    for i in range(max_winsize, len(fusion_frames) + 1): 
        winsize = np.random.randint(1, max_winsize + 1)
        event_frame = torch.sum(fusion_frames[i - winsize : i], 0)
        #print(event_frame)
        event_frames = torch.concat((event_frames, event_frame[None, ...]), dim=0)
        splits = torch.concat((splits, Tensor([[i-winsize, i]])), dim=0)
    return event_frames, splits

def shuffle_array(arr):
    """ Fisher-Yates shuffle"""
    n = len(arr)
    for i in range(n - 1, 0, -1):
        j = np.random.randint(0, i)
        arr[i], arr[j] = arr[j], arr[i]
    return arr

class EventSampler:
    """Event sampler from accumulation frames"""

    def __init__(self, file_path: Path, pos_thre: Tensor, neg_thre: Tensor, max_winsize=50, cam_cnt=1001, h=260, w=346, batch_size=1024, device="cuda:0"):
        if file_path == "":  # only test
            self.pos_frames, self.neg_frames = 2 * torch.ones((cam_cnt-1, h, w)), -1 * torch.ones((cam_cnt-1, h, w))
        else:
            self.pos_frames, self.neg_frames = event_split(file_path=file_path, cam_cnt=cam_cnt, h=h, w=w)
        #print(self.pos_frames.shape, self.neg_frames.shape)
        self.pos_thre = pos_thre
        self.neg_thre = neg_thre
        self.max_winsize = max_winsize
        self.cam_cnt = cam_cnt
        self.h = h
        self.w = w
        self.batch_size = batch_size
        self.device = device
        self.pos_frames = self.pos_frames.to(device)
        self.neg_frames = self.neg_frames.to(device)
        self.color_mask = torch.zeros((self.h, self.w, 3))
        if True:
            self.color_mask[0::2, 0::2, 0] = 1 # R
            self.color_mask[0::2, 1::2, 1] = 1 # G
            self.color_mask[1::2, 0::2, 1] = 1 # G
            self.color_mask[1::2, 1::2, 2] = 1 # B
        else:
            self.color_mask[...] = 1
        #print(self.pos_frames.device, self.neg_frames.device)

    def __iter__(self):
        self.event_frames, self.splits = event_fusion(self.pos_frames, self.neg_frames, self.pos_thre, self.neg_thre, self.max_winsize)
        self.nonzero_indices_3d = torch.nonzero(self.event_frames)
        self.count = 0
        self.frames_order = shuffle_array(list(range(len(self.event_frames))))
        self.frames_order_idx = 0
        self.nonzero_indices_2d = torch.nonzero(self.event_frames[self.frames_order[self.frames_order_idx]])
        #print(self.frame_nonzero_indices.shape)
        #print(self.frames_order)
        #print(self.event_frames)
        #print(self.splits)
        #print(self.nonzero_indices.shape)
        #print(self.nonzero_indices)
        return self
    
    def sample_ordered(self, batch_size):
        if self.count >= len(self.nonzero_indices_3d):
            self.__iter__()
        coords_3d = self.nonzero_indices_3d[self.count : self.count + batch_size].to("cpu")
        event_frame_selected = self.event_frames[coords_3d[:, 0], coords_3d[:, 1], coords_3d[:, 2]]
        splits = self.splits[coords_3d[:, 0]]
        #print(splits[:, 0][..., None].shape)
        #print(coords[:, 1:].shape)
        ray_indices = torch.concat((torch.concat((splits[:, 0][..., None], coords_3d[:, 1:]), dim=-1),
                                    torch.concat((splits[:, 1][..., None], coords_3d[:, 1:]), dim=-1)), dim=0).int()
        color_mask = self.color_mask[coords_3d[:, 1], coords_3d[:, 2]]
        self.count += batch_size

        #if len(event_frame_selected) < batch_size:
        #    ray_indices2, event_frame_selected2, color_mask2 = self.get_event_frame_selected(batch_size - len(event_frame_selected))
        #    event_frame_selected = torch.concat((event_frame_selected, event_frame_selected2))
        #    ray_indices = torch.concat((ray_indices, ray_indices2))
        #    color_mask = torch.concat((color_mask, color_mask2))
        return ray_indices, event_frame_selected, color_mask
    
    def sample_random_frames(self, batch_size):
        if self.frames_order_idx >= len(self.frames_order):
            self.__iter__()
        frame_idx = self.frames_order[self.frames_order_idx]
        if self.count == 0:
            self.nonzero_indices_2d = torch.nonzero(self.event_frames[frame_idx])
        coords_2d = self.nonzero_indices_2d[self.count: self.count + batch_size].to("cpu")
        event_frame_selected = self.event_frames[frame_idx, coords_2d[:, 0], coords_2d[:, 1]]
        splits = self.splits[frame_idx][None, ...].tile(len(coords_2d), 1)
        #print(event_frame_selected.shape)
        #print(splits.shape)
        #print(coords_2d.shape)
        ray_indices = torch.concat((torch.concat((splits[:, 0][..., None], coords_2d), dim=-1),
                                    torch.concat((splits[:, 1][..., None], coords_2d), dim=-1)), dim=0).int()
        #print(ray_indices.shape)
        color_mask = self.color_mask[coords_2d[:, 0], coords_2d[:, 1]]
        self.count += batch_size
        if self.count >= len(self.nonzero_indices_2d):
            self.frames_order_idx += 1
            self.count = 0
        return ray_indices, event_frame_selected, color_mask
    
    def sample_random_3d(self, batch_size):
        if self.count >= len(self.nonzero_indices_3d):
            self.__iter__()
        selected_indices = np.random.choice(self.nonzero_indices_3d.shape[0], size=(batch_size, ))
        coords_3d = self.nonzero_indices_3d[selected_indices].to("cpu")
        #print(coords_3d.shape)
        #print(coords_3d)
        event_frame_selected = self.event_frames[coords_3d[:, 0], coords_3d[:, 1], coords_3d[:, 2]]
        splits = self.splits[coords_3d[:, 0]]
        #print(splits[:, 0][..., None].shape)
        #print(coords[:, 1:].shape)
        ray_indices = torch.concat((torch.concat((splits[:, 0][..., None], coords_3d[:, 1:]), dim=-1),
                                    torch.concat((splits[:, 1][..., None], coords_3d[:, 1:]), dim=-1)), dim=0).int()
        color_mask = self.color_mask[coords_3d[:, 1], coords_3d[:, 2]]
        self.count += batch_size
        return ray_indices, event_frame_selected, color_mask
    
    def __next__(self):
        ray_indices, event_frame, color_mask = self.sample_ordered(self.batch_size)
        #ray_indices, event_frame, color_mask = self.sample_random_frames(self.batch_size)
        #ray_indices, event_frame, color_mask = self.sample_random_3d(self.batch_size)
        #print(ray_indices.shape)
        #print(event_frame.shape)
        event_frame = event_frame[..., None].tile(1, 3) * color_mask
        batch = {"event_frame_selected" : event_frame,
                 "color_mask" : color_mask}
        return ray_indices, batch

if __name__ == "__main__":
    #pos, neg = event_split("/DATA/wyj/EventNeRF/data/lego1/test1/train/events/test_lego1_color.npz")
    #event_sampler = EventSampler("/DATA/wyj/EventNeRF/data/nextnextgen/bottle/train/events/worgb-2022_11_16_15_46_53.npz", pos_thre, neg_thre, cam_cnt, h, w)
    cam_cnt, h, w = 4, 2, 2
    pos_thre = torch.ones((cam_cnt - 1, h, w), device="cuda:0")
    neg_thre = torch.ones((cam_cnt - 1, h, w), device="cuda:0")
    event_sampler = EventSampler("", pos_thre=pos_thre, neg_thre=neg_thre, cam_cnt=cam_cnt, h=h, w=w, max_winsize=2, batch_size=2)
    event_iter = iter(event_sampler)
    batch = next(event_iter)
    batch = next(event_iter)
    #print(pos[0, :10, :10])
    #print(pos[50, 120:130, 170:180])
    #print(neg[0, :10, :10])

