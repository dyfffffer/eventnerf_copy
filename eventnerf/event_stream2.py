import numpy as np
from pathlib import Path
from typing import List
import torch
from torch import nn
from torch import Tensor
from torch.utils.data.dataset import Dataset
from nerfstudio.utils.rich_utils import CONSOLE
import timeit
import numba

@numba.jit()
def accumulate_events(xs, ys, ts, ps, out):
    for i in range(len(xs)):
        x, y, t, p = xs[i], ys[i], ts[i], ps[i]
        out[y, x] += p

class EventStream2:

    def __init__(self, event_path: Path, cam_cnt = 1001, h=260, w=346, is_color=True, max_winsize=150, **kwargs):
        self.max_winsize = max_winsize
        self.event_path = Path(event_path)
        evs = np.load(self.event_path)
        self.xs = np.array(evs['x'])
        self.ys = np.array(evs['y'])
        self.ps = np.array(evs['p'])
        self.ts = np.array(evs['t'])
        self.cam_cnt = cam_cnt
        #print("event last time:", self.ts[-10:-1])
        self.h = h
        self.w = w
        self.is_color = is_color
        self.color_mask = torch.zeros((self.h, self.w, 3))
        if self.is_color:
            self.color_mask[0::2, 0::2, 0] = 1 # R
            self.color_mask[0::2, 1::2, 1] = 1 # G
            self.color_mask[1::2, 0::2, 1] = 1 # G
            self.color_mask[1::2, 1::2, 2] = 1 # B
        else:
            self.color_mask[...] = 1
        
    def __len__(self):
        return self.cam_cnt

    def __iter__(self):
        event_frames = []
        splits = []
        for i in range(1+self.max_winsize, self.cam_cnt):
            winsize = np.random.randint(1, self.max_winsize + 1)
            start_time = (i - winsize)/(self.cam_cnt-1)
            end_time = (i)/(self.cam_cnt-1)

            ts = self.ts
            xs = self.xs
            ys = self.ys
            ps = self.ps
            start = np.searchsorted(ts, start_time*ts.max(), side="right") + 1
            end = np.searchsorted(ts, end_time*ts.max())
        
            if start <= end:
                xs, ys, ts, ps = xs[start:end], ys[start:end], ts[start:end], ps[start:end]
            else:
                xs, ys, ts, ps = np.concatenate((xs[start:], xs[:end])), \
                    np.concatenate((ys[start:], ys[:end])), np.concatenate((ts[start:], ts[:end])), np.concatenate((ps[start:], ps[:end]))

            event_frame = np.zeros((self.h, self.w))
            accumulate_events(xs, ys, ts, ps, event_frame)
            event_frame = np.tile(event_frame[..., None], (1, 1, 3))
            event_frames.append(event_frame)
            splits.append([i-winsize, i])
        self.event_frames = torch.from_numpy(np.array(event_frames, dtype=np.float32)) * self.color_mask
        self.splits = torch.from_numpy(np.array(splits, dtype=np.float32))
        print(self.event_frames.shape)
        print(self.splits.shape)
        self.nonzero_indices = torch.nonzero(self.event_frames)
        self.zero_indices = torch.nonzero(self.event_frames.sum(3) == 0)
        print(self.zero_indices.shape)

        return self
    
    def __next__(self):

        i = np.random.randint(0, len(self.event_frames))
        batch = {"event_frame": self.event_frames[i],
                 "split" : self.splits[i],
                 "event_frames": self.event_frames,
                 "splits": self.splits,
                 "nonzero_indices": self.nonzero_indices,
                 "zero_indices": self.zero_indices,
                 }
        return batch

if __name__ == "__main__":
    es = EventStream2("/DATA/wyj/EventNeRF/data/lego1/test1/train/events/test_lego1_color.npz")
    myiter = iter(es)
    batch = next(myiter)
    print(batch)
    batch = next(myiter)
