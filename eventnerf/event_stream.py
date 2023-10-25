import numpy as np
from pathlib import Path
from typing import List
import torch
from torch import nn
from torch import Tensor
from torch.utils.data.dataset import Dataset
from nerfstudio.utils.rich_utils import CONSOLE
import timeit

class EventStream:

    def __init__(self, event_path: Path, cmr_cnt = 1001, h=260, w=346, is_color=True, downscale_factor=1):
        self.event_path = Path(event_path)
        #print(self.event_path.name + ".npz")
        evs = np.load(self.event_path)
        self.xs = np.array(evs['x'])
        self.ys = np.array(evs['y'])
        self.ps = np.array(evs['p'])
        self.ts = np.array(evs['t'])
        self.cmr_cnt = cmr_cnt
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

        t_start = timeit.default_timer()
        try:
            data = np.load(self.event_path.name + ".npz")
            #print(data.files)
            self.acc_maps = Tensor(data["acc_maps"])
            #self.acc_masks = Tensor(data["acc_masks"]).int()
        except OSError:
            acc_maps = torch.zeros((cmr_cnt, self.h, self.w, 3), dtype=torch.int32) 
            t_slice = self.ts.max() / (cmr_cnt - 1)
            for i in range(len(self.xs)):
                x, y, p, t = self.xs[i], self.ys[i], self.ps[i], self.ts[i]
                acc_maps[int(t // t_slice), y, x] += p
            #acc_maps = torch.concat([acc_maps, acc_maps], dim=0)
            #print(acc_maps.shape)
            acc_maps = torch.cumsum(acc_maps, dim=0)
            acc_maps = acc_maps * self.color_mask
            self.acc_maps  = acc_maps
            np.savez(self.event_path.name + ".npz", acc_maps = acc_maps.numpy())
        print("init event time:", timeit.default_timer() - t_start)
        if downscale_factor != 1:
            self.h /= downscale_factor
            self.w /= downscale_factor
            self.acc_maps = self.acc_maps.permute(0, 3, 1, 2)
            self.acc_maps = torch.nn.functional.conv2d(input=self.acc_maps, weight=torch.ones((3, 1, 2, 2))/2, stride=2, groups=3).permute(0, 2, 3, 1)
            CONSOLE.log(self.acc_maps.shape)

    def __len__(self):
        return self.cmr_cnt

    def __iter__(self):
        return self
    
    def __next__(self):
        #split = (self.__random_sample2(min_width=0.05, max_width=0.35)*1000).int()
        split = (self.__random_sample(min_width=0.002, max_width=0.050)*(self.cmr_cnt)).int()[0]
        if split[0] == 0:
            acc_map = self.acc_maps[split[1] - 1]
        else:
            acc_map = self.acc_maps[split[1] - 1] - self.acc_maps[split[0] - 1]
        batch = {"acc_map": acc_map,
                 "split" : split}
        return batch

    def __random_sample2(self, min_width: float=.01, max_width: float=.15, decimals=3):
        splits = torch.rand(2).numpy()
        splits = np.round(splits, decimals=decimals)
        #print(splits)
        width = splits[1] * (max_width - min_width) + min_width
        p_start = 1.0 * splits[0]
        splits[0] = p_start
        splits[1] = p_start + width
        #print(splits)
        return Tensor(splits)

    def __random_sample(self, split_num: int=1, min_width: float=.01, max_width: float=.15, decimals=3):
        """
        split_num: 事件流分割的数量, split_num > 0
        offset: 事件流分割的最小间隔,  split_end - split_start >= offset
        return:
          Tensor[split_num * 2], 0 <= value <= 1. + max_width, example: [split_start0, split_end0, split_start1, split_end1, ...]
        """
        assert(split_num >= 0)
        start = timeit.default_timer()
        splits = torch.rand(int(2 * split_num)).numpy()
        min_width /= 2
        for i in range(len(splits) // 2):
            l, r = 2 * i, 2 * i + 1
            if splits[l] > splits[r]:
                splits[l], splits[r] = splits[r], splits[l]
            splits[l] -= min_width
            splits[r] += min_width
            if splits[r] - splits[l] > max_width:
                splits[r] = splits[l] + max_width
            if splits[l] < 0.:
                splits[r] -= splits[l]
                splits[l] = 0.
            if splits[r] > 1.: 
                splits[l] = splits[l] - (splits[r] - 1)
                splits[r] = 1.
        #splits = np.round(splits, decimals=decimals).reshape(-1, 2)
        #print("random sample time:", timeit.default_timer() - start)
        return Tensor(splits).reshape(-1, 2)

if __name__ == "__main__":
    es = EventStream("/DATA/wyj/EventNeRF/data/lego1/test1/train/events/test_lego1_color.npz")
    #print(es.acc_maps[0, 0, 0])
    #print(es.acc_maps[0, 0, 1])
    #print(es.acc_maps[0, 1, 0])
    #print(es.acc_maps[0, 1, 1])
    #es.__random_accumate_event(2000)
    myiter = iter(es)
    batch = next(myiter)
    print(batch)
    #split = batch['split']
    #nonind = torch.nonzero(acc_map)[0]
    #print(nonind)
    #print(acc_map[nonind[0], nonind[1]])
    ##X = torch.arange(1, 10).reshape(3, 3)
    ##K = torch.ones((2, 2))
    ##print(X)
    ##print(K)
    ##print(coorr2d(X, K))
    ##mask = Tensor([[1, 0], [0, 0]])
    ##data = torch.arange(12).reshape(3, 2, 2)
    ##print(data * mask)
    #CONSOLE.print("hello")
    #CONSOLE.log("hello")
    #n = torch.ones((1,1024,1024,3))
    #n = n.permute(0,3,1,2)
    #r = torch.nn.functional.conv2d(input=n,weight=torch.ones((3,1,2,2)),stride=2,groups=3)
    #r = r.permute(0,2,3,1)
    #print(r.shape)
    #print(r[0, :3, :3])
    #r = torch.nn.functional.avg_pool2d(n,kernel_size=2, stride=2, divisor_override=1)
