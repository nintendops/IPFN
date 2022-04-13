from torch.utils.data import Dataset
import os
import json
import csv
import torch
import utils.io as io
import numpy as np
import glob
from pathlib import Path
import torchvision.transforms as transforms
from PIL import Image

class SDFDataset(Dataset):
    def __init__(self, opt):
        super(SDFDataset, self).__init__()
        self.opt = opt
        self.channel_dim = 1 if self.opt.model.channel_dim <= 1 else self.opt.model.channel_dim
        self.dataset_mode = opt.run_mode
        self.bs = opt.batch_size
        self.device = opt.device
        self.sdf_files = glob.glob(os.path.join(opt.dataset.path,'*.npy'))

        assert len(self.sdf_files) > 0
        self.sdf_files *= opt.dataset.repeat

        sample = np.load(self.sdf_files[0])
        if sample.ndim == 2:
            res = sample.shape[0]
            grid_res = round(res**(1/3))
            assert grid_res**3 == res
            self.res = grid_res
        elif sample.ndim >= 3:
            self.res = sample.shape[0]
        else:
            raise ValueError(f"Expected sample to have dim 2 or 4, but got {sample.shape}")

        if self.opt.model.crop_res < 1.0:
            sample_interval = int(1/self.opt.model.crop_res)
            sample = sample.reshape(-1,self.res, self.res, self.res)
            sample = sample[:,::sample_interval,::sample_interval,::sample_interval]
            self.original_res = self.res
            self.res = sample.shape[1]
            # self.interval = sample_interval
        else:
            self.original_res = self.res
            self.res = int(self.opt.model.crop_res)

        if self.opt.model.source_scale < 1.0:
            sample_interval = int(1/self.opt.model.source_scale)
            sample = sample.reshape(-1,self.res, self.res, self.res)
            sample = sample[:,::sample_interval,::sample_interval,::sample_interval]
            self.interval = sample_interval
        else:
            self.interval = 1.0

    def _get_original_size(self):
        return [self.original_res]

    def _get_cropped_size(self):
        return self.res
            
    def __len__(self):
        return len(self.sdf_files)

    def __getitem__(self, idx):
        sdf_data = np.load(self.sdf_files[idx])

        if sdf_data.ndim == 2:
            sdfv = sdf_data[:,-1] if self.channel_dim == 1 else sdf_data 
            sdfv = sdfv.reshape(self.channel_dim, self.original_res, self.original_res, self.original_res)
        else:
            # sdfv = sdf_data[:,:,:,-1] if sdf_data.ndim == 4 else sdf_data
            sdfv = sdfv.reshape(self.channel_dim, *sdfv.shape)
        
        sdf_data = torch.from_numpy(sdfv) * self.opt.dataset.sdf_scale
        sdf_data = torch.clamp(sdf_data,-1.0,1.0)
        
        if self.opt.model.source_scale < 1.0:
            sdf_data = sdf_data[:, ::self.interval,::self.interval,::self.interval]

        return self._random_crop(sdf_data), sdf_data, None

    def _random_crop(self,sdf):
        if self.original_res == self.res:
            return sdf

        nz = round(sdf.shape[-1] // self.interval)
        global_res = round(self.original_res // self.interval)
        dx, dy, dz = np.random.randint(0,global_res - self.res - 1,3)

        if self.opt.shift_type == 'xy':
            sdf = sdf[:,dx:dx + self.res, dy:dy+self.res, -self.res:]
        else:
            sdf = sdf[:,dx:dx + self.res, dy:dy+self.res, dz:dz + self.res]

        return sdf


class FoamLoader(Dataset):
    '''
        A particularly specific loader for the foam sdf shape
    '''
    def __init__(self, opt):
        super(FoamLoader, self).__init__()
        self.opt = opt
        self.dataset_mode = opt.run_mode
        self.bs = opt.batch_size
        self.device = opt.device
        self.sdf_file = opt.dataset.path

        # Assume to be [256,256,64,4]
        sdf = np.load(self.sdf_file)
        sdf = sdf.reshape(200,200,128,-1)
        
        self.res = 32
        self.N = 200
        self.sdf = sdf
        self.slice = self.sdf[:,:,90,-1]

        if self.opt.model.source_scale < 1.0:
            interval = int(1/self.opt.model.source_scale)
            self.N = self.N // interval
            self.sdf = self.sdf[::interval, ::interval, ::interval]
            self.slice = self.slice[::interval, ::interval]

    def __len__(self):
        return self.opt.dataset.repeat

    def __getitem__(self, idx):
        '''
            randomly crop the foam slice in x,y direction.
            Cropped volume res preset to be 32,32,32
        '''
        N = self.N
        kernel_size = self.res
        g_min = 5.0  
        g_max = 12.0 

        if self.opt.model.source_scale < 1.0:
            interval = int(1/self.opt.model.source_scale)
            N = N // interval

        x_i, y_i = np.random.randint(0, N-kernel_size,2)
        sdf_slice = self.sdf[x_i:x_i+kernel_size, y_i:y_i+kernel_size, -kernel_size:, -1]
        sdf_slice = torch.from_numpy(sdf_slice)
        sdf_slice = torch.clamp(sdf_slice * self.opt.dataset.sdf_scale, -1.0, 1.0)
        footprint = self.slice[x_i:x_i+kernel_size, y_i:y_i+kernel_size]
        # Expected guidance value range centers at 0.25 and mostly fall between 0.0-0.6
        gv = np.abs(footprint).sum()
        gv = np.clip((gv - g_min) / (g_max - g_min), 0.0, 1.0) 
        return sdf_slice.reshape(1,self.res,self.res,self.res), None, torch.full([1,self.res,self.res,self.res], gv)

