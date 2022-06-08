from torch.utils.data import Dataset
import os
import json
import csv
import torch
import utils.io as io
import numpy as np
from pathlib import Path
import torchvision.transforms as transforms
from PIL import Image
import cv2
import parse 

# def exp_distribution(x, sigma=0.05):
#     return 1 - np.exp(-x**2 / sigma)


class TextureImageDataset(Dataset):

    def __init__(self, opt):
        super(TextureImageDataset, self).__init__()
        
        self.opt = opt
        self.dataset_mode = opt.run_mode
        self.bs = opt.batch_size
        self.device = opt.device
        self.path_dir = Path(opt.dataset.path)
        self.use_single = True # opt.dataset.use_single
        self.samples = self.get_samples()
        self.dataset_length = len(self.samples)
        
        filepath = self.samples[0]
        file = io.load_image(str(Path(filepath)))
        h,w,_ = file.shape
        
        self.default_size = int(min(w,h) * opt.dataset.image_scale)
        self.default_w = int(w * opt.dataset.image_scale)
        self.default_h = int(h * opt.dataset.image_scale)
        
        if opt.model.crop_res <= 1.0:
            self.crop_res = int(opt.model.crop_res * self.default_size)
        else:
            self.crop_res = int(opt.model.crop_res)

        #########################
        # transform_type = "positioned_crop"
        transform_type = "default"
        ########################

        self.initialize_transform(transform_type)

    def __len__(self):
        return self.dataset_length

    def _get_original_size(self):
        return [self.default_w, self.default_h]

    def _get_cropped_size(self):
        return self.crop_res

    def initialize_transform(self, transform_type):
        d = None

        if transform_type == 'simple' or transform_type == 'positioned_crop':            
            self.transforms = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize([self.default_h,self.default_w]),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            self.transforms_crop = transforms.Compose([
                transforms.RandomCrop(self.crop_res),
            ])
        elif transform_type == 'default':
            self.transforms = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize([self.default_h,self.default_w]),
                transforms.RandomCrop(self.crop_res),
                # transforms.RandomHorizontalFlip(p=0.5),
                # transforms.RandomVerticalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])

        self.transform_type = transform_type

    def crop(self, x):
        top = np.random.randint(0,self.default_h - self.crop_res,1)[0] if self.default_h > self.crop_res else 0
        left = np.random.randint(0,self.default_w - self.crop_res,1)[0] if self.default_w > self.crop_res else 0
        x = transforms.functional.crop(x, top, left, self.crop_res, self.crop_res)
        # x = transforms.functional.resize(x, [self.image_res, self.image_res])
        return x, torch.from_numpy(np.array([top,left], dtype=np.float32))

    def get_samples(self):

        if self.path_dir.is_dir():    
            samples = []
            types = ('*.jpeg', '*.jpg', '*.png')

            split = 'test' if self.dataset_mode == 'val' else self.dataset_mode

            if self.use_single or self.dataset_mode == 'test':
                for type in types:
                    samples += list(Path(self.path_dir / 'test').glob('{}'.format(type)))
            else:
                for type in types:
                    samples += list(Path(self.path_dir / split).glob('{}'.format(type)))
            samples.sort()
        else:
            samples = [str(self.path_dir)]

        if self.dataset_mode == 'train':
            samples = samples * self.bs * self.opt.dataset.repeat  # 1000 can be removed if data set is big
        else:
            samples = samples * self.bs

        print("DATASET SAMPLES LOADED. LENGTH: ", len(samples))

        return samples

    def __getitem__(self, idx):
        idx = 0 if self.use_single else idx
        d = torch.from_numpy(np.array([0.0,0.0], dtype=np.float32))
        
        filepath = self.samples[idx]
        # file = io.load_image(str(Path(filepath)),(256,256))
        file = io.load_image(str(Path(filepath)))
        file = io.numpy_to_pytorch(file)
        file = file[:3]

        file_patch = self.transforms(file)
        if self.transform_type == 'positioned_crop':
            file_patch, d = self.crop(file_patch)

        return file_patch, file, d #, Path(filepath).stem


class TerrainImageDataset(Dataset):

    def __init__(self, opt):
        super(TerrainImageDataset, self).__init__()
        
        self.opt = opt
        self.scale_factor = 1 / opt.model.portion
        self.bs = opt.batch_size
        self.dataset_mode = opt.run_mode

        self.device = opt.device

        self.path_dir = Path(opt.dataset.path)
        self.p = opt.model.portion
        self.samples = self.get_samples()
        self.dataset_length = len(self.samples)
        
        filepath = self.samples[0]
        file = io.load_image(str(Path(filepath)))
        h,w,_ = file.shape
        
        self.default_size = int(min(w,h) * opt.dataset.image_scale)
        self.default_w = int(w * opt.dataset.image_scale)
        self.default_h = int(h * opt.dataset.image_scale)
        
        if opt.model.crop_res <= 1.0:
            self.crop_res = int(opt.model.crop_res * self.default_size)
        else:
            self.crop_res = int(opt.model.crop_res)

        self.initialize_transform()
        self.ref_idx = 0

    def __len__(self):
        return self.dataset_length

    def _get_original_size(self):
        return [self.default_w, self.default_h]

    def _get_cropped_size(self):
        return self.crop_res

    def initialize_transform(self):          


        self.transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize([int(self.opt.model.portion * self.default_h),
                               int(self.opt.model.portion * self.default_w)]),
            transforms.RandomCrop(self.crop_res),
            # transforms.CenterCrop(self.crop_res),
            # transforms.RandomHorizontalFlip(p=0.5),
            # transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        self.transforms_original = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize([self.default_h,self.default_w]),
            transforms.RandomCrop(int(self.crop_res * self.scale_factor)),
            # transforms.CenterCrop(int(self.crop_res * self.scale_factor)),
            # transforms.RandomHorizontalFlip(p=0.5),
            # transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        
    def get_samples(self):
        if self.path_dir.is_dir():    
            samples = []
            types = ('*.jpeg', '*.jpg', '*.png',"*.JPG")
            for t in types:
                samples += list(Path(self.path_dir).glob('{}'.format(t)))
            samples.sort()
        else:
            samples = [str(self.path_dir)]

        if self.dataset_mode == 'train':
            samples = samples * self.opt.dataset.repeat  
        else:
            samples = samples * self.opt.dataset.repeat

        print("DATASET SAMPLES LOADED. LENGTH: ", len(samples))
        return samples

    def zero_padding(self, img):
        # assume torch tensor input (3, h, w)
        h, w = img.shape[-2:]
        p = h - int(h * self.p)
        padded_img = torch.cat([img[:,:p], torch.zeros_like(img[:,p:])], 1)
        return padded_img

    def center_cropping(self, img):
        h, w = img.shape[-2:]
        hc = int(self.opt.model.portion * h)
        wc = int(self.opt.model.portion * w)
        return img[:, h//2 - hc//2:h//2 + hc//2, w//2 - wc//2:w//2 + wc//2]

    def _load_img(self, idx, mode='crop'):
        transform = self.transforms if mode == 'crop' else self.transforms_original

        filepath = self.samples[idx]
        file = io.load_image(str(Path(filepath)))
        file = io.numpy_to_pytorch(file)
        file = file[:3]
        
        return transform(file)        

    def __getitem__(self, idx):

        self.ref_idx = idx
        ############################
        # idx = 0
        # self.ref_idx = 1
        ############################
        
        # real_img = self._load_img(idx)
        ref_img = self._load_img(self.ref_idx, mode='original')

        ############################################
        # top = 64
        # left = min(idx * 4, self.default_size - int(self.crop_res * self.scale_factor))
        # x = ref_img
        # x = transforms.functional.crop(x, top, left, int(self.crop_res * self.scale_factor), int(self.crop_res * self.scale_factor))
        # ref_img = x
        ###########################################
        
        # ref_img_padded = self.zero_padding(ref_img)
        ref_img_padded = self.center_cropping(ref_img)

        ######################################
        real_img = transforms.functional.resize(ref_img, (self.crop_res, self.crop_res))
        ####################################
        
        
        return real_img, ref_img, ref_img_padded
