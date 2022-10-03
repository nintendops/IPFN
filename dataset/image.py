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

        transform_type = "positioned_crop"
        self.initialize_transform(transform_type)

    def __len__(self):
        return self.dataset_length

    def _get_original_size(self):
        return [self.default_h, self.default_w]

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

    def weighted_crop(self, x, upscale_factor=4):
        def exp_distribution(x, sigma=0.05):
            return 1 - np.exp(-x**2 / sigma)
        top, left = np.random.rand(2) 
        sign_t, sign_l = np.random.randint(0,2,2)

        x = transforms.functional.resize(x,[upscale_factor * self.default_h, upscale_factor * self.default_w])

        if self.default_h <= self.default_size:
            top = 0
        elif sign_t == 0:
            top = int((1 - exp_distribution(top, self.sample_sigma)) * upscale_factor * (self.default_h - self.default_size) / 2)
        else:
            top = int((exp_distribution(top, self.sample_sigma)+1) * upscale_factor * (self.default_h - self.default_size) / 2)

        left = np.random.randint(0, upscale_factor * (self.default_w - self.default_size),1)[0] if self.default_w > self.default_size else 0
        
        # if self.default_w <= self.global_res:
        #     left = 0
        # elif sign_l == 0:
        #     left = int((1 - exp_distribution(left, self.sample_sigma))* (self.default_w - self.global_res) / 2)
        # else:
        #     left = int(exp_distribution(left, self.sample_sigma)*(self.default_w - self.global_res))

        x = transforms.functional.crop(x, top, left, upscale_factor * self.global_res, upscale_factor * self.global_res)
        x = transforms.functional.resize(x, [self.image_res, self.image_res])
        return x, torch.from_numpy(np.array([top/upscale_factor,left/upscale_factor], dtype=np.float32))

    def crop(self, x, det=False):
        top = np.random.randint(0,self.default_h - self.crop_res,1)[0] if not det and self.default_h > self.crop_res else 0
        left = np.random.randint(0,self.default_w - self.crop_res,1)[0] if not det and self.default_w > self.crop_res else 0        
        x = transforms.functional.crop(x, top, left, self.crop_res, self.crop_res)
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
            file_patch, d = self.crop(file_patch, self.opt.shift_type == 'none')

        return file_patch, file, d #, Path(filepath).stem


# class Texture3dDataset(TextureImageDataset):
#     def get_samples(self):
#         basecolor = next(Path(self.path_dir).glob('*basecolor.jpg'))
#         height = next(Path(self.path_dir).glob('*height.png'))
#         normal = next(Path(self.path_dir).glob('*normal.jpg'))
#         ambientOcclusion = next(Path(self.path_dir).glob('*ambientOcclusion.jpg'))
#         samples = [basecolor, height, normal, ambientOcclusion]
#         return samples

#     def __len__(self):
#         return self.bs * self.opt.dataset.repeat

#     def __getitem__(self, idx):
#         # idx = 0 if self.use_single else idx
#         d = torch.from_numpy(np.array([0.0,0.0], dtype=np.float32))
        
#         rgb_path = self.samples[0]
#         height_path = self.samples[1]
#         normal_path = self.samples[2]
#         ao_path = self.samples[3]
#         # file = io.load_image(str(Path(filepath)),(256,256))
#         files = []
#         file_patches = []

#         top, left = np.random.rand(2) 
#         top = int(top * (self.default_h - self.global_res))
#         left = int(left * (self.default_w - self.global_res))

#         for filepath in [rgb_path, height_path, normal_path]:
#             file = io.load_image(str(Path(filepath)))
#             file = io.numpy_to_pytorch(file)
#             file = file[:3]
#             file_patch = self.transforms(file)
#             # file_patch = self.weighted_crop(file_patch, top, left)
#             files.append(file)
#             file_patches.append(file_patch)

#         files = torch.cat(files, 0)
#         file_patches = torch.cat(file_patches, 0)
#         file_patches = self.transforms_crop(file_patches)
#         return file_patches, files, d #, Path(filepath).stem

# class RGBDDataset(TextureImageDataset):
#     def get_samples(self):
#         color_files = []
#         depth_files = []

#         for cf in Path(self.path_dir).glob('*color.png'):
#             xi, yi = parse.parse("{:d}_{:d}_color.png", cf.name)
#             depth_files.append(Path(self.path_dir) / f"{xi}_{yi}_depth.png")
#             color_files.append(str(cf.resolve()))
#             # print(depth_files[-1])
#             # assert depth_files[-1].exists()

#         self.depth_files = [str(p) for p in depth_files]
#         return color_files

#     def __len__(self):
#         return len(self.samples) * self.opt.dataset.repeat

#     def __getitem__(self, idx):
#         idx = idx // self.opt.dataset.repeat
#         d = torch.from_numpy(np.array([0.0,0.0], dtype=np.float32))
#         rgb_path = self.samples[idx]
#         depth_path = self.depth_files[idx]
#         rgb = io.load_image(rgb_path)
#         rgb = io.numpy_to_pytorch(rgb)
#         depth = io.load_image(depth_path)
#         depth = io.numpy_to_pytorch(depth)
#         rgbd = torch.cat([rgb, depth], 0)
#         file_patch = self.transforms(rgbd)
#         return file_patch[:3], rgbd[:3], d #, Path(filepath).stem

#     def initialize_transform(self, transform_type):
#         d = None
#         if transform_type == 'simple' or transform_type == 'positioned_crop':            
#             self.transforms = transforms.Compose([
#                 transforms.ToPILImage(),
#                 transforms.Resize([self.default_h,self.default_w]),
#                 transforms.ToTensor(),
#                 transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5)),
#             ])
#             self.transforms_crop = transforms.Compose([
#                 transforms.RandomCrop(self.global_res),
#             ])
#         elif transform_type == 'default':
#             self.transforms = transforms.Compose([
#                 transforms.ToPILImage(),
#                 transforms.Resize([self.default_h,self.default_w]),
#                 transforms.RandomCrop(self.global_res),
#                 # transforms.RandomHorizontalFlip(p=0.5),
#                 # transforms.RandomVerticalFlip(p=0.5),
#                 transforms.ToTensor(),
#                 transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5)),
#             ])
#         self.transform_type = transform_type