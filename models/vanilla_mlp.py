import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import json
import vgtk
import core.network_blocks as M
import utils.helper as H
import numpy as np

class ImplicitGenerator(nn.Module):
    def __init__(self, opt, param):
        super(ImplicitGenerator, self).__init__()

        self.dim = opt.model.image_dim
        self.sigma = 0.2
        self.l = param['encoding_order']
        self.noise_intp = opt.model.noise_interpolation
        self.k_type = opt.model.k_type
        self.k_threshold = opt.model.k_threshold

        if hasattr(opt.model, 'condition_on_guidance'):
            if opt.model.guidance_feature_type == 'modx' or opt.model.guidance_feature_type == 'mody':
                self.g_in = 1
            else:
                self.g_in = 2
            param['g_in'] = self.g_in
        else:
            self.g_in = 0

        param['c_in'] = self.l * self.dim * 2
        param['c_in'] = param['c_in'] + param['nz'] + self.g_in # + 2
        # shape = [opt.batch_size, opt.model.image_dim, opt.model.image_res, opt.model.image_res]

        # if self.g_in > 0:
        #     self.mlp = M.netG_mlp_coupling(param)
        # else:
        #     self.mlp = M.netG_mlp_v1(param)

        # self.mlp = M.netG_mlp_multihead(param)
        self.mlp = M.netG_mlp_v1(param)

        self.device = opt.device
        self.nz = param['nz']
        self.rand_shape = [opt.batch_size, self.nz]
        self.octaves = opt.model.octaves
        self.batch_size = opt.batch_size
        self.dist_z = H.get_distribution_type(self.rand_shape, 'uniform')
        self.noise = opt.model.noise
        if self.noise == 'perlin':
            from custom_ops.noise.noise import Noise
            self.noise_sampler = Noise().to(self.device)
            self.perlin_scale = 0.5
        elif self.noise.startswith('stationary'):
            self.noise_dim = 32

        if self.k_type == 'scale':
            K = torch.zeros(self.dim) + param['k_initial_scale']
            self.register_parameter('K',nn.Parameter(K))
        elif self.k_type == 'affine':
            self.transformer = M.transformer_block(self.dim)
            self.K = torch.stack([self.transformer.K[0,0],self.transformer.K[1,1]],0)
        elif self.k_type == 'affine_guidance':
            self.transformer = M.transformer_guided_block(self.dim)
            self.K = torch.zeros(self.dim)      
        elif self.k_type == 'none':
            self.K = torch.zeros(self.dim) + param['k_initial_scale']      
        else:
            raise NotImplementedError(f"type of k type {self.k_type} is not recognized!")

        # K_noise = torch.zeros(self.dim) + param['k_initial_scale']
        # self.register_parameter('K_noise',nn.Parameter(K_noise))

    def forward(self, x, noise_factor=1.0, fix_sample=False):
        '''
            input: cartesian coordinates nb, dim, np
            output: color nb, 3, np
        '''
        g = None
        if x.shape[1] > self.dim:
            g = x[:,self.dim:]
            x = x[:,:self.dim]
            if g.ndim == 3:
                g = g.unsqueeze(1)

        position = x
        # coords = torch.sin(0.5 * np.pi * x)

        # learned scaled coordinates
        if self.k_type == 'scale':
            if self.k_threshold > 0:
                self.K = 0.5 * (1 + torch.tanh(self.K*0.55)) * self.k_threshold
            x = x * (self.K**-1).reshape(-1,self.dim,1,1)
        elif self.k_type == 'affine':
            x = self.transformer(x)
        elif self.k_type == 'affine_guidance':
            x = self.transformer(x, g)
        else:
            # no scaling
            x = x

        x_encoding = H.positional_encoding(x, l=self.l)
        nb, nc, h, w = x_encoding.shape

        ### enable this if we also want to scale the noise
        # position = position * (self.K**-1).reshape(-1,self.dim,1,1)
        # position = position * (self.K_noise**-1).reshape(-1,self.dim,1,1)        
        #####################################

        if hasattr(self,'z_sample') and fix_sample:
            z_sample = self.z_sample
        else:
            if self.noise.startswith('stationary'):
                if self.noise.startswith('stationary1d'):
                    z_sample = self.dist_z.sample([self.noise_dim]).permute([1,2,0]).contiguous()
                elif self.dim == 2:
                    z_shape = [self.noise_dim, self.noise_dim]
                    z_sample = self.dist_z.sample(z_shape).permute([2,3,0,1]).contiguous()
                elif self.dim == 3:
                    z_shape = [self.noise_dim, self.noise_dim,self.noise_dim]
                    z_sample = self.dist_z.sample(z_shape).permute([3,4,0,1,2]).contiguous()
                else:
                    raise NotImplementedError(f"image dim of {self.dim} is not supported!")
                
                if nb != self.batch_size:
                    octaves = nb // self.batch_size
                    z_sample = z_sample.unsqueeze(1).expand(self.batch_size,octaves,self.nz, *z_shape)\
                                      .reshape(-1,self.nz,*z_shape).contiguous()

            else:
                z_sample = self.dist_z.sample()
                # z_sample = torch.zeros_like(z_sample).to(self.device)
                if nb != self.batch_size:
                    octaves = nb // self.batch_size
                    z_sample = z_sample.unsqueeze(1).expand(self.batch_size,octaves,self.nz).reshape(-1,self.nz).contiguous()
            z_sample = z_sample.to(self.device)
            self.z_sample = z_sample

        if self.noise == 'perlin':
            seed = z_sample.reshape(nb,-1,2,1,1)[:,:,0]
            seed = seed.expand(nb,self.nz//2,h,w).reshape(-1).contiguous()
            position = position.permute([0,2,3,1]).unsqueeze(1) \
                        .expand(nb,self.nz//2,h,w,2).reshape(-1,2).contiguous()
            z = self.noise_sampler(self.perlin_scale * position,seed)
            z = z.reshape(2,nb,self.nz//2,h,w).permute([1,0,2,3,4]).reshape(nb,self.nz,h,w).contiguous()
            # z = self.dist_z.sample([h,w]).permute([2,3,0,1]).contiguous().to(self.device)
        elif self.noise == 'const':
            z = z_sample[...,None,None].expand(nb,self.nz,h,w).contiguous()
        elif self.noise == 'none':
            z = torch.zeros(nb, self.nz, h, w).to(self.device)
        elif self.noise.startswith('stationary'):
            # if self.octaves > 1:
            #     factor = 2**(self.octaves-2scale)
            # else:
            #     factor = noise_factor / 2
            factor = noise_factor / 2

            if self.noise.startswith('stationary1d'):
                if self.noise == 'stationary1d_x':
                    pos_in = position[:,1]
                elif self.noise == 'stationary1d_y':
                    pos_in = position[:,0]
                else:
                    raise NotImplementedError(f"type of noise {self.noise} is not recognized!")
                z = H.stationary_noise_1d(factor*pos_in, z_sample, mode=self.noise_intp, sigma=self.sigma)
            elif self.dim == 2:
                z = H.stationary_noise(factor*position, z_sample, mode=self.noise_intp, sigma=self.sigma)
            elif self.dim == 3:
                z = H.stationary_noise_3d(factor*position, z_sample, mode=self.noise_intp, sigma=self.sigma)
            else:
                raise NotImplementedError(f"image dim of {self.dim} is not supported!")
        else:
            raise NotImplementedError(f"type of noise {self.noise} is not recognized!")
       
        # x_encoding = torch.cat([x_encoding, z], 1)
        # if g is not None:
        #     x_recon = self.mlp(x_encoding, g)
        # else:
        #     x_recon = self.mlp(x_encoding)

        if g is not None:
            x_encoding = torch.cat([x_encoding, z, g], 1)
        else:
            x_encoding = torch.cat([x_encoding, z], 1)
            # x_encoding = torch.cat([x_encoding, position, z], 1)
        x_recon = self.mlp(x_encoding)
        
        return x_recon, z

def build_model_from(opt, outfile_path=None, c_out=3):
    device = opt.device
    k_scale = 0.5 * opt.dataset.crop_size / opt.model.global_res

    model_param = {
        # 'n_features' : [128,128,128,128,128],
        'n_features' : [128,128,128,128,128,128,128,128,128,128],
        'encoding_order' : 5,
        'c_out': c_out,
        'k_initial_scale': 1.0, #torch.from_numpy(np.array([1.0,3.0])).float(),
        'nz' : opt.model.latent_dim,
        'non_linearity': 'relu',
        'bn': 'none',
    }

    model = ImplicitGenerator(opt, model_param).to(device)

    return model
