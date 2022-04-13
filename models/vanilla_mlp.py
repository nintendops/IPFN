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
        self.sigma = opt.model.sigma
        self.l = param['encoding_order']
        self.noise_intp = opt.model.noise_interpolation
        self.k_type = opt.model.k_type
        self.k_threshold = opt.model.k_threshold

        if opt.model.guidance_feature_type != 'none':
            self.g_in = opt.model.guidance_channel
            param['g_in'] = self.g_in
        else:
            self.g_in = 0

        param['c_in'] = self.l * self.dim * 2
        param['c_in'] = param['c_in'] + param['nz'] + self.g_in 

        self.mlp = M.netG_mlp_v1(param)
        self.device = opt.device
        self.nz = param['nz']
        self.rand_shape = [opt.batch_size, self.nz]
        self.octaves = opt.model.octaves
        self.batch_size = opt.batch_size
        self.dist_z = H.get_distribution_type(self.rand_shape, 'uniform')
        self.noise = opt.model.noise

        # TODO: set size of the latent field dynamically to make model more efficient
        self.noise_dim = 32 # 16, 8

        if self.k_type == 'scale':
            K = torch.zeros(self.dim) + param['k_initial_scale']
            self.register_parameter('K',nn.Parameter(K))
        elif self.k_type == 'affine':
            self.transformer = M.transformer_block(self.dim)
            self.K = torch.stack([self.transformer.K[0,0],self.transformer.K[1,1]],0)
        elif self.k_type == 'none':
            self.K = torch.zeros(self.dim) + param['k_initial_scale']      
        else:
            raise NotImplementedError(f"type of k type {self.k_type} is not recognized!")

    def getK(self):
        return self.K

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

        # learned scaled coordinates
        if self.k_type == 'scale':
            if self.k_threshold > 0:
                self.K = 0.5 * (1 + torch.tanh(self.K*0.55)) * self.k_threshold
            x = x * (self.K**-1).reshape(-1,self.dim,1,1)
        elif self.k_type == 'affine':
            x = self.transformer(x)
        else:
            # no scaling
            x = x

        x_encoding = H.positional_encoding(x, l=self.l)
        nb, nc, h, w = x_encoding.shape

        ### enable this if we also want to scale the noise as well (recommended during eval only)
        # position = position * (self.K**-1).reshape(-1,self.dim,1,1)

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
            else:
                z_sample = self.dist_z.sample()
                # z_sample = torch.zeros_like(z_sample).to(self.device)
            z_sample = z_sample.to(self.device)
            self.z_sample = z_sample

        if self.noise == 'const':
            z = z_sample[...,None,None].expand(nb,self.nz,h,w).contiguous()
        elif self.noise == 'none':
            z = torch.zeros(nb, self.nz, h, w).to(self.device)
        elif self.noise.startswith('stationary'):
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
       
        if g is not None:
            x_encoding = torch.cat([x_encoding, z, g], 1)
        else:
            x_encoding = torch.cat([x_encoding, z], 1)
        x_recon = self.mlp(x_encoding)        
        return x_recon, z

def build_model_from(opt, outfile_path=None, c_out=3):
    device = opt.device
    nc = 128

    model_param = {
        'n_features' : [nc,nc,nc,nc,nc,nc,nc,nc,nc,nc],
        'encoding_order' : 5,
        'c_out': opt.model.channel_dim,
        'k_initial_scale': 1.0, 
        'nz' : opt.model.latent_dim,
        'non_linearity': 'relu',
        'bn': 'none',
    }

    model = ImplicitGenerator(opt, model_param).to(device)
    return model
