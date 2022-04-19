import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.helper as H
import core.network_blocks as M


class ConvGenerator(nn.Module):
    def __init__(self, opt, param):
        super(ConvGenerator, self).__init__()

        nb = opt.batch_size
        nz = param['nz']
        self.p = int(param['portion'] * opt.model.image_res)

        # param for encoder
        param_encoder = param['encoder']
        param_encoder['res'] = opt.model.image_res

        self.encoder = M.encoder_2d(param['encoder'])

        # param for convG
        param_generator = param['convG']
        param_generator['c_in'] = nz + param_encoder['latent']
        param_generator['portion'] = param['portion']
        self.generator = M.PartialConvGenerator(param['convG'])

        # nfeatures = param['n_features']
        # kernels = param['kernels']

        self.opt = opt
        self.device = opt.device
        self.rand_shape = [nb, nz]
        self.sampler = H.get_distribution_type(self.rand_shape, 'normal')


    def forward(self, x):
        '''
            input: a reference image bn, 3, H, W (partially padded with zero)
            encode input + noise map, then generate from the encoded vector and a noise vector
        '''
        bn, _, h, w = x.shape

        # keep the non-padded part of the reference image
        source_img = x[:, :, :self.p]        
        z = self.encoder(x)
        noise = self.sampler.sample().to(self.device)
        z = torch.cat([z, noise], 1)
        y, _ = self.generator(z)     
        # upsampling
        y = torch.nn.functional.interpolate(y, \
                     size=[h - self.p,w], mode='bilinear')
        composed_y = torch.cat([source_img, y], 2)
        return composed_y, z

def build_model_from(opt, p, outfile_path=None):
    device = opt.device
    model_param = {
        'portion' : p,
        'encoder' : {
            'n_features': 64,
            'c_in':3,
            'latent': 64,
            'stride': 2,
            'kernel_size': 2,
            'non_linearity': 'relu',
        },
        'convG' : {
            'n_features': [512, 256, 128, 64],
            'stride':2,
            'kernel_size':4,
            'non_linearity': 'relu',
        },
        'nz': opt.model.latent_dim,
    }
    model = ConvGenerator(opt, model_param).to(device)

    return model
