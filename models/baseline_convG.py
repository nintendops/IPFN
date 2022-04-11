import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.helper as H

class ConvGenerator(nn.Module):
    def __init__(self, opt, param):
        super(ConvGenerator, self).__init__()

        nfeatures = param['n_features']
        nz = param['nz']
        res = param['res']
        c_in = param['c_in']
        kernels = param['kernels']
        strides = param['strides']
        nb = opt.batch_size
        s_i = 0

        self.opt = opt
        self.device = opt.device
        self.z_res = int(res / 2**(1+len(nfeatures)))
        if self.opt.model.noise == "const":
            self.rand_shape = [nb, nz, 1, 1]
        else:
            self.rand_shape = [nb, nz, self.z_res, self.z_res]
        self.sampler = H.get_distribution_type(self.rand_shape, 'uniform')

        c_in = nz

        blocks = nn.ModuleList()        
        for i,nc in enumerate(nfeatures):
            padding = 1 if i > 0 else 0
            blocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, nc, kernels[i], stride=strides[i], padding=padding),
                # nn.BatchNorm2d(nc),
                nn.ReLU(True),
            ))            
            c_in = nc

        deconv_out = nn.ConvTranspose2d(c_in, 3, 2, stride=strides[-1])
        self.blocks = blocks
        self.deconv_out = deconv_out
        self.tanh = nn.Tanh()

    def forward(self, x, noise_factor=None):
        '''
            input: latent vector
            output: color nb, 3, h, w
        '''
        
        x = self.sampler.sample().to(self.device)
        nb, nz = x.shape[:2]
        if self.opt.model.noise == 'const':
            x = x.expand(nb, nz, self.z_res, self.z_res)
        # x = torch.rand(self.rand_shape).to(self.device)
        
        output = x

        for idx, block in enumerate(self.blocks):
            output = block(output)
            # print(f'----- netG layer {idx} -------')
            # print(output.shape)
        output = self.deconv_out(output)
        output = self.tanh(output)

        return output, x

def build_model_from(opt, outfile_path=None):
    device = opt.device
    model_param = {
        'n_features' :  [512,256,128,64],
        'strides' : [2,2,2,2,2],
        'kernels' : [2,4,4,4,4],
        'c_in' : 2**16 ,
        'nz': opt.model.latent_dim,
        'res': opt.model.image_res,
        # 'non_linearity': 'relu',
        # 'bn': 'none',
    }
    model = ConvGenerator(opt, model_param).to(device)

    return model
