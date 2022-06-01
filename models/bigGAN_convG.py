import os
import numpy as np
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.helper as H
import core.network_blocks as M
from torch.nn import init


class bigGANGenerator(nn.Module):
    def __init__(self, opt, param, init='ortho'):
        super(bigGANGenerator, self).__init__()
        self.opt = opt
        self.param = param
        self.init = init
        self.dim_z = param['nz']
        self.bottom_width = 4

        self.linear_layer_at_first = True
        self.crop_index = 0
        
        self.padding_func = H.add_padding
        self.downsample_func = nn.AvgPool2d(int(1/self.opt.model.portion))
        
        # random variable generator
        if self.linear_layer_at_first:
            rand_shape = [self.opt.batch_size, self.dim_z]
            # first linear layer
            self.linear = nn.Linear(self.dim_z, param['convG']['in_channels'][0] * (self.bottom_width**2))
        else:
            rand_shape = [self.opt.batch_size, param['convG']['in_channels'][0], self.bottom_width, self.bottom_width]

        self.sampler = H.get_distribution_type(rand_shape, 'normal')

        # main blocks
        # vanilla version of conv2d
        conv = functools.partial(nn.Conv2d, kernel_size=3, padding=1)
        # vanilla version of bn
        bn = functools.partial(nn.BatchNorm2d)
        activation = nn.ReLU(inplace=False)

        blocks_decoder = []
        blocks_encoder = []
        i = 0
        self.skip_index = []

        for c_in, c_out, upsample, res in zip( self.param['convG']['in_channels'],
                                               self.param['convG']['out_channels'],
                                               self.param['convG']['upsample'],
                                               self.param['convG']['resolution']):

            upsample_func = functools.partial(F.interpolate, scale_factor=2) if upsample else None
            downsample_func =  nn.AvgPool2d(2) if upsample else None

            # conv layer
            blocks_encoder += [ M.DBlock(
                                c_out, 
                                c_in, 
                                which_conv=conv, 
                                wide=True, 
                                activation = activation,
                                preactivation = False,
                                downsample=downsample_func)]

            blocks_decoder += [ M.GBlock(c_in, c_out, which_conv=conv, which_bn=bn, upsample=upsample_func)]

            self.skip_index += [i]

            # attention layer
            if self.param['convG']['attention'][res]:
                print('Adding attention layer in G at resolution %d' % res)
                blocks_decoder += [M.Attention(c_out, which_conv=conv)]
                self.skip_index += [-1]

            i += 1

        self.blocks_decoder = nn.ModuleList(blocks_decoder)
        blocks_encoder.append(M.DBlock(3,c_out,which_conv=conv,wide=True,activation=activation, preactivation=True,downsample=None))
        blocks_encoder.reverse()
        self.blocks_encoder = nn.ModuleList(blocks_encoder)

        # output layer
        self.output_layer = nn.Sequential(nn.BatchNorm2d(self.param['convG']['out_channels'][-1]), 
                                          activation, 
                                          conv(self.param['convG']['out_channels'][-1], 3))

        self.init_weights()

    def forward(self, x):
        '''
            input: a reference image bn, 3, H, W 
            encode input + noise map, then generate from the encoded vector and a noise vector
        '''

        # encoder branch
        image_feats = [x]
        for index, block in enumerate(self.blocks_encoder):
            x = block(x)
            image_feats.append(x)

        # noise branch
        z = self.sampler.sample().to(self.opt.device)
        h = z        
        if self.linear_layer_at_first:
            h = self.linear(h)
            h = h.view(h.shape[0], -1, self.bottom_width, self.bottom_width)

        h = h + x

        # decoder branch
        for index, block in enumerate(self.blocks_decoder):
            # if self.crop_index == index:
            #     h = H.upsample_and_crop(h, k=4)    

            if self.skip_index[index] > 0:           
                y = self.downsample_func(self.padding_func(image_feats[-(self.skip_index[index]+2)]))
                h = block(h) + y
            else:
                h = block(h)
            
        h = self.output_layer(h)

        return torch.tanh(h), z

      # Initialize
    def init_weights(self):
        self.param_count = 0
        for module in self.modules():
          if (isinstance(module, nn.Conv2d) 
              or isinstance(module, nn.Linear) 
              or isinstance(module, nn.Embedding)):
            if self.init == 'ortho':
              init.orthogonal_(module.weight)
            elif self.init == 'N02':
              init.normal_(module.weight, 0, 0.02)
            elif self.init in ['glorot', 'xavier']:
              init.xavier_uniform_(module.weight)
            else:
              print('Init style not recognized...')
            self.param_count += sum([p.data.nelement() for p in module.parameters()])
        print('Param count for G''s initialized parameters: %d' % self.param_count)


def build_model_from(opt, p=1.0, outfile_path=None):
    device = opt.device

    attention = '64'
    ch = 64
    arch = {}
    arch[128] = {'in_channels' :  [ch * item for item in [16, 16, 8, 4, 2]],
                'out_channels' : [ch * item for item in [16, 8, 4, 2, 1]],
                'upsample' : [True] * 5,
                'resolution' : [8, 16, 32, 64, 128],
                'attention' : {2**i: (2**i in [int(item) for item in attention.split('_')])
                              for i in range(3,8)}}

    arch[256] = {'in_channels' :  [ch * item for item in [16, 16, 8, 8, 4, 2]],
                'out_channels' : [ch * item for item in [16,  8, 8, 4, 2, 1]],
                'upsample' : [True] * 6,
                'resolution' : [8, 16, 32, 64, 128, 256],
                'attention' : {2**i: (2**i in [int(item) for item in attention.split('_')])
                              for i in range(3,9)}}

    arch[512] = {'in_channels' :  [ch * item for item in [16, 16, 8, 8, 4, 2, 1]],
               'out_channels' : [ch * item for item in [16,  8, 8, 4, 2, 1, 1]],
               'upsample' : [True] * 7,
               'resolution' : [8, 16, 32, 64, 128, 256, 512],
               'attention' : {2**i: (2**i in [int(item) for item in attention.split('_')])
                              for i in range(3,10)}}

    model_param = {
        'portion' : p,
        'encoder' : {
            'n_features': 16,
            'c_in':3,
            'latent': 64,
            'stride': 2,
            'kernel_size': 2,
            'non_linearity': 'relu',
        },
        'convG' : arch[opt.model.image_res],
        'nz': 128,
    }
    model = bigGANGenerator(opt, model_param).to(device)

    return model
