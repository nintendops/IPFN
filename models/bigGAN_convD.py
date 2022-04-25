import os
import numpy as np
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.helper as H
import core.network_blocks as M


class bigGANDiscriminator(nn.Module):
    def __init__(self, opt, param, init='ortho'):
        super(bigGANDiscriminator, self).__init__()
        self.opt = opt
        self.param = param
        self.init = init

        output_dim = 1

        # main blocks
        # vanilla version of conv2d
        conv = functools.partial(F.conv2d, kernel_size=3, padding=1)
        self.activation = nn.ReLU(inplace=False)

        blocks = []
        i = 0

        for c_in, c_out, downsample, res in zip(  self.param['convD']['in_channels'],
                                                  self.param['convD']['out_channels'],
                                                  self.param['convD']['downsample'],
                                                  self.param['convD']['resolution']):
            
            downsample_func = nn.AvgPool2d(2) if downsample else None

            # conv layer
            blocks += [ M.DBlock(c_in, 
                                c_out, 
                                which_conv=conv, 
                                wide=True, 
                                activation = self.activation,
                                preactivation = (i > 0),
                                downsample=downsample_func)]

            # attention layer
            if self.param['convD']['attention'][res]:
                print('Adding attention layer in D at resolution %d' % res)
                blocks += [M.Attention(c_out, which_conv=conv)]
            i += 1
    
        self.blocks = nn.ModuleList(blocks)

        # output layer
        self.linear = self.which_linear(self.param['convD']['out_channels'][-1], output_dim)

        self.init_weights()

    def forward(self, x):
        h = x
        for index, block in enumerate(self.blocks):
            h = block(x)
        
        # Apply global sum pooling as in SN-GAN
        h = torch.sum(self.activation(h), [2, 3])

        # class-unconditional output
        out = self.linear(h)
        return out

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

    arch = {}
    ch = 64

    arch[256] = {'in_channels' :  [3] + [ch*item for item in [1, 2, 4, 8, 8, 16]],
                   'out_channels' : [item * ch for item in [1, 2, 4, 8, 8, 16, 16]],
                   'downsample' : [True] * 6 + [False],
                   'resolution' : [128, 64, 32, 16, 8, 4, 4 ],
                   'attention' : {2**i: 2**i in [int(item) for item in attention.split('_')]
                                  for i in range(2,8)}}
    arch[128] = {'in_channels' :  [3] + [ch*item for item in [1, 2, 4, 8, 16]],
                   'out_channels' : [item * ch for item in [1, 2, 4, 8, 16, 16]],
                   'downsample' : [True] * 5 + [False],
                   'resolution' : [64, 32, 16, 8, 4, 4],
                   'attention' : {2**i: 2**i in [int(item) for item in attention.split('_')]
                                  for i in range(2,8)}}
    arch[64]  = {'in_channels' :  [3] + [ch*item for item in [1, 2, 4, 8]],
                   'out_channels' : [item * ch for item in [1, 2, 4, 8, 16]],
                   'downsample' : [True] * 4 + [False],
                   'resolution' : [32, 16, 8, 4, 4],
                   'attention' : {2**i: 2**i in [int(item) for item in attention.split('_')]
                                  for i in range(2,7)}}
    arch[32]  = {'in_channels' :  [3] + [item * ch for item in [4, 4, 4]],
                   'out_channels' : [item * ch for item in [4, 4, 4, 4]],
                   'downsample' : [True, True, False, False],
                   'resolution' : [16, 16, 16, 16],
                   'attention' : {2**i: 2**i in [int(item) for item in attention.split('_')]
                                  for i in range(2,6)}}
    model_param = {
        'convD' : arch[opt.model.image_res],
    }
    model = bigGANDiscriminator(opt, model_param).to(device)

    return model
