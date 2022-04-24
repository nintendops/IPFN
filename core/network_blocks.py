import torch
import torch.nn.functional as F
from torch import nn
import functools
from core.core_layers import *

class transformer_block(nn.Module):
    def __init__(self, dim=2, initial_scale=1.0):
        super(transformer_block, self).__init__()
        K = torch.eye(dim) * initial_scale
        self.register_parameter('K',nn.Parameter(K))

    def forward(self, x):
        x = x.permute([0,2,3,1]) @ self.K
        x = x.permute([0,3,1,2]).contiguous()
        return x


# mlp generator
class netG_mlp_v1(nn.Module):
    def __init__(self, param, dim=2, end_activation='tanh'):
        super(netG_mlp_v1, self).__init__()
        self.param = param

        if dim == 2:
            ConvBlock = Conv2dBlock
        elif dim == 3:
            ConvBlock = Conv3dBlock
        else:
            raise NotImplementedError(f"a dimension of {dim} is not supported!")
        ###### params##############
        n_features = param['n_features']
        nf_in = param['c_in']
        nf_out = param['c_out']
        non_linearity = param['non_linearity']
        bn = param['bn']
        # dropout = param['dropout_ratio']
        ##########################

        self.blocks = nn.ModuleList()
        c = nf_in
        for idx, c_out in enumerate(n_features):
            block_i = ConvBlock(c, c_out, 1, 1, 0, bn, non_linearity)
            self.blocks.append(block_i)
            c = c_out

        self.last_conv = ConvBlock(c, nf_out, 1, 1, 0, None, None)
        self.blocks.append(self.last_conv)
        self.activation = nn.ReLU() if end_activation == 'relu' else nn.Tanh()

    def forward(self, x, y=None):
        for block in self.blocks:
            x = block(x)
        x = self.activation(x)
        if y is not None:
            x = x + y
        return x

# multihead mlp generator
class netG_mlp_multihead(nn.Module):
    def __init__(self, param, dim=2, end_activation='tanh'):
        super(netG_mlp_multihead, self).__init__()
        self.param = param

        if dim == 2:
            ConvBlock = Conv2dBlock
        elif dim == 3:
            ConvBlock = Conv3dBlock
        else:
            raise NotImplementedError(f"a dimension of {dim} is not supported!")
        ###### params##############
        n_features = param['n_features']
        nf_in = param['c_in']
        nf_out = param['c_out']
        non_linearity = param['non_linearity']
        bn = param['bn']
        # dropout = param['dropout_ratio']
        ##########################

        share_blocks = nn.ModuleList()
        c = nf_in
        for idx, c_out in enumerate(n_features):
            block_i = ConvBlock(c, c_out, 1, 1, 0, bn, non_linearity)
            share_blocks.append(block_i)
            c = c_out
        self.share_blocks = share_blocks

        multi_blocks = nn.ModuleList()
        for i in range(nf_out//3):
            blocks = nn.ModuleList()
            for idx, c_out in enumerate(n_features[:3]):
                block_i = ConvBlock(c, c_out, 1, 1, 0, bn, non_linearity)
                blocks.append(block_i)
                c = c_out
            last_conv = ConvBlock(c, 3, 1, 1, 0, None, None)
            blocks.append(last_conv)
            multi_blocks.append(blocks)
        # self.last_convs = nn.ModuleList([ConvBlock(c, 3, 1, 1, 0, None, None) for i in range(nf_out//3)])
        # self.blocks.append(self.last_conv)
        self.blocks = multi_blocks
        self.activation = nn.ReLU() if end_activation == 'relu' else nn.Tanh()

    def forward(self, x, y=None):
        # share network + multibranch networks + skip connection
        for block in self.share_blocks:
            x = block(x)
        xs = []        
        for branch in self.blocks:
            feat = x
            for idx, block in enumerate(branch):
                feat = block(feat)
                if idx == len(branch) - 2:
                    feat = feat + x
            xs.append(feat)
        xs = torch.cat(xs, 1)
        xs = self.activation(xs)
        if y is not None:
            xs = xs + y
        return xs

class netG_mlp_unet(nn.Module):
    def __init__(self, param):
        super(netG_mlp_unet, self).__init__()
        self.param = param

        ###### params##############
        n_feautres = param['n_features']
        nf_in = param['c_in']
        nf_out = param['c_out']
        non_linearity = param['non_linearity']
        bn = param['bn']
        # dropout = param['dropout_ratio']
        ##########################

        self.blocks = nn.ModuleList()
        c = nf_in
        for idx, c_out in enumerate(n_feautres):
            block_i = Conv2dBlock(c, c_out, 1, 1, 0, bn, non_linearity)
            self.blocks.append(block_i)
            c = c_out

        self.last_conv = Conv2dBlock(c + 3, nf_out, 1, 1, 0, None, None)
        self.blocks.append(self.last_conv)
        self.tanh = nn.Tanh()

    def forward(self, x, y):
        # skip_x = x[:,:3]
        for idx, block in enumerate(self.blocks):
            # if idx == len(self.blocks) - 1:
            #     x = torch.cat([x,y],1)
            x = block(x)
        x = self.tanh(x)
        return x + y

class netG_mlp_ns(nn.Module):
    def __init__(self, param):
        super(netG_mlp_ns, self).__init__()
        self.param = param

        ###### params for encoder ##############
        param_encoder = {
            'n_features': param['n_features_encoder'],
            'c_in': param['g_in'],
            'c_out': param['n_features'][0],
            'non_linearity' : param['non_linearity'],
            'bn' : param['bn'],
        }
        ##########################################

        self.encoder = netG_mlp_v1(param_encoder, 'relu')
        param['c_in'] = param['g_in'] + param['n_features'][0] + param['c_in']
        self.generator = netG_mlp_v1(param)

    def forward(self, x, g):
        g_feat = self.encoder(g)
        x = torch.cat([x, g_feat, g],1)
        return self.generator(x)


# discriminator
class netD_v1(nn.Module):
    def __init__(self, param):
        super(netD_v1, self).__init__()
        self.param = param
        ###### params##############
        dim = param['n_features']
        nf_in = param['c_in']
        # nf_out = param['c_out']
        non_linearity = param['non_linearity']        
        res = param['res']
        kernel_size = 3
        ##########################
        # should not contain batchnorm
        # 4 layer 2 stride
        blocks = nn.Sequential(
            nn.Conv2d(nf_in, dim, kernel_size, 2, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(dim, 2 * dim, kernel_size, 2, 1),   
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(2 * dim, 4 * dim, kernel_size, 2, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(4 * dim, 8 * dim, kernel_size, 2, 1),
            nn.LeakyReLU(inplace=True),
        )
        self.blocks = blocks
        n_conv = 4
        c_in = int(8 * dim * res * res / 4**n_conv)
        self.linear = nn.Linear(c_in, 1)

    def forward(self, x):
        bs, nc, w, h = x.shape
        for idx, block in enumerate(self.blocks):
            # print(f"----- Before NetD layer {idx} -----")
            # print(x.shape)
            x = block(x)
            # print(f"----- After NetD layer {idx} -----")
            # print(x.shape)
        x = x.view(bs,-1)
        x = self.linear(x)
        return x

class netD_3d(nn.Module):
    def __init__(self, param):
        super(netD_3d, self).__init__()
        self.param = param
        ###### params##############
        dim = param['n_features']
        nf_in = param['c_in']
        # nf_out = param['c_out']
        non_linearity = param['non_linearity']        
        res = param['res']
        kernel_size = 4
        ##########################
        # should not contain batchnorm
        blocks = nn.Sequential(
            nn.Conv3d(nf_in, dim, kernel_size, 2, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(dim, 2 * dim, kernel_size, 2, 1),   
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(2 * dim, 4 * dim, kernel_size, 2, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(4 * dim, 8 * dim, kernel_size, 2, 1),
            nn.LeakyReLU(inplace=True),
        )
        self.blocks = blocks
        n_conv = 4
        c_in = int(8 * dim * (res**3) / 8**n_conv)
        self.linear = nn.Linear(c_in, 1)

    def forward(self, x):
        bs = x.shape[0]
        for idx, block in enumerate(self.blocks):
            x = block(x)

        x = x.view(bs,-1)
        x = self.linear(x)
        return x


class encoder_2d(nn.Module):
    def __init__(self, param):
        super(encoder_2d, self).__init__()
        self.param = param
        ###### params##############
        dim = param['n_features']
        nf_in = param['c_in']
        # nf_out = param['c_out']
        non_linearity = param['non_linearity']        
        res = param['res']
        latent_dim = param['latent']
        kernel_size = param['kernel_size']
        stride = param['stride']
        ##########################

        # should not contain batchnorm
        blocks = nn.Sequential(
            nn.Conv2d(nf_in, dim, kernel_size, stride, 0),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(dim, 2 * dim, kernel_size, stride, 0),   
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(2 * dim, 4 * dim, kernel_size, stride, 0),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(4 * dim, 8 * dim, kernel_size, stride, 0),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(8 * dim, latent_dim, 1, 1, 0),
            nn.LeakyReLU(inplace=True),
        )
        self.blocks = blocks
        n_conv = 4

        # c_in = int(8 * dim * (res**2) / 4**n_conv)
        # self.linear = nn.Linear(c_in, latent_dim)



    def forward(self, x):
        bs = x.shape[0]
        for idx, block in enumerate(self.blocks):
            # print(f"Encoder: input tensor of size {x.shape}")
            x = block(x)
            # print(f"Encoder: output tensor of size {x.shape}")
            # import ipdb; ipdb.set_trace()

        # x = x.view(bs,-1)
        # x = self.linear(x)
        return x   


# Conv Blocks
class PartialConvGenerator(nn.Module):
    '''
        Generate partial portion of the synthesized image:
            z -> (3, H/k,W)

    '''
    def __init__(self, param):
        super(PartialConvGenerator, self).__init__()

        nfeatures = param['n_features']
        c_in = param['c_in']
        p = param['portion']
        kernel_size = param['kernel_size']
        stride = param['stride']
        s_i = 0

        self.ratio = int(1/p)
        # stride_w = int(stride)
        # stride_h = int(stride * 1/p)

        blocks = nn.ModuleList()        

        # 4-layer 2 strides deconv
        for i,nc in enumerate(nfeatures):
            # padding = 1 if i > 0 else 0
            blocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, nc, kernel_size, stride=stride, padding=(1,1)),
                # nn.ConvTranspose2d(c_in, nc, kernel_size, stride=(stride_w, stride_h)),
                # nn.BatchNorm2d(nc),
                nn.ReLU(True),
            ))            
            c_in = nc

        deconv_out = nn.ConvTranspose2d(c_in, 3, kernel_size, stride=stride, padding=(1,1))
        self.blocks = blocks
        self.deconv_out = deconv_out
        self.tanh = nn.Tanh()

    def forward(self, x, noise_factor=None):
        '''
            input: latent vector
            output: color nb, 3, h, w
        '''        
        # y = x[...,None,None]
        # y = y.expand([*y.shape[:2], 2, 2 * self.ratio]).contiguous()
        y = x

        for idx, block in enumerate(self.blocks):
            # print(f'----- BEFORE netG layer {idx} -------')
            # print(y.shape)
            y = block(y)
            # print(f'----- AFTER netG layer {idx} -------')
            # print(y.shape)
            # import ipdb; ipdb.set_trace()
        y = self.deconv_out(y)
        # print(f'----- FINAL netG layer {idx} -------')
        # print(y.shape)
        # import ipdb; ipdb.set_trace()
        y = self.tanh(y)
        return y, x






class ResnetBlock(nn.Module):
    """ A single Res-Block module """

    def __init__(self, dim, use_bias=True):
        super(ResnetBlock, self).__init__()

        # A res-block without the skip-connection, pad-conv-norm-relu-pad-conv-norm
        self.conv_block = nn.Sequential(nn.utils.spectral_norm(nn.Conv2d(dim, dim // 4, kernel_size=1, bias=use_bias)),
                                        nn.BatchNorm2d(dim // 4),
                                        nn.LeakyReLU(0.2, True),
                                        nn.ReflectionPad2d(1),
                                        nn.utils.spectral_norm(nn.Conv2d(dim // 4, dim // 4, kernel_size=3, bias=use_bias)),
                                        nn.BatchNorm2d(dim // 4),
                                        nn.LeakyReLU(0.2, True),
                                        nn.utils.spectral_norm(nn.Conv2d(dim // 4, dim, kernel_size=1, bias=use_bias)),
                                        nn.BatchNorm2d(dim))

    def forward(self, input_tensor):
        # The skip connection is applied here
        return input_tensor + self.conv_block(input_tensor)

