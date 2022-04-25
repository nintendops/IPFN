import torch
import torch.nn.functional as F
from torch.nn import Parameter as P
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

# BigGAN Generator blocks
# Note that this class assumes the kernel size and padding (and any other
# settings) have been selected in the main generator module and passed in
# through the which_conv arg. Similar rules apply with which_bn (the input
# size [which is actually the number of channels of the conditional info] must 
# be preselected)

class GBlock(nn.Module):
  def __init__(self, in_channels, out_channels,
               which_conv=nn.Conv2d, which_bn=F.batch_norm, activation=nn.ReLU(inplace=False), 
               upsample=None):
    super(GBlock, self).__init__()

    # upsample = functools.partial(F.interpolate, scale_factor=2)
    
    self.in_channels, self.out_channels = in_channels, out_channels
    self.which_conv, self.which_bn = which_conv, which_bn
    self.activation = activation
    self.upsample = upsample
    # Conv layers
    self.conv1 = self.which_conv(self.in_channels, self.out_channels)
    self.conv2 = self.which_conv(self.out_channels, self.out_channels)
    self.learnable_sc = in_channels != out_channels or upsample
    if self.learnable_sc:
      self.conv_sc = self.which_conv(in_channels, out_channels, 
                                     kernel_size=1, padding=0)
    # Batchnorm layers
    self.bn1 = self.which_bn(in_channels)
    self.bn2 = self.which_bn(out_channels)
    # upsample layers
    self.upsample = upsample

  def forward(self, x):
    h = self.activation(self.bn1(x))
    if self.upsample:
      h = self.upsample(h)
      x = self.upsample(x)
    h = self.conv1(h)
    h = self.activation(self.bn2(h))
    h = self.conv2(h)
    if self.learnable_sc:       
      x = self.conv_sc(x)
    return h + x

# Residual block for the discriminator
class DBlock(nn.Module):
  def __init__(self, in_channels, out_channels, which_conv=nn.Conv2d, wide=True,
               preactivation=False, activation=None, downsample=None,):
    super(DBlock, self).__init__()
    self.in_channels, self.out_channels = in_channels, out_channels
    # If using wide D (as in SA-GAN and BigGAN), change the channel pattern
    self.hidden_channels = self.out_channels if wide else self.in_channels
    self.which_conv = which_conv
    self.preactivation = preactivation
    self.activation = activation
    self.downsample = downsample
        
    # Conv layers
    self.conv1 = self.which_conv(self.in_channels, self.hidden_channels)
    self.conv2 = self.which_conv(self.hidden_channels, self.out_channels)
    self.learnable_sc = True if (in_channels != out_channels) or downsample else False
    if self.learnable_sc:
      self.conv_sc = self.which_conv(in_channels, out_channels, 
                                     kernel_size=1, padding=0)
  def shortcut(self, x):
    if self.preactivation:
      if self.learnable_sc:
        x = self.conv_sc(x)
      if self.downsample:
        x = self.downsample(x)
    else:
      if self.downsample:
        x = self.downsample(x)
      if self.learnable_sc:
        x = self.conv_sc(x)
    return x
    
  def forward(self, x):
    if self.preactivation:
      h = F.relu(x)
    else:
      h = x    
    h = self.conv1(h)
    h = self.conv2(self.activation(h))
    if self.downsample:
      h = self.downsample(h)        
    return h + self.shortcut(x)


# A non-local block as used in SA-GAN
class Attention(nn.Module):
  def __init__(self, ch, which_conv=nn.Conv2d, name='attention'):
    super(Attention, self).__init__()
    # Channel multiplier
    self.ch = ch
    self.which_conv = which_conv
    self.theta = self.which_conv(self.ch, self.ch // 8, kernel_size=1, padding=0, bias=False)
    self.phi = self.which_conv(self.ch, self.ch // 8, kernel_size=1, padding=0, bias=False)
    self.g = self.which_conv(self.ch, self.ch // 2, kernel_size=1, padding=0, bias=False)
    self.o = self.which_conv(self.ch // 2, self.ch, kernel_size=1, padding=0, bias=False)
    # Learnable gain parameter
    self.gamma = P(torch.tensor(0.), requires_grad=True)
  def forward(self, x):
    # Apply convs
    theta = self.theta(x)
    phi = F.max_pool2d(self.phi(x), [2,2])
    g = F.max_pool2d(self.g(x), [2,2])    
    # Perform reshapes
    theta = theta.view(-1, self. ch // 8, x.shape[2] * x.shape[3])
    phi = phi.view(-1, self. ch // 8, x.shape[2] * x.shape[3] // 4)
    g = g.view(-1, self. ch // 2, x.shape[2] * x.shape[3] // 4)
    # Matmul and softmax to get attention maps
    beta = F.softmax(torch.bmm(theta.transpose(1, 2), phi), -1)
    # Attention map times g path
    o = self.o(torch.bmm(g, beta.transpose(1,2)).view(-1, self.ch // 2, x.shape[2], x.shape[3]))
    return self.gamma * o + x


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

