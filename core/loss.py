import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from core.network_blocks import VGGFeatures

# WGAN-GP: https://arxiv.org/pdf/1704.00028.pdf]
def calc_gradient_penalty(netD, real_data, fake_data, multiscale=False, scale_weights=None, LAMBDA=10):
    '''
      credit: https://github.com/caogang/wgan-gp/blob/master/gan_cifar10.py
    '''
    if isinstance(real_data, list):
        # TODO: remove assumption on 2D data
        assert isinstance(fake_data, list)
        real_data_global = real_data[0].data
        real_data_local = real_data[1].data
        fake_data_global = fake_data[0].data
        fake_data_local = fake_data[1].data
        device = real_data_global.device
        nb, dim, w, h = real_data_global.shape
        _, _, wl, hl = real_data_local.shape
        alpha = torch.rand(nb, 1)
        alpha_global = alpha.expand(nb, dim*w*h).contiguous().view(nb, dim, w, h)
        alpha_global = alpha_global.to(device)
        alpha_local = alpha.expand(nb, dim*wl*hl).contiguous().view(nb, dim, wl, hl)
        alpha_local = alpha_local.to(device)
        interpolates_global = alpha_global * real_data_global + ((1 - alpha_global) * fake_data_global)
        interpolates_global = autograd.Variable(interpolates_global, requires_grad=True)
        interpolates_local = alpha_local * real_data_local + ((1 - alpha_local) * fake_data_local)
        interpolates_local = autograd.Variable(interpolates_local, requires_grad=True)
        interpolates = [interpolates_global, interpolates_local]
    else:
        device = real_data.device
        real_data = real_data.data
        fake_data = fake_data.data
        
        if multiscale:
            nb, octaves, dim = real_data.shape[:3]
            res = real_data.shape[3:]
            real_data = real_data.reshape(nb*octaves, dim, *res)
            fake_data = fake_data.reshape(nb*octaves, dim, *res)
            nb = nb * octaves
        else:
            nb, dim = real_data.shape[:2]
            res = real_data.shape[2:]
            fake_data = fake_data[:nb]

        alpha = torch.rand(nb, 1)
        alpha = alpha.expand(nb, dim*np.prod(res)).contiguous().view(nb, dim, *res)
        alpha = alpha.to(device)
        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        interpolates = autograd.Variable(interpolates, requires_grad=True)
        if multiscale:
            interpolates = interpolates.reshape(-1, octaves, dim, *res)
    if scale_weights is not None:
      disc_interpolates = netD(interpolates, scale_weights)
    else:
      disc_interpolates = netD(interpolates)

    if isinstance(disc_interpolates, tuple) or isinstance(disc_interpolates, list):
      disc_interpolates = disc_interpolates[0]

    grad_outputs = torch.ones(disc_interpolates.size()).to(device)
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=grad_outputs,
                              create_graph=True, only_inputs=True)
    norm = 0.0
    if isinstance(real_data, list):
      for grad in gradients:
        grad = grad.view(grad.size(0), -1)
        norm += ((grad.norm(2, dim=1) - 1) ** 2).mean()
    else:
      gradients = gradients[0]
      gradients = gradients.view(gradients.size(0), -1)
      norm = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    gradient_penalty = norm * LAMBDA
    return gradient_penalty, norm.item()

class VGGLoss(nn.Module):
    def __init__(self, opt):
      super(VGGLoss, self).__init__()
      self.vgg = VGGFeatures()
      self.signed_to_unsigned = lambda x : (x + 1) / 2
      self.loss_type = get_loss_type('mse')

    def forward(self, y, gt):
      loss_style = torch.tensor(0.0, device=y.device)
      vggf_y = self.vgg(self.signed_to_unsigned(y))
      vggf_gt = self.vgg(self.signed_to_unsigned(gt))
      gram_y = list(map(gram_matrix, vggf_y))
      gram_gt = list(map(gram_matrix, vggf_gt))

      for gy,ggt in zip(gram_y, gram_gt):
        loss_style += self.loss_type(gy, ggt)

      # recon loss (debug only)
      for fy,fgt in zip(vggf_y, vggf_gt):
        loss_style += self.loss_type(fy, fgt)
        
      return loss_style

def gram_matrix(x):
    b, c, h, w = x.size()
    features = x.view(b, c, h * w)
    gram_matrix = torch.bmm(features, features.transpose(1, 2))
    gram_matrix.div_(h * w)
    return gram_matrix
