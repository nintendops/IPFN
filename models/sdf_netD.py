import os
import numpy as np
import torch
import core.network_blocks as M
import utils.helper as H

def build_model_from(opt, outfile_path=None):
    device = opt.device

    if opt.model.guidance_feature_type != 'none':
        g_in = opt.model.guidance_channel
    else:
        g_in = 0

    c_in = opt.model.channel_dim + g_in

    model_param = {
        'n_features' : 64,
        'c_in' : c_in,
        'non_linearity': 'relu',
        'bn': 'none',
        'res': opt.model.image_res,
    }

    model = M.netD_3d(model_param).to(device)
    return model
