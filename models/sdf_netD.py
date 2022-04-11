import os
import numpy as np
import torch
import core.network_blocks as M
import utils.helper as H

def build_model_from(opt, outfile_path=None, c_in=None):
    device = opt.device

    if c_in is None:
        c_in = 1

    model_param = {
        'n_features' : 64,
        'c_in' : c_in,
        'non_linearity': 'relu',
        'bn': 'none',
        'res': opt.model.image_res,
    }

    model = M.netD_3d(model_param).to(device)
    return model
