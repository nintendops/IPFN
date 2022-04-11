import os
import numpy as np
import torch
import core.network_blocks as M
import utils.helper as H

def build_model_from(opt, outfile_path=None, c_in=None):
    device = opt.device

    if c_in is None:
        if hasattr(opt.model, 'condition_on_lr'):
            c_in = 6
        elif hasattr(opt.model, 'condition_on_guidance'):
            if opt.model.guidance_feature_type == 'modx' or opt.model.guidance_feature_type == 'mody':
                c_in = 4
            else:
                c_in = 5
        else:
            c_in = 3

    model_param = {
        'n_features' : 64,
        'c_in' : c_in,
        'non_linearity': 'relu',
        'bn': 'none',
        'res': opt.model.image_res,
    }

    model = M.netD_v1(model_param).to(device)

    return model
