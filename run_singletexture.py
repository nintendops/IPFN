from configs.default import opt
from importlib import import_module
from tqdm import tqdm
import os
import numpy as np
import torch
import vgtk
from core.trainer import BasicTrainer
from core.loss import VGGLoss
import utils.helper as H
import utils.io as io
import utils.visualizer as V


class Trainer(BasicTrainer):
    def __init__(self, opt):
        super(Trainer, self).__init__(opt)
        self.summary.register(['Style Loss'])


    def _setup_metric(self):
        self.metric = VGGLoss(self.opt).to(self.opt.device)

    def _optimize(self, data):
        img_patch, img_ref, _ = data   
        img_ref = img_ref.to(self.opt.device)     
        img_patch = img_patch.to(self.opt.device)
        p_recon, x_recon = self.forward()
        self.optimizer.zero_grad()
        self.loss = self.metric(p_recon, img_patch)
        self.loss.backward()
        self.optimizer.step()

        log_info = {
                'Style Loss': self.loss.item(),
        }

        self.summary.update(log_info)

        # visuals
        if self.iter_counter % self.opt.log_freq == 0:
            self.visuals = {'train_patch': V.tensor_to_visual(img_patch),\
                            'train_ref': V.tensor_to_visual(img_ref), 
                            'train_patch_recon': V.tensor_to_visual(p_recon),  
                            'train_ref_recon': V.tensor_to_visual(x_recon), 
            }

            # add flatline to deal with a dumb visdom bug
            self.losses = {'Style Loss': self.loss.item(), 'flatline': 0.0}

    def _get_input(self):
        if 'conv' in self.opt.model.model:
            return None
        else:
            size = (self.opt.model.image_res, self.opt.model.image_res)
            coords = H.get_position( size, self.opt.model.image_dim, \
                                     self.opt.device, self.opt.batch_size)
            return coords
            
    def eval(self):
        self.logger.log('Testing','Evaluating test set!')
        
        self.model.eval()
        self.metric.eval()
        with torch.no_grad():
            for it, data in enumerate(self.dataset):
                img_patch, img_ref, filaname = data
                p_recon, x_recon = self.forward()
                self.visuals = {'eval_patch': V.tensor_to_visual(img_patch),\
                                'eval_ref': V.tensor_to_visual(img_ref), 
                                'eval_patch_recon': V.tensor_to_visual(p_recon),  
                                'eval_ref_recon': V.tensor_to_visual(x_recon), 
                     }

        self.vis.display_current_results(self.visuals, -1)

        self.model.train()
        self.metric.train()
        
    def forward(self):
        x = self._get_input()
        patch_recon, x_recon = self.model(x)
        return patch_recon, x_recon


if __name__ == '__main__':

    if opt.run_mode == 'test' or opt.run_mode == 'eval':
        opt = io.process_test_args(opt)
    opt.model.octaves = 1

    trainer = Trainer(opt)

    if opt.run_mode == 'train':
        trainer.train()
    else:
        trainer.eval()