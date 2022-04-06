from configs.default import opt
from importlib import import_module
from tqdm import tqdm
import os
import numpy as np
import torch
import vgtk
from core.trainer import BasicTrainer
from core.loss import *
from dataset import TextureImageDataset 
import utils.helper as H
import utils.io as io
import utils.visualizer as V
from torch.autograd import Variable


class Trainer(BasicTrainer):
    def __init__(self, opt):
        super(Trainer, self).__init__(opt)
        self.dataset_iter = iter(self.dataset)
        self.summary.register(['LossD', 'LossG', 'Gradient Norm'])        
        self.dist_shift = H.get_distribution_type([self.opt.batch_size, self.opt.model.image_dim], 'uniform')
        self.octaves = opt.model.octaves
        self.scale_factor = self.opt.dataset.crop_size / self.opt.model.image_res
        print(f"[MODEL] choosing a scale factor of {self.scale_factor}!!!")

    def _setup_datasets(self):
        dataset = TextureImageDataset(self.opt, self.opt.model.octaves, transform_type='multiscale')
        self.opt.dataset.crop_size = dataset.default_size
        self.dataest_size = len(dataset)
        self.n_iters = self.dataest_size / self.opt.batch_size
        self.dataset = torch.utils.data.DataLoader( dataset, \
                                                    batch_size=self.opt.batch_size, \
                                                    shuffle=False, \
                                                    num_workers=self.opt.num_thread)
    def _setup_model(self):     
        if self.opt.run_mode == 'train':
            param_outfile = os.path.join(self.root_dir, "params.json")
        else:
            param_outfile = None
        module = import_module('models')
        # modelG_name = 'multiscale_mlp'
        modelG_name = 'vanilla_mlp'        
        print(f"Using network model {modelG_name}!")
        self.model = getattr(module, modelG_name).build_model_from(self.opt, param_outfile)

        self.modelG = self.model
        module = import_module('models')
        modelD_name = 'multiscale_netD'
        print(f"Using discriminator model {modelD_name}!")
        self.modelD = getattr(module, modelD_name).build_model_from(self.opt, None)

    def _setup_optim(self):
        self.logger.log('Setup', 'Setup optimizer!')
        self.optimizerG = optim.Adam(self.modelG.parameters(), 
                                    lr=self.opt.train_lr.init_lr, betas=(0.5, 0.9))        
        self.optimizerD = optim.Adam(self.modelD.parameters(), 
                                    lr=self.opt.train_lr.init_lr, betas=(0.5, 0.9))
        # self.lr_schedule = vgtk.LearningRateScheduler(self.optimizer,
        #                                               **vars(self.opt.train_lr))
        self.logger.log('Setup', 'Optimizer all-set!')


    def _setup_metric(self):
        self.metric = None
        # self.l1_loss = get_loss_type('l1') 

    def train_epoch(self):
        while self.epoch_counter <= self.opt.num_epochs:
            try:
                self._optimize()
                self.iter_counter += 1
            except StopIteration:
                self.epoch_counter += 1
                self.iter_counter = 0
                self.dataset_iter = iter(self.dataset)           
                if self.epoch_counter > 0 and self.epoch_counter % self.opt.save_freq == 0:
                    self._save_network(f'Epoch{self.epoch_counter}')
    
            if self.iter_counter % self.opt.log_freq == 0:
                self._print_running_stats(f'Epoch {self.epoch_counter}')
                if hasattr(self, 'visuals'):
                    self.vis.display_current_results(self.visuals)
                if hasattr(self, 'losses'):
                    counter_ratio = self.opt.critic_steps * self.iter_counter / self.n_iters 
                    self.vis.plot_current_losses(self.epoch_counter, counter_ratio, self.losses)


    def _optimize(self):

        # one = torch.FloatTensor([1]).to(self.opt.device)
        # mone =(-1*one).to(self.opt.device)

        # -------------------- Train D -------------------------------------------
        # for p in self.modelG.parameters():
        #     p.requires_grad = False

        scale_weights = self._get_scale_weights()

        for p in self.modelD.parameters():
            p.requires_grad = True
        for i in range(self.opt.critic_steps):
            data = next(self.dataset_iter)
            img_patch, img_ref, filaname = data
            bn, octaves, bc, h, w = img_patch.shape
            self.modelD.zero_grad()
            
            # real data
            real_data = img_patch.to(self.opt.device)
            real_data = Variable(real_data)

            D_real = self.modelD(real_data, scale_weights)
            D_real = -1 * D_real.mean()
            D_real.backward()

            # fake data
            coords = self._get_input()
            fake = []
            for i,coord in enumerate(coords):          
                fake_i, _ = self.modelG(coord)
                fake_i = torch.nn.functional.interpolate(fake_i, \
                         size=[self.opt.model.image_res,self.opt.model.image_res], mode='bilinear')
                fake.append(fake_i)
            fake_data = torch.stack(fake,1).to(self.opt.device)
            fake_data = Variable(fake_data.data)#.reshape(bn, octaves, bc, h, w)
            D_fake = self.modelD(fake_data, scale_weights).mean()
            D_fake.backward()

            gradient_penality, gradient_norm = calc_gradient_penalty(self.modelD, real_data.data, fake_data.data, scale_weights=scale_weights)
            gradient_penality.backward()
            D_cost = D_fake + D_real + gradient_penality
            # Wasserstein_D = D_real - D_fake
            self.optimizerD.step()

        # -------------------- Train G -------------------------------------------
        for p in self.modelD.parameters():
            p.requires_grad = False
        self.modelG.zero_grad()   

        coords = self._get_input()
        fake = []
        zs = []
        for i,coord in enumerate(coords):                
            fake_i, z = self.modelG(coord)
            fake_i = torch.nn.functional.interpolate(fake_i, \
                     size=[self.opt.model.image_res,self.opt.model.image_res], mode='bilinear')
            fake.append(fake_i)
            zs.append(z)

        fake_data = torch.stack(fake,1).to(self.opt.device)             
        # p_recon, z = self.modelG(self._get_input())
        # fake_data = p_recon.reshape(self.opt.batch_size, self.octaves, bc, \
        #                             self.opt.model.image_res, self.opt.model.image_res)
        G_cost = self.modelD(fake_data, scale_weights)
        G_cost = -G_cost.mean()
        G_cost.backward()
        self.optimizerG.step()

        log_info = {
                'LossG': G_cost.item(),
                'LossD': D_cost.item(),
                'Gradient Norm': gradient_norm,
        }

        self.summary.update(log_info)

        # visuals
        if self.iter_counter % self.opt.log_freq == 0:
            self.visuals = {# 'train_patch': V.tensor_to_visual(img_patch),\
                            'train_ref': V.tensor_to_visual(img_ref), 
                            # 'train_patch_recon': V.tensor_to_visual(p_recon), 
            }

            # z = z.reshape(self.opt.batch_size, self.octaves, -1, self.opt.model.image_res, self.opt.model.image_res)
            for i in range(self.octaves):
                self.visuals[f'recon_scale{i}'] = V.tensor_to_visual(fake_data[:,i])
                self.visuals[f'noise_scale{i}'] = V.tensor_to_visual(zs[i][:,:3])
                self.visuals[f'gt_scale{i}'] = V.tensor_to_visual(img_patch[:,i])

            self.losses = {                
                    'LossG': G_cost.item(),
                    'LossD': D_cost.item(),
                    'Gradient Norm': gradient_norm,}

    def _get_input(self, factor=4):
        # get input from coarse [-1,1] to fine [-factor^octaves, factor^octaves]
        octaves = self.octaves
        res = self.opt.model.image_res
        # size = (res, res)
        coords = []

        for i in range(octaves):
            ######
            # debug_factor = 2**(octaves-i-1)
            ############

            scale_factor = factor**(octaves-i-1)
            res_at_iscale = res * scale_factor
            size = (res_at_iscale,res_at_iscale)
            coord = scale_factor * H.get_position(size, self.opt.model.image_dim, \
                                 self.opt.device, self.opt.batch_size)
            if opt.run_mode == 'train':
                shift = self.scale_factor * self.dist_shift.sample()[:,:,None,None]
                coord = coord + shift.to(self.opt.device)
            coords.append(coord)

        # octaves = torch.arange(octaves).reshape(1,octaves,1,1,1)
        # octaves = factor**(self.octaves - octaves - 1).to(self.opt.device)
        # coords = coords.unsqueeze(1) * octaves
        # bn, octave, image_dim, h, w = coords.shape
        # output: [bn, octave, image_dim, res, res]        
        # if self.opt.run_mode == 'train':
        #     # [nb,1,2,1,1]
        #     shift = self.scale_factor * self.dist_shift.sample()[:,None,:,None,None]
        #     shift = shift.expand(bn, octave, self.opt.model.image_dim, 1, 1).contiguous().to(self.opt.device)
        #     coords = coords + shift
        # coords = coords.reshape(bn*octave, image_dim, h, w)
        return coords
    
    def _get_high_res_input(self, res, scale=None):
        if 'conv' in self.opt.model.model:
            return None
        else:            
            size = (res, res)
            scale = self.scale_factor if scale is None else scale
            coords = H.get_position( size, self.opt.model.image_dim, \
                                     self.opt.device, self.opt.batch_size)
            return scale * coords 

    def _get_scale_weights(self, base_factor=1.4):
        it = self.iter_counter + self.n_iters * self.epoch_counter
        max_it = self.n_iters * self.opt.num_epochs
        it = max(it, max_it)

        # base_factor -> 1
        factor = base_factor ** ((max_it - it) * 1.0 / max_it)

        # start with more weights towards coarser level
        factor = factor**(-1)
        weights = factor**(np.arange(self.octaves))
        weights = weights / weights.sum()

        #######################
        # weights = np.array([1.0,0.0])
        ##########################

        return torch.from_numpy(weights).to(self.opt.device)

    def eval(self):
        import time
        self.logger.log('Testing','Evaluating test set!')
        
        self.model.eval()
        with torch.no_grad():

            countdown = 5
            for i in range(countdown):
                self.vis.yell(f"Getting ready to test! Start in {5-i}...")
                time.sleep(1)
            
            # self.vis.yell(f"First is random noise input!")
            # sample = 10
            # for i in range(sample):
            #     recon, _ = self.modelG(self._get_input())

            #     self.visuals = {
            #                     'generated image': V.tensor_to_visual(recon[:,-1]),\
            #     }
            #     self.vis.display_current_results(self.visuals)
            #     time.sleep(1)

            # self.vis.yell(f"Random high-res noise input!")
            # sample = 20
            # res = self.opt.dataset.crop_size
                
            # for i in range(sample):
            #     recon, _ = self.modelG(self._get_high_res_input(64))
            #     self.visuals = {
            #                     'generated image': V.tensor_to_visual(recon),\
            #     }
            #     self.vis.display_current_results(self.visuals)
            #     time.sleep(1)

            self.vis.yell(f"zomming test!")
            for scale in np.arange(0.0, 3, 0.00001):
                scale_i = 4 - scale
                recon = self.modelG(self._get_high_res_input(64, scale=scale_i), True)
                self.visuals = {
                                'Zooming test': V.tensor_to_visual(recon),\
                }
                self.vis.display_current_results(self.visuals,10)
                self.vis.yell(f"zomming level at {scale_i}!")

            # self.vis.yell(f"panning test!")
            # for i in np.arange(-3, 3, 0.001):
            #     shift = torch.from_numpy(np.array([0,i],dtype=np.float32)[None,:,None,None]).to(self.opt.device)

            #     recon = self.modelG(self._get_input(shift=shift), True)
            #     self.visuals = {
            #                     'Panning test': V.tensor_to_visual(recon),\
            #     }
            #     self.vis.display_current_results(self.visuals,11)
                
            # for i in np.arange(-3, 3, 0.01):
            #     shift = torch.from_numpy(np.array([i,3],dtype=np.float32)[None,:,None,None]).to(self.opt.device)
            #     recon = self.modelG(self._get_input(shift=shift), True)
            #     self.visuals = {
            #                     'Panning test': V.tensor_to_visual(recon),\
            #     }
            #     self.vis.display_current_results(self.visuals,11)
                

if __name__ == '__main__':
    if opt.run_mode == 'test' or opt.run_mode == 'eval':
        opt = io.process_test_args(opt)
    trainer = Trainer(opt)
    if opt.run_mode == 'train':
        trainer.train()
    else:
        trainer.eval()
