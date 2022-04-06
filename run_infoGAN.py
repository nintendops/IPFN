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
        self.summary.register(['LossD', 'LossG', 'Gradient Norm', 'Info Loss'])        
        self.dist_shift = H.get_distribution_type([self.opt.batch_size, self.opt.model.image_dim], 'uniform')

    def _setup_datasets(self):
        '''
        In this version of multiscale training, the generator generates at a single large scale, while the discriminator disriminates at multiple scale
        '''
        dataset = TextureImageDataset(self.opt, octaves=1, transform_type='bottleneck')
        self.opt.dataset.crop_size = dataset.default_size
        self.dataest_size = len(dataset)
        self.n_iters = self.dataest_size / self.opt.batch_size
        self.dataset = torch.utils.data.DataLoader( dataset, \
                                                    batch_size=self.opt.batch_size, \
                                                    shuffle=False, \
                                                    num_workers=self.opt.num_thread)
    def _setup_model(self):
        self.opt.model.octaves = 2       
        if self.opt.run_mode == 'train':
            param_outfile = os.path.join(self.root_dir, "params.json")
        else:
            param_outfile = None
        module = import_module('models')
        modelG_name = 'bottleneck_mlp'
        print(f"Using network model {modelG_name}!")
        self.model = getattr(module, modelG_name).build_model_from(self.opt, param_outfile)
        self.modelG = self.model
        modelD_name = 'infoGAN_netD'
        print(f"Using discriminator model {modelD_name}!")
        self.modelD = getattr(module, modelD_name).build_model_from(self.opt, None)

    def _setup_optim(self):
        self.logger.log('Setup', 'Setup optimizer!')
        # torch.autograd.set_detect_anomaly(True)
        self.optimizerG = optim.Adam(self.modelG.parameters(), 
                                    lr=self.opt.train_lr.init_lr, betas=(0.5, 0.9))        
        self.optimizerD = optim.Adam(self.modelD.parameters(), 
                                    lr=self.opt.train_lr.init_lr, betas=(0.5, 0.9))
        # self.lr_schedule = vgtk.LearningRateScheduler(self.optimizer,
        #                                               **vars(self.opt.train_lr))
        self.logger.log('Setup', 'Optimizer all-set!')


    def _setup_metric(self):
        self.metric = get_loss_type('normal')
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
        scale = self.opt.model.image_res / self.opt.dataset.crop_size
        scale_weights = self._get_scale_weights()

        for p in self.modelD.parameters():
            p.requires_grad = True
        for i in range(self.opt.critic_steps):
            data = next(self.dataset_iter)
            img_patch, img_ref, filaname = data
            self.modelD.zero_grad()
            
            # real data
            img_ref = img_ref.to(self.opt.device)            
            img_patch = img_patch.to(self.opt.device)
            img_global = torch.nn.functional.interpolate(img_ref, \
                         size=[self.opt.model.global_res,self.opt.model.global_res], mode='bilinear')

            if self.opt.model.global_res == self.opt.model.image_res:
                real_data = Variable(torch.stack([img_global,img_patch],dim=1))
            else:
                real_data = [Variable(img_global),Variable(img_patch)]

            D_real, _, _ = self.modelD(real_data, scale_weights)
            D_real = -1 * D_real.mean()
            D_real.backward()

            # fake data        
            x_in = self._get_multiscale_input(scale=scale)

            if self.opt.model.global_res == self.opt.model.image_res:
                fake, _ = self.modelG(x_in, img_global)
                fake_data = fake.reshape(self.opt.batch_size,2,-1,self.opt.model.image_res,self.opt.model.image_res)
                fake_data = Variable(fake_data.data)
            else:
                fake_global, _ = self.modelG(x_in[0], img_global)
                fake_local, _ = self.modelG(x_in[1], img_global)
                fake_data = [Variable(fake_global.data), Variable(fake_local.data)]

            D_fake, _, _ = self.modelD(fake_data, scale_weights)
            D_fake = D_fake.mean()
            D_fake.backward()

            gradient_penality, gradient_norm = calc_gradient_penalty(self.modelD, real_data, fake_data, scale_weights)
            gradient_penality.backward()
            D_cost = D_fake + D_real + gradient_penality
            # Wasserstein_D = D_real - D_fake
            self.optimizerD.step()

        # -------------------- Train G -------------------------------------------
        for p in self.modelD.parameters():
            p.requires_grad = False
        self.modelG.zero_grad()        

        x_in = self._get_multiscale_input(scale=scale)
        if self.opt.model.global_res == self.opt.model.image_res:
            fake, z = self.modelG(x_in, img_global)
            fake_data = fake.reshape(self.opt.batch_size,2,-1,self.opt.model.image_res,self.opt.model.image_res)
        else:
            fake_global, z_global = self.modelG(x_in[0], img_global)
            fake_local, z_local = self.modelG(x_in[1], img_global)
            fake_data = [fake_global, fake_local]
            z = torch.cat([z_local, z_local], 0)
        # p_recon, z = self.modelG(self._get_multiscale_input(scale=scale), img_global)
        # fake_data = p_recon.reshape(self.opt.batch_size,2,-1,self.opt.model.image_res,self.opt.model.image_res)
        
        nb, nz, h, w = z.shape
        G_cost, mu, var = self.modelD(fake_data)    
        mutual_loss_weight = 10.0
        mutual_loss = self.metric(z.reshape(nb//2,2,nz,-1).mean(-1), mu, var)
        G_cost = -G_cost.mean() + mutual_loss_weight * mutual_loss
        G_cost.backward()
        self.optimizerG.step()

        log_info = {
                'LossG': G_cost.item(),
                'LossD': D_cost.item(),
                'Gradient Norm': gradient_norm,
                'Info Loss': mutual_loss.item(),
        }

        self.summary.update(log_info)

        z_visual = z[1:,:3]
        if self.opt.model.global_res == self.opt.model.image_res:
            fake_data_global_visual = fake_data[:,0]
            fake_data_local_visual = fake_data[:,1]
        else:
            fake_data_global_visual = fake_data[0]
            fake_data_local_visual = fake_data[1]

        # visuals
        if self.iter_counter % self.opt.log_freq == 0:
            self.visuals = {'train_patch': V.tensor_to_visual(img_patch),\
                            'train_ref': V.tensor_to_visual(img_global), \
                            'train_global_recon': V.tensor_to_visual(fake_data_global_visual),\
                            'train_patch_recon': V.tensor_to_visual(fake_data_local_visual), \
                            'noise_visual' : V.tensor_to_visual(z_visual),  
            }
            self.losses = {                
                    'LossG': G_cost.item(),
                    'LossD': D_cost.item(),
                    'Gradient Norm': gradient_norm,
                    'Info Loss': mutual_loss.item(),
                    }
    
    def _get_scale_weights(self, base_factor=2.0):
        it = self.iter_counter + self.n_iters * self.epoch_counter
        max_it = self.n_iters * self.opt.num_epochs
        it = max(it, max_it)

        # base_factor -> 1
        factor = base_factor ** (1.0 - ((max_it - it) / max_it))

        # progress by giving more weights towards finer level
        weights = factor**(np.arange(2))
        weights = weights / weights.sum()

        ##########################
        weights = np.array([0.0, 1.0])
        #########################

        return torch.from_numpy(weights).to(self.opt.device)


    def _get_multiscale_input(self, scale):
        global_coords = self._get_input(scale=1.0, res=self.opt.model.global_res)
        local_coords = self._get_input(scale=scale)
        b,c,h,w = local_coords.shape

        if self.opt.model.global_res == self.opt.model.image_res:
                coords = torch.cat([global_coords.unsqueeze(1), local_coords.unsqueeze(1)], 1).reshape(-1,c,h,w)
        else:
            coords = [global_coords, local_coords]
        return coords

    def _get_input(self, shift=0.0, scale=1.0, res=None):
        if 'conv' in self.opt.model.model:
            return None
        else:            
            res = self.opt.model.image_res if res is None else res
            size = (res, res)
            coords = H.get_position( size, self.opt.model.image_dim, \
                                     self.opt.device, self.opt.batch_size)

            if self.opt.run_mode == 'train':
                shift = self.dist_shift.sample()[...,None,None]
                shift = shift.expand(self.opt.batch_size, self.opt.model.image_dim, \
                                     res, res).contiguous().to(self.opt.device)
                return scale * coords + (1.0-scale)* shift
            else:
                return scale * (coords + shift)
    
    def _get_high_res_input(self, res=256):
        if 'conv' in self.opt.model.model:
            return None
        else:            
            size = (res, res)
            # scale = 0.5 * res / self.opt.model.image_res
            coords = H.get_position( size, self.opt.model.image_dim, \
                                     self.opt.device, self.opt.batch_size)
            # nb, c, h, w = coords.shape
            return coords
            
    def eval(self):
        import time
        self.logger.log('Testing','Evaluating test set!')
        self.model.eval()
        scale = self.opt.model.image_res / self.opt.dataset.crop_size
        with torch.no_grad():
            data = next(self.dataset_iter)
            img_patch, img_ref, filaname = data
            self.modelD.zero_grad()
            
            # real data
            img_ref = img_ref.to(self.opt.device)            
            img_patch = img_patch.to(self.opt.device)
            img_global = torch.nn.functional.interpolate(img_ref, \
                         size=[self.opt.model.image_res,self.opt.model.image_res], mode='bilinear')

            countdown = 5
            for i in range(countdown):
                self.vis.yell(f"Getting ready to test! Start in {5-i}...")
                time.sleep(1)
            
            self.vis.yell(f"First is random noise input!")
            sample = 10
            for i in range(sample):
                recon, z = self.modelG(self._get_input(scale=scale), img_global)
                self.visuals = {
                                'generated image': V.tensor_to_visual(recon),\
                                'noise_visual' : V.tensor_to_visual(z[:,:3]),  
                }
                self.vis.display_current_results(self.visuals)
                time.sleep(1)

            self.vis.yell(f"Random high-res noise input!")
            sample = 20
            for i in range(sample):
                recon, z = self.modelG(self._get_high_res_input(), img_global)
                self.visuals = {
                                'generated image': V.tensor_to_visual(recon),\
                                'noise_visual' : V.tensor_to_visual(z[:,:3]), 
                }
                self.vis.display_current_results(self.visuals)
                time.sleep(1)

            # self.vis.yell(f"zomming test!")
            # for i in np.arange(0.5, 2, 0.001):
            #     recon = self.modelG(self._get_input(scale=i), True)
            #     self.visuals = {
            #                     'Zooming test': V.tensor_to_visual(recon),\
            #     }
            #     self.vis.display_current_results(self.visuals,10)
                

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
    opt.model.octaves = 1
    trainer = Trainer(opt)
    if opt.run_mode == 'train':
        trainer.train()
    else:
        trainer.eval()