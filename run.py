from configs.default import opt
from importlib import import_module
from tqdm import tqdm
import os
import time
import numpy as np
import torch
from dataset import TextureImageDataset, Texture3dDataset, RGBDDataset 
from core.trainer import BasicTrainer
from core.loss import *
import utils.helper as H
import utils.io as io
import utils.visualizer as V
from torch.autograd import Variable


class Trainer(BasicTrainer):
    def __init__(self, opt):
        super(Trainer, self).__init__(opt)
        self.dataset_iter = iter(self.dataset)
        self.summary.register(['LossD', 'LossG', 'kscale_x','kscale_y','Gradient Norm'])        
        self.dist_shift = H.get_distribution_type([self.opt.batch_size, self.opt.model.image_dim], 'uniform')
        self.scale_factor = 8.0 #self.opt.dataset.crop_size / self.opt.model.global_res # self.opt.model.image_res

        print(f"[MODEL] choosing a scale factor of {self.scale_factor}!!!")
        print(f"[MODEL] Patch original resolution at {self.opt.model.global_res}, which is downsampled to {self.opt.model.image_res}!!!")


    def _setup_datasets(self):
        '''
        In this version of multiscale training, the generator generates at a single large scale, while the discriminator disriminates at multiple scale
        '''
        dataset = TextureImageDataset(self.opt, octaves=1, transform_type='default')
        # dataset = RGBDDataset(self.opt, octaves=1, transform_type='default')
        self.opt.dataset.crop_size = dataset.default_size
        self.opt.model.global_res = dataset.global_res
        self.dataest_size = len(dataset)
        self.n_iters = self.dataest_size / self.opt.batch_size
        self.dataset = torch.utils.data.DataLoader( dataset, \
                                                    batch_size=self.opt.batch_size, \
                                                    shuffle=True, \
                                                    num_workers=self.opt.num_thread)
    def _setup_model(self):
        super(Trainer, self)._setup_model()
        self.modelG = self.model
        module = import_module('models')
        print(f"Using discriminator model {self.opt.model.modelD}!")
        self.modelD = getattr(module, self.opt.model.modelD).build_model_from(self.opt, None)


    def _setup_optim(self):
        self.logger.log('Setup', 'Setup optimizer!')
        # torch.autograd.set_detect_anomaly(True)
        self.optimizerG = optim.Adam(self.modelG.parameters(), 
                                    lr=self.opt.train_lr.init_lr, betas=(0.5, 0.9))        
        self.optimizerD = optim.Adam(self.modelD.parameters(), 
                                    lr=self.opt.train_lr.init_lr, betas=(0.5, 0.9))
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

        # -------------------- Train D -------------------------------------------
        # for p in self.modelG.parameters():
        #     p.requires_grad = False

        scale = 1.0

        for p in self.modelD.parameters():
            p.requires_grad = True
        for i in range(self.opt.critic_steps):
            data = next(self.dataset_iter)
            img_patch, img_ref, _ = data
            self.modelD.zero_grad()

            
            # real data
            img_patch = img_patch.to(self.opt.device)
            img_patch = torch.nn.functional.interpolate(img_patch, \
                         size=[self.opt.model.image_res,self.opt.model.image_res], mode='bilinear')
            real_data = Variable(img_patch)

            D_real = self.modelD(real_data)
            D_real = -1 * D_real.mean()
            D_real.backward()

            # fake data
            g_in = self._get_input(scale=scale)
            fake, _ = self.modelG(g_in, noise_factor=self.opt.model.noise_factor)
            fake_data = fake.to(self.opt.device)
            fake_data = Variable(fake_data.data)
            D_fake = self.modelD(fake_data).mean()
            D_fake.backward()

            gradient_penality, gradient_norm = calc_gradient_penalty(self.modelD, real_data.data, fake_data.data)
            gradient_penality.backward()
            D_cost = D_fake + D_real + gradient_penality
            # Wasserstein_D = D_real - D_fake
            self.optimizerD.step()

        # -------------------- Train G -------------------------------------------
        for p in self.modelD.parameters():
            p.requires_grad = False

        for i in range(self.opt.g_steps):
            self.modelG.zero_grad()
            # p_recon, z = self.modelG(self._get_input(scale=scale), noise_factor=self.opt.model.noise_factor)
            p_recon, z = self.modelG(g_in, noise_factor=self.opt.model.noise_factor, fix_sample=self.opt.fix_sample)
            fake_data = p_recon
            G_cost = self.modelD(fake_data)
            G_cost = -G_cost.mean()
            G_cost.backward()
            self.optimizerG.step()

        kx = self.modelG.K[1].item() if 'conv' not in self.opt.model.model else 1.0
        ky = self.modelG.K[0].item() if 'conv' not in self.opt.model.model else 1.0

        log_info = {
                'LossG': G_cost.item(),
                'LossD': D_cost.item(),
                'kscale_x' : kx,
                'kscale_y' : ky,
                'Gradient Norm': gradient_norm,
        }

        self.summary.update(log_info)

        # visuals
        if self.iter_counter % self.opt.log_freq == 0:
            self.model.eval()
            with torch.no_grad():
                global_recon, global_z = self.modelG(\
                    self._get_input(scale=self.scale_factor, \
                                    no_shift=True, \
                                    up_factor=self.scale_factor), \
                    noise_factor=self.opt.model.noise_factor)
            self.model.train()

            self.visuals = {'train_patch': V.tensor_to_visual(img_patch[:,:3]),\
                            'train_ref': V.tensor_to_visual(img_ref[:,:3]), 
                            'train_patch_recon': V.tensor_to_visual(p_recon[:,:3]), 
                            'train_global_recon': V.tensor_to_visual(global_recon[:,:3]),     
                            'noise_visual': V.tensor_to_visual(z[:,:3]),
                            'global_noise': V.tensor_to_visual(global_z[:,:3]),                        
            }

            self.losses = {                
                    'LossG': G_cost.item(),
                    'LossD': D_cost.item(),
                    'Gradient Norm': gradient_norm,}

    def _get_input(self, shift=0.0, scale=1.0, no_shift=False, up_factor=1.0):
        if 'conv' in self.opt.model.model:
            return None
        else:            
            res = self.opt.model.image_res
            size = (int(up_factor*res), int(up_factor*res))
            coords = H.get_position( size, self.opt.model.image_dim, \
                                     self.opt.device, self.opt.batch_size)

            if no_shift:
                return scale * coords
            elif self.opt.shift_type != 'none' and self.opt.run_mode == 'train':
                if self.opt.shift_type == 'mid':
                    shift = torch.zeros(self.opt.batch_size, self.opt.model.image_dim, \
                                     res, res).contiguous().to(self.opt.device)
                    if np.random.rand() > 0.5:
                        shift += 0.5
                else:
                    shift = self.dist_shift.sample()[...,None,None]
                    shift = shift.expand(self.opt.batch_size, self.opt.model.image_dim, \
                                         res, res).contiguous().to(self.opt.device)

                if self.opt.shift_type == 'y':
                    shift = torch.stack([shift[:,0], torch.zeros_like(shift[:,0]).to(self.opt.device)],1)
                    shift_factor = 2 * self.scale_factor
                elif self.opt.shift_type == 'x':
                    shift = torch.stack([torch.zeros_like(shift[:,1]).to(self.opt.device), shift[:,1]],1)
                    shift_factor = 2 * self.scale_factor
                elif self.opt.shift_type == 'mid':
                    shift_factor = 1.0
                else:
                    shift_factor = 1 - scale if scale < 1 else self.scale_factor
                return scale * coords + shift_factor * shift
            else:
                return scale * (coords + shift)
    
    def _get_high_res_input(self, res, train_scale=1.0):
        if 'conv' in self.opt.model.model:
            return None
        else:      
            scale_factor = self.scale_factor # 8.0      
            size = (int(scale_factor*res), int(scale_factor*res))
            if train_scale < 1:
                scale = 1.0
            else:
                scale = scale_factor # res / self.opt.model.image_res
            coords = H.get_position( size, self.opt.model.image_dim, \
                                     self.opt.device, self.opt.batch_size)
            return scale * coords 
    
    def eval(self):
        import time
        self.logger.log('Testing','Evaluating test set!')
        
        self.model.eval()

        # save path
        image_path = os.path.join(os.path.dirname(self.root_dir), 'visuals')
        os.makedirs(image_path,exist_ok=True)
        print(f"Saving output to {image_path}!!!")

        with torch.no_grad():
            # scale = self.scale_factor
            scale = 1.0
            countdown = 5
            for i in range(countdown):
                self.vis.yell(f"Getting ready to test! Start in {5-i}...")
                time.sleep(1)
            
            # self.vis.yell(f"First is random noise input!")
            # sample = 10
            # for i in range(sample):
            #     recon, z = self.modelG(self._get_input(scale=scale), noise_factor=1/scale)
            #     self.visuals = {
            #                     'generated image': V.tensor_to_visual(recon),\
            #                     'noise_visual': V.tensor_to_visual(z[:,:3]),
            #     }
            #     self.vis.display_current_results(self.visuals)
            #     time.sleep(1)

            self.vis.yell(f"Random high-res noise input!")
            sample = 50
            for i in range(sample):
                start = time.time()
                res = self.opt.model.image_res #self.opt.dataset.crop_size
                recon, z = self.modelG(self._get_high_res_input(res, scale), noise_factor=self.opt.model.noise_factor)
                end  = time.time()                
                self.visuals = {
                                'generated image': V.tensor_to_visual(recon),\
                                'noise_visual': V.tensor_to_visual(z[:,:3]),
                }
                self.vis.display_current_results(self.visuals)
                print(f"Inference Time: {end-start} second!")
                filename = f"sample_idx{i}.png"
                io.write_images(os.path.join(image_path,filename),V.tensor_to_visual(recon),1)
                time.sleep(1)

            # self.vis.yell(f"zomming test!")
            # for i in np.arange(0.5, 2, 0.001):
            #     recon = self.modelG(self._get_input(scale=i), fix_sample=True)
            #     self.visuals = {
            #                     'Zooming test': V.tensor_to_visual(recon),\
            #     }
            #     self.vis.display_current_results(self.visuals,10)
                

            # self.vis.yell(f"panning test!")
            # for i in np.arange(-16, 16, 0.02):
            #     shift = torch.from_numpy(np.array([0,i],dtype=np.float32)[None,:,None,None]).to(self.opt.device)

            #     recon, z = self.modelG(self._get_input(shift=shift), fix_sample=True)
            #     self.visuals = {
            #                     'Panning test': V.tensor_to_visual(recon),\
            #                     'noise_visual': V.tensor_to_visual(z[:,:3]),
            #     }
            #     self.vis.display_current_results(self.visuals,11)
                
            # for i in np.arange(-3, 3, 0.01):
            #     shift = torch.from_numpy(np.array([i,3],dtype=np.float32)[None,:,None,None]).to(self.opt.device)
            #     recon = self.modelG(self._get_input(shift=shift), True)
            #     self.visuals = {
            #                     'Panning test': V.tensor_to_visual(recon),\
            #     }
            #    self.vis.display_current_results(self.visuals,11)
                


if __name__ == '__main__':
    if opt.run_mode == 'test' or opt.run_mode == 'eval':
        opt = io.process_test_args(opt)
    opt.model.octaves = 1
    trainer = Trainer(opt)
    if opt.run_mode == 'train':
        trainer.train()
    else:
        trainer.eval()