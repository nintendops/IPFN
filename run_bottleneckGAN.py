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
        self.source_res = int(self.opt.model.image_res * opt.model.source_scale)
        self.scale_factor = self.opt.dataset.crop_size / self.opt.model.global_res # self.opt.model.image_res

        print(f"[MODEL] choosing a scale factor of {self.scale_factor}!!!")
        print(f"[MODEL] Patch original resolution at {self.opt.model.global_res}, which is downsampled to {self.opt.model.image_res}!!!")


    def _setup_datasets(self):
        '''
        In this version of multiscale training, the generator generates at a single large scale, while the discriminator disriminates at multiple scale
        '''
        dataset = TextureImageDataset(self.opt, octaves=1, transform_type='bottleneck')
        self.opt.dataset.crop_size = dataset.default_size
        self.opt.model.global_res = dataset.global_res
        self.dataest_size = len(dataset)
        self.n_iters = self.dataest_size / self.opt.batch_size
        self.dataset = torch.utils.data.DataLoader( dataset, \
                                                    batch_size=self.opt.batch_size, \
                                                    shuffle=False, \
                                                    num_workers=self.opt.num_thread)
    def _setup_model(self):
        # self.opt.model.octaves = 2       
        if self.opt.run_mode == 'train':
            param_outfile = os.path.join(self.root_dir, "params.json")
        else:
            param_outfile = None
        module = import_module('models')
        modelG_name = 'bottleneck_mlp'
        print(f"Using network model {modelG_name}!")
        self.model = getattr(module, modelG_name).build_model_from(self.opt, param_outfile)
        self.modelG = self.model
        # modelD_name = 'multiscale_netD'
        modelD_name = 'vanilla_netD_v2'
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
        self.metric = None
        self.l1_loss = get_loss_type('l1') 

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
        # scale = self.opt.dataset.crop_size / self.opt.model.image_res 
        # scale_weights = self._get_scale_weights()

        for p in self.modelD.parameters():
            p.requires_grad = True
        for i in range(self.opt.critic_steps):
            data = next(self.dataset_iter)
            img_patch, img_ref, filaname = data
            self.modelD.zero_grad()
            
            # real data
            img_ref = img_ref.to(self.opt.device)     
            # img_ref_lr = torch.nn.functional.interpolate(img_ref, size=self.source_res, mode='bilinear')

            img_patch = img_patch.to(self.opt.device)
            img_patch = torch.nn.functional.interpolate(img_patch, \
                         size=[self.opt.model.image_res,self.opt.model.image_res], mode='bilinear')
            img_source = torch.nn.functional.interpolate(img_ref, \
                         size=[self.source_res,self.source_res], mode='bilinear')
            
            # lr_res = int(self.opt.model.image_res * self.source_res / self.opt.dataset.crop_size)

            # # downsample and upsample to blur the patch for conditional trainig
            # img_patch_lr = torch.nn.functional.interpolate(img_patch, size=[lr_res, lr_res], mode='bilinear')
            # img_patch_lr = torch.nn.functional.interpolate(img_patch_lr, size=[self.opt.model.image_res, self.opt.model.image_res], mode='bilinear')

            real_data = Variable(img_patch)
            # real_data = Variable(torch.cat([img_patch_lr,img_patch], dim=1))
            D_real = self.modelD(real_data)
            # D_real = self.modelD(real_data, scale_weights)
            D_real = -1 * D_real.mean()
            D_real.backward()

            # fake data        
            # x_in = self._get_multiscale_input(scale=scale)
            x_in = self._get_input(scale=1.0)

            # fake_global, _ = self.modelG(x_in[0], img_source, scale, fix_sample=True)
            fake_local, fake_lr = self.modelG(x_in, img_source, noise_factor=self.opt.model.noise_factor, z_scale=self.scale_factor)
            # fake_data = [Variable(fake_global.data), Variable(fake_local.data)]

            # fake_local = torch.cat([fake_lr, fake_local], dim=1)
            fake_data = Variable(fake_local.data)

            # D_fake = self.modelD(fake_data, scale_weights).mean()
            D_fake = self.modelD(fake_data).mean()            
            D_fake.backward()

            gradient_penality, gradient_norm = calc_gradient_penalty(self.modelD, real_data, fake_data)
            gradient_penality.backward()
            D_cost = D_fake + D_real + gradient_penality
            # Wasserstein_D = D_real - D_fake
            self.optimizerD.step()

        # -------------------- Train G -------------------------------------------
        for p in self.modelD.parameters():
            p.requires_grad = False

        for i in range(self.opt.g_steps):
            self.modelG.zero_grad()        

            x_in = self._get_input(scale=1.0)
            loss_reg = None

            fake_local, z_local = self.modelG(x_in, img_source, noise_factor=self.opt.model.noise_factor, z_scale=self.scale_factor)
            # fake_data = [fake_global, fake_local]
            # fake_data = torch.cat([z_local,fake_local], dim=1)        
            
            z = z_local # torch.cat([z_local, z_local], 0)

            #######################################
            # optional l1_loss?
            # fake_global, z_global = self.modelG(x_in[0], img_source, scale, fix_sample=True)
            # loss_reg = self.l1_loss(fake_global, img_global)
            #######################################
            
            G_cost = self.modelD(fake_local)
            G_cost = -G_cost.mean() # + 10.0 * loss_reg
            G_cost.backward()
            self.optimizerG.step()
        
        log_info = {
                'LossG': G_cost.item(),
                'LossD': D_cost.item(),
                'Gradient Norm': gradient_norm,
        }

        self.summary.update(log_info)

        z_visual = z[:,:3]

        # visuals
        if self.iter_counter % self.opt.log_freq == 0:
            # self.modelG.eval()
            # with torch.no_grad():
            #     fake_global, z_global = self.modelG(x_in[0], img_source, scale)
            # self.modelG.train()

            self.visuals = {'patch': V.tensor_to_visual(img_patch),\
                            # 'ref': V.tensor_to_visual(img_global), \
                            # 'global_recon': V.tensor_to_visual(fake_global),\
                            'patch_recon': V.tensor_to_visual(fake_local), \
                            'z_lr' : V.tensor_to_visual(z_visual),  
                            # 'patch_lr': V.tensor_to_visual(img_patch_lr),
            }

            self.losses = {           
                    'LossG': G_cost.item(),
                    'LossD': D_cost.item(),
                    'Gradient Norm': gradient_norm,}

            if loss_reg is not None:
                self.losses['Loss_reg'] = loss_reg.item()
                # self.losses['Loss_reg_2'] = loss_reg.item()
    
    # def _get_scale_weights(self, base_factor=2.0):
    #     it = self.iter_counter + self.n_iters * self.epoch_counter
    #     max_it = self.n_iters * self.opt.num_epochs
    #     it = max(it, max_it)

    #     # base_factor -> 1
    #     factor = base_factor ** (1.0 - ((max_it - it) / max_it))

    #     # progress by giving more weights towards finer level
    #     weights = factor**(np.arange(2))
    #     weights = weights / weights.sum()

    #     #####################################
    #     # weights = np.array([1.0, 0.0])
    #     #####################################

    #     return torch.from_numpy(weights).to(self.opt.device)



    def _get_input(self, shift=0.0, scale=1.0, no_shift=False, up_factor=1.0):
        if 'conv' in self.opt.model.model:
            return None
        else:            
            res = self.opt.model.image_res
            size = (int(up_factor*res), int(up_factor*res))
            coords = H.get_position( size, self.opt.model.image_dim, \
                                     self.opt.device, self.opt.batch_size)
            return scale * coords

            # if no_shift:
            #     return scale * coords
            # elif self.opt.shift_type != 'none' and self.opt.run_mode == 'train':
            #     shift = self.dist_shift.sample()[...,None,None]
            #     shift = shift.expand(self.opt.batch_size, self.opt.model.image_dim, \
            #                          res, res).contiguous().to(self.opt.device)

            #     if self.opt.shift_type == 'y':
            #         shift = torch.stack([shift[:,0], torch.zeros_like(shift[:,0]).to(self.opt.device)],1)
            #         shift_factor = 2 * self.scale_factor
            #     elif self.opt.shift_type == 'x':
            #         shift = torch.stack([torch.zeros_like(shift[:,1]).to(self.opt.device), shift[:,1]],1)
            #         shift_factor = 2 * self.scale_factor
            #     else:
            #         shift_factor = 1 - scale if scale < 1 else self.scale_factor
            #     return scale * coords + shift_factor * shift
            # else:
            #     return scale * (coords + shift)

    # def _get_input(self, shift=0.0, scale=1.0, shift_scale=0.0, res=None):
    #     if 'conv' in self.opt.model.model:
    #         return None
    #     else:            
    #         res = self.opt.model.image_res if res is None else res
    #         size = (res, res)
    #         coords = H.get_position( size, self.opt.model.image_dim, \
    #                                  self.opt.device, self.opt.batch_size)

    #         if self.opt.run_mode == 'train':
    #             shift = self.dist_shift.sample()[...,None,None]
    #             shift = shift.expand(self.opt.batch_size, self.opt.model.image_dim, \
    #                                  res, res).contiguous().to(self.opt.device)
    #             return scale * coords + shift_scale * shift
    #         else:
    #             return scale * (coords + shift)


    # def _get_multiscale_input(self, scale):
    #     global_coords = self._get_input(scale=scale, res=self.opt.model.global_res)
    #     local_coords = self._get_input(scale=1.0, shift_scale=scale-1)
    #     b,c,h,w = local_coords.shape

    #     # if self.opt.model.global_res == self.opt.model.image_res:
    #     #     coords = torch.cat([global_coords.unsqueeze(1), local_coords.unsqueeze(1)], 1).reshape(-1,c,h,w)
    #     # else:
    #     coords = [global_coords, local_coords]

    #     return coords

    
    def _get_high_res_input(self, scale, res):
        if 'conv' in self.opt.model.model:
            return None
        else:            
            size = (res, res)
            # scale = 0.5 * res / self.opt.model.image_res
            coords = H.get_position( size, self.opt.model.image_dim, \
                                     self.opt.device, self.opt.batch_size)
            # nb, c, h, w = coords.shape
            return scale * coords
            
    def eval(self):
        import time
        self.logger.log('Testing','Evaluating test set!')
        self.model.eval()
        scale = self.opt.dataset.crop_size / self.opt.model.image_res 

        # save path
        image_path = os.path.join(os.path.dirname(self.root_dir), 'visuals')
        os.makedirs(image_path,exist_ok=True)
        print(f"Saving output to {image_path}!!!")
        
        with torch.no_grad():
            data = next(self.dataset_iter)
            img_patch, img_ref, filaname = data
            self.modelD.zero_grad()
            
            # real data
            img_ref = img_ref.to(self.opt.device)            
            img_patch = img_patch.to(self.opt.device)
            # img_global = torch.nn.functional.interpolate(img_ref, \
            #              size=[self.opt.model.global_res,self.opt.model.global_res], mode='bilinear')

            img_source = torch.nn.functional.interpolate(img_ref, \
                         size=[self.source_res,self.source_res], mode='bilinear')
            img_source = torch.nn.functional.interpolate(img_source, \
                         size=[self.opt.model.global_res,self.opt.model.global_res], mode='bilinear')

            countdown = 5
            for i in range(countdown):
                self.vis.yell(f"Getting ready to test! Start in {5-i}...")
                time.sleep(1)
            
            # self.vis.yell(f"First is random noise input!")
            # sample = 10
            # for i in range(sample):
            #     recon, z = self.modelG(self._get_input(scale=1.0), img_global, scale)
            #     self.visuals = {
            #                     'generated': V.tensor_to_visual(recon),\
            #                     'noise_visual' : V.tensor_to_visual(z[:,:3]),  
            #     }
            #     self.vis.display_current_results(self.visuals)
            #     time.sleep(1)

            self.vis.yell(f"Random high-res noise input!")
            sample = 30
            for i in range(sample):

                high_res_input = self._get_high_res_input(scale, self.opt.dataset.crop_size)
                recon, z = self.modelG(high_res_input, img_source, scale)
                self.visuals = {
                                'generated': V.tensor_to_visual(recon),\
                                'noise_visual' : V.tensor_to_visual(z[:,:3]), 
                }
                self.vis.display_current_results(self.visuals)
                time.sleep(1)

                filename = f"sample_idx{i}.png"
                io.write_images(os.path.join(image_path,filename),V.tensor_to_visual(recon),1)
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
    # opt.model.condition_on_lr = True
    trainer = Trainer(opt)
    if opt.run_mode == 'train':
        trainer.train()
    else:
        trainer.eval()
