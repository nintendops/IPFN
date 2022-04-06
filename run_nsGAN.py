from configs.default import opt
from importlib import import_module
from tqdm import tqdm
import os
import numpy as np
import torch
import vgtk
from core.trainer import BasicTrainer
from core.loss import *
import utils.helper as H
import utils.io as io
import utils.visualizer as V
from dataset import TextureImageDataset 
from torch.autograd import Variable


class Trainer(BasicTrainer):
    def __init__(self, opt):
        # torch.autograd.set_detect_anomaly(True)
        super(Trainer, self).__init__(opt)
        self.dataset_iter = iter(self.dataset)
        self.summary.register(['LossD', 'LossG', 'kscale_x','kscale_y','Gradient Norm'])        
        self.dist_shift = H.get_distribution_type([self.opt.batch_size, self.opt.model.image_dim], 'uniform')
        self.scale_factor = self.opt.dataset.crop_size / self.opt.model.global_res
        print(f"[MODEL] choosing a scale factor of {self.scale_factor}!!!")
        print(f"[MODEL] Patch original resolution at {self.opt.model.global_res}, which is downsampled to {self.opt.model.image_res}!!!")

        # self.sample_tlc = []
        # self.source_tlc = []

    def _setup_datasets(self):
        '''
        In this version of multiscale training, the generator generates at a single large scale, while the discriminator disriminates at multiple scale
        '''
        dataset = TextureImageDataset(self.opt, octaves=1, transform_type='positioned_crop')
        self.opt.dataset.crop_size = dataset.default_size
        self.opt.model.global_res = dataset.global_res
        self.source_w = dataset.default_w
        self.source_h = dataset.default_h
        self.dataest_size = len(dataset)
        self.n_iters = self.dataest_size / self.opt.batch_size
        self.dataset = torch.utils.data.DataLoader( dataset, \
                                                    batch_size=self.opt.batch_size, \
                                                    shuffle=False, \
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
                # self.vis.yell(f"Top value is {self.top.item()}, Left value is {self.left.item()}...")

    

    def mod_coords(self, coords):
        gr = self.opt.model.global_res
        h = self.source_h
        w = self.source_w

        gf = self.opt.model.guidance_factor
        if gf > 0:
            gf = gf * gr / h if self.opt.model.guidance_feature_type == 'mody' else gf * gr / w 

        if self.opt.model.guidance_feature_type == 'mody':
            coords_y = coords[:,0]
            if gf > 0:
                mod_y = torch.floor(gf * coords_y) / gf
                mod_y = torch.clamp(mod_y, - h/gr, h/gr) / (h/gr)
            else:
                mod_y = torch.clamp(coords_y / (h/gr), -1.0, 1.0)

            mod_y = (mod_y + 1)/2
            return mod_y.unsqueeze(1)
        elif self.opt.model.guidance_feature_type == 'modx':
            coords_x = coords[:,1]
            if gf > 0:
                mod_x = torch.floor(gf * coords_x) / gf
                mod_x = torch.clamp(mod_x, - w/gr, w/gr) / (w/gr)
            else:
                mod_x = torch.clamp(coords_x / (w/gr), -1.0, 1.0)
            mod_x = (mod_x + 1)/2
            return mod_x.unsqueeze(1)
        elif self.opt.model.guidance_feature_type == 'modxy':
            coords = torch.clamp(coords / (self.opt.dataset.crop_size/gr), -1.0, 1.0)
            coords = (coords + 1) / 2
            return coords

    def _image_coords_to_spatial_coords(self, crop_pos):
        coords = H.get_position([self.opt.model.image_res, self.opt.model.image_res], self.opt.model.image_dim, \
                                     self.opt.device, self.opt.batch_size)
        gr = self.opt.model.global_res
        h = self.source_h
        w = self.source_w
        # self.top = crop_pos[0,0]
        # self.left = crop_pos[0,1]
        top_coord = (h - 2 * crop_pos[:,0]) / gr
        left_coord = (2 * crop_pos[:,1] - w) / gr
        coords_shift = torch.stack([1 - top_coord, left_coord + 1],1)
        # coords_shift = torch.stack([left_coord + 1, top_coord - 1],1)
        coords = coords + coords_shift[...,None,None].to(self.opt.device)
        # tl_corner = coords[:,:,0,0].cpu().numpy()
        # self.source_tlc.append(tl_corner)
        return coords

    def _optimize(self):

        # one = torch.FloatTensor([1]).to(self.opt.device)
        # mone =(-1*one).to(self.opt.device)

        # -------------------- Train D -------------------------------------------
        # for p in self.modelG.parameters():
        #     p.requires_grad = False

        # scale = self.scale_factor
        scale = 1.0

        for p in self.modelD.parameters():
            p.requires_grad = True
        for i in range(self.opt.critic_steps):
            data = next(self.dataset_iter)
            img_patch, img_ref, crop_pos = data
            img_coords = self._image_coords_to_spatial_coords(crop_pos)
            guidance_real = self.mod_coords(img_coords)

            self.modelD.zero_grad()
            
            # real data
            img_patch = img_patch.to(self.opt.device)
            img_patch = torch.nn.functional.interpolate(img_patch, \
                         size=[self.opt.model.image_res,self.opt.model.image_res], mode='bilinear')
            # add with guidance feature
            img_patch = torch.cat([img_patch, guidance_real], 1)
            real_data = Variable(img_patch)

            D_real = self.modelD(real_data)
            D_real = -1 * D_real.mean()
            D_real.backward()

            # fake data
            coords, guidance = self._get_input(scale=scale)
            g_in = torch.cat([coords,guidance],1)
            fake_data, z = self.modelG(g_in.detach(), noise_factor=self.opt.model.noise_factor)
            fake_data = torch.cat([fake_data, guidance],1)
            fake = Variable(fake_data.data)
            D_fake = self.modelD(fake).mean()
            # D_fake.backward()
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
            # coords, guidance = self._get_input(scale=scale)
            p_recon, z = self.modelG(g_in.detach(), noise_factor=self.opt.model.noise_factor)
            fake_data = (torch.cat([p_recon, guidance],1)).to(self.opt.device)
            G_cost = self.modelD(fake_data)
            G_cost = -G_cost.mean()
            # G_cost.backward()
            G_cost.backward()         
            self.optimizerG.step()

        log_info = {
                'LossG': G_cost.item(),
                'LossD': D_cost.item(),
                'kscale_x' : self.modelG.K[1].item(),
                'kscale_y' : self.modelG.K[0].item(),
                'Gradient Norm': gradient_norm,
        }

        self.summary.update(log_info)

        # visuals
        if self.iter_counter % self.opt.log_freq == 0:

            # src_tlc = np.concatenate(self.source_tlc,0)
            # smp_tlc = np.concatenate(self.sample_tlc,0)
            # np.savetxt('log/src_tlc.csv',src_tlc,delimiter=',')
            # np.savetxt('log/smp_tlc.csv',smp_tlc,delimiter=',')
            # import ipdb; ipdb.set_trace()

            self.model.eval()
            with torch.no_grad():
                global_coords, global_guidance = self._get_input(scale=self.scale_factor, no_shift=True, up_factor=self.scale_factor)
                global_recon, global_z = self.modelG(torch.cat([global_coords,global_guidance],1), noise_factor=self.opt.model.noise_factor)
            self.model.train()

            self.visuals = {'train_patch': V.tensor_to_visual(img_patch[:,:3]),\
                            'train_ref': V.tensor_to_visual(img_ref), 
                            'train_patch_recon': V.tensor_to_visual(p_recon[:,:3]), 
                            'train_global_recon': V.tensor_to_visual(global_recon[:,:3]),     
                            'noise_visual': V.tensor_to_visual(z[:,:3]),
                            'global_guidance': V.tensor_to_visual(global_guidance),   
                            'global_noise': V.tensor_to_visual(global_z[:,:3]),   
                            'guidance_real': V.tensor_to_visual(guidance_real),   
                            'guidance_fake': V.tensor_to_visual(guidance),                     
            }

            self.losses = {                
                    'LossG': G_cost.item(),
                    'LossD': D_cost.item(),
                    'Gradient Norm': gradient_norm,}
            
    def _sample(self, dim=None):
        if dim is None:
            return H.exp_distribution(self.dist_shift.sample(), self.opt.sample_sigma)
        else:
            sample = self.dist_shift.sample()
            coords = []
            ndim = sample.shape[1]
            for i in range(ndim):
                coord = sample[:,i]
                if i == dim:
                    coord = H.exp_distribution(coord,self.opt.sample_sigma)
                coords.append(coord)
            return torch.stack(coords,1)
                
    def _get_input(self, shift=0.0, scale=1.0, no_shift=False, up_factor=1.0):
        if 'conv' in self.opt.model.model:
            return None
        else:            
            res = self.opt.model.image_res
            size = (int(up_factor*res), int(up_factor*res))
            coords = H.get_position( size, self.opt.model.image_dim, \
                                     self.opt.device, self.opt.batch_size)

            if no_shift:
                coords = scale * coords
            elif self.opt.shift_type != 'none' and self.opt.run_mode == 'train':
                # weighted sampling at y axis only
                shift = self._sample(0)[...,None,None]
                shift = shift.expand(self.opt.batch_size, self.opt.model.image_dim, \
                                     res, res).contiguous().to(self.opt.device)

                if self.opt.shift_type == 'y':
                    shift = torch.stack([shift[:,0], torch.zeros_like(shift[:,0]).to(self.opt.device)],1)
                elif self.opt.shift_type == 'x':
                    scale_factor_y = self.scale_factor - 1
                    scale_factor_x = 8.0
                    # shift = torch.stack([torch.zeros_like(shift[:,1]).to(self.opt.device), shift[:,1]],1)
                    shift = torch.stack([ scale_factor_y * shift[:,0], scale_factor_x * shift[:,1]],1)
                    coords = scale * coords + shift
                else:
                    shift_factor = self.scale_factor - 1 # 1 - scale if scale < 1 else self.scale_factor
                    coords = scale * coords + shift_factor * shift
            else:
                coords = scale * (coords + shift)

            guidance = self.mod_coords(coords)
            # tl_corner = coords[:,:,0,0].cpu().numpy()
            # self.sample_tlc.append(tl_corner)
            return coords, guidance
    
    def _get_high_res_input(self, res, train_scale=1.0):
        if 'conv' in self.opt.model.model:
            return None
        else:            
            scale_factor = self.scale_factor        
            size = (int(scale_factor * res), int(scale_factor * res))
            coords = H.get_position( size, self.opt.model.image_dim, \
                                     self.opt.device, self.opt.batch_size)

            guidance = self.mod_coords(scale_factor * coords)
            return scale_factor * coords, guidance 
    
    def eval(self):
        import time
        self.logger.log('Testing','Evaluating test set!')
        
        self.model.eval()
        self.logger.log('Model', 'Learned kernel scaling:')
        print(self.model.K)

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
                res = self.opt.model.image_res # self.opt.dataset.crop_size

                coords, guidance = self._get_high_res_input(res)
                recon, z = self.modelG(torch.cat([coords,guidance],1), noise_factor=self.opt.model.noise_factor)
                self.visuals = {
                                'generated image': V.tensor_to_visual(recon),\
                                'noise_visual': V.tensor_to_visual(z[:,:3]),
                                'guidance': V.tensor_to_visual(guidance),
                }
                self.vis.display_current_results(self.visuals)
                
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
                

            self.vis.yell(f"panning test!")
            for i in np.arange(-8, 8, 0.005):
                shift = torch.from_numpy(np.array([0,i],dtype=np.float32)[None,:,None,None]).to(self.opt.device)
                
                coords, guidance = self._get_input(shift=shift, scale=self.scale_factor*scale,up_factor=self.scale_factor)
                recon, z = self.modelG(torch.cat([coords,guidance],1), noise_factor=self.opt.model.noise_factor, fix_sample=True)
                
                self.visuals = {
                                'Panning test': V.tensor_to_visual(recon),\
                                'noise_visual': V.tensor_to_visual(z[:,:3]),
                }
                self.vis.display_current_results(self.visuals,11)
                
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
    opt.model.condition_on_guidance = True

    trainer = Trainer(opt)
    if opt.run_mode == 'train':
        trainer.train()
    else:
        trainer.eval()
