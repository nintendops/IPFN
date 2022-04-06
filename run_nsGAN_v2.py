from configs.default import opt
from importlib import import_module
from tqdm import tqdm
import os
import numpy as np
import torch
import vgtk
from core.trainer import ImageTrainer
from core.loss import *
import utils.helper as H
import utils.io as io
import utils.visualizer as V
from dataset import TextureImageDataset 
from torch.autograd import Variable


class Trainer(ImageTrainer):

    def _setup_model(self):
        super(Trainer, self)._setup_model()
        self.modelG = self.model
        module = import_module('models')
        modelD_name = 'vanilla_netD_v2'
        print(f"Using discriminator model {modelD_name}!")
        self.modelD = getattr(module, modelD_name).build_model_from(self.opt, None)

    def _image_coords_to_spatial_coords(self):
        coords = H.get_position([self.source_h, self.source_w], self.opt.model.image_dim, \
                                     self.opt.device, self.opt.batch_size)
        scale = 1 / self.opt.model.patch_scale
        return scale * coords

    def mod_coords(self, coords):
        coords_y = coords[:,0]
        mod_y = torch.clamp(coords_y * self.opt.model.patch_scale, -1.0, 1.0)
        mod_y = (mod_y + 1)/2
        return mod_y.unsqueeze(1)

    def _optimize(self):

        scale = 1.0

        for p in self.modelD.parameters():
            p.requires_grad = True
        for i in range(self.opt.critic_steps):
            data = next(self.dataset_iter)
            img_patch, img_ref, _ = data
            img_coords = self._image_coords_to_spatial_coords()
            guidance_real = self.mod_coords(img_coords)

            self.modelD.zero_grad()
            
            # real data
            img_patch = img_patch.to(self.opt.device)

            # add with guidance feature
            img_patch = torch.cat([img_patch, guidance_real], 1)
            real_data = Variable(img_patch)

            D_real = self.modelD(real_data)
            D_real = -1 * D_real.mean()
            D_real.backward()

            # fake data
            coords, guidance = self._get_input()
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
            # self.model.eval()
            # with torch.no_grad():
            #     global_coords, global_guidance = self._get_input(scale=self.scale_factor, no_shift=True, up_factor=self.scale_factor)
            #     global_recon, global_z = self.modelG(torch.cat([global_coords,global_guidance],1), noise_factor=self.opt.model.noise_factor)
            # self.model.train()

            self.visuals = {'train_patch': V.tensor_to_visual(img_patch[:,:3]),\
                            'train_ref': V.tensor_to_visual(img_ref), 
                            'train_patch_recon': V.tensor_to_visual(p_recon[:,:3]), 
                            # 'train_global_recon': V.tensor_to_visual(global_recon[:,:3]),     
                            'noise_visual': V.tensor_to_visual(z[:,:3]),
                            # 'global_guidance': V.tensor_to_visual(global_guidance),   
                            # 'global_noise': V.tensor_to_visual(global_z[:,:3]),   
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
                

    def _get_input(self, shift=None):
        size = (self.source_h, self.source_w)
        scale = 1 / self.opt.model.patch_scale

        coords = H.get_position(size, self.opt.model.image_dim, \
                         self.opt.device, self.opt.batch_size)

        if self.opt.shift_type != 'none' and self.opt.run_mode == 'train':
            # only shift at horizontal direction
            shift = self._sample(0)[...,None,None]
            shift = shift.expand(self.opt.batch_size, self.opt.model.image_dim,*size).contiguous().to(self.opt.device)
            shift = torch.stack([torch.zeros_like(shift[:,0]), self.opt.shift_scale*shift[:,1]],1)
        else:
            shift = 0.0 if shift is None else shift
        coords = scale * (coords + shift) 
        return coords, self.mod_coords(coords)


    
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
            

            # self.vis.yell(f"Random high-res noise input!")
            # sample = 50
            # for i in range(sample):
            #     res = self.opt.model.image_res # self.opt.dataset.crop_size

            #     coords, guidance = self._get_high_res_input(res)
            #     recon, z = self.modelG(torch.cat([coords,guidance],1), noise_factor=self.opt.model.noise_factor)
            #     self.visuals = {
            #                     'generated image': V.tensor_to_visual(recon),\
            #                     'noise_visual': V.tensor_to_visual(z[:,:3]),
            #                     'guidance': V.tensor_to_visual(guidance),
            #     }
            #     self.vis.display_current_results(self.visuals)
                
            #     filename = f"sample_idx{i}.png"
            #     io.write_images(os.path.join(image_path,filename),V.tensor_to_visual(recon),1)
            #     time.sleep(1)

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
                

                


if __name__ == '__main__':
    if opt.run_mode == 'test' or opt.run_mode == 'eval':
        opt = io.process_test_args(opt)
    
    opt.shift_type = 'x'
    opt.shift_scale = 2.0
    opt.model.octaves = 1
    opt.model.global_res = 1.0
    opt.model.condition_on_guidance = True
    opt.dataset.transform_type = 'simple'

    trainer = Trainer(opt)
    if opt.run_mode == 'train':
        trainer.train()
    else:
        trainer.eval()
