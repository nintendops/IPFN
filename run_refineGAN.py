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
import json
from copy import deepcopy

CONDITION_ON_SOURCE = False

class Trainer(BasicTrainer):
    def __init__(self, opt):
        super(Trainer, self).__init__(opt)
        self.dataset_iter = iter(self.dataset)
        self.summary.register(['LossD', 'LossG', 'Gradient Norm'])        

        self._corase_model_resume_from_ckpt(self.opt.coarse_path)

        # load this factor from checkpoint
        self.dist_shift = H.get_distribution_type([self.opt.batch_size, self.opt.model.image_dim], 'uniform')
        self.scale_factor = self.opt.dataset.crop_size / self.opt.model.global_res
        print(f"[MODEL] choosing a scale factor of {self.scale_factor}!!!")
        print(f"[MODEL] Patch original resolution at {self.opt.model.global_res}, which is downsampled to {self.opt.model.image_res}!!!")

    def _load_ckpt_config(self):
        resume_path = self.opt.coarse_path

        if resume_path is None:
            raise NotImplementedError('No pretrained coarse model is provided!')
        
        opt_path = os.path.join(os.path.dirname(os.path.dirname(resume_path)),'opt.txt')

        with open(opt_path,'r') as w:
            data = w.read()
            options = json.loads(data)

        self.c_options = options

    def _corase_model_resume_from_ckpt(self, resume_path):

        if resume_path is None:
            raise NotImplementedError('No pretrained coarse model is provided!')
        self.logger.log('Setup', f'Coarse model resumed from checkpoint: {resume_path}')

        state_dicts = torch.load(resume_path)

        # self.model = nn.DataParallel(self.model)
        self.modelC.load_state_dict(state_dicts)
        for p in self.modelC.parameters():
            p.requires_grad = False

        self.coarse_size = self.c_options['model']['global_res'] * self.opt.dataset.crop_size
        self.coarse_factor =  self.c_options['model']['global_res'] / (self.opt.model.global_res /  self.opt.dataset.crop_size)
        self.coarse_res = self.c_options['model']['image_res']


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
        self._load_ckpt_config()

        # TODO: setup based on model type
        opt.model.guidance_factor = self.c_options['model']['guidance_factor']
        opt.model.guidance_feature_type = self.c_options['model']['guidance_feature_type']

        if self.opt.run_mode == 'train':
            param_outfile = os.path.join(self.root_dir, "params.json")
        else:
            param_outfile = None

        module = import_module('models')
        modelG_name = 'refine_mlp'
        print(f"Using network model {modelG_name}!")
        self.model = getattr(module, modelG_name).build_model_from(self.opt, param_outfile)
        self.modelG = self.model

        c_opt = deepcopy(opt)
        c_opt.model.condition_on_guidance = True
        self.modelC = getattr(module, 'vanilla_mlp').build_model_from(c_opt, None)
        modelD_name = 'vanilla_netD'
        print(f"Using discriminator model {modelD_name}!")
        self.modelD = getattr(module, modelD_name).build_model_from(self.opt, None, c_in=6)
        self.downsample_factor = 0.5


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
    

    def mod_coords(self, coords):
        gr = self.coarse_size
        h = self.source_h
        w = self.source_w

        gf = self.opt.model.guidance_factor
        if gf > 0:
            gf = gf * gr / h

        if self.opt.model.guidance_feature_type == 'mody':
            coords_y = coords[:,0]
            if gf > 0:
                mod_y = torch.floor(gf * coords_y) / gf
                mod_y = torch.clamp(mod_y, - h/gr, h/gr) / (h/gr)
            else:
                mod_y = torch.clamp(coords_y / (h/gr), -1.0, 1.0)

            mod_y = (mod_y + 1)/2
            return mod_y.unsqueeze(1)

    def _image_coords_to_spatial_coords(self, crop_pos):
        coords = H.get_position([self.opt.model.image_res, self.opt.model.image_res], self.opt.model.image_dim, \
                                     self.opt.device, self.opt.batch_size)
        gr = self.opt.model.global_res
        h = self.source_h
        w = self.source_w
        top_coord = (h - 2 * crop_pos[:,0]) / gr
        left_coord = (2 * crop_pos[:,1] - w) / gr
        coords_shift = torch.stack([1 - top_coord, left_coord + 1],1)
        # coords_shift = torch.stack([left_coord + 1, top_coord - 1],1)
        coords = coords + coords_shift[...,None,None].to(self.opt.device)

        return coords

    def _resample(self, image, scale=1):
        if self.downsample_factor < 1.0:
            image = torch.nn.functional.interpolate(image, \
                            scale_factor= self.downsample_factor, \
                            mode='bilinear')
        image = torch.nn.functional.interpolate(image, \
                        size=[int(scale*self.opt.model.image_res),int(scale*self.opt.model.image_res)], \
                        mode='bilinear')
        return image

    def _optimize(self):

        scale = 1.0
        
        for p in self.modelD.parameters():
            p.requires_grad = True
        for i in range(self.opt.critic_steps):
            data = next(self.dataset_iter)
            img_patch, img_ref, crop_pos = data
            
            # img_coords = self._image_coords_to_spatial_coords(crop_pos)
            # guidance_real = self.mod_coords(img_coords)

            self.modelD.zero_grad()
            
            # real data
            img_patch = img_patch.to(self.opt.device)

            if CONDITION_ON_SOURCE:
                real_coarse_img = self._resample(img_patch)
            else:
                real_coarse_img = torch.nn.functional.interpolate(img_patch, \
                                size=[int((self.coarse_factor**-1) * self.opt.model.image_res),\
                                      int((self.coarse_factor**-1) * self.opt.model.image_res)], \
                                mode='bilinear')
                real_coarse_img = self._resample(real_coarse_img)

            img_patch = torch.nn.functional.interpolate(img_patch, \
                         size=[self.opt.model.image_res,self.opt.model.image_res], mode='bilinear')

            real_data = Variable(torch.cat([real_coarse_img,img_patch],1))
            D_real = self.modelD(real_data)
            D_real = -1 * D_real.mean()
            D_real.backward()

            # fake data
            coords, coarse_coords = self._get_input(scale=scale)

            if CONDITION_ON_SOURCE:
                coarse_img_fake = real_coarse_img
            else:
                coarse_img_fake, _ = self.modelC(coarse_coords)
                coarse_img_fake = self._resample(coarse_img_fake)

            fake, _ = self.modelG(torch.cat([coarse_img_fake, coords],1), noise_factor=1/scale)
            fake = torch.cat([coarse_img_fake,fake],1)
            # fake_data = (torch.cat([fake, guidance],1)).to(self.opt.device)
            fake_data = Variable(fake.data)
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
        self.modelG.zero_grad()        

        coords, coarse_coords = self._get_input(scale=scale)

        if CONDITION_ON_SOURCE:
            coarse_img_fake = real_coarse_img
        else:
            coarse_img_fake, _ = self.modelC(coarse_coords)
            coarse_img_fake = self._resample(coarse_img_fake)

        p_recon, z = self.modelG(torch.cat([coarse_img_fake, coords],1), noise_factor=1/scale)
        # fake_data = (torch.cat([p_recon, guidance],1)).to(self.opt.device)

        fake = torch.cat([coarse_img_fake,p_recon],1)
        G_cost = self.modelD(fake)
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
            self.model.eval()
            with torch.no_grad():
                global_coords, global_ccorrds = self._get_high_res_input()                    
                if CONDITION_ON_SOURCE:
                    global_coarse = torch.nn.functional.interpolate(img_ref.to(self.opt.device), \
                        size = (int(self.scale_factor * self.opt.model.image_res), int(self.scale_factor * self.opt.model.image_res)),
                        mode='bilinear')
                    global_coarse = self._resample(global_coarse, scale=self.scale_factor)
                else:
                    global_coarse, _ = self.modelC(global_ccorrds)
                    global_coarse = self._resample(global_coarse, scale=self.scale_factor)
                global_recon, global_z = self.modelG(torch.cat([global_coarse, global_coords],1), noise_factor=1/scale)
            self.model.train()

            self.visuals = {'train_patch': V.tensor_to_visual(img_patch),\
                            'train_ref': V.tensor_to_visual(img_ref), 
                            'train_patch_recon': V.tensor_to_visual(p_recon), 
                            'real_coarse_visual': V.tensor_to_visual(real_coarse_img),     
                            'fake_coarse_visual': V.tensor_to_visual(z),
                            'global_coarse': V.tensor_to_visual(global_coarse), 
                            'global_recon': V.tensor_to_visual(global_recon),   
            }

            self.losses = {                
                    'LossG': G_cost.item(),
                    'LossD': D_cost.item(),
                    'Gradient Norm': gradient_norm,}

    def _get_input(self, shift=0.0, scale=1.0, no_shift=False, up_factor=1.0, bs=None):
        if 'conv' in self.opt.model.model:
            return None
        else:            
            bs = self.opt.batch_size if bs is None else bs
            res = self.opt.model.image_res
            size = (int(up_factor*res), int(up_factor*res))
            coords = H.get_position( size, self.opt.model.image_dim, \
                                     self.opt.device, bs)

            ##### corase scale coords ##########
            coarse_res = int(up_factor * self.coarse_res / self.coarse_factor)
            coarse_size = (coarse_res,coarse_res)
            coarse_coords = H.get_position(coarse_size, self.opt.model.image_dim, \
                                     self.opt.device, bs)

            # DEBUG: shift corase coords
            if self.opt.shift_type != 'none' and self.opt.run_mode == 'train':
                coarse_shift = self.dist_shift.sample()[...,None,None]
                coarse_shift = coarse_shift.expand(self.opt.batch_size, self.opt.model.image_dim, \
                                     coarse_res, coarse_res).contiguous().to(self.opt.device)
                coarse_scale_factor = 1 / self.c_options['model']['global_res'] - 1 / self.coarse_factor
                coarse_coords = (1/self.coarse_factor) * coarse_coords + coarse_scale_factor * coarse_shift

            # if no_shift:
            #     coords = scale * coords
            # elif self.opt.shift_type != 'none' and self.opt.run_mode == 'train':
            #     shift = self.dist_shift.sample()[...,None,None]
            #     shift = shift.expand(self.opt.batch_size, self.opt.model.image_dim, \
            #                          res, res).contiguous().to(self.opt.device)

            #     if self.opt.shift_type == 'y':
            #         shift = torch.stack([shift[:,0], torch.zeros_like(shift[:,0]).to(self.opt.device)],1)
            #     elif self.opt.shift_type == 'x':
            #         scale_factor_y = self.scale_factor - 1
            #         scale_factor_x = 8.0
            #         # shift = torch.stack([torch.zeros_like(shift[:,1]).to(self.opt.device), shift[:,1]],1)
            #         shift = torch.stack([ scale_factor_y * shift[:,0], scale_factor_x * shift[:,1]],1)
            #         coords = scale * coords + shift
            #     else:
            #         shift_factor = self.scale_factor - 1 # 1 - scale if scale < 1 else self.scale_factor
            #         coords = scale * coords + shift_factor * shift
            # else:
            #     coords = scale * (coords + shift)

            coords = scale * coords
            coarse_coords = scale * coarse_coords
            coarse_guidance = self.mod_coords(coarse_coords)

            return coords, torch.cat([coarse_coords, coarse_guidance],1)

    def _get_high_res_input(self, bs=None):
        bs = self.opt.batch_size if bs is None else bs
        coarse_res = int(self.coarse_res / self.c_options['model']['global_res'])
        coarse_scale = 1 / self.c_options['model']['global_res']
        fine_res = int(self.scale_factor * self.opt.model.image_res)
        fine_scale = self.scale_factor


        coarse_coords = coarse_scale * H.get_position((coarse_res,coarse_res), self.opt.model.image_dim, \
                        self.opt.device, bs) 
        coarse_guidance = self.mod_coords(coarse_coords)

        coords = fine_scale * H.get_position((fine_res,fine_res), self.opt.model.image_dim, \
                                     self.opt.device, bs)
    
        return coords, torch.cat([coarse_coords, coarse_guidance],1)

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
            
            self.vis.yell(f"Random high-res noise input!")
            sample = 50

            data = next(self.dataset_iter)
            img_patch, img_ref, crop_pos = data

            for i in range(sample):                
                coords, coarse_coords = self._get_high_res_input()
                coarse_img_fake, _ = self.modelC(coarse_coords)
                coarse_img_fake = torch.nn.functional.interpolate(coarse_img_fake, \
                                scale_factor=self.downsample_factor, \
                                mode='bilinear')
                coarse_img_fake = torch.nn.functional.interpolate(coarse_img_fake, \
                                size=[  int(self.scale_factor*self.opt.model.image_res), \
                                        int(self.scale_factor*self.opt.model.image_res)], \
                                        mode='bilinear')

                ##################################
                # img_ref = img_ref.to(self.opt.device)
                # resampled_size = int(self.scale_factor * self.opt.model.image_res)
                # img_ref = torch.nn.functional.interpolate(img_ref, size=[resampled_size,resampled_size],mode='bilinear')
                # coarse_img_fake = torch.nn.functional.interpolate(img_ref, \
                #                     size=[int(self.downsample_factor * self.c_options['model']['global_res'] * resampled_size),\
                #                           int(self.downsample_factor * self.c_options['model']['global_res'] * resampled_size)], \
                #                     mode='bilinear') 
                # coarse_img_fake = torch.nn.functional.interpolate(coarse_img_fake, \
                #                 size=[  resampled_size, \
                #                         resampled_size], \
                #                         mode='bilinear')

                ##################################

                recon, z = self.modelG(torch.cat([coarse_img_fake, coords],1), noise_factor=1/scale)
                self.visuals = {
                                'generated image': V.tensor_to_visual(recon),\
                                # 'noise_visual': V.tensor_to_visual(z[:,:3]),
                                'coarse_img': V.tensor_to_visual(coarse_img_fake),
                                'img_ref': V.tensor_to_visual(img_ref),
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
                

            # self.vis.yell(f"panning test!")
            # for i in np.arange(-8, 8, 0.005):
            #     shift = torch.from_numpy(np.array([0,i],dtype=np.float32)[None,:,None,None]).to(self.opt.device)
                
            #     coords, guidance = self._get_input(shift=shift, scale=self.scale_factor*scale,up_factor=self.scale_factor)
            #     recon, z = self.modelG(torch.cat([coords,guidance],1), noise_factor=1/scale, fix_sample=True)
                
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
    # opt.model.condition_on_guidance = True

    trainer = Trainer(opt)
    if opt.run_mode == 'train':
        trainer.train()
    else:
        trainer.eval()