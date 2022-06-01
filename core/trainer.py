import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import utils.helper as H
import utils.io as io
import utils.visualizer as V
import utils.guidance as GU
from configs.default import opt
from importlib import import_module
from tqdm import tqdm
from torch.autograd import Variable
from core.loss import *
from dataset import TextureImageDataset, SDFDataset
from core.app import Trainer as BasicTrainer

class IPFNTrainer(BasicTrainer):
    def __init__(self, opt):
        super(IPFNTrainer, self).__init__(opt)
        self._setup_visualizer()

    def _initialize_settings(self):        
        self._initialize_guidance_function()
        if self.opt.model.input_type == '2d':
            self.modelG = 'vanilla_mlp'
            self.modelD = 'vanilla_netD'
        elif self.opt.model.input_type == '3d':
            self.modelG = 'sdf_mlp'
            self.modelD = 'sdf_netD'
        else:
            raise NotImplementedError(f'Unsupported input type {self.opt.model.input_type}!')

        self.summary.register(self._get_summary())
        self.epoch_counter = 0
        self.iter_counter = 0

    def _initialize_guidance_function(self):
        if self.opt.model.guidance_feature_type == 'custom':
            self.guidance = GU.CustomGuidanceMapping()
        elif self.opt.model.guidance_feature_type == 'x':
            self.guidance = GU.ModXMapping()
        elif self.opt.model.guidance_feature_type == 'y':
            self.guidance = GU.ModYMapping()
        else:
            self.guidance = None
        self.ifconditional = self.guidance is not None
        if self.ifconditional:
            self.opt.model.guidance_channel = self.guidance._get_channel()

    def _get_summary(self):
        return ['LossD', 'LossG', 'Gradient Norm']        

    def _get_dataset(self):
        if self.opt.model.input_type == '2d':
            dataset = TextureImageDataset(self.opt)
        elif self.opt.model.input_type == '3d':
            dataset = TextureImageDataset(self.opt)
        else:
            raise NotImplementedError(f'Unsupported input type {self.opt.model.input_type}!')
        return dataset

    def _get_scale_factor(self):
        # scale factor is the ratio of the original image size over the cropped patch size
        size = min(self.original_size) if isinstance(self.original_size, list) else self.original_size
        return 1 / self.opt.model.crop_res if self.opt.model.crop_res <= 1.0 else size / self.opt.model.crop_res

    def _get_dist_shift(self):
        return H.get_distribution_type([self.opt.batch_size, self.opt.model.image_dim], 'uniform')

    def _setup_datasets(self):
        dataset = self._get_dataset()

        # size info of the input exemplar
        self.original_size = dataset._get_original_size()
        self.crop_size = dataset._get_cropped_size()
        self.opt.model.crop_res = self.crop_size
        self.size_info = [self.crop_size] + self.original_size
        self.scale_factor = self._get_scale_factor()
        self.dist_shift = self._get_dist_shift()

        # settings for conditional model
        if self.ifconditional:
            self.opt.shift_factor = self.scale_factor   

        self.dataest_size = len(dataset)
        self.n_iters = self.dataest_size / self.opt.batch_size
        self.dataset = torch.utils.data.DataLoader( dataset, \
	                                                batch_size=self.opt.batch_size, \
	                                                shuffle=False, \
	                                                num_workers=self.opt.num_thread)
        self.dataset_iter = iter(self.dataset)

        print(f"[Dataset] choosing a scale factor of {self.scale_factor}!!!")
        print(f"[Dataset] Input original size at {self.original_size}, which is cropped to {self.crop_size}!!!")

    def _setup_model(self):        
        if self.opt.run_mode == 'train':
            param_outfile = os.path.join(self.root_dir, "params.json")
        else:
            param_outfile = None
        module = import_module('models')
        print(f"Using network model {self.modelG}!")
        self.model = getattr(module, self.modelG).build_model_from(self.opt, param_outfile)
        self.modelG = self.model
        module = import_module('models')
        print(f"Using discriminator model {self.modelD}!")
        self.modelD = getattr(module, self.modelD).build_model_from(self.opt, None)

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

    def _setup_visualizer(self):
        self.vis = V.Visualizer(self.opt)


    def _setup_model_multi_gpu(self):
        if torch.cuda.device_count() > 1:
            self.logger.log('Setup', 'Using Multi-gpu and DataParallel!')
            self._use_multi_gpu = True
            self.modelG = nn.DataParallel(self.modelG)
        else:
            self.logger.log('Setup', 'Using Single-gpu!')
            self._use_multi_gpu = False

    def _save_network(self, step, label=None):
        label = self.opt.experiment_id if label is None else label
        save_filename_netG = '%s_netG_%s.pth' % (label, step)
        save_path_netG = os.path.join(self.root_dir, 'ckpt', save_filename_netG)
        save_filename_netD = '%s_netD_%s.pth' % (label, step)
        save_path_netD = os.path.join(self.root_dir, 'ckpt', save_filename_netD)

        if self._use_multi_gpu:
            params = self.modelG.module.cpu().state_dict()
            params_netD = self.modelD.module.cpu().state_dict()
        else:
            params = self.modelG.cpu().state_dict()
            params_netD = self.modelD.cpu().state_dict()

        torch.save(params, save_path_netG)
        torch.save(params_netD, save_path_netD)

        if torch.cuda.is_available():
            # torch.cuda.device(gpu_id)
            self.modelG.to(self.opt.device)
            self.modelD.to(self.opt.device)
            
        self.logger.log('Training', f'Checkpoint saved to: {save_path_netG}!')

    def _resume_from_ckpt(self, resume_path):
        if resume_path is None:
            self.logger.log('Setup', f'Seems like we train from scratch!')
            return
        self.logger.log('Setup', f'Resume from checkpoint: {resume_path}')

        resume_path_netD = resume_path.replace('netG', 'netD')
        state_dicts = torch.load(resume_path)
        state_dicts_netD = torch.load(resume_path_netD)

        # self.model = nn.DataParallel(self.model)
        self.model.load_state_dict(state_dicts)
        if self.opt.run_mode == 'train':
            self.modelD.load_state_dict(state_dicts_netD)
        # self.model = self.model.module
        # self.optimizer.load_state_dict(state_dicts['optimizer'])
        # self.start_epoch = state_dicts['epoch']
        # self.start_iter = state_dicts['iter']
        self.logger.log('Setup', f'Resume finished! Great!')


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

        # set the scale of input grid in real coordinate space for adversarial training
        scale = 1.0

        for p in self.modelD.parameters():
            p.requires_grad = True
        for i in range(self.opt.critic_steps):

            data = next(self.dataset_iter)
            data_patch, data_ref, gu = data
            
            self.modelD.zero_grad()
            
            # real data
            data_patch = data_patch.to(self.opt.device)

            # (optional) subsample data further (deprecated)
            # data_patch = torch.nn.functional.interpolate(data_patch, \
            #              size=[self.opt.model.image_res,self.opt.model.image_res], mode='bilinear')

            # conditional model
            if self.ifconditional:
                guidance_real = self.guidance.compute(data_patch, gu, self.size_info)
                data_patch = torch.cat([data_patch, guidance_real.to(self.opt.device)], 1)

            real_data = Variable(data_patch)
            D_real = self.modelD(real_data)
            D_real = -1 * D_real.mean()
            D_real.backward()

            # fake data
            g_in = H._get_input(self.crop_size, self.dist_shift, self.opt)
            if self.ifconditional:
                guidance_fake = self.guidance.compute(None, g_in, [0,self.scale_factor,self.scale_factor])
                g_in = torch.cat([g_in, guidance_fake.to(self.opt.device)],1)

            fake, _ = self.modelG(g_in, noise_factor=self.opt.model.noise_factor)

            if self.ifconditional:
                fake = torch.cat([fake, guidance_fake.to(self.opt.device)],1)

            fake_data = Variable(fake.data)
            D_fake = self.modelD(fake_data).mean()
            D_fake.backward()

            gradient_penality, gradient_norm = calc_gradient_penalty(self.modelD, real_data.data, fake_data.data)
            gradient_penality.backward()
            D_cost = D_fake + D_real + gradient_penality
            self.optimizerD.step()

        # -------------------- Train G -------------------------------------------
        for p in self.modelD.parameters():
            p.requires_grad = False

        for i in range(self.opt.g_steps):
            self.modelG.zero_grad()
            p_recon, z = self.modelG(g_in, noise_factor=self.opt.model.noise_factor, fix_sample=self.opt.fix_sample)
            fake_data = p_recon if self.guidance is None else torch.cat([p_recon, guidance_fake.to(self.opt.device)], 1)
            G_cost = self.modelD(fake_data)
            G_cost = -G_cost.mean()
            G_cost.backward()
            self.optimizerG.step()

        log_info = {
                'LossG': G_cost.item(),
                'LossD': D_cost.item(),
                'Gradient Norm': gradient_norm,
        }

        self.summary.update(log_info)

        # visualization (only applicable in 2D cases or can be modified to visualize slices of the 3D volume)
        if self.opt.model.input_type == '2d' and self.iter_counter % self.opt.log_freq == 0:
            self.model.eval()
            with torch.no_grad():
                # eval: synthesize a larger texture (4x scale)
                if self.ifconditional:
                    g_in = H._get_input(self.crop_size * self.scale_factor, self.dist_shift, self.opt, scale=self.scale_factor, shift=0.0)
                    guidance_fake = self.guidance.compute(None, g_in, [0,self.scale_factor,self.scale_factor])
                    g_in = torch.cat([g_in, guidance_fake.to(self.opt.device)],1)
                else:
                    g_in = H._get_input(self.crop_size * 4.0, self.dist_shift, self.opt, scale=4.0)
                global_recon, global_z = self.modelG(g_in, noise_factor=self.opt.model.noise_factor)

            self.model.train()

            self.visuals = {'train_patch': V.tensor_to_visual(data_patch[:,:3]),\
                            'train_ref': V.tensor_to_visual(data_ref[:,:3]), 
                            'train_patch_recon': V.tensor_to_visual(p_recon[:,:3]), 
                            'train_global_recon': V.tensor_to_visual(global_recon[:,:3]),     
                            'noise_visual': V.tensor_to_visual(z[:,:3]),
                            'global_noise': V.tensor_to_visual(global_z[:,:3]),                        
            }

            if self.ifconditional:
                self.visuals['guidance_real'] = V.tensor_to_visual(guidance_real)
                self.visuals['guidance_fake'] = V.tensor_to_visual(guidance_fake)

            self.losses = {                
                    'LossG': G_cost.item(),
                    'LossD': D_cost.item(),
                    'Gradient Norm': gradient_norm,}


    # For epoch-based training
    def epoch_step(self):
        self.iter_counter = 0
        i = self.epoch_counter
        for it, data in tqdm(enumerate(self.dataset)):
            self._optimize(data)
            self.iter_counter += 1

            if it % self.opt.log_freq == 0:
                self._print_running_stats(f'Epoch {i} Iter{it}/{self.n_iters}')
                if hasattr(self, 'visuals'):
                    self.vis.display_current_results(self.visuals)
                if hasattr(self, 'losses'):
                    counter_ratio = it / self.n_iters
                    self.vis.plot_current_losses(i, counter_ratio, self.losses)

        if i > 0 and i % self.opt.save_freq == 0:
            # self.test()
            self._save_network(f'Epoch{i}')
        self.epoch_counter += 1

    # For iter-based training
    def step(self):
        try:
            data = next(self.dataset_iter)
            if data['input'].shape[0] < self.opt.batch_size:
                raise StopIteration
        except StopIteration:
            # New epoch
            self.epoch_counter += 1
            # print("[DataLoader]: At Epoch %d!"%self.epoch_counter)
            self.dataset_iter = iter(self.dataset)
            data = next(self.dataset_iter)
        self._optimize(data)
        self.iter_counter += 1

    def _print_running_stats(self, step):
        stats = self.summary.get()
        self.logger.log('Training', f'{step}: {stats}')
        # self.summary.reset(['Loss', 'Pos', 'Neg', 'Acc', 'InvAcc'])

    def _synthesize(self, res, scale=1.0, shift=0.0):
        if self.ifconditional:
            g_in = H._get_input(res, self.dist_shift, self.opt, scale=scale, shift=shift)
            guidance_fake = self.guidance.compute(None, g_in, [0,scale,scale])
            g_in = torch.cat([g_in, guidance_fake.to(self.opt.device)],1)
        else:
            g_in = H._get_input(res, self.dist_shift, self.opt, scale=scale, shift=shift)
            guidance_fake = None
        global_recon, global_z = self.modelG(g_in, noise_factor=self.opt.model.noise_factor)
        return global_recon, global_z, guidance_fake


    def eval(self):
        self.logger.log('Testing','Evaluating test set!')        
        self.model.eval()

        '''
        Eval options: 
            - Synthesize test: synthesize enlarged samples of the pattern
            - Zoom out test: create gif of a zooming out views of the pattern
            - Panning test: create gif of a moving views of the pattern

        '''
        # save path
        image_path = os.path.join(os.path.dirname(self.root_dir), 'visuals')
        os.makedirs(image_path,exist_ok=True)
        self.logger.log('Testing', f"Saving output to {image_path}!!!")

        with torch.no_grad():

            self.crop_size *= self.opt.test_scale
            self.scale_factor = self.opt.test_scale

            countdown = 5
            for i in range(countdown):
                self.vis.yell(f"Getting ready to test! Start in {5-i}...")
                time.sleep(1)
            
            # self.vis.yell(f"zomming test!")
            # max_res = 16.0
            # zoom_images = []
            # for i in np.arange(4.0, max_res, 0.05):
            #     g_in = H._get_input(self.crop_size, None, self.opt, scale=i, shift=0.0)
            #     recon, z = self.modelG(g_in, noise_factor=self.opt.model.noise_factor, fix_sample=True)
            #     recon_np = V.tensor_to_visual(recon)
            #     self.visuals = {
            #                     'Zooming test': recon_np,\
            #     }
            #     zoom_images.append(recon_np)
            #     self.vis.display_current_results(self.visuals,10)
            

            # io.write_gif(os.path.join(image_path, 'sample_zoom.gif'), zoom_images)
            # import ipdb; ipdb.set_trace()

            self.vis.yell(f"Random synthesized pattern at {self.scale_factor}X size!")
            sample = 10

            for i in range(sample):

                if self.ifconditional:
                    g_in = H._get_input(self.crop_size, self.dist_shift, self.opt, scale=self.scale_factor, shift=0.0)
                    guidance_fake = self.guidance.compute(None, g_in, [0,self.scale_factor,self.scale_factor])
                    g_in = torch.cat([g_in, guidance_fake.to(self.opt.device)],1)
                else:
                    g_in = H._get_input(self.crop_size, self.dist_shift, self.opt, scale=self.scale_factor)
                recon, z = self.modelG(g_in, noise_factor=self.opt.model.noise_factor)

                self.visuals = {
                                'generated image': V.tensor_to_visual(recon),\
                                'noise_visual': V.tensor_to_visual(z[:,:3]),
                }
                self.vis.display_current_results(self.visuals)
                
                filename = f"sample_{self.scale_factor}x_idx{i}.png"
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
            #     recon, z = self.modelG(torch.cat([coords,guidance],1), noise_factor=self.opt.model.noise_factor, fix_sample=True)
                
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
                
