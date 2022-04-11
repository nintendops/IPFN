from configs.default import opt
from importlib import import_module
from tqdm import tqdm
import app
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import utils.helper as H
import utils.io as io
import utils.visualizer as V
import utils.guidance as GU
from dataset import TextureImageDataset, SDFDataset


class BasicTrainer(app.Trainer):
    def __init__(self, opt):
        super(BasicTrainer, self).__init__(opt)
        self._setup_visualizer()

    def _initialize_settings(self):        
        if self.opt.model.input_type == '2d':
            self.input_dim = 2
            self.modelG = 'vanilla_mlp'
            self.modelD = 'vanilla_netD'
        elif self.opt.model.input_type == '3d':
            self.input_dim = 3
            self.modelG = 'sdf_mlp'
            self.modelD = 'sdf_netD'
        else:
            raise NotImplementedError(f'Unsupported input type {self.opt.model.input_type}!')

        self.summary.register(self._get_summary())
        self.epoch_counter = 0
        self.iter_counter = 0

    def _get_guidance_function(self):
        if self.opt.guidance_feature_type == 'custom':
            self.guidance = GU.CustomGuidanceMapping()
        elif self.opt.guidance_feature_type == 'x':
            self.guidance = GU.ModXMapping()
        elif self.opt.guidance_feature_type == 'y':
            self.guidance = GU.ModYMapping()
        else:
            self.guidance = None

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
        size = min(self.original_size) if isinstance(self.original_size, tuple) else self.original_size
        return 1 / self.opt.model.crop_res if self.opt.model.crop_res <= 1.0 else size / self.opt.model.crop_res

    def _get_dist_shift(self):
        return H.get_distribution_type([self.opt.batch_size, self.input_dim], 'uniform')


    def _setup_datasets(self):
        dataset = self._get_dataset()

        # size info of the input exemplar
        self.original_size = dataset._get_original_size()
        self.cropped_size = dataset._get_cropped_size()
        self.size_info = list(self.cropped_size) + list(self.original_size)
        self.scale_factor = self._get_scale_factor()
        self.dist_shift = self._get_dist_shift()

        self.dataest_size = len(dataset)
        self.n_iters = self.dataest_size / self.opt.batch_size
        self.dataset = torch.utils.data.DataLoader( dataset, \
	                                                batch_size=self.opt.batch_size, \
	                                                shuffle=False, \
	                                                num_workers=self.opt.num_thread)
        self.dastaset_iter = iter(self.dataset)


        print(f"[Dataset] choosing a scale factor of {self.scale_factor}!!!")
        print(f"[Dataset] Input original size at {self.original_size}, which is cropped to {self.original_size / self.scale_factor}!!!")

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
            if self.guidance is not None:
                guidance_real = self.guidance.compute(data_patch, gu, self.size_info)
                data_patch = torch.cat([data_patch, guidance_real.to(self.opt.device)], 1)

            real_data = Variable(data_patch)
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

    def _get_input(self):
        pass

    def _print_running_stats(self, step):
        stats = self.summary.get()
        self.logger.log('Training', f'{step}: {stats}')
        # self.summary.reset(['Loss', 'Pos', 'Neg', 'Acc', 'InvAcc'])

    def test(self):
        pass

    def eval(self):
        pass


# class ImageTrainer(BasicTrainer):
#     def __init__(self, opt):
#         super(ImageTrainer, self).__init__(opt)
#         self.dataset_iter = iter(self.dataset)
#         self.summary.register(['LossD', 'LossG', 'kscale_x','kscale_y','Gradient Norm'])
#         self.dist_shift = H.get_distribution_type([self.opt.batch_size, self.opt.model.image_dim], 'uniform')
#         self.scale_factor = self.opt.dataset.crop_size / self.opt.model.global_res
#         print(f"[MODEL] choosing a scale factor of {self.scale_factor}!!!")
#         print(f"[MODEL] Patch original resolution at {self.opt.model.global_res}, which is downsampled to {self.opt.model.image_res}!!!")

#     def _setup_datasets(self):
#         '''
#         In this version of multiscale training, the generator generates at a single large scale, while the discriminator disriminates at multiple scale
#         '''
#         dataset = TextureImageDataset(self.opt, octaves=1, transform_type=self.opt.dataset.transform_type)
#         self.opt.dataset.crop_size = dataset.default_size
#         self.opt.model.global_res = dataset.global_res
#         self.source_w = dataset.default_w
#         self.source_h = dataset.default_h
#         self.opt.source_w = self.source_w
#         self.opt.source_h = self.source_h
#         self.dataest_size = len(dataset)
#         self.n_iters = self.dataest_size / self.opt.batch_size
#         self.dataset = torch.utils.data.DataLoader( dataset, \
#                                                     batch_size=self.opt.batch_size, \
#                                                     shuffle=False, \
#                                                     num_workers=self.opt.num_thread)

#     def _setup_optim(self):
#         self.logger.log('Setup', 'Setup optimizer!')
#         # torch.autograd.set_detect_anomaly(True)
#         self.optimizerG = optim.Adam(self.modelG.parameters(),
#                                     lr=self.opt.train_lr.init_lr, betas=(0.5, 0.9))
#         self.optimizerD = optim.Adam(self.modelD.parameters(),
#                                     lr=self.opt.train_lr.init_lr, betas=(0.5, 0.9))
#         # self.lr_schedule = app.LearningRateScheduler(self.optimizer,
#         #                                               **vars(self.opt.train_lr))
#         self.logger.log('Setup', 'Optimizer all-set!')


#     def _setup_metric(self):
#         self.metric = None
#         # self.l1_loss = get_loss_type('l1')

#     def train_epoch(self):
#         while self.epoch_counter <= self.opt.num_epochs:
#             try:
#                 self._optimize()
#                 self.iter_counter += 1
#             except StopIteration:
#                 self.epoch_counter += 1
#                 self.iter_counter = 0
#                 self.dataset_iter = iter(self.dataset)
#                 if self.epoch_counter > 0 and self.epoch_counter % self.opt.save_freq == 0:
#                     self._save_network(f'Epoch{self.epoch_counter}')

#             if self.iter_counter % self.opt.log_freq == 0:
#                 self._print_running_stats(f'Epoch {self.epoch_counter}')
#                 if hasattr(self, 'visuals'):
#                     self.vis.display_current_results(self.visuals)
#                 if hasattr(self, 'losses'):
#                     counter_ratio = self.opt.critic_steps * self.iter_counter / self.n_iters
#                     self.vis.plot_current_losses(self.epoch_counter, counter_ratio, self.losses)
#                 # self.vis.yell(f"Top value is {self.top.item()}, Left value is {self.left.item()}...")


#     def mod_coords(self, coords):
#         gr = self.opt.model.global_res
#         h = self.source_h
#         w = self.source_w

#         gf = self.opt.model.guidance_factor
#         if gf > 0:
#             gf = gf * gr / h if self.opt.model.guidance_feature_type == 'mody' else gf * gr / w

#         if self.opt.model.guidance_feature_type == 'mody':
#             coords_y = coords[:,0]
#             if gf > 0:
#                 mod_y = torch.floor(gf * coords_y) / gf
#                 mod_y = torch.clamp(mod_y, - h/gr, h/gr) / (h/gr)
#             else:
#                 mod_y = torch.clamp(coords_y / (h/gr), -1.0, 1.0)

#             mod_y = (mod_y + 1)/2
#             return mod_y.unsqueeze(1)
#         elif self.opt.model.guidance_feature_type == 'modx':
#             coords_x = coords[:,1]
#             if gf > 0:
#                 mod_x = torch.floor(gf * coords_x) / gf
#                 mod_x = torch.clamp(mod_x, - w/gr, w/gr) / (w/gr)
#             else:
#                 mod_x = torch.clamp(coords_x / (w/gr), -1.0, 1.0)
#             mod_x = (mod_x + 1)/2
#             return mod_x.unsqueeze(1)
#         elif self.opt.model.guidance_feature_type == 'modxy':
#             coords = torch.clamp(coords / (self.opt.dataset.crop_size/gr), -1.0, 1.0)
#             coords = (coords + 1) / 2
#             return coords

#     def _image_coords_to_spatial_coords(self, crop_pos):
#         coords = H.get_position([self.opt.model.image_res, self.opt.model.image_res], self.opt.model.image_dim, \
#                                      self.opt.device, self.opt.batch_size)
#         gr = self.opt.model.global_res
#         h = self.source_h
#         w = self.source_w
#         top_coord = (h - 2 * crop_pos[:,0]) / gr
#         left_coord = (2 * crop_pos[:,1] - w) / gr
#         coords_shift = torch.stack([1 - top_coord, left_coord + 1],1)
#         coords = coords + coords_shift[...,None,None].to(self.opt.device)
#         return coords
