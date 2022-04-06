from configs.default import opt
from importlib import import_module
from tqdm import tqdm
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import vgtk
import utils.helper as H
import utils.io as io
import utils.visualizer as V
from dataset import TextureImageDataset

class BasicTrainer(vgtk.Trainer):
    def __init__(self, opt):
        super(BasicTrainer, self).__init__(opt)
        self.epoch_counter = 0
        self.iter_counter = 0

        self._setup_visualizer()

    def _setup_datasets(self):
        dataset = TextureImageDataset(self.opt)
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
        print(f"Using network model {self.opt.model.model}!")
        self.model = getattr(module, self.opt.model.model).build_model_from(self.opt, param_outfile)

    def _setup_metric(self):
    	pass

    def _setup_visualizer(self):
        self.vis = V.Visualizer(self.opt)

    def train_epoch(self):
        for i in range(self.opt.num_epochs):
            self.lr_schedule.step()
            self.epoch_step()

            if i > 0 and i % self.opt.save_freq == 0:
                self._save_network(f'Epoch{i}')


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

    def _optimize(self, data):
        pass

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

class ImageTrainer(BasicTrainer):
    def __init__(self, opt):
        super(ImageTrainer, self).__init__(opt)
        self.dataset_iter = iter(self.dataset)
        self.summary.register(['LossD', 'LossG', 'kscale_x','kscale_y','Gradient Norm'])
        self.dist_shift = H.get_distribution_type([self.opt.batch_size, self.opt.model.image_dim], 'uniform')
        self.scale_factor = self.opt.dataset.crop_size / self.opt.model.global_res
        print(f"[MODEL] choosing a scale factor of {self.scale_factor}!!!")
        print(f"[MODEL] Patch original resolution at {self.opt.model.global_res}, which is downsampled to {self.opt.model.image_res}!!!")

    def _setup_datasets(self):
        '''
        In this version of multiscale training, the generator generates at a single large scale, while the discriminator disriminates at multiple scale
        '''
        dataset = TextureImageDataset(self.opt, octaves=1, transform_type=self.opt.dataset.transform_type)
        self.opt.dataset.crop_size = dataset.default_size
        self.opt.model.global_res = dataset.global_res
        self.source_w = dataset.default_w
        self.source_h = dataset.default_h
        self.opt.source_w = self.source_w
        self.opt.source_h = self.source_h
        self.dataest_size = len(dataset)
        self.n_iters = self.dataest_size / self.opt.batch_size
        self.dataset = torch.utils.data.DataLoader( dataset, \
                                                    batch_size=self.opt.batch_size, \
                                                    shuffle=False, \
                                                    num_workers=self.opt.num_thread)

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
        top_coord = (h - 2 * crop_pos[:,0]) / gr
        left_coord = (2 * crop_pos[:,1] - w) / gr
        coords_shift = torch.stack([1 - top_coord, left_coord + 1],1)
        coords = coords + coords_shift[...,None,None].to(self.opt.device)
        return coords
