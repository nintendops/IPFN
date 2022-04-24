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
from dataset import TerrainImageDataset
from core.app import Trainer as BasicTrainer


class ProceduralTrainer(BasicTrainer):
    def __init__(self, opt):
        super(ProceduralTrainer, self).__init__(opt)
        self._setup_visualizer()

    def _initialize_settings(self):        
        self.modelG = 'procedural_convG'
        self.modelD = 'vanilla_netD'
        self.summary.register(self._get_summary())
        self.epoch_counter = 0
        self.iter_counter = 0

    def _get_summary(self):
        return ['LossD', 'LossG', 'Gradient Norm']        

    def _get_dataset(self):
        dataset = TerrainImageDataset(self.opt)
        return dataset

    def _setup_datasets(self):
        dataset = self._get_dataset()

        # size info of the input exemplar
        self.original_size = dataset._get_original_size()
        self.crop_size = dataset._get_cropped_size()
        self.opt.model.image_res = self.crop_size
        self.crop_portion = int(self.crop_size * self.opt.model.portion)
        self.size_info = [self.crop_size] + self.original_size
        
        self.dataest_size = len(dataset)
        self.n_iters = self.dataest_size / self.opt.batch_size
        self.dataset = torch.utils.data.DataLoader( dataset, \
	                                                batch_size=self.opt.batch_size, \
	                                                shuffle=False, \
	                                                num_workers=self.opt.num_thread)
        self.dataset_iter = iter(self.dataset)


    def _setup_model(self):        
        if self.opt.run_mode == 'train':
            param_outfile = os.path.join(self.root_dir, "params.json")
        else:
            param_outfile = None
        module = import_module('models')
        print(f"Using network model {self.modelG}!")
        self.model = getattr(module, self.modelG).build_model_from(self.opt, self.opt.model.portion, param_outfile)
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

        for p in self.modelD.parameters():
            p.requires_grad = True
        for i in range(self.opt.critic_steps):

            data = next(self.dataset_iter)

            data_real, data_ref_original, data_ref = data

            # nb, nc, hr, wr = data_real.shape
            # nb, _, href, wref = data_ref.shape
            # assert wref == wr and hr > href
            # # padding reference image with zeros
            # paddings = torch.zeros([nb, nc, href-hr, wr])
            # data_patch = torch.cat([paddings, data_ref],2)

            self.modelD.zero_grad()
            
            # real data
            data_real = data_real.to(self.opt.device)

            # (optional) subsample data further (deprecated)
            # data_patch = torch.nn.functional.interpolate(data_patch, \
            #              size=[self.opt.model.image_res,self.opt.model.image_res], mode='bilinear')

            real_data = Variable(data_real)
            D_real = self.modelD(real_data)
            D_real = -1 * D_real.mean()
            D_real.backward()

            # fake data
            g_in = data_ref.to(self.opt.device)            
            fake, _ = self.modelG(g_in)
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
            p_recon, z = self.modelG(g_in)
            fake_data = p_recon 
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

            self.visuals = {'train_real': V.tensor_to_visual(data_real[:,:3]), 
                            'train_ref_recon': V.tensor_to_visual(p_recon[:,:3]), 
                            'train_ref_original': V.tensor_to_visual(data_ref_original[:,:3]),
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

    def _print_running_stats(self, step):
        stats = self.summary.get()
        self.logger.log('Training', f'{step}: {stats}')
        # self.summary.reset(['Loss', 'Pos', 'Neg', 'Acc', 'InvAcc'])

    def eval(self):
        self.logger.log('Testing','Evaluating test set!')        
        self.model.eval()

        '''
        Eval options: 


        '''
        # save path
        image_path = os.path.join(os.path.dirname(self.root_dir), 'visuals')
        os.makedirs(image_path,exist_ok=True)
        self.logger.log('Testing', f"Saving output to {image_path}!!!")

        with torch.no_grad():
            pass
