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
from dataset import FoamLoader_v2 as FoamLoader
from torch.autograd import Variable


class Trainer(BasicTrainer):
    def __init__(self, opt):
        super(Trainer, self).__init__(opt)
        self.summary.register(['LossD', 'LossG', 'kscale_x','kscale_y','Gradient Norm'])
        self.dist_shift = H.get_distribution_type([self.opt.batch_size, self.opt.model.image_dim], 'uniform')
        self.scale_factor = 2.0
        print(f"[MODEL] Grid resolution at {self.opt.model.image_res}!")


    def _setup_datasets(self):
        if self.opt.run_mode == 'train':
            dataset = FoamLoader(self.opt)
            self.opt.model.image_res = dataset.res
            self.dataest_size = len(dataset)
            self.n_iters = self.dataest_size / self.opt.batch_size
            self.dataset = torch.utils.data.DataLoader( dataset, \
                                                        batch_size=self.opt.batch_size, \
                                                        shuffle=True, \
                                                        num_workers=self.opt.num_thread)
            self.dataset_iter = iter(self.dataset)
        else:
            self.dataset = None

    def _setup_model(self):
        if self.opt.run_mode == 'train':
            param_outfile = os.path.join(self.root_dir, "params.json")
        else:
            param_outfile = None
        module = import_module('models')
        modelG_name = 'sdf_mlp'
        print(f"Using network model {modelG_name}!")
        self.model = getattr(module, modelG_name).build_model_from(self.opt, param_outfile)
        self.modelG = self.model

        modelD_name = "sdf_netD"
        # modelD_name = "vanilla_netD"
        print(f"Using discriminator model {modelD_name}!")
        self.modelD = getattr(module, modelD_name).build_model_from(self.opt, None, c_in=2)

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

    def _optimize(self):
        scale = 1.0

        for p in self.modelD.parameters():
            p.requires_grad = True
        for i in range(self.opt.critic_steps):
            data = next(self.dataset_iter)
            sdf, g_real = data
            # g_real = torch.zeros_like(g_real) + 1.0

            self.modelD.zero_grad()

            # real data
            sdf = sdf.to(self.opt.device)
            g_real = g_real.to(self.opt.device)

            # real_data = Variable(sdf)
            real_data = Variable(torch.cat([sdf,g_real],1))

            D_real = self.modelD(real_data)

            D_real = -1 * D_real.mean()
            D_real.backward()

            # # fake data
            x_in = self._get_input()
            # x_in = x_in[:,:2,...,0]

            g_shape = [self.opt.batch_size,\
                       1,\
                       self.opt.model.image_res,\
                       self.opt.model.image_res,\
                       self.opt.model.image_res]

            g_in = self._get_guidance(g_shape, 'uniform')
            x_in = torch.cat([x_in, g_in], 1)

            fake, _ = self.modelG(x_in.detach(), noise_factor=self.opt.model.noise_factor)
            fake = torch.cat([fake,g_in], 1)
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
            p_recon, z = self.modelG(x_in.detach(), noise_factor=self.opt.model.noise_factor, fix_sample=True)
            fake_data = p_recon
            fake_data = torch.cat([p_recon,g_in],1)
            G_cost = self.modelD(fake_data)
            G_cost = -G_cost.mean()
            G_cost.backward()
            self.optimizerG.step()

        log_info = {
                'LossG': G_cost.item(),
                'LossD': D_cost.item(),
                'kscale_x' : self.modelG.getK()[1].item(),
                'kscale_y' : self.modelG.getK()[0].item(),
                'Gradient Norm': gradient_norm,
        }

        self.summary.update(log_info)

        # visuals
        if self.iter_counter % self.opt.log_freq == 0:
            # self.model.eval()
            # with torch.no_grad():
            #     global_recon, global_z = self.modelG(self._get_input(scale=self.scale_factor, \
            #         no_shift=True, \
            #         up_factor=self.scale_factor), \
            #         noise_factor=self.opt.model.noise_factor)
            # self.model.train()

            random_slice = int(self.opt.model.image_res // 2)
            self.vis.yell(f"Current guidance factor is, g_real {g_real.mean().item():.2f}, g_fake {g_in.mean().detach().item():.2f}!")
            
            # self.visuals = {'train_patch': V.tensor_to_visual(sdf, normalize=True),\
            #                 'train_patch_recon': V.tensor_to_visual(p_recon, normalize=True),
            #                 'noise_visual': V.tensor_to_visual(z[:,:3]),
            # }
            self.visuals = {'train_patch': V.tensor_to_visual(sdf[..., random_slice], normalize=True),\
                            'train_patch_recon': V.tensor_to_visual(p_recon[..., random_slice], normalize=True),
                            # 'train_global_recon': V.tensor_to_visual(global_recon[..., random_slice], normalize=True),
                            'noise_visual': V.tensor_to_visual(z[:,:3,...,random_slice]),
                            # 'global_noise': V.tensor_to_visual(global_z[:,:3,...,random_slice]),
            }

            self.losses = {
                    'LossG': G_cost.item(),
                    'LossD': D_cost.item(),
                    'Gradient Norm': gradient_norm,}

    def _get_guidance(self, shape, dist='uniform', val=1.0):
        if dist == 'uniform':
            return torch.full(shape, torch.rand(1).item()).to(self.opt.device)
        elif dist == 'gaussian':
            g = np.clip(torch.normal(0.4,0.1,(1,1)).item(),0.0,1.0)
            return torch.full(shape, g).to(self.opt.device)
        elif dist == 'const':
            return torch.full(shape, val).to(self.opt.device)
        else:
            raise NotImplementedError(f"type of dist {dist} is not recognized!")

    def _get_input(self, shift=0.0, scale=1.0, no_shift=False, up_factor=1.0):
        res = self.opt.model.image_res
        size = int(up_factor*res)
        coords = H.get_position_3d(size, self.opt.device, self.opt.batch_size)

        if no_shift:
            return scale * coords
        elif self.opt.shift_type != 'none' and self.opt.run_mode == 'train':
            shift = self.dist_shift.sample()[...,None,None,None]
            shift = shift.expand(self.opt.batch_size, self.opt.model.image_dim, \
                                 size, size, size).contiguous().to(self.opt.device)

            padding = torch.zeros_like(shift[:,0]).to(self.opt.device)
            if self.opt.shift_type == 'x':
                shift = torch.stack([shift[:,0], padding, padding],1)
            elif self.opt.shift_type == 'y':
                shift = torch.stack([padding, shift[:,1], padding],1)
            elif self.opt.shift_type == 'xy':
                shift = torch.stack([shift[:,0], shift[:,1], padding],1)
            shift_factor = self.scale_factor
            return scale * coords + shift_factor * shift
        else:
            return scale * (coords + shift)

    def _get_high_res_input(self, res, shift=0.0, scale=1.0):
        coords = H.get_position_3d(res, self.opt.device, self.opt.batch_size)
        return scale * (coords + shift)

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

            
            sample = 50
            res = self.opt.model.image_res
            voxel_origin = [-1, -1, -1]
            voxel_size = 2.0 / (res - 1)
                
            up_factor = 8.0
            hres = [int(up_factor * res),int(up_factor * res), res]
            chunk_size = 4096

            for i in np.arange(0,1,0.1):
                self.vis.yell(f"Current guidance factor at {i}:")
                g_shape = [self.opt.batch_size,\
                           1,\
                           *hres]
                g_in = self._get_guidance(g_shape, 'const', val=i)
                # g_in = self._get_guidance(g_shape, 'const', val=1)
                x_in = self._get_high_res_input(hres, scale=scale)
                # x_in = x_in[:,:2,...,0]
                temperature = 0.25
                # g_in = 0.9 * torch.sigmoid(temperature * x_in.sum(1)).unsqueeze(1)
                
                x_in = torch.cat([x_in, g_in], 1)
                x_in = x_in.reshape(x_in.shape[0],x_in.shape[1],-1)
                recon = []
                zs = []
                npoints = x_in.shape[-1]
                for j in range(0, npoints, chunk_size):
                    fix_sample = j > 0
                    print(f"Current progress at {100 * j/npoints:.2f}%...", end='\r')
                    start = j
                    end = min(j+chunk_size, npoints)
                    x_slice = x_in[:,:,start:end].unsqueeze(-1).unsqueeze(-1)
                    y, z = self.modelG(x_slice, noise_factor=self.opt.model.noise_factor, fix_sample=fix_sample)
                    zs.append(z[...,0,0])
                    recon.append(y[...,0,0])

                recon = torch.cat(recon,-1).reshape(self.opt.batch_size, 1, *hres)
                zs = torch.cat(zs,-1).reshape(self.opt.batch_size, -1, *hres)

                random_slice = int(self.opt.model.image_res // 2)
                self.visuals = {
                                'generated image': V.tensor_to_visual(recon[...,random_slice], normalize=True),\
                                'noise_visual': V.tensor_to_visual(zs[:,:3,...,random_slice]),
                }
                self.vis.display_current_results(self.visuals)

                filename = f"sample_g_{i}.png"
                io.write_images(os.path.join(image_path,filename),V.tensor_to_visual(recon[...,random_slice], normalize=True),1)
                if recon.min() < 0:
                    self.vis.yell(f"Input saved!")
                    recon = recon[0].cpu().numpy().reshape(*hres)                
                    filename = f"hr_marching_cube_idx{i}.ply"
                    io.convert_sdf_samples_to_ply(recon, voxel_origin, voxel_size, \
                               os.path.join(image_path,filename))
                else:
                    self.vis.yell(f"sdf is degenerated!")
                time.sleep(1)




if __name__ == '__main__':
    if opt.run_mode == 'test' or opt.run_mode == 'eval':
        opt = io.process_test_args(opt)

    opt.model.octaves = 1
    # opt.model.image_dim = 3
    opt.model.image_dim = 3
    opt.model.image_res = 32
    opt.model.condition_on_guidance = True
    opt.model.guidance_feature_type = 'modx'
    opt.model.noise = 'stationary2d'

    trainer = Trainer(opt)
    if opt.run_mode == 'train':
        trainer.train()
    else:
        trainer.eval()
