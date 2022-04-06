from configs.default import opt
from importlib import import_module
from tqdm import tqdm
import os
import time
import numpy as np
import torch
import vgtk
from core.trainer import BasicTrainer
from core.loss import *
import utils.helper as H
import utils.io as io
import utils.visualizer as V
from dataset import SDFDataset
from torch.autograd import Variable


class Trainer(BasicTrainer):
    def __init__(self, opt):
        super(Trainer, self).__init__(opt)
        self.summary.register(['LossD', 'LossG', 'kscale_x','kscale_y','Gradient Norm'])
        self.dist_shift = H.get_distribution_type([self.opt.batch_size, self.opt.model.image_dim], 'uniform')
        self.scale_factor = 2.0
        # self.scale_factor = self.opt.dataset.crop_size / self.opt.model.global_res # self.opt.model.image_res
        # print(f"[MODEL] choosing a scale factor of {self.scale_factor}!!!")
        print(f"[MODEL] Grid resolution at {self.opt.model.image_res}!")


    def _setup_datasets(self):
        if self.opt.run_mode == 'train':
            dataset = SDFDataset(self.opt)
            # self.opt.dataset.crop_size = dataset.default_size
            # self.opt.model.global_res = dataset.global_res
            self.opt.model.image_res = dataset.res
            self.dataest_size = len(dataset)
            self.n_iters = self.dataest_size / self.opt.batch_size
            self.dataset = torch.utils.data.DataLoader( dataset, \
                                                        batch_size=self.opt.batch_size, \
                                                        shuffle=False, \
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

        if self.opt.slice:
            modelD_name = "vanilla_netD"
        else:
            modelD_name = "sdf_netD"

        print(f"Using discriminator model {modelD_name}!")
        self.modelD = getattr(module, modelD_name).build_model_from(self.opt, None, c_in=1)

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
            sdf = data

            # sdf_exp = sdf[0].detach().cpu().numpy()
            # sdf_slice = sdf_exp[...,32][0][...,None]
            # import ipdb; ipdb.set_trace()

            self.modelD.zero_grad()

            # real data
            sdf = sdf.to(self.opt.device)
            real_data = Variable(sdf)

            if self.opt.slice:
                random_slice = np.random.randint(self.opt.model.image_res)
                real_data_slice = real_data[...,random_slice]
                D_real = self.modelD(real_data_slice)
            else:
                D_real = self.modelD(real_data)

            D_real = -1 * D_real.mean()
            D_real.backward()

            # # fake data
            g_in = self._get_input()
            fake, _ = self.modelG(g_in.detach(), noise_factor=self.opt.model.noise_factor)
            fake_data = fake.to(self.opt.device)
            fake_data = Variable(fake_data.data)

            if self.opt.slice:
                fake_data_slice = fake_data[..., random_slice]
                D_fake = self.modelD(fake_data_slice).mean()
            else:
                D_fake = self.modelD(fake_data).mean()
            D_fake.backward()

            if self.opt.slice:
                gradient_penality, gradient_norm = calc_gradient_penalty(self.modelD, real_data_slice.data, fake_data_slice.data)
            else:
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
            g_in = self._get_input()
            p_recon, z = self.modelG(g_in.detach(), noise_factor=self.opt.model.noise_factor)
            fake_data = p_recon
            if self.opt.slice:
                fake_data_slice = fake_data[..., random_slice]
                G_cost = self.modelD(fake_data_slice)
            else:
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

            random_slice = self.opt.model.image_res //2 # np.random.randint(self.opt.model.image_res)

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

    def _get_input(self, shift=0.0, scale=1.0, no_shift=False, up_factor=1.0):
        res = self.opt.model.image_res
        size = int(up_factor*res)
        coords = H.get_position_3d(size, self.opt.device, self.opt.batch_size)

        if no_shift:
            return scale * coords
        elif self.opt.shift_type != 'none' and self.opt.run_mode == 'train':
            
            if self.opt.shift_type == 'mid':
                shift = torch.zeros(self.opt.batch_size, self.opt.model.image_dim, \
                                 size, size, size).contiguous().to(self.opt.device)
                if np.random().rand() > 0.5:
                    shift += 0.5
            else:
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
        image_path = os.path.join(os.path.dirname(self.ckpt_dir), 'visuals')
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
            res = self.opt.model.image_res
            voxel_origin = [-1, -1, -1]
            voxel_size = 2.0 / (res - 1)

            # for i in range(sample):
            #     # res = self.opt.model.image_res #self.opt.dataset.crop_size
            #     recon, z = self.modelG(self._get_input(no_shift=True), noise_factor=self.opt.model.noise_factor)
            #     random_slice = np.random.randint(res)
            #     self.visuals = {
            #                     'generated image': V.tensor_to_visual(recon[...,random_slice], normalize=True),\
            #                     'noise_visual': V.tensor_to_visual(z[:,:3,...,random_slice]),
            #     }
            #     self.vis.display_current_results(self.visuals)

            #     recon = recon[0].cpu().numpy().reshape(res,res,res)
            #     if recon.min() < 0:
            #         self.vis.yell(f"Input saved!")
            #         filename = f"marching_cube_idx{i}.ply"
            #         io.convert_sdf_samples_to_ply(recon, voxel_origin, voxel_size, \
            #                    os.path.join(image_path,filename))
            #     else:
            #         self.vis.yell(f"sdf is degenerated!")
            #     time.sleep(1)

            up_factor = 8.0
            hres = [int(up_factor * res),int(up_factor * res),int(res)]
            voxel_size = 2.0 / (res - 1)
            chunk_size = 65536
            zs = []
            recon = []

            for i in range(50):
                # res = self.opt.model.image_res #self.opt.dataset.crop_size
                x_in = self._get_high_res_input(hres)
                x_in = x_in.reshape(x_in.shape[0],x_in.shape[1],-1)
                recon = []
                zs = []
                npoints = x_in.shape[-1]

                start = time.time()
                for j in range(0, npoints, chunk_size):
                    fix_sample = j > 0
                    print(f"Current progress at {100 * j/npoints:.2f}%...", end='\r')
                    start = j
                    end = min(j+chunk_size, npoints)
                    x_slice = x_in[:,:,start:end].unsqueeze(-1).unsqueeze(-1)
                    y, z = self.modelG(x_slice, noise_factor=self.opt.model.noise_factor, fix_sample=fix_sample)
                    zs.append(z[...,0,0])
                    recon.append(y[...,0,0])
                end = time.time()
                # print(end-start)
                # import ipdb; ipdb.set_trace()

                # recon, z = self.modelG(g_in, noise_factor=self.opt.model.noise_factor)
                recon = torch.cat(recon,-1).reshape(self.opt.batch_size, 1, *hres)
                zs = torch.cat(zs,-1).reshape(self.opt.batch_size, -1, *hres)

                random_slice = res // 2
                self.visuals = {
                                'generated image': V.tensor_to_visual(recon[...,random_slice], normalize=True),\
                                'noise_visual': V.tensor_to_visual(zs[:,:3,...,random_slice]),
                }
                self.vis.display_current_results(self.visuals)
                filename = f"slice_{i}.png"
                io.write_images(os.path.join(image_path,filename),V.tensor_to_visual(recon[...,random_slice], normalize=True),1)

                recon = recon[0].cpu().numpy().reshape(*hres)
                # recon_slice = recon[:,:,16][...,None]
                # import ipdb; ipdb.set_trace()

                if recon.min() < 0:
                    self.vis.yell(f"Input saved!")
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
    opt.model.image_dim = 3
    trainer = Trainer(opt)
    if opt.run_mode == 'train':
        trainer.train()
    else:
        trainer.eval()
