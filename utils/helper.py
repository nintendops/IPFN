import torch
import torch.distributions as dists
import torch.nn.functional as F
import numpy as np
import kornia
import math

PI = np.pi


def upsample_and_crop(z, p=2, k=1):
    nb, nd, h, w = z.shape
    assert h > p and w > p
    z = F.interpolate(z, scale_factor=p*k)

    cropped_zs = []
    for i in range(nb):
        crop_h = np.random.randint(0,k*(p*h - h))    
        crop_w = np.random.randint(0,k*(p*w - w))
        cropped_z = z[i,:,crop_h:crop_h + k*h, crop_w:crop_w+ k*w].unsqueeze(0)
        cropped_z = F.interpolate(cropped_z, scale_factor=1/k)
        cropped_zs += cropped_z
        
    z = torch.stack(cropped_zs, 0)
    return z.contiguous()


def add_padding(x, mode='constant'):
    # zero padding by default
    h, w = x.shape[-2:]
    x = F.pad(x, (h//2,h//2,w//2,w//2), mode=mode)
    return x

def exp_distribution(x, sigma=0.05):
    # maps a random distribution in [-1,1] to be biased towards the boundaries
    sign = torch.randint(2,x.shape).float().to(x.device)
    sign = 2 * (sign - 0.5)
    x = 1 - torch.exp(-x**2 / sigma)
    return sign * x

def get_distribution_type(shape, type='normal'):
    if type == 'normal':
        return dists.Normal(torch.zeros(shape), torch.ones(shape))
    elif type == 'uniform':
        return dists.Uniform(torch.zeros(shape) - 1, torch.ones(shape))
    else:
        raise NotImplementedError


def flip_positions(positions):
    positions = positions.transpose(2,3)
    positions = torch.cat([positions[:,0][:,None],-positions[:,1][:,None]],1)
    return positions.contiguous()

def get_position(size, dim, device, batch_size):
    height, width = size
    aspect_ratio = width / height
    position = kornia.utils.create_meshgrid(width, height, device=torch.device(device)).permute(0, 3, 1, 2)
    position[:, 1] = -position[:, 1] * aspect_ratio  # flip y axis

    if dim == 1:
        x, y = torch.split(position, 1, dim=1)
        position = x
    if dim == 3:
        x, y = torch.split(position, 1, dim=1)
        z = torch.ones_like(x) * torch.rand(1, device=device) * 2 - 1
        a = torch.randint(0, 3, (1,)).item()
        if a == 0:
            xyz = [x, y, z]
        elif a == 1:
            xyz = [z, x, y]
        else:
            xyz = [x, z, y]
        # xyz =  [x,z,y]
        position = torch.cat(xyz, dim=1)
    position = position.expand(batch_size, dim, width, height)
    return position if dim == 3 else flip_positions(position)


def get_position_3d(N, device, batch_size):
    if isinstance(N, int):
        overall_index = np.arange(N ** 3)
        samples = np.zeros([N ** 3,3], np.float32)
        voxel_origin = [-1, -1, -1]
        voxel_size = 2.0 / (N - 1)
        n1 = N
        n2 = N
        n3 = N
    else:
        assert len(N) == 3
        n1,n2,n3 = N
        overall_index = np.arange(n1*n2*n3)
        samples = np.zeros([n1*n2*n3,3], np.float32)
        voxel_origin = [-1 * n1 / min(n1,n2,n3), -1 * n2 / min(n1,n2,n3), -1 * n3 / min(n1,n2,n3)]
        voxel_size = 2.0 / (min(n1,n2,n3) - 1)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % n3
    samples[:, 1] = (overall_index // n3) % n2
    samples[:, 0] = (overall_index // (n3*n2)) % n1
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[0]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[2]

    samples = torch.from_numpy(samples.T.reshape(1,3,n1,n2,n3)).to(device)
    samples = samples.expand(batch_size,3,n1,n2,n3).contiguous()
    return samples

# x -> [cos(2^0 pi x), ..., sin(2^9) pi x]
# b,dim,... -> b,2*dim*l,...
def positional_encoding(x, l=5, beta=None):
    bs,dim = x.shape[:2]
    res = x.shape[2:]
    x = x.unsqueeze(2).expand(bs,dim,l,*res)
    octaves = 2**(torch.arange(l)).to(x.device)
    if beta is not None:        
        octaves = octaves[None] * beta
        b, _ = octaves.shape
        octaves = octaves[:,None].expand(b,bs//b,l).reshape(-1,l).contiguous()
        for r in res:
            octaves = octaves.unsqueeze(-1)
        x = x * octaves[:,None,...] * PI
    else:
        for r in res:
            octaves = octaves.unsqueeze(-1)
        x = x * octaves[None,None,...] * PI
    x = torch.cat((torch.sin(x).unsqueeze(2), torch.cos(x).unsqueeze(2)),2)
    return x.reshape(bs,-1,*res)


def _get_input(res, dist_shift, opt, scale=1.0, shift=None):
    
    res = int(res)

    # sample a grid of coordinates
    if opt.model.input_type == '2d':
        size = (res, res)    
        coords = get_position( size, 2, opt.device, opt.batch_size)
    elif opt.model.input_type == '3d':
        size = (res, res, res)    
        coords = get_position_3d( size, opt.device, opt.batch_size)
    else:
        raise NotImplementedError(f"Unknown input type {opt.model.input_type}")

    # if no shift applied
    if opt.shift_type == 'none':
        return scale * coords
    
    # if manually specified shift
    if shift is not None:
        return scale * coords + shift

    # if randomly applying shift
    if opt.model.input_type == '2d':
        shift = dist_shift.sample()[...,None,None]
        shift = shift.expand(opt.batch_size, 2, res, res).contiguous().to(opt.device)
    else:
        shift = dist_shift.sample()[...,None,None,None]
        shift = shift.expand(opt.batch_size, 3, res, res, res).contiguous().to(opt.device)

    # further maneuvering of how shifts are applied
    padding = torch.zeros_like(shift[:,0]).to(opt.device)
    if opt.shift_type == 'y':
        shift = torch.stack([shift[:,0], padding, padding],1)
    elif opt.shift_type == 'x':
        shift = torch.stack([padding, shift[:,1], padding],1)
    elif opt.shift_type == 'xy':
        shift = torch.stack([shift[:,0], shift[:,1], padding],1)
    if opt.model.input_type == '2d':
        shift = shift[:,:2]

    return scale * coords + opt.shift_factor * shift


def dir_from_tlc(d, res, batch_size, device):
    grid_shift = get_position([res,res],2,device,batch_size).transpose(2,3)
    topleftcorner = grid_shift[:,:,0,0].unsqueeze(-1).unsqueeze(-1)
    grid_shift = grid_shift - topleftcorner
    grid_offset = d[...,None,None] + grid_shift
    grid_dir = grid_offset / torch.norm(grid_offset, dim=1, keepdim=True)
    return grid_dir

################## functions for generating latent noise field ######################
def batched_index_select_2d(index, feature):
    batch_size, image_dim, res1, res2 = index.shape
    _, f_dim, grid_dim, _ = feature.shape
    indexf = index.reshape(batch_size,image_dim,-1)
    indexf2 = indexf[:,0] * grid_dim + indexf[:,1]
    def select(x, idx):
        x_select = torch.index_select(x.reshape(f_dim, -1),1,idx)
        x_select = x_select.reshape(f_dim, res1, res2)
        return x_select[None]
    xyz_select = torch.cat([select(x,idx) for x,idx in zip(feature, indexf2)], dim=0)
    return xyz_select


def stationary_noise_1d(positions, feature, mode='gaussian',sigma=0.2):
    '''
    positions: nb,w,h
    feature: nb,c,x
    ''' 
    nb, w, h = positions.shape
    _, nc, n_feature = feature.shape
    offset = n_feature // 2
    index_1 = torch.clamp(torch.floor(positions).long() + offset, 0, n_feature-1)
    index_1 = index_1.unsqueeze(1).expand(nb,nc,w,h).reshape(nb,nc,-1)
    index_2 = torch.clamp(index_1 + 1, 0, n_feature-1)

    d = positions - torch.floor(positions)

    # nb, 1, w, h, 2
    dists = torch.stack([1-d, d],3).unsqueeze(1)

    # nb, c, w, h
    fl = torch.gather(feature, 2, index_1).reshape(nb,nc,w,h)
    fr = torch.gather(feature, 2, index_2).reshape(nb,nc,w,h)
    grouped_feats = torch.stack([fl,fr],4)

    if mode == 'gaussian':
        dists = torch.nn.functional.softmax(dists/sigma,-1)
        return torch.sum(dists * grouped_feats, -1)
    elif mode == 'linear':
        return torch.sum(dists * grouped_feats, -1)
    else:
        raise NotImplementedError(f"type of interpolation {mode} is not recognized!")

def stationary_noise(positions, feature, mode='gaussian',sigma=0.2):
    '''
    positions: nb, 2, w,h
    feature: nb, 2,x,y,z
    '''
    # index-select from features
    
    offset = feature.shape[2] // 2
    # pmin = torch.floor(positions[0].min()).long() # if pmin is None else math.floor(pmin)
    # pmax = torch.floor(positions.max())
    # assert pmax - pmin < feature.shape[2]
    index_1 = torch.clamp(torch.floor(positions).long() + offset, 0, feature.shape[2]-1)
    index_2 = torch.clamp(index_1 + 1, 0, feature.shape[2]-1)
    index_3 = torch.clamp(torch.cat([index_1[:,0].unsqueeze(1) + 1, index_1[:,1].unsqueeze(1)],1), 0, feature.shape[2]-1)
    index_4 = torch.clamp(torch.cat([index_1[:,0].unsqueeze(1), index_1[:,1].unsqueeze(1)+1],1), 0, feature.shape[2]-1)

    # distance to corners
    px = positions[:,0]
    py = positions[:,1]
    x1 = (px - torch.floor(px))
    x2 = (torch.floor(px) + 1 - px)
    y1 = (py - torch.floor(py))
    y2 = (torch.floor(py) + 1 - py)  
    
    if mode == 'gaussian':
        f_grouped = torch.cat([batched_index_select_2d(index, feature)[...,None] \
                    for index in [index_1,index_2,index_3,index_4]],dim=-1)
        dist1 = torch.sqrt(x1**2 + y1**2)[...,None]
        dist2 = torch.sqrt(x2**2 + y2**2)[...,None]
        dist3 = torch.sqrt(x2**2 + y1**2)[...,None]
        dist4 = torch.sqrt(x1**2 + y2**2)[...,None]
        dists = torch.cat([dist1,dist2,dist3,dist4], 3)
        dists = torch.nn.functional.softmax(-dists/sigma, -1).unsqueeze(1)
        return torch.sum(dists * f_grouped, -1)
    elif mode == 'bilinear':
        tr,tl,br,bl = [batched_index_select_2d(index, feature)\
                    for index in [index_1,index_3,index_4,index_2]]
        bx = x1.unsqueeze(1) * bl + x2.unsqueeze(1) * br
        tx = x1.unsqueeze(1) * tl + x2.unsqueeze(1) * tr        
        return y1.unsqueeze(1) * bx + y2.unsqueeze(1) * tx        
    else:
        raise NotImplementedError(f"type of interpolation {mode} is not recognized!")

def stationary_noise_2d(positions, feature, mode='gaussian',sigma=0.2):
    return stationary_noise(positions, feature, mode=mode, sigma=sigma)

def batched_index_select_3d(index, feature):
    batch_size = index.shape[0]
    image_dim = index.shape[1]
    res = index.shape[2:]
    _, f_dim, grid_dim, _, _ = feature.shape
    indexf = index.reshape(batch_size,image_dim,-1)
    indexf2 = indexf[:,0] * grid_dim * grid_dim + indexf[:,1] * grid_dim + indexf[:,2]
    def select(x, idx):
        x_select = torch.index_select(x.reshape(f_dim, -1),1,idx)
        x_select = x_select.reshape(f_dim, *res)
        return x_select[None]
    xyz_select = torch.cat([select(x,idx) for x,idx in zip(feature, indexf2)], dim=0)
    return xyz_select

def stationary_noise_3d(positions, feature, mode='gaussian',sigma=0.2):
    '''
    positions: nb, 3, ...
    feature: nb, dim, x,y,z
    '''
    # index-select from features
    nb = positions.shape[0]
    assert positions.shape[1] == 3
    pos_res = positions.shape[2:]
    positions = positions.reshape(nb, 3, -1)

    offset = feature.shape[2] // 2
    c000_idx = torch.clamp(torch.floor(positions).long() + offset, 0, feature.shape[2]-1)
    c100_idx = torch.clamp(torch.cat([c000_idx[:,0].unsqueeze(1)+1,\
                                      c000_idx[:,1].unsqueeze(1),\
                                      c000_idx[:,2].unsqueeze(1)],1),\
                                      0, feature.shape[2]-1)
    c001_idx = torch.clamp(torch.cat([c000_idx[:,0].unsqueeze(1),\
                                      c000_idx[:,1].unsqueeze(1)+1,\
                                      c000_idx[:,2].unsqueeze(1)],1),\
                                      0, feature.shape[2]-1)
    c101_idx = torch.clamp(torch.cat([c000_idx[:,0].unsqueeze(1)+1,\
                                      c000_idx[:,1].unsqueeze(1)+1,\
                                      c000_idx[:,2].unsqueeze(1)],1),\
                                      0, feature.shape[2]-1)
    c011_idx = torch.clamp(torch.cat([c000_idx[:,0].unsqueeze(1),\
                                      c000_idx[:,1].unsqueeze(1)+1,\
                                      c000_idx[:,2].unsqueeze(1)+1],1),\
                                      0, feature.shape[2]-1)
    c010_idx = torch.clamp(torch.cat([c000_idx[:,0].unsqueeze(1),\
                                      c000_idx[:,1].unsqueeze(1),\
                                      c000_idx[:,2].unsqueeze(1)+1],1),\
                                      0, feature.shape[2]-1)
    c110_idx = torch.clamp(torch.cat([c000_idx[:,0].unsqueeze(1)+1,\
                                      c000_idx[:,1].unsqueeze(1),\
                                      c000_idx[:,2].unsqueeze(1)+1],1),\
                                      0, feature.shape[2]-1)
    c111_idx = torch.clamp(torch.cat([c000_idx[:,0].unsqueeze(1)+1,\
                                      c000_idx[:,1].unsqueeze(1)+1,\
                                      c000_idx[:,2].unsqueeze(1)+1],1),\
                                      0, feature.shape[2]-1)

    # distance to corners
    px = positions[:,0]
    py = positions[:,1]
    pz = positions[:,2]

    x1 = (px - torch.floor(px))
    x2 = (torch.floor(px) + 1 - px)
    y1 = (py - torch.floor(py))
    y2 = (torch.floor(py) + 1 - py)  
    z1 = (pz - torch.floor(pz))
    z2 = (torch.floor(pz) + 1 - pz)  

    ######## TOO SLOW ############################    
    # f_grouped = torch.cat([batched_index_select_3d(index, feature)[...,None] \
    #             for index in [c000_idx,c100_idx,c001_idx,c010_idx,\
    #                           c101_idx,c011_idx,c110_idx,c111_idx]],dim=-1)
    #################################################
    # [nb, 3, res**k, 8]
    index = torch.stack([c000_idx,c100_idx,c001_idx,c010_idx,c101_idx,c011_idx,c110_idx,c111_idx],-1)
    
    _, _, res, _ = index.shape
    _, c, gdim, _, _ = feature.shape
    index_flattened = index[:,0] * gdim**2 + index[:,1] * gdim + index[:,2]
    index_flattened = index_flattened.reshape(nb, 1, -1).expand(nb, c, res*8)
    feature_flattened = feature.reshape(nb,c,-1)
    f_grouped = torch.gather(feature_flattened, 2, index_flattened)
    f_grouped = f_grouped.reshape(nb,c,res,8)

    dist1 = torch.sqrt(x1**2 + y1**2 + z1**2)[...,None]
    dist2 = torch.sqrt(x2**2 + y1**2 + z1**2)[...,None]
    dist3 = torch.sqrt(x1**2 + y2**2 + z1**2)[...,None]
    dist4 = torch.sqrt(x1**2 + y1**2 + z2**2)[...,None]
    dist5 = torch.sqrt(x2**2 + y2**2 + z1**2)[...,None]
    dist6 = torch.sqrt(x1**2 + y2**2 + z2**2)[...,None]
    dist7 = torch.sqrt(x2**2 + y1**2 + z2**2)[...,None]
    dist8 = torch.sqrt(x2**2 + y2**2 + z2**2)[...,None]
    dists = torch.cat([dist1,dist2,dist3,dist4,dist5,dist6,dist7,dist8], -1)
    if mode == 'gaussian':
        dists = torch.nn.functional.softmax(-dists/sigma, -1).unsqueeze(1)
    elif mode == 'linear':
        dists = dists.max(-1,keepdim=True)[0] - dists
        dists = dists / dists.sum(-1,keepdim=True)
        dists = dists.unsqueeze(1)
    else:
        raise NotImplementedError(f"type of interpolation {mode} is not recognized!")

    # [nb, c, res]
    feat = torch.sum(dists * f_grouped, -1)
    return feat.reshape(nb, c, *pos_res)
