import os
import numpy as np
import random
import cv2
import imageio
import torch
import torchvision

def convert_to_int(array):
    array *= 255
    array[array > 255] = 255.0

    if type(array).__module__ == 'numpy':
        return array.astype(np.uint8)

    elif type(array).__module__ == 'torch':
        return array.byte()
    else:
        raise NotImplementedError

def convert_to_float(array):
    max_value = np.iinfo(array.dtype).max
    array[array > max_value] = max_value

    if type(array).__module__ == 'numpy':
        return array.astype(np.float32) / max_value

    elif type(array).__module__ == 'torch':
        return array.float() / max_value
    else:
        raise NotImplementedError

def numpy_to_pytorch(array, is_batch=False, flip=True):
    if flip:
        dest = 1 if is_batch else 0
        source = array.ndim - 1
        array = np.moveaxis(array, source, dest)

    array = torch.from_numpy(array)
    array = array.float()

    return array

def pytorch_to_numpy(array, is_batch=True, flip=True):
    array = array.detach().cpu().numpy()

    if flip:
        source = 1 if is_batch else 0
        dest = array.ndim - 1
        array = np.moveaxis(array, source, dest)

    return array

def write_gif(path, images, n_row, duration=0.1, loop=0):
    np_images = []
    for image in images:
        image = torchvision.utils.make_grid(image, n_row)
        image = pytorch_to_numpy(image, is_batch=False)
        image = convert_to_int(image)
        np_images.append(image)

    imageio.mimwrite(path, np_images, duration=duration, loop=loop)


def write_images(path, images, n_row=1):
    # image = torchvision.utils.make_grid(images)
    # image = pytorch_to_numpy(image, is_batch=False)
    images = images.transpose(1,2,0)
    image = convert_to_int(images)
    if image.ndim == 3:
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
    cv2.imwrite('{}'.format(str(path)), np.squeeze(image))



def load_image(path, res=None):
    path = str(path)

    if not os.path.isfile(path):
        raise FileNotFoundError(str(path))

    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    if res is not None:
        height, width = res
        image = cv2.resize(image, (width, height))

    if image.ndim == 3:
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            raise NotImplementedError

    if image.ndim == 2:
        image = image[..., np.newaxis]

    if image is None:
        raise FileNotFoundError

    if image.dtype == np.uint8 or image.dtype == np.uint16:
        image = convert_to_float(image)

    return image


def process_args(opt):
    if opt.model.input_type = '2d':
        opt.model.image_dim = 2
        if opt.model.channel_dim < 0:
            # default to rgb color
            opt.model.channel_dim = 3 
    elif opt.model.input_type = '3d':
        opt.model.image_dim = 3
        if opt.model.channel_dim < 0:
            # default to sdf
            opt.model.channel_dim = 1 
    if opt.run_mode != 'train':
        opt.batch_size = 1
    return opt


import plyfile
import skimage.measure
import time

def convert_sdf_samples_to_ply(
    sdf,
    voxel_grid_origin,
    voxel_size,
    ply_filename_out,
    offset=None,
    scale=None,
):
    """
    Convert sdf samples to .ply

    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to

    This function adapted from: https://github.com/RobotLocomotion/spartan
    """
    start_time = time.time()

    numpy_3d_sdf_tensor = sdf

    verts, faces, normals, values = skimage.measure.marching_cubes(
        numpy_3d_sdf_tensor, level=0.0, spacing=[voxel_size] * 3
    )

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    # try writing to the ply file

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(mesh_points[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    print("saving mesh to %s" % (ply_filename_out))
    ply_data.write(ply_filename_out)

    print(
        "converting to ply format and writing to file took {} s".format(
            time.time() - start_time
        )
    )
