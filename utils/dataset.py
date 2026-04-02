import numpy as np
import scipy.io as sio
import h5py
import tifffile


def normalize(cube, eps=1e-6):
    cube = cube.astype(np.float32)
    return (cube - cube.min()) / (cube.max() - cube.min() + eps)


def load_mat_cube(path, key):
    data = sio.loadmat(path)
    return data[key]


def load_h5_cube(path, key):
    with h5py.File(path, 'r') as f:
        return np.array(f[key]).astype(np.float32)


def load_tiff_cube(path):
    cube = tifffile.imread(path)
    if cube.shape[0] < cube.shape[-1]:
        cube = np.transpose(cube, (1, 2, 0))
    return cube.astype(np.float32)


def spatial_split(cube, test_ratio=0.3):
    H = cube.shape[0]
    split = int(H * (1 - test_ratio))
    return cube[:split], cube[split:]
