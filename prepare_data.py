import numpy as np
from utils.dataset import *
from utils.patching import extract_overlapping_patches
from utils.dataloader import create_dataloaders


def prepare_chikusei(
    path,
    patch_size=(64, 64, 48),
    stride=(32, 32),
    batch_size=128
):
    cube = load_h5_cube(path, "chikusei")
    cube = normalize(cube).transpose(1, 2, 0)

    train_cube, _ = spatial_split(cube)
    patches = extract_overlapping_patches(
        train_cube,
        patch_size=patch_size,
        stride=stride
    )

    return create_dataloaders(patches, batch_size=batch_size)
