import numpy as np


def extract_overlapping_patches(
    cube,
    patch_size=(64, 64, 48),
    stride=(32, 32),
    pad_mode="reflect"
):
    H, W, C = cube.shape
    ph, pw, pc = patch_size
    sh, sw = stride

    pad_h = (sh - (H - ph) % sh) % sh
    pad_w = (sw - (W - pw) % sw) % sw

    cube = np.pad(cube, ((0, pad_h), (0, pad_w), (0, 0)), mode=pad_mode)
    H, W, _ = cube.shape

    patches = []

    for k in range(0, C, pc):
        spec = cube[:, :, k:k + pc]
        if spec.shape[2] < pc:
            pad = np.zeros((H, W, pc), dtype=cube.dtype)
            pad[:, :, :spec.shape[2]] = spec
            spec = pad

        for i in range(0, H - ph + 1, sh):
            for j in range(0, W - pw + 1, sw):
                patches.append(spec[i:i + ph, j:j + pw])

    return np.stack(patches)
