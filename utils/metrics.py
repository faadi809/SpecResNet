import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio
from pytorch_msssim import ms_ssim


def compute_psnr(gt_cube, recon_cube):
    """
    Compute whole-cube PSNR by treating the entire HSI as a single 3D volume.
    """
    return peak_signal_noise_ratio(gt_cube, recon_cube, data_range=1.0)


def compute_msssim_cube(gt_cube, recon_cube):
    """
    Compute band-averaged MS-SSIM by applying MS-SSIM independently
    to each spectral band and averaging the scores.
    """
    assert gt_cube.shape == recon_cube.shape
    bands = gt_cube.shape[0]
    scores = []

    custom_weights = [0.1, 0.2, 0.25, 0.2, 0.15, 0.1]

    for b in range(bands):
        gt_band = torch.tensor(gt_cube[b]).unsqueeze(0).unsqueeze(0).float()
        rec_band = torch.tensor(recon_cube[b]).unsqueeze(0).unsqueeze(0).float()

        score = ms_ssim(
            rec_band, gt_band,
            data_range=1.0,
            size_average=True,
            win_size=3,
            weights=custom_weights
        )
        scores.append(score.item())

    return sum(scores) / len(scores)


def compute_sam(img1, img2):
    """
    Compute mean Spectral Angle Mapper (SAM) in radians between two
    hyperspectral images of shape [H, W, C].
    """
    dot_product = np.sum(img1 * img2, axis=2)
    norm1 = np.linalg.norm(img1, axis=2)
    norm2 = np.linalg.norm(img2, axis=2)
    epsilon = 1e-8

    cos_theta = dot_product / (norm1 * norm2 + epsilon)
    cos_theta = np.clip(cos_theta, -1, 1)
    sam_map = np.arccos(cos_theta)

    return np.mean(sam_map)
