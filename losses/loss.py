import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#====== charbonneir Loss ========#

def charbonnier_loss(pred, target, epsilon=1e-3):
    diff = pred - target
    loss = torch.sqrt(diff * diff + epsilon * epsilon)
    return loss.mean()

# -------------------------
# 3D Total Variation Loss
# -------------------------
def tv_loss_3d(x):
    loss = torch.mean(torch.abs(x[:, 1:, :, :] - x[:, :-1, :, :])) + \
           torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])) + \
           torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
    return loss

# ============= SAM Loss ================#

def sam_loss(original, reconstructed, eps=1e-5):
    # [B,C,D,H,W] -> treat each pixel's spectrum independently
    B, C, D, H, W = original.shape
    orig_spec = original.permute(0, 2, 3, 4, 1).reshape(-1, C)      # [Npix, C]
    recon_spec = reconstructed.permute(0, 2, 3, 4, 1).reshape(-1, C)
    dot = (orig_spec * recon_spec).sum(dim=1)
    norm_orig = orig_spec.norm(dim=1)
    norm_recon = recon_spec.norm(dim=1)
    cosine = dot / (norm_orig * norm_recon + eps)
    angle = torch.acos(torch.clamp(cosine, -1 + eps, 1 - eps))
    return angle.mean()


# ============ Combined Loss ============= #

def loss_function(reconstructed, original,
                  lambda_tv=0.1,
                  lambda_sam=0.1,
                  use_charbonnier=True,
                  tv_loss_fn=None):
    
    recon_loss = charbonnier_loss(reconstructed, original) if use_charbonnier else F.mse_loss(reconstructed, original)
    
    tv = tv_loss_fn(reconstructed) if tv_loss_fn else 0
    sam = sam_loss(original, reconstructed)
    
    total = recon_loss + lambda_tv * tv + lambda_sam * sam  
    return total