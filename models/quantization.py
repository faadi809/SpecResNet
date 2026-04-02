import torch
import torch.nn as nn

class GradualQuantization(nn.Module):
    def __init__(self, num_levels=256, max_epoch=30):
        super().__init__()
        self.num_levels = num_levels
        self.max_epoch = max_epoch
        self.register_buffer('epoch_progress', torch.tensor(0.0))

    def forward(self, x):
        if self.training:
            # Progressively increase noise over epochs
            scale = torch.clamp(self.epoch_progress / self.max_epoch, 0.0, 1.0)
            noise = torch.empty_like(x).uniform_(-0.5, 0.5)
            x_noisy = x + scale * noise
            return x_noisy
        else:
            # Real quantization (for testing or exporting)
            x_min, x_max = x.min(), x.max()
            delta = (x_max - x_min) / (self.num_levels - 1)
            return torch.round((x - x_min) / delta) * delta + x_min

    def set_epoch(self, epoch):
        self.epoch_progress.fill_(epoch)