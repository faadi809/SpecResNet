import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock3D(nn.Module):
    """Squeeze-and-Excitation block for 3D features."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, d, h, w = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y

class HybridResGroupedBlock3D(nn.Module):
    """
    Residual block with grouped 3D convolutions.
    Preserves identity mapping while reducing computational cost.
    """
    def __init__(self, channels, groups=8):
        super().__init__()

        assert channels % groups == 0, "channels must be divisible by groups"

        self.conv1 = nn.Conv3d(
            channels, channels,
            kernel_size=3, padding=1,
            groups=groups, bias=False
        )
        self.norm1 = nn.BatchNorm3d(channels)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = nn.Conv3d(
            channels, channels,
            kernel_size=3, padding=1,
            groups=groups, bias=False
        )
        self.norm2 = nn.BatchNorm3d(channels)

        self.act2 = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        identity = x
        out = self.act1(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out = out + identity
        return self.act2(out)


# Spectral Calibration Module
class SC_Module(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(in_channels))
        nn.init.ones_(self.weights)
        self.projection = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        B, C, D, H, W = x.shape
        x = x * self.weights.view(1, C, 1, 1, 1)
        return self.projection(x)