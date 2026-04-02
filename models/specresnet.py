import torch
import torch.nn as nn
from .blocks import SimpleResBlock3D, SEBlock3D, ASW_Module
from .quantization import GradualQuantization


class Autoencoder3D(nn.Module):
    def __init__(self, in_channels=48, base_channels=64, num_levels=256, max_epoch=30):
        super().__init__()

        # ================= Encoder ================= #

        self.enc1 = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, 3, padding=1),
            HybridResGroupedBlock3D(base_channels, groups=8),
            SEBlock3D(base_channels),
        )

        self.enc2 = nn.Sequential(
            nn.Conv3d(base_channels, base_channels // 2, 3, stride=2, padding=1),
            HybridResGroupedBlock3D(base_channels // 2, groups=8),
            HybridResGroupedBlock3D(base_channels // 2, groups=8),
            HybridResGroupedBlock3D(base_channels // 2, groups=8),
            SEBlock3D(base_channels // 2)
        )

        self.enc3 = nn.Sequential(
            nn.Conv3d(base_channels // 2, base_channels // 4, 3, stride=2, padding=1),
            HybridResGroupedBlock3D(base_channels // 4, groups=8),
            SEBlock3D(base_channels // 4)
        )

        self.enc31 = nn.Sequential(
            nn.Conv3d(base_channels // 4, base_channels // 4, 3, stride=2, padding=1),
            HybridResGroupedBlock3D(base_channels // 4, groups=8),
            SEBlock3D(base_channels // 4)
        )

        self.quantizer = GradualQuantization(num_levels=256, max_epoch=50)

        # ================= Decoder ================= #

        self.dec01 = nn.Sequential(
            nn.ConvTranspose3d(
                base_channels // 4, base_channels // 4,
                3, stride=2, padding=1, output_padding=(0,1,1)
            ),
            HybridResGroupedBlock3D(base_channels // 4, groups=8),
            SEBlock3D(base_channels // 4)
        )

        self.dec1 = nn.Sequential(
            nn.ConvTranspose3d(
                base_channels // 4, base_channels // 2,
                3, stride=2, padding=1, output_padding=(0,1,1)
            ),
            HybridResGroupedBlock3D(base_channels // 2, groups=8),
            SEBlock3D(base_channels // 2)
        )

        self.dec11 = nn.Sequential(
            nn.ConvTranspose3d(
                base_channels // 2, base_channels // 2,
                3, stride=1, padding=1
            ),
            HybridResGroupedBlock3D(base_channels // 2, groups=8),
            HybridResGroupedBlock3D(base_channels // 2, groups=8),
            HybridResGroupedBlock3D(base_channels // 2, groups=8),
            SEBlock3D(base_channels // 2)
        )

        self.dec2 = nn.Sequential(
            nn.ConvTranspose3d(
                base_channels // 2, base_channels,
                3, stride=2, padding=(0,1,1), output_padding=(0,1,1)
            ),
            HybridResGroupedBlock3D(base_channels, groups=8),
            SEBlock3D(base_channels)
        )

        self.final = nn.Sequential(
            nn.Conv3d(base_channels, in_channels, 3, padding=(0,1,1)),
            SC_Module(in_channels)
        )

    def forward(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x1 = self.enc3(x)
        x2 = self.enc31(x1)

        x1_quant = self.quantizer(x2)

        x = self.dec01(x1_quant)
        x = self.dec1(x)
        x = self.dec11(x)
        x = self.dec2(x)
        x = self.final(x)

        return x, x1_quant