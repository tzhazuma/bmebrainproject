"""Modality-specific 3D encoders for multimodal PET/MR."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock3D(nn.Module):
    """3D Conv + GroupNorm + ReLU block."""

    def __init__(self, in_ch, out_ch, kernel=3, stride=1, groups=8):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel, stride=stride, padding=kernel // 2)
        self.gn = nn.GroupNorm(min(groups, out_ch), out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.gn(self.conv(x)))


class ResBlock3D(nn.Module):
    """3D Residual block."""

    def __init__(self, ch, groups=8):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock3D(ch, ch, 3, groups=groups),
            ConvBlock3D(ch, ch, 3, groups=groups),
        )

    def forward(self, x):
        return x + self.block(x)


class MRIEncoder(nn.Module):
    """
    3D encoder for T1w MRI (structural brain anatomy).
    Outputs μ and log_σ² for variational latent.
    """

    def __init__(self, in_channels=1, base_ch=32, latent_dim=256):
        super().__init__()
        self.conv1 = ConvBlock3D(in_channels, base_ch, 5, stride=2)
        self.conv2 = ConvBlock3D(base_ch, base_ch * 2, 3, stride=2)
        self.conv3 = ConvBlock3D(base_ch * 2, base_ch * 4, 3, stride=2)
        self.conv4 = ConvBlock3D(base_ch * 4, base_ch * 8, 3, stride=2)

        self.res_blocks = nn.Sequential(
            ResBlock3D(base_ch * 8),
            ResBlock3D(base_ch * 8),
        )

        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc_mu = nn.Linear(base_ch * 8, latent_dim)
        self.fc_logvar = nn.Linear(base_ch * 8, latent_dim)

    def forward(self, x):
        """
        Args:
            x: [B, 1, D, H, W]
        Returns:
            mu: [B, latent_dim]
            logvar: [B, latent_dim]
            features: intermediate features for decoder skip connections
        """
        f1 = self.conv1(x)
        f2 = self.conv2(f1)
        f3 = self.conv3(f2)
        f4 = self.conv4(f3)
        f4 = self.res_blocks(f4)

        pooled = self.pool(f4).flatten(1)
        mu = self.fc_mu(pooled)
        logvar = self.fc_logvar(pooled)

        return mu, logvar, [f1, f2, f3, f4]


class PETEncoder(nn.Module):
    """
    3D encoder for PET scans (FDG or TAU).
    Uses same architecture as MRIEncoder but expects SUVR-normalized input.
    """

    def __init__(self, in_channels=1, base_ch=32, latent_dim=256):
        super().__init__()
        self.mri_enc = MRIEncoder(in_channels, base_ch, latent_dim)

    def forward(self, x):
        return self.mri_enc(x)


class ModalityEncoder(nn.Module):
    """
    Unified encoder factory: creates modality-specific encoder.

    Args:
        modality: "t1w", "fdg_pet", "tau_pet", "asl"
    """

    def __init__(self, modality='t1w', **kwargs):
        super().__init__()
        self.modality = modality
        if modality in ('t1w', 'mri'):
            self.encoder = MRIEncoder(**kwargs)
        else:
            self.encoder = PETEncoder(**kwargs)

    def forward(self, x):
        return self.encoder(x)


class SimpleEncoder3D(nn.Module):
    """Lightweight 3D CNN encoder for faster iteration."""

    def __init__(self, in_channels=1, latent_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, 32, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 256, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

    def forward(self, x):
        h = self.conv(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h), []
