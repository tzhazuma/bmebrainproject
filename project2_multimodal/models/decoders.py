"""Cross-modal decoders for synthesizing missing modality images."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock3DDecoder(nn.Module):
    """3D residual block for decoder."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.gn1 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1)
        self.gn2 = nn.GroupNorm(8, out_ch)
        self.shortcut = nn.Conv3d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        h = F.relu(self.gn1(self.conv1(x)))
        h = self.gn2(self.conv2(h))
        return F.relu(h + self.shortcut(x))


class ModalityDecoder(nn.Module):
    """
    3D decoder that reconstructs a modality image from latent code.

    Uses transposed convolutions to upsample from latent to full 3D volume.
    """

    def __init__(self, latent_dim=256, out_channels=1, base_ch=256, target_size=(64, 64, 32)):
        super().__init__()
        self.target_size = target_size

        # Initial FC to spatial feature map
        init_spatial = tuple(d // 16 for d in target_size)  # e.g., (8, 8, 4) → 256
        init_ch = base_ch
        self.init_fc = nn.Linear(latent_dim, int(init_ch * torch.prod(torch.tensor(init_spatial)).item()))

        # Decoder blocks (4x upsampling in total: 16x → target)
        self.dec1 = nn.Sequential(
            nn.ConvTranspose3d(init_ch, init_ch // 2, 4, stride=2, padding=1),
            nn.GroupNorm(8, init_ch // 2),
            nn.ReLU(inplace=True),
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose3d(init_ch // 2, init_ch // 4, 4, stride=2, padding=1),
            nn.GroupNorm(8, init_ch // 4),
            nn.ReLU(inplace=True),
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose3d(init_ch // 4, init_ch // 8, 4, stride=2, padding=1),
            nn.GroupNorm(8, init_ch // 8),
            nn.ReLU(inplace=True),
        )
        self.dec4 = nn.Sequential(
            nn.ConvTranspose3d(init_ch // 8, init_ch // 16, 4, stride=2, padding=1),
            nn.GroupNorm(8, init_ch // 16),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Sequential(
            nn.Conv3d(init_ch // 16, init_ch // 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(init_ch // 32, out_channels, 1),
            nn.Tanh(),
        )

    def forward(self, z):
        """
        Args:
            z: [B, latent_dim]
        Returns:
            image: [B, out_channels, D, H, W]
        """
        B = z.shape[0]
        h = self.init_fc(z)
        h = h.view(B, -1, *self.target_size)
        h = h.view(B, -1, *(d // 16 for d in self.target_size))

        h = self.dec1(h)
        h = self.dec2(h)
        h = self.dec3(h)
        h = self.dec4(h)

        # Adjust to exact target size
        h = F.interpolate(h, size=self.target_size, mode='trilinear', align_corners=False)
        return self.final(h)


class CrossModalDecoder(nn.Module):
    """
    Decoder that reconstructs a target modality from latent code.

    During training with modality dropout, this learns to synthesize
    missing modalities (cross-modal translation).
    """

    def __init__(self, modality_name, **kwargs):
        super().__init__()
        self.modality = modality_name
        self.decoder = ModalityDecoder(**kwargs)

    def forward(self, z):
        return self.decoder(z)
