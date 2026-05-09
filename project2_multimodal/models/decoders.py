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
    Automatically adjusts number of upsampling layers based on target size.
    """

    def __init__(self, latent_dim=256, out_channels=1, base_ch=256, target_size=(64, 64, 32)):
        super().__init__()
        self.target_size = target_size

        # Compute how many upsampling stages needed
        min_dim = min(target_size)
        n_upsample = max(2, int(torch.log2(torch.tensor(min_dim)).item()) - 2)
        init_spatial = tuple(max(2, d // (2 ** n_upsample)) for d in target_size)

        self.init_spatial = init_spatial
        init_ch = base_ch

        # Initial FC to spatial feature map
        n_elements = init_ch
        for d in init_spatial:
            n_elements *= d
        self.init_fc = nn.Linear(latent_dim, n_elements)

        # Decoder blocks
        ch = init_ch
        self.upsample_blocks = nn.ModuleList()
        for i in range(n_upsample):
            out_ch = max(32, ch // 2)
            self.upsample_blocks.append(
                nn.Sequential(
                    nn.ConvTranspose3d(ch, out_ch, 4, stride=2, padding=1),
                    nn.GroupNorm(min(8, out_ch), out_ch),
                    nn.ReLU(inplace=True),
                )
            )
            ch = out_ch

        self.final = nn.Sequential(
            nn.Conv3d(ch, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, out_channels, 1),
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
        h = h.view(B, -1, *self.init_spatial)

        for block in self.upsample_blocks:
            h = block(h)

        # Adjust to exact target size
        if list(h.shape[2:]) != list(self.target_size):
            h = F.interpolate(
                h, size=self.target_size, mode='trilinear', align_corners=False
            )
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
