"""
3D U-Net backbone for T2T-Bridge.

Based on I2SB U-Net (NVlabs/I2SB) adapted for 3D medical images.
Supports:
  - Time-step conditioning via GroupNorm + scale/shift
  - Age embedding injection (CFG)
  - Residual prediction head
"""

import torch
import torch.nn as nn
import math


def sinusoidal_embedding(timesteps, dim, max_period=10000):
    """Sinusoidal time/age embedding (as in DDPM/NCSN)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(0, half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps.float().unsqueeze(-1) * freqs.unsqueeze(0)
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class ResBlock3D(nn.Module):
    """3D Residual block with time embedding conditioning."""

    def __init__(self, in_ch, out_ch, time_emb_dim, dropout=0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_ch)
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_ch * 2),  # scale + shift
        )
        self.norm2 = nn.GroupNorm(32, out_ch)
        self.dropout = nn.Dropout3d(dropout)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1)

        if in_ch != out_ch:
            self.shortcut = nn.Conv3d(in_ch, out_ch, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, t_emb):
        h = self.norm1(x)
        h = nn.SiLU()(h)
        h = self.conv1(h)

        # Time conditioning: scale + shift
        scale_shift = self.time_mlp(t_emb).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        scale, shift = scale_shift.chunk(2, dim=1)
        h = h * (1 + scale) + shift

        h = self.norm2(h)
        h = nn.SiLU()(h)
        h = self.dropout(h)
        h = self.conv2(h)
        return h + self.shortcut(x)


class Downsample3D(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv3d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x, t_emb=None):
        return self.conv(x)


class Upsample3D(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv3d(ch, ch, 3, padding=1)

    def forward(self, x, t_emb=None):
        h = nn.functional.interpolate(x, scale_factor=2, mode="trilinear")
        return self.conv(h)


class UNet3D(nn.Module):
    """
    3D U-Net with time embedding conditioning.

    Args:
        in_channels: Input channels (1 for T1w MRI)
        out_channels: Output channels (1 for image, 3 for tissue maps)
        model_channels: Base feature dimension
        channel_mult: Channel multipliers per level (e.g., [1,2,4,8])
        num_res_blocks: ResBlocks per level
        dropout: Dropout rate
    """

    def __init__(
        self,
        in_channels=2,        # thick-slice + x_t
        out_channels=1,       # predicted residual
        model_channels=64,
        channel_mult=(1, 2, 4, 8),
        num_res_blocks=2,
        dropout=0.0,
        time_emb_dim=None,
    ):
        super().__init__()
        if time_emb_dim is None:
            time_emb_dim = model_channels * 4

        self.time_emb_dim = time_emb_dim

        # Time embedding MLP
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Age embedding MLP (Classifier-Free Guidance)
        self.age_embed = nn.Sequential(
            nn.Linear(model_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Input convolution
        self.input_conv = nn.Conv3d(in_channels, model_channels, 3, padding=1)

        # Encoder (downsampling path)
        self.down_blocks = nn.ModuleList()
        ch = model_channels
        chs = [ch]
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                self.down_blocks.append(ResBlock3D(ch, model_channels * mult, time_emb_dim, dropout))
                ch = model_channels * mult
                chs.append(ch)
            if level != len(channel_mult) - 1:
                self.down_blocks.append(Downsample3D(ch))
                chs.append(ch)

        # Middle block
        self.mid_block1 = ResBlock3D(ch, ch, time_emb_dim, dropout)
        self.mid_block2 = ResBlock3D(ch, ch, time_emb_dim, dropout)

        # Decoder (upsampling path)
        self.up_blocks = nn.ModuleList()
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for _ in range(num_res_blocks + 1):
                self.up_blocks.append(
                    ResBlock3D(ch + chs.pop(), model_channels * mult, time_emb_dim, dropout)
                )
                ch = model_channels * mult
            if level != 0:
                self.up_blocks.append(Upsample3D(ch))

        # Output convolution
        self.out_norm = nn.GroupNorm(32, ch)
        self.out_conv = nn.Conv3d(ch, out_channels, 3, padding=1)

    def forward(self, x, t, age=None):
        """
        Args:
            x: Input tensor [B, C, D, H, W] (thick-slice concatenated with x_t)
            t: Timestep [B] or [B, 1]
            age: Age in months [B] or [B, 1], optional for CFG
        """
        # Time embedding
        t_emb_dim = self.time_embed[0].in_features
        t_emb = sinusoidal_embedding(t, t_emb_dim).to(x.dtype)
        t_emb = self.time_embed(t_emb)

        # Age embedding (CFG)
        if age is not None:
            age_emb = sinusoidal_embedding(age, t_emb_dim).to(x.dtype)
            age_emb = self.age_embed(age_emb)
            t_emb = t_emb + age_emb  # Combine

        # Encoder
        h = self.input_conv(x)
        hs = [h]
        for block in self.down_blocks:
            h = block(h, t_emb)
            hs.append(h)

        # Middle
        h = self.mid_block1(h, t_emb)
        h = self.mid_block2(h, t_emb)

        # Decoder
        for block in self.up_blocks:
            if isinstance(block, ResBlock3D):
                h = torch.cat([h, hs.pop()], dim=1)
            h = block(h, t_emb)

        # Output
        h = self.out_norm(h)
        h = nn.SiLU()(h)
        return self.out_conv(h)


class UNet3DEnhanced(nn.Module):
    """
    Enhanced U-Net with attention at lower resolutions.

    Adds self-attention blocks at the lowest resolution levels
    for capturing long-range spatial dependencies.
    """

    def __init__(self, *args, attn_resolutions=(8, 16), **kwargs):
        super().__init__()
        self.unet = UNet3D(*args, **kwargs)
        # TODO: Add attention blocks at specified resolution levels

    def forward(self, x, t, age=None):
        return self.unet(x, t, age)
