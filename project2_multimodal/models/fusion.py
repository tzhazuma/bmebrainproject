"""Cross-attention fusion module for multimodal PET/MR."""

import torch
import torch.nn as nn


class CrossAttentionFusion(nn.Module):
    """
    Fuse latent representations from multiple modalities via cross-attention.

    Given K modality-specific latent codes z₁,...,z_K, computes:
      - Cross-attention between each pair
      - Weighted fusion with learnable modality importance weights
    """

    def __init__(self, latent_dim=256, num_modalities=3, num_heads=4, dropout=0.1):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_modalities = num_modalities

        # Cross-attention between modalities
        self.cross_attn = nn.MultiheadAttention(
            latent_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.self_attn = nn.MultiheadAttention(
            latent_dim, num_heads, dropout=dropout, batch_first=True
        )

        # Modality importance weights
        self.modality_weights = nn.Parameter(torch.ones(num_modalities) / num_modalities)

        # Output projection
        self.fc = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, z_list):
        """
        Args:
            z_list: List of [B, latent_dim] tensors (one per available modality)
        Returns:
            fused_z: [B, latent_dim]
            weights: Attention weights for each modality
        """
        B = z_list[0].shape[0]
        K = len(z_list)
        device = z_list[0].device

        # Stack modalities: [B, K, D]
        z_stack = torch.stack(z_list, dim=1)

        # Self-attention across modalities
        z_self, self_weights = self.self_attn(z_stack, z_stack, z_stack)
        z_self = z_self.mean(dim=1)  # [B, D]

        # Cross-attention: use first modality as query (or mean)
        if K > 1:
            query = z_stack[:, 0:1, :]  # [B, 1, D]
            z_cross, cross_weights = self.cross_attn(query, z_stack, z_stack)
            z_cross = z_cross.squeeze(1)  # [B, D]
        else:
            z_cross = z_stack.squeeze(1)

        # Combine
        z_concat = torch.cat([z_self, z_cross], dim=-1)  # [B, 2D]
        fused_z = self.fc(z_concat)

        return fused_z


class AdaptiveFusion(nn.Module):
    """
    Adaptive fusion that handles arbitrary modality subsets at inference time.

    Uses learnable per-modality projection to a shared space,
    then weighted sum with learned importance.
    """

    def __init__(self, latent_dim=256, num_modalities=3):
        super().__init__()
        self.projection = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim, latent_dim),
                nn.LayerNorm(latent_dim),
                nn.ReLU(inplace=True),
            )
            for _ in range(num_modalities)
        ])
        self.fusion_fc = nn.Linear(latent_dim, latent_dim)

    def forward(self, z_list):
        """
        Args:
            z_list: List of [B, latent_dim] — variable length!
        Returns:
            fused: [B, latent_dim]
        """
        projected = [proj(z) for proj, z in zip(self.projection[:len(z_list)], z_list)]
        fused = torch.stack(projected, dim=0).mean(dim=0)  # simple mean
        return self.fusion_fc(fused)
