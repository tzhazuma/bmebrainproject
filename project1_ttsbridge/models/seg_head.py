"""
Tissue segmentation head for Structure Consistency Loss.

Outputs GM, WM, CSF probability maps used to enforce
anatomical plausibility of generated thin-slice MRI.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SegmentationHead(nn.Module):
    """
    Lightweight Conv3D decoder that takes U-Net bottleneck features or generated image
    and outputs tissue probability maps (3 classes: GM, WM, CSF).
    """

    def __init__(self, in_channels=64, num_classes=3, hidden=32):
        super().__init__()
        num_groups = min(8, hidden) if hidden > 1 else 1
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, hidden, 3, padding=1),
            nn.GroupNorm(num_groups if in_channels > 1 else 1, hidden) if num_groups > 1 else nn.BatchNorm3d(hidden),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(hidden, hidden, 3, padding=1),
            nn.GroupNorm(num_groups, hidden) if num_groups > 1 else nn.BatchNorm3d(hidden),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Conv3d(hidden, num_classes, 1)

    def forward(self, features):
        """
        Args:
            features: U-Net encoder features or image [B, C, D, H, W]
        Returns:
            tissue_probs: [B, num_classes, D, H, W]
        """
        h = self.conv1(features)
        h = self.conv2(h)
        return self.conv3(h)


class SharedSegHead(nn.Module):
    """
    Segmentation head that shares features with the diffusion denoiser.
    Takes features from multiple U-Net encoder levels for better detail.
    """

    def __init__(self, feature_channels=(64, 128, 256, 512), num_classes=3):
        super().__init__()
        self.upsample_layers = nn.ModuleList()
        self.conv_layers = nn.ModuleList()

        for ch in feature_channels:
            self.upsample_layers.append(
                nn.Sequential(
                    nn.Conv3d(ch, 32, 3, padding=1),
                    nn.GroupNorm(8, 32),
                    nn.ReLU(inplace=True),
                )
            )

        self.final = nn.Sequential(
            nn.Conv3d(32 * len(feature_channels), 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, num_classes, 1),
        )

    def forward(self, features_list):
        """
        Args:
            features_list: List of multi-scale features [f1(B,C1,D,H,W), f2, f3, f4]
        Returns:
            tissue_probs: [B, num_classes, D, H, W]
        """
        upsampled = []
        target_size = features_list[0].shape[2:]

        for feat, upsample in zip(features_list, self.upsample_layers):
            f = upsample(feat)
            f = F.interpolate(f, size=target_size, mode='trilinear', align_corners=False)
            upsampled.append(f)

        concat = torch.cat(upsampled, dim=1)
        return self.final(concat)


def tissue_to_onehot(tissue_map, num_classes=3):
    """
    Convert tissue label map to one-hot encoding.

    Args:
        tissue_map: [B, 1, D, H, W] with integer labels {0,1,2}
    Returns:
        onehot: [B, num_classes, D, H, W]
    """
    B, C, D, H, W = tissue_map.shape
    onehot = torch.zeros(B, num_classes, D, H, W, device=tissue_map.device)
    return onehot.scatter_(1, tissue_map.long(), 1)
