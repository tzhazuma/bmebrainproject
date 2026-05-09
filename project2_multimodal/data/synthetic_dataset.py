"""
Synthetic multimodal data generator for debugging.
"""

import torch
from torch.utils.data import Dataset
import random


class SyntheticMultimodalDataset(Dataset):
    """
    Generates synthetic multimodal data for debugging.

    Creates random 3D volumes representing different modalities:
      - t1w: structural with tissue classes
      - fdg_pet: metabolic activity
      - tau_pet: tau pathology
    """

    DIAGNOSIS_MAP = {'NC': 0, 'MCI': 1, 'AD': 2}

    def __init__(self, num_samples=100, size=(64, 64, 32)):
        self.num_samples = num_samples
        self.size = size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        d, h, w = self.size

        # Diagnosis distribution: 40% NC, 30% MCI, 30% AD
        diag_idx = idx % 10
        if diag_idx < 4:
            diagnosis = 'NC'
        elif diag_idx < 7:
            diagnosis = 'MCI'
        else:
            diagnosis = 'AD'

        label = self.DIAGNOSIS_MAP[diagnosis]

        # Generate base brain structure
        zz, yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, d),
            torch.linspace(-1, 1, h),
            torch.linspace(-1, 1, w),
            indexing='ij',
        )

        # Brain mask
        brain_mask = (zz**2 + yy**2 + xx**2) < 0.7**2
        ventricle_mask = ((zz - 0.05)**2 + yy**2 + xx**2) < 0.15**2

        # T1w: structural image [1, 1, D, H, W] (B, C, D, H, W)
        t1w = brain_mask.float() + torch.randn_like(brain_mask.float()) * 0.05
        t1w = t1w.unsqueeze(0).unsqueeze(0)

        # Normalize to [-1, 1]
        t1w = t1w / (t1w.abs().max() + 1e-8)
        t1w = t1w * 2 - 1

        # FDG-PET: metabolic (higher in gray matter, lower in AD)
        fdg = brain_mask.float() * (0.8 + 0.2 * torch.rand(1).item())
        if diagnosis == 'AD':
            fdg = fdg * 0.7  # hypometabolism in AD
        fdg = fdg + torch.randn_like(fdg) * 0.03
        fdg = fdg.unsqueeze(0).unsqueeze(0)
        fdg = fdg / (fdg.abs().max() + 1e-8)
        fdg = fdg * 2 - 1

        # TAU-PET: pathology (higher in AD, lower in NC)
        tau = brain_mask.float() * 0.3
        if diagnosis == 'AD':
            tau = tau + ventricle_mask.float() * 0.5 + torch.randn_like(tau) * 0.1
        elif diagnosis == 'MCI':
            tau = tau + ventricle_mask.float() * 0.2 + torch.randn_like(tau) * 0.05
        tau = tau + torch.randn_like(tau) * 0.02
        tau = tau.unsqueeze(0).unsqueeze(0)
        tau = tau / (tau.abs().max() + 1e-8)
        tau = tau * 2 - 1

        images = {
            't1w': t1w,
            'fdg_pet': fdg,
            'tau_pet': tau,
        }

        return {
            'images': images,
            'label': label,
            'available_modalities': list(images.keys()),
            'subject': f'synth_{idx:04d}',
        }
