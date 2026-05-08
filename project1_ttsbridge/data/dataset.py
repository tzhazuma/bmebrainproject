"""
Dataset loader for T2T-Bridge.

Handles:
  - Loading preprocessed dHCP / CBCP paired thick-thin volumes
  - On-the-fly thick-slice simulation
  - Age-aware sampling (stratified by age group)
  - Tissue map loading for structure consistency
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib
from pathlib import Path
import random


class InfMRI3DDataset(Dataset):
    """
    3D paired dataset for thick-to-thin slice MRI reconstruction.

    Directory structure:
        data/dhcp/processed/
        ├── subject_001/
        │   ├── thin_t1w.nii.gz        # 0.8mm isotropic
        │   ├── tissue_map.nii.gz       # integer labels {0,1,2}
        │   └── meta.json               # {age_months: 6, sex: "F"}
        └── ...
    """

    AGE_GROUPS = [(0, 2), (2, 6), (6, 12), (12, 24), (24, 48), (48, 72)]

    def __init__(
        self,
        data_dir,
        split='train',
        patch_size=(128, 128, 32),
        augment=True,
        simulate_thick=True,
        thick_method='fourier',
        target_age_group=None,  # if set, only load this age range (min, max)
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.patch_size = patch_size
        self.augment = augment
        self.simulate_thick = simulate_thick
        self.target_age_group = target_age_group
        self.split = split

        # Discover subjects
        self.subjects = []
        subject_dirs = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])

        for subj_dir in subject_dirs:
            thin_path = subj_dir / 'thin_t1w.nii.gz'
            if thin_path.exists():
                self.subjects.append(subj_dir)

        # Filter by age group
        if target_age_group is not None:
            min_age, max_age = target_age_group
            filtered = []
            for subj in self.subjects:
                meta = self._load_meta(subj)
                if min_age <= meta.get('age_months', 0) <= max_age:
                    filtered.append(subj)
            self.subjects = filtered

        if len(self.subjects) == 0:
            raise ValueError(f"No subjects found in {data_dir}")

        # Create thick-slice simulator
        if simulate_thick:
            from .corruption import ThickSliceSimulator
            self.thick_sim = ThickSliceSimulator(
                thin_spacing=(0.8, 0.8, 0.8),
                thick_spacing=(5.2, 0.4, 0.4),
                method=thick_method,
            )
        else:
            self.thick_sim = None

    def _load_meta(self, subj_dir):
        import json
        meta_path = subj_dir / 'meta.json'
        if meta_path.exists():
            with open(meta_path) as f:
                return json.load(f)
        return {}

    def _load_nifti(self, path, dtype=np.float32):
        img = nib.load(str(path))
        data = img.get_fdata(dtype=dtype)
        return data

    def _random_crop(self, volume, target_size):
        """Random 3D crop with fallback to center crop if volume too small."""
        d, h, w = volume.shape[-3:]
        td, th, tw = target_size

        if d < td or h < th or w < tw:
            return self._center_crop(volume, target_size)

        d0 = random.randint(0, d - td)
        h0 = random.randint(0, h - th)
        w0 = random.randint(0, w - tw)
        return volume[..., d0:d0+td, h0:h0+th, w0:w0+tw]

    def _center_crop(self, volume, target_size):
        d, h, w = volume.shape[-3:]
        td, th, tw = target_size
        d0 = max(0, (d - td) // 2)
        h0 = max(0, (h - th) // 2)
        w0 = max(0, (w - tw) // 2)
        return volume[..., d0:d0+td, h0:h0+th, w0:w0+tw]

    def _augment(self, volume):
        """Apply 3D augmentations: random flip, rotation."""
        if random.random() > 0.5:
            volume = torch.flip(volume, dims=[-1])  # left-right flip
        if random.random() > 0.5:
            volume = torch.flip(volume, dims=[-2])  # ant-post flip
        if random.random() > 0.3:
            noise = torch.randn_like(volume) * 0.01
            volume = volume + noise
        return volume

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        subj_dir = self.subjects[idx]
        meta = self._load_meta(subj_dir)

        # Load thin-slice
        thin_data = self._load_nifti(subj_dir / 'thin_t1w.nii.gz')
        thin = torch.from_numpy(thin_data).float().unsqueeze(0)  # [1, D, H, W]

        # Normalize to [-1, 1]
        thin = thin / thin.max()
        thin = thin * 2 - 1

        # Crop to patch
        thin = self._random_crop(thin, self.patch_size)

        # Simulate or load thick-slice
        if self.simulate_thick and self.thick_sim is not None:
            thick = self.thick_sim(thin)
        else:
            thick_data = self._load_nifti(subj_dir / 'thick_t1w.nii.gz')
            thick = torch.from_numpy(thick_data).float().unsqueeze(0)
            thick = thick / thick.max()
            thick = thick * 2 - 1
            thick = self._random_crop(thick, self.patch_size)

        # Load tissue map
        tissue_path = subj_dir / 'tissue_map.nii.gz'
        tissue = torch.zeros(1, *self.patch_size) if not tissue_path.exists() else \
                 torch.from_numpy(
                     self._load_nifti(tissue_path, dtype=np.int32)
                 ).unsqueeze(0)

        if tissue_path.exists():
            tissue = self._random_crop(tissue, self.patch_size)

        # Age
        age = meta.get('age_months', 6)

        # Augmentation
        if self.augment:
            if random.random() > 0.5:
                perm = torch.randperm(3)
                perm = tuple(perm.numpy() + 1)
                thin = thin.permute(0, perm[0], perm[1], perm[2])
                thick = thick.permute(0, perm[0], perm[1], perm[2])
                tissue = tissue.permute(0, perm[0], perm[1], perm[2])

        return {
            'thin': thin,               # [1, D, H, W] target
            'thick': thick,             # [1, D, H, W] source
            'tissue': tissue.long(),    # [1, D, H, W] labels
            'age': age,
            'subject': subj_dir.name,
        }


class SyntheticDataGenerator(Dataset):
    """
    Generates fully synthetic brain-like volumes for debugging.
    Uses spheres, ellipsoids, and random noise.
    """

    def __init__(self, num_samples=100, size=(64, 64, 32)):
        self.num_samples = num_samples
        self.size = size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        d, h, w = self.size

        # Create simple synthetic brain: sphere + noise
        zz, yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, d),
            torch.linspace(-1, 1, h),
            torch.linspace(-1, 1, w),
            indexing='ij',
        )

        # Main brain sphere
        center = (0, 0, 0)
        r = 0.7
        brain = ((zz - center[0])**2 + (yy - center[1])**2 + (xx - center[2])**2) < r**2
        brain = brain.float()

        # Ventricle sphere
        vcenter = (0, 0, 0)
        vr = 0.2
        ventricle = ((zz - vcenter[0])**2 + (yy - vcenter[1])**2 + (xx - vcenter[2])**2) < vr**2
        ventricle = ventricle.float()

        # Add noise
        brain = brain + torch.randn_like(brain) * 0.05

        # Thin: full resolution
        thin = brain.unsqueeze(0)

        # Thick: blurred along z
        thick = thin.clone()
        kernel = torch.ones(5) / 5
        for c in range(thick.shape[1]):
            thick[0, c] = torch.nn.functional.conv1d(
                thick[0, c].unsqueeze(0).unsqueeze(0),
                kernel.view(1, 1, -1),
                padding=2,
            ).squeeze()

        # Tissue map
        tissue = torch.zeros_like(brain).long()
        tissue[brain > 0.5] = 1  # GM
        tissue[ventricle > 0.5] = 2  # CSF

        return {
            'thin': thin,
            'thick': thick,
            'tissue': tissue.unsqueeze(0),
            'age': idx % 72,
            'subject': f'synth_{idx:04d}',
        }
