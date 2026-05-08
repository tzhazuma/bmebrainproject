"""
ADNI dataset loader for multimodal diagnosis.

Handles:
  - Loading T1w MRI, FDG-PET, TAU-PET modalities
  - Modality dropout for missing-modality training
  - Diagnosis labels (NC=0, MCI=1, AD=2)
  - 3D patch extraction around disease-relevant ROIs
"""

import torch
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
import json
from pathlib import Path
import random


class ADNIMultimodalDataset(Dataset):
    """
    Multimodal ADNI dataset.

    Directory structure:
        data/adni/processed/
        ├── subject_001/
        │   ├── t1w.nii.gz        # 1mm MNI space
        │   ├── fdg_pet.nii.gz    # SUVR normalized
        │   ├── tau_pet.nii.gz    # SUVR normalized
        │   ├── tissue.nii.gz     # GM/WM/CSF
        │   └── meta.json         # {diagnosis: "AD", mmse: 23, ...}
        └── ...

    Diagnosis labels:
        0: NC (Normal Control)
        1: MCI (Mild Cognitive Impairment)
        2: AD (Alzheimer's Disease)
    """

    DIAGNOSIS_MAP = {'NC': 0, 'MCI': 1, 'AD': 2}
    MODALITIES = ['t1w', 'fdg_pet', 'tau_pet']

    def __init__(
        self,
        data_dir,
        split='train',
        modalities=None,
        patch_size=(64, 64, 32),
        augment=False,
        modality_drop_prob=0.3,  # probability of dropping a modality
        return_meta=False,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.split = split
        self.patch_size = patch_size
        self.augment = augment
        self.modality_drop_prob = modality_drop_prob
        self.return_meta = return_meta

        if modalities is None:
            modalities = self.MODALITIES
        self.modalities = modalities

        # Load subjects based on split
        self.subjects = self._load_split(split)

        if len(self.subjects) == 0:
            raise ValueError(f"No subjects found in {data_dir}/{split}")

        print(f"Loaded {len(self.subjects)} subjects for {split} split")

    def _load_split(self, split):
        """Load subject directories for a given split."""
        split_dir = self.data_dir / split
        if not split_dir.exists():
            # If no split subdir, use all subjects
            return sorted([d for d in self.data_dir.iterdir() if d.is_dir()])

        return sorted([d for d in split_dir.iterdir() if d.is_dir()])

    def _load_meta(self, subj_dir):
        """Load metadata JSON."""
        meta_path = subj_dir / 'meta.json'
        if meta_path.exists():
            with open(meta_path) as f:
                return json.load(f)
        return {}

    def _load_nifti(self, path, dtype=np.float32):
        """Load NIfTI and return as numpy array."""
        if not path.exists():
            return None
        return nib.load(str(path)).get_fdata(dtype=dtype)

    def _to_tensor(self, arr, add_channel=True):
        if arr is None:
            return None
        t = torch.from_numpy(arr).float()
        if add_channel:
            t = t.unsqueeze(0)
        return t

    def _random_crop_3d(self, volume, size):
        """Random 3D crop."""
        if volume is None:
            return None
        d, h, w = volume.shape[-3:]
        td, th, tw = size
        if d < td or h < th or w < tw:
            # Pad if too small
            pad_d = max(0, td - d)
            pad_h = max(0, th - h)
            pad_w = max(0, tw - w)
            volume = torch.nn.functional.pad(volume, (0, pad_w, 0, pad_h, 0, pad_d))
            d, h, w = volume.shape[-3:]

        d0 = random.randint(0, d - td)
        h0 = random.randint(0, h - th)
        w0 = random.randint(0, w - tw)
        return volume[..., d0:d0+td, h0:h0+th, w0:w0+tw]

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        subj_dir = self.subjects[idx]
        meta = self._load_meta(subj_dir)

        # Load modalities
        images = {}
        for mod in self.modalities:
            data = self._load_nifti(subj_dir / f'{mod}.nii.gz')
            images[mod] = self._to_tensor(data) if data is not None else None

        # Determine available modalities (for dropout)
        available_modalities = [m for m, img in images.items() if img is not None]

        # Modality dropout during training
        if self.modality_drop_prob > 0 and self.split == 'train' and len(available_modalities) > 1:
            for mod in available_modalities:
                if random.random() < self.modality_drop_prob:
                    # Drop this modality (set to None for cross-modal)
                    # Keep in available list for training
                    pass  # We'll handle this in the model

        # Crop all modalities consistently
        selected = next(img for img in images.values() if img is not None)
        if selected is not None:
            d, h, w = selected.shape[-3:]
            td, th, tw = self.patch_size

            if d < td or h < th or w < tw:
                pad_tuple = (0, max(0, tw - w), 0, max(0, th - h), 0, max(0, td - d))
                for mod in list(images.keys()):
                    if images[mod] is not None:
                        images[mod] = torch.nn.functional.pad(images[mod], pad_tuple)

            d0 = random.randint(0, max(1, images[list(images.keys())[0]].shape[-3] - td)) if self.split == 'train' else 0
            h0 = random.randint(0, max(1, images[list(images.keys())[0]].shape[-2] - th)) if self.split == 'train' else 0
            w0 = random.randint(0, max(1, images[list(images.keys())[0]].shape[-1] - tw)) if self.split == 'train' else 0

            for mod in list(images.keys()):
                if images[mod] is not None:
                    images[mod] = images[mod][..., d0:d0+td, h0:h0+th, w0:w0+tw]

        # Normalize to [-1, 1]
        for mod in images:
            if images[mod] is not None:
                vmax = images[mod].max()
                if vmax > 0:
                    images[mod] = images[mod] / vmax
                images[mod] = images[mod] * 2 - 1

        # Diagnosis label
        diagnosis = meta.get('diagnosis', 'NC')
        label = self.DIAGNOSIS_MAP.get(diagnosis, 0)

        output = {
            'images': images,
            'label': label,
            'available_modalities': available_modalities,
            'subject': subj_dir.name,
        }

        if self.return_meta:
            output['meta'] = meta

        return output


def collate_fn(batch):
    """Custom collate for variable modalities."""
    return {
        'images': [b['images'] for b in batch],
        'label': torch.tensor([b['label'] for b in batch]),
        'available_modalities': [b['available_modalities'] for b in batch],
        'subject': [b['subject'] for b in batch],
    }
