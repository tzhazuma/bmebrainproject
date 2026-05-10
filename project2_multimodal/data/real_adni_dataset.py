import os, json, torch, random, nibabel as nib
import numpy as np
from torch.utils.data import Dataset
from collections import defaultdict

DIAGNOSIS_TO_LABEL = {'CN': 0, 'EMCI': 1, 'MCI': 1, 'LMCI': 2, 'AD': 2}

class RealADNIDataset(Dataset):
    def __init__(self, data_dir, modalities=('t1w', 'fdg_pet', 'tau_pet'), patch_size=(32,32,16), augment=False, modality_drop_prob=0.3):
        super().__init__()
        self.data_dir = data_dir
        self.modalities = list(modalities)
        self.patch_size = patch_size
        self.augment = augment
        self.drop_prob = modality_drop_prob
        self.subjects = []
        for s in sorted(os.listdir(data_dir)):
            dp = os.path.join(data_dir, s)
            if not os.path.isdir(dp): continue
            meta_file = os.path.join(dp, 'meta.json')
            if not os.path.exists(meta_file): continue
            has_t1 = os.path.exists(os.path.join(dp, 't1w.nii')) or os.path.exists(os.path.join(dp, 't1w.nii.gz'))
            if not has_t1: continue
            self.subjects.append(s)

    def __len__(self): return len(self.subjects)

    def _find_nii(self, dp, name):
        for ext in ['.nii.gz', '.nii']:
            p = os.path.join(dp, name + ext)
            if os.path.exists(p): return p
        return None

    def _load_volume(self, path):
        data = nib.load(path).get_fdata(dtype=np.float32)
        data = torch.from_numpy(data).unsqueeze(0)
        vmax = data.abs().max()
        if vmax > 0: data = data / vmax
        data = data * 2 - 1
        return data

    def _crop_or_pad(self, vol):
        t = self.patch_size
        s = list(vol.shape[1:])
        result = torch.zeros(1, *t)
        crop = [min(a,b) for a,b in zip(s,t)]
        d0 = [(a-b)//2 for a,b in zip(s,crop)]
        result[:,:crop[0],:crop[1],:crop[2]] = vol[:, d0[0]:d0[0]+crop[0], d0[1]:d0[1]+crop[1], d0[2]:d0[2]+crop[2]]
        return result

    def __getitem__(self, idx):
        s = self.subjects[idx]
        dp = os.path.join(self.data_dir, s)
        images = {}
        available = []
        for mod in self.modalities:
            p = self._find_nii(dp, mod)
            if p:
                vol = self._load_volume(p)
                vol = self._crop_or_pad(vol)
                images[mod] = vol
                available.append(mod)

        with open(os.path.join(dp, 'meta.json')) as f:
            meta = json.load(f)
        label = DIAGNOSIS_TO_LABEL.get(meta.get('Group', 'CN'), 0)
        age = float(meta.get('Age', 70))
        return {'images': images, 'label': label, 'age': age, 'available_modalities': available, 'subject': s}
