"""
Brain tissue segmentation using MONAI DynUNet.

Lightweight replacement for FreeSurfer infantFS.
Performs 3-class segmentation: GM, WM, CSF on infant brain MRI.
"""

import torch
import torch.nn as nn
import nibabel as nib
import numpy as np
from pathlib import Path
from monai.networks.nets import DynUNet
from monai.inferers import sliding_window_inference
from monai.transforms import (
    NormalizeIntensity,
    EnsureType,
    EnsureChannelFirstd,
    ScaleIntensityRanged,
)


class InfantBrainSegmenter(nn.Module):
    """
    3-class infant brain tissue segmentation (GM=1, WM=2, CSF=0).

    Uses MONAI DynUNet with 3D sliding window inference.
    """

    def __init__(
        self,
        in_channels=1,
        out_channels=3,
        roi_size=(64, 64, 32),
        device='cuda',
    ):
        super().__init__()
        self.device = device
        self.roi_size = roi_size

        self.model = DynUNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=[3, 3, 3, 3],
            strides=[1, 2, 2, 1],
            upsample_kernel_size=[2, 2, 1],
            filters=[32, 64, 128, 256],
            dropout=0.1,
            deep_supervision=False,
        ).to(device)

        self.model.eval()

    def forward(self, x):
        return self.model(x)

    @torch.no_grad()
    def predict(self, image_path, output_path=None, postprocess=True):
        """
        Segment a 3D brain MRI into tissue classes.

        Args:
            image_path: Path to NIfTI file or numpy array [D, H, W]
            output_path: Optional output path for tissue map
            postprocess: Apply largest connected component per class

        Returns:
            tissue_map: numpy array [D, H, W] with labels {0, 1, 2}
        """
        if isinstance(image_path, (str, Path)):
            img = nib.load(str(image_path))
            data = img.get_fdata(dtype=np.float32)
        else:
            data = image_path
            if data.max() != 0:
                data = data / data.max()

        # Normalize to [0, 1]
        if data.max() > data.min():
            data = (data - data.min()) / (data.max() - data.min())

        # Add batch and channel dims: [D, H, W] → [1, 1, D, H, W]
        tensor = torch.from_numpy(data).float().unsqueeze(0).unsqueeze(0).to(self.device)

        # Sliding window inference
        logits = sliding_window_inference(
            tensor,
            self.roi_size,
            sw_batch_size=4,
            predictor=self.model,
            overlap=0.5,
            mode='gaussian',
        )

        # Argmax to get class labels
        tissue_map = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()

        if postprocess:
            tissue_map = self._postprocess(tissue_map)

        if output_path and isinstance(image_path, (str, Path)):
            if isinstance(image_path, (str, Path)):
                ref_img = nib.load(str(image_path))
                out_img = nib.Nifti1Image(
                    tissue_map.astype(np.int16),
                    ref_img.affine,
                    ref_img.header,
                )
                nib.save(out_img, str(output_path))

        return tissue_map

    def _postprocess(self, tissue_map):
        """Simple post-processing: remove small islands."""
        from scipy.ndimage import label

        processed = np.zeros_like(tissue_map)
        for c in range(3):
            mask = tissue_map == c
            if mask.sum() == 0:
                continue
            labeled, n_features = label(mask)
            if n_features > 1:
                sizes = np.bincount(labeled.ravel())[1:]
                max_label = sizes.argmax() + 1
                processed[labeled == max_label] = c
            else:
                processed[mask] = c
        return processed

    def save_checkpoint(self, path):
        torch.save({'model': self.model.state_dict()}, path)

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model'])
        self.model.eval()


def create_default_segmenter(device='cuda'):
    """Create a segmenter with default configuration."""
    return InfantBrainSegmenter(
        in_channels=1,
        out_channels=3,
        roi_size=(64, 64, 32),
        device=device,
    )
