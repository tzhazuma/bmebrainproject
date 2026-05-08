"""
Corruption module: simulate 2D thick-slice from 3D thin-slice MRI.

Methods:
  1. Fourier low-pass filter + subsampling (clinically realistic)
  2. Gaussian blur + averaging (simple baseline)
  3. k-space zero-filling (most physically accurate)
"""

import torch
import torch.nn as nn
import torch.fft


class ThickSliceSimulator:
    """
    Simulate 2D thick-slice acquisition from 3D thin-slice.

    Through-plane (z-axis):
      - Thin:  0.8 mm isotropic
      - Thick: 5.2 mm (0.8 × 6.5)

    In-plane (x, y axes):
      - Thin:  0.8 mm
      - Thick: 0.4 mm (can optionally upsample in-plane)
    """

    def __init__(
        self,
        thin_spacing=(0.8, 0.8, 0.8),  # mm (z, y, x)
        thick_spacing=(5.2, 0.4, 0.4),  # mm (z, y, x)
        method="fourier",
    ):
        self.thin_spacing = thin_spacing
        self.thick_spacing = thick_spacing
        self.method = method
        self.slice_factor = thick_spacing[0] / thin_spacing[0]  # ~6.5

    def __call__(self, thin_slice, return_meta=False):
        """
        Args:
            thin_slice: 3D volume [B, 1, D, H, W] or [1, D, H, W]
        Returns:
            thick_slice: Simulated thick-slice volume
        """
        if self.method == "fourier":
            thick = self._fourier_lowpass(thin_slice)
        elif self.method == "gaussian":
            thick = self._gaussian_blur(thin_slice)
        elif self.method == "average":
            thick = self._slice_average(thin_slice)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        if return_meta:
            return thick, {"method": self.method, "slice_factor": self.slice_factor}
        return thick

    def _fourier_lowpass(self, volume):
        """Fourier-based through-plane low-pass filtering."""
        # FFT along z-axis
        kspace = torch.fft.fft(volume, dim=-3)
        kspace = torch.fft.fftshift(kspace, dim=-3)

        # Low-pass filter: keep only (1/slice_factor) of k-space along z
        D = volume.shape[-3]
        cutoff = int(D / self.slice_factor / 2)

        mask = torch.zeros(D, device=volume.device)
        center = D // 2
        mask[center - cutoff : center + cutoff + 1] = 1.0
        mask = mask.view(1, 1, D, 1, 1)

        kspace_filtered = kspace * mask
        kspace_filtered = torch.fft.ifftshift(kspace_filtered, dim=-3)
        result = torch.fft.ifft(kspace_filtered, dim=-3).real

        return result

    def _gaussian_blur(self, volume):
        """Through-plane Gaussian blur + downsampling."""
        sigma = self.slice_factor / 2.355  # FWHM ≈ 2.355σ
        kernel_size = int(sigma * 6 + 1) | 1  # make odd

        # Create 1D Gaussian kernel along z
        coords = torch.arange(kernel_size, device=volume.device) - kernel_size // 2
        kernel = torch.exp(-0.5 * (coords / sigma) ** 2)
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, kernel_size, 1, 1)

        # Apply convolution along z
        padding = kernel_size // 2
        blurred = nn.functional.conv3d(
            volume.unsqueeze(0) if volume.dim() == 4 else volume,
            kernel.expand(1, 1, kernel_size, 1, 1),
            padding=(padding, 0, 0),
        )

        if volume.dim() == 4:
            blurred = blurred.squeeze(0)

        return blurred

    def _slice_average(self, volume):
        """Simple slice averaging (nearest-neighbor downsampling)."""
        D = volume.shape[-3]
        factor = int(self.slice_factor)

        # Pad if needed
        if D % factor != 0:
            pad = factor - (D % factor)
            volume = nn.functional.pad(volume, (0, 0, 0, 0, 0, pad))

        # Reshape and average
        shape = volume.shape
        volume = volume.view(shape[0], shape[1], -1, factor, shape[-2], shape[-1])
        volume = volume.mean(dim=-3)
        return volume


class PairedDataGenerator:
    """Generate paired thick-thin data from thin-slice volumes."""

    def __init__(self, simulator, upscale_inplane=True):
        self.simulator = simulator
        self.upscale_inplane = upscale_inplane

    def generate_pair(self, thin_volume):
        thick = self.simulator(thin_volume)

        if self.upscale_inplane:
            # Optionally upsample in-plane to 0.4mm from 0.8mm
            thick = nn.functional.interpolate(
                thick.unsqueeze(0) if thick.dim() == 4 else thick,
                scale_factor=(1.0, 2.0, 2.0),
                mode="trilinear",
            )
            if thick.dim() == 5:
                thick = thick.squeeze(0)

        return thick, thin_volume
