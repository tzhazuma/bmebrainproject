#!/usr/bin/env python3
"""
Verify all installed tools and dependencies for the project.

Usage:
    python scripts/verify_tools.py
"""

import sys
from pathlib import Path

EXIT_CODE = 0
CHECKS_PASSED = 0
CHECKS_FAILED = 0
RESULTS = []


def check(name, description, fn):
    global EXIT_CODE, CHECKS_PASSED, CHECKS_FAILED
    try:
        fn()
        RESULTS.append(('pass', name, description))
        CHECKS_PASSED += 1
    except Exception as e:
        RESULTS.append(('fail', name, str(e)))
        EXIT_CODE = 1
        CHECKS_FAILED += 1


if __name__ == '__main__':
    print('=' * 60)
    print(' BME Brain Project — Tool Verification')
    print('=' * 60)

    # ---- Core Python ----
    def _py(): assert sys.version_info >= (3, 10), f'Python {sys.version}'
    check("Python", ">= 3.10", _py)

    # ---- Deep Learning ----
    def _torch():
        import torch
        assert torch.__version__ >= '2', torch.__version__
        assert torch.cuda.is_available(), 'CUDA not available'
        p = torch.cuda.get_device_properties(0)
    check("PyTorch", ">= 2.0 + CUDA", _torch)

    def _tv():
        import torchvision
        _ = torchvision.__version__
    check("TorchVision", "Image utilities", _tv)

    def _monai():
        import monai
        from monai.networks.nets import DynUNet
        m = DynUNet(
            spatial_dims=3, in_channels=1, out_channels=3,
            kernel_size=[3, 3, 3, 3], strides=[1, 2, 2, 1],
            upsample_kernel_size=[2, 2, 1],
        )
        _ = m.__class__.__name__
    check("MONAI", "Medical imaging toolkit", _monai)

    # ---- Image Processing ----
    def _nib(): import nibabel; _ = nibabel.__version__
    check("Nibabel", "NIfTI I/O", _nib)

    def _sitk():
        import SimpleITK as sitk
        img = sitk.GaussianSource(sitk.sitkFloat32, [16, 16, 8])
        assert img.GetSize() == (16, 16, 8)
    check("SimpleITK", "Registration & resampling", _sitk)

    def _hdbet():
        import subprocess
        result = subprocess.run(['hd-bet', '-h'], capture_output=True, text=True, timeout=30)
        assert 'usage' in result.stdout.lower() + result.stderr.lower()
    check("HD-BET", "Brain extraction", _hdbet)

    def _scipy(): import scipy; _ = scipy.__version__
    check("Scipy", "Scientific computing", _scipy)

    # ---- Acceleration ----
    def _bnb(): import bitsandbytes; _ = bitsandbytes.__version__
    check("bitsandbytes", "8-bit quantization", _bnb)

    def _accel(): import accelerate; _ = accelerate.__version__
    check("accelerate", "Distributed training", _accel)

    # ---- Data ----
    def _npy(): import numpy; _ = numpy.__version__
    check("numpy", "Numerical computing", _npy)

    def _skl(): import sklearn; _ = sklearn.__version__
    check("scikit-learn", "ML metrics", _skl)

    def _yaml(): import yaml; _ = yaml.__version__
    check("PyYAML", "Config files", _yaml)

    # ---- Optional ----
    def _gdown(): import gdown; _ = gdown.__version__
    check("gdown", "Google Drive downloader", _gdown)

    def _ants():
        try:
            import antspyx
            _ = antspyx.__version__
        except ImportError:
            # ANTsPy not available on this Python — SimpleITK covers all needs
            pass
    check("ANTsPy", "Advanced registration (optional, SimpleITK used instead)", _ants)

    # ---- Project Modules ----
    def _p1():
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from project1_ttsbridge.data.dataset import SyntheticDataGenerator
        from project1_ttsbridge.models.unet3d import UNet3D
        ds = SyntheticDataGenerator(2, (32, 32, 16))
        _ = ds[0]
    check("project1_ttsbridge", "T2T-Bridge models", _p1)

    def _p2():
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from project2_multimodal.data.synthetic_dataset import SyntheticMultimodalDataset
        from project2_multimodal.models.classifier import DiagnosisHead
        ds = SyntheticMultimodalDataset(2, (32, 32, 16))
        _ = ds[0]
    check("project2_multimodal", "Multimodal models", _p2)

    def _ckpt_dir():
        Path('checkpoints').mkdir(parents=True, exist_ok=True)
        Path('results').mkdir(parents=True, exist_ok=True)
    check("Checkpoint dirs", "Model storage", _ckpt_dir)

    # ---- Report ----
    print()
    for status, name, msg in RESULTS:
        icon = '✅' if status == 'pass' else '❌'
        print(f'  {icon} {name}: {msg}')

    print(f'\n{"=" * 60}')
    print(f' Result: {CHECKS_PASSED}/{CHECKS_PASSED + CHECKS_FAILED} passed')
    if EXIT_CODE:
        print(' Some tools missing — see ❌ above.\n')
    else:
        print(' All tools ready!\n')
    sys.exit(EXIT_CODE)
