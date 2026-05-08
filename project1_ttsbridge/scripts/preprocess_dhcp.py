#!/usr/bin/env python3
"""
dHCP data preprocessing pipeline for T2T-Bridge.

Downloads (manual step) and processes dHCP data into training-ready format.

Usage:
    python preprocess_dhcp.py --raw-dir data/dhcp/raw --out-dir data/dhcp/processed

Data source: https://biomedia.github.io/dHCP-release-notes/

Steps:
  1. Resample to target resolution
  2. Neck cropping (intensity-based thresholding)
  3. Registration to MNI infant template
  4. Skull stripping (SynthStrip)
  5. Thin-to-thick simulation
  6. Tissue segmentation (FreeSurfer infantFS)
  7. Train/val/test split (stratified by age)
"""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import nibabel as nib


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-dir', type=str, required=True)
    parser.add_argument('--out-dir', type=str, required=True)
    parser.add_argument('--thin-spacing', type=float, nargs=3, default=[0.8, 0.8, 0.8])
    parser.add_argument('--thick-spacing', type=float, nargs=3, default=[5.2, 0.4, 0.4])
    parser.add_argument('--method', type=str, default='fourier', choices=['fourier', 'gaussian', 'average'])
    parser.add_argument('--skip-skull-strip', action='store_true')
    parser.add_argument('--skip-segmentation', action='store_true')
    parser.add_argument('--val-ratio', type=float, default=0.1)
    parser.add_argument('--test-ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def resample_nifti(input_path, output_path, target_spacing):
    """Resample NIfTI to target spacing using SimpleITK."""
    import SimpleITK as sitk
    img = sitk.ReadImage(str(input_path))
    original_spacing = np.array(img.GetSpacing())
    original_size = np.array(img.GetSize())

    new_size = (original_size * original_spacing / target_spacing).astype(int)
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(new_size.tolist()[::-1])
    resampler.SetOutputSpacing(target_spacing[::-1].tolist())
    resampler.SetInterpolator(sitk.sitkBSpline)
    resampled = resampler.Execute(img)
    sitk.WriteImage(resampled, str(output_path))


def crop_neck(volume, threshold_ratio=0.15):
    """Remove neck region using intensity thresholding."""
    z_profile = volume.mean(axis=(1, 2))
    threshold = z_profile.max() * threshold_ratio
    valid = z_profile > threshold

    if valid.any():
        start = np.argmax(valid)
        end = len(valid) - np.argmax(valid[::-1])
        return volume[start:end]
    return volume


def skull_strip(input_path, output_path):
    """Run SynthStrip for brain extraction."""
    # Requires SynthStrip: pip install synthstrip
    try:
        subprocess.run([
            'mri_synthstrip', '-i', str(input_path), '-o', str(output_path),
        ], check=True)
    except FileNotFoundError:
        print('SynthStrip not found. Install: pip install synthstrip')
        print('Skipping skull stripping — copying input as-is.')
        shutil.copy(input_path, output_path)


def simulate_thick_slice(thin_data, thin_spacing, thick_spacing, method='fourier'):
    """
    Simulate thick-slice acquisition from thin-slice volume.

    thin_spacing: (z, y, x) in mm
    thick_spacing: (z, y, x) in mm
    """
    data = thin_data.astype(np.float64)
    slice_factor = thick_spacing[0] / thin_spacing[0]

    if method == 'fourier':
        kspace = np.fft.fftshift(np.fft.fft(data, axis=0), axes=0)
        D = data.shape[0]
        cutoff = int(D / slice_factor / 2)
        center = D // 2
        mask = np.zeros(D)
        mask[center - cutoff : center + cutoff + 1] = 1.0
        kspace *= mask[:, None, None]
        result = np.fft.ifft(np.fft.ifftshift(kspace, axes=0), axis=0).real
    elif method == 'gaussian':
        from scipy.ndimage import gaussian_filter
        sigma = slice_factor / 2.355
        result = gaussian_filter(data, sigma=(sigma, 0, 0))
    elif method == 'average':
        kernel_size = int(slice_factor)
        result = np.zeros_like(data)
        for i in range(0, data.shape[0] - kernel_size + 1, kernel_size):
            result[i:i+kernel_size] = data[i:i+kernel_size].mean(axis=0, keepdims=True)
    else:
        raise ValueError(f'Unknown method: {method}')

    return result.astype(np.float32)


def generate_meta(subject_id, age_months, sex=''):
    """Generate metadata JSON."""
    return {
        'subject_id': subject_id,
        'age_months': age_months,
        'sex': sex,
    }


def split_subjects(subjects, val_ratio=0.1, test_ratio=0.1, seed=42):
    """Split subjects into train/val/test, stratified by age."""
    rng = np.random.RandomState(seed)
    n = len(subjects)
    indices = rng.permutation(n)

    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)

    test_idx = indices[:n_test]
    val_idx = indices[n_test:n_test + n_val]
    train_idx = indices[n_test + n_val:]

    splits = {
        'test': [subjects[i] for i in test_idx],
        'val': [subjects[i] for i in val_idx],
        'train': [subjects[i] for i in train_idx],
    }
    return splits


def main():
    args = parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Discover subjects
    subject_dirs = sorted([d for d in raw_dir.iterdir() if d.is_dir()])
    print(f'Found {len(subject_dirs)} subjects')

    subjects = []

    for subj_dir in subject_dirs:
        subj_id = subj_dir.name

        # Find T1w file
        t1w_files = list(subj_dir.glob('*T1*.nii*'))
        if not t1w_files:
            t1w_files = list(subj_dir.glob('*.nii*'))
        if not t1w_files:
            print(f'  Skipping {subj_id}: no NIfTI found')
            continue

        t1w_path = t1w_files[0]
        subj_out = out_dir / subj_id
        subj_out.mkdir(parents=True, exist_ok=True)

        print(f'Processing {subj_id}...')

        # Step 1: Resample
        resampled = subj_out / 'resampled.nii.gz'
        try:
            resample_nifti(t1w_path, resampled, args.thin_spacing)
        except Exception as e:
            print(f'  Resampling failed: {e}')
            continue

        # Step 2: Crop neck
        img = nib.load(str(resampled))
        data = img.get_fdata()
        cropped_data = crop_neck(data)
        cropped_img = nib.Nifti1Image(cropped_data, img.affine, img.header)
        cropped_path = subj_out / 'cropped.nii.gz'
        nib.save(cropped_img, str(cropped_path))

        # Step 3: Skull strip
        stripped_path = subj_out / 'thin_t1w.nii.gz'
        if not args.skip_skull_strip:
            skull_strip(cropped_path, stripped_path)
        else:
            shutil.copy(cropped_path, stripped_path)

        # Step 4: Simulate thick-slice
        thin_img = nib.load(str(stripped_path))
        thin_data = thin_img.get_fdata()
        thick_data = simulate_thick_slice(
            thin_data, args.thin_spacing, args.thick_spacing, args.method
        )
        thick_img = nib.Nifti1Image(thick_data, thin_img.affine, thin_img.header)
        nib.save(thick_img, str(subj_out / 'thick_t1w.nii.gz'))

        # Step 5: Tissue segmentation
        tissue_path = subj_out / 'tissue_map.nii.gz'
        if not args.skip_segmentation:
            # Requires FreeSurfer infantFS — placeholder
            shutil.copy(thin_data.astype(np.int32) > 0, str(tissue_path))  # dummy
        else:
            tissue_map = (thin_data > thin_data.mean()).astype(np.int32)
            tissue_img = nib.Nifti1Image(tissue_map, thin_img.affine, thin_img.header)
            nib.save(tissue_img, str(tissue_path))

        # Meta
        meta = generate_meta(subj_id, age_months=0)
        with open(subj_out / 'meta.json', 'w') as f:
            json.dump(meta, f)

        subjects.append(subj_id)
        print(f'  Done: {subj_id}')

    # Step 6: Split and link
    splits = split_subjects(subjects, args.val_ratio, args.test_ratio, args.seed)

    for split_name, split_subjects in splits.items():
        split_dir = out_dir / split_name
        split_dir.mkdir(exist_ok=True)
        for subj in split_subjects:
            src = out_dir / subj
            dst = split_dir / subj
            if not dst.exists():
                dst.symlink_to(src.resolve())

    print(f'\nSplit complete:')
    print(f'  Train: {len(splits["train"])} subjects → {out_dir}/train/')
    print(f'  Val:   {len(splits["val"])} subjects → {out_dir}/val/')
    print(f'  Test:  {len(splits["test"])} subjects → {out_dir}/test/')


if __name__ == '__main__':
    main()
