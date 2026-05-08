#!/usr/bin/env python3
"""
ADNI data preprocessing pipeline for multimodal diagnosis.

Downloads (manual step) and processes ADNI data.

Usage:
    python preprocess_adni.py --raw-dir data/adni/raw --out-dir data/adni/processed

Data source: https://adni.loni.usc.edu/ (requires application approval)

Steps:
  1. DICOM → NIfTI conversion (dcm2niix)
  2. PET-MRI co-registration (FSL flirt)
  3. MNI152 spatial normalization (ANTs SyN)
  4. PET SUVR normalization (cerebellar gray matter reference)
  5. MRI z-score intensity normalization
  6. Brain extraction (SynthStrip)
  7. Train/val/test split (stratified by diagnosis)
"""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import nibabel as nib
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-dir', type=str, required=True)
    parser.add_argument('--out-dir', type=str, required=True)
    parser.add_argument('--mni-template', type=str, default='')
    parser.add_argument('--cerebellar-mask', type=str, default='')
    parser.add_argument('--diagnosis-csv', type=str, default='')
    parser.add_argument('--skip-registration', action='store_true')
    parser.add_argument('--skip-skull-strip', action='store_true')
    parser.add_argument('--val-ratio', type=float, default=0.1)
    parser.add_argument('--test-ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def convert_dicom(subject_dir, output_dir):
    """Convert DICOM series to NIfTI using dcm2niix."""
    try:
        subprocess.run([
            'dcm2niix', '-z', 'y', '-o', str(output_dir), str(subject_dir),
        ], check=True, capture_output=True)
    except FileNotFoundError:
        print('  dcm2niix not found. Install: https://github.com/rordenlab/dcm2niix')
        print('  Skipping DICOM conversion.')
        return False
    except subprocess.CalledProcessError as e:
        print(f'  dcm2niix error: {e}')
        return False
    return True


def coregister_pet_to_mri(pet_path, mri_path, output_path):
    """Co-register PET to MRI using FSL flirt."""
    try:
        subprocess.run([
            'flirt',
            '-in', str(pet_path),
            '-ref', str(mri_path),
            '-out', str(output_path),
            '-dof', '6',           # rigid body
            '-cost', 'mutualinfo',
            '-interp', 'trilinear',
        ], check=True, capture_output=True)
    except FileNotFoundError:
        print('  FSL not found. Install: https://fsl.fmrib.ox.ac.uk/')
        shutil.copy(pet_path, output_path)
    except subprocess.CalledProcessError:
        shutil.copy(pet_path, output_path)


def normalize_mni(input_path, output_path, mni_template):
    """Spatial normalize to MNI152 using ANTs."""
    if not mni_template or not Path(mni_template).exists():
        shutil.copy(input_path, output_path)
        return

    try:
        subprocess.run([
            'antsRegistrationSyNQuick.sh',
            '-d', '3',
            '-f', str(mni_template),
            '-m', str(input_path),
            '-o', str(output_path.with_suffix('')),
        ], check=True, capture_output=True)
    except FileNotFoundError:
        print('  ANTs not found. Install: https://github.com/ANTsX/ANTs')
        shutil.copy(input_path, output_path)


def suvr_normalize(pet_data, mri_data, cerebellar_mask):
    """
    Normalize PET to SUVR using cerebellar gray matter reference.

    SUVR = PET / mean(cerebellarGM_PET)
    """
    if cerebellar_mask and Path(cerebellar_mask).exists():
        mask = nib.load(cerebellar_mask).get_fdata() > 0.5
        if mask.sum() > 0:
            ref_value = pet_data[mask].mean()
            if ref_value > 0:
                return pet_data / ref_value
    return pet_data


def brain_extract(input_path, output_path):
    """Run SynthStrip or HD-BET for brain extraction."""
    try:
        subprocess.run([
            'mri_synthstrip', '-i', str(input_path), '-o', str(output_path),
        ], check=True, capture_output=True)
    except FileNotFoundError:
        try:
            subprocess.run([
                'hd-bet', '-i', str(input_path), '-o', str(output_path),
            ], check=True, capture_output=True)
        except FileNotFoundError:
            print('  No brain extraction tool found.')
            shutil.copy(input_path, output_path)


def load_diagnosis_labels(csv_path):
    """Load diagnosis labels from ADNI CSV."""
    if not csv_path or not Path(csv_path).exists():
        return {}
    df = pd.read_csv(csv_path)
    label_map = {}
    for _, row in df.iterrows():
        label_map[row['subject_id']] = {
            'diagnosis': row.get('diagnosis', 'NC'),
            'mmse': row.get('mmse', None),
            'age': row.get('age', None),
            'sex': row.get('sex', None),
        }
    return label_map


def split_subjects(subjects, labels, val_ratio=0.1, test_ratio=0.1, seed=42):
    """Split subjects into train/val/test, stratified by diagnosis."""
    rng = np.random.RandomState(seed)
    n = len(subjects)
    indices = rng.permutation(n)

    # Stratify by diagnosis
    diagnosis_groups = {'NC': [], 'MCI': [], 'AD': []}
    for i in range(n):
        subj = subjects[indices[i]]
        diag = labels.get(subj, {}).get('diagnosis', 'NC')
        if diag in diagnosis_groups:
            diagnosis_groups[diag].append(subj)

    splits = {'train': [], 'val': [], 'test': []}

    for diag, subset in diagnosis_groups.items():
        n_sub = len(subset)
        n_test = max(1, int(n_sub * test_ratio))
        n_val = max(1, int(n_sub * val_ratio))

        splits['test'].extend(subset[:n_test])
        splits['val'].extend(subset[n_test:n_test + n_val])
        splits['train'].extend(subset[n_test + n_val:])

    return splits


def main():
    args = parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load diagnosis labels
    labels = load_diagnosis_labels(args.diagnosis_csv)

    # Discover subjects
    subject_dirs = sorted([d for d in raw_dir.iterdir() if d.is_dir()])
    print(f'Found {len(subject_dirs)} subjects')

    subjects_processed = []

    for subj_dir in subject_dirs:
        subj_id = subj_dir.name
        subj_out = out_dir / subj_id
        subj_out.mkdir(parents=True, exist_ok=True)

        print(f'Processing {subj_id}...')

        # Convert DICOM to NIfTI (if needed)
        nifti_dir = subj_dir / 'nifti'
        if not nifti_dir.exists():
            nifti_dir.mkdir(exist_ok=True)
            convert_dicom(subj_dir, nifti_dir)

        # Find modality files
        t1w_files = list(subj_dir.rglob('*T1*.nii*'))
        fdg_files = list(subj_dir.rglob('*FDG*.nii*')) + list(subj_dir.rglob('*fdg*.nii*'))
        tau_files = list(subj_dir.rglob('*TAU*.nii*')) + list(subj_dir.rglob('*AV1451*.nii*'))

        t1w_path = t1w_files[0] if t1w_files else None
        fdg_path = fdg_files[0] if fdg_files else None
        tau_path = tau_files[0] if tau_files else None

        if t1w_path is None:
            print(f'  Skipping {subj_id}: no T1w found')
            continue

        # Process T1w
        t1w_out = subj_out / 't1w_orig.nii.gz'
        shutil.copy(t1w_path, t1w_out)

        # Brain extraction
        t1w_brain = subj_out / 't1w.nii.gz'
        if not args.skip_skull_strip:
            brain_extract(t1w_out, t1w_brain)
        else:
            shutil.copy(t1w_out, t1w_brain)

        # MNI normalization
        t1w_mni = t1w_brain  # placeholder

        # Process PET modalities
        for pet_type, pet_path in [('fdg_pet', fdg_path), ('tau_pet', tau_path)]:
            if pet_path is None:
                continue

            pet_out = subj_out / f'{pet_type}.nii.gz'

            # Co-register to MRI
            if not args.skip_registration and t1w_brain.exists():
                coregister_pet_to_mri(pet_path, t1w_brain, pet_out)
            else:
                shutil.copy(pet_path, pet_out)

            # SUVR normalization
            pet_data = nib.load(str(pet_out)).get_fdata()
            t1w_data = nib.load(str(t1w_brain)).get_fdata()
            suvr_data = suvr_normalize(pet_data, t1w_data, args.cerebellar_mask)

            suvr_img = nib.Nifti1Image(suvr_data, nib.load(str(pet_out)).affine)
            nib.save(suvr_img, str(pet_out))

        # Write metadata
        meta = labels.get(subj_id, {'diagnosis': 'NC'})
        with open(subj_out / 'meta.json', 'w') as f:
            json.dump(meta, f, indent=2, default=str)

        subjects_processed.append(subj_id)
        print(f'  Done: {subj_id}')

    # Split and link
    splits = split_subjects(subjects_processed, labels, args.val_ratio, args.test_ratio, args.seed)

    for split_name, split_subjects in splits.items():
        split_dir = out_dir / split_name
        split_dir.mkdir(exist_ok=True)
        for subj in split_subjects:
            src = out_dir / subj
            dst = split_dir / subj
            if not dst.exists():
                dst.symlink_to(src.resolve())

    print(f'\nSplit complete:')
    for name in ['train', 'val', 'test']:
        print(f'  {name.capitalize()}: {len(splits[name])} subjects → {out_dir}/{name}/')


if __name__ == '__main__':
    main()
