# Dataset Guide

This document describes all datasets used in the Brain MRI Research Projects.

---

## Project 1: T2T-Bridge (Infant MRI Super-Resolution)

### 1. dHCP (Developing Human Connectome Project)

- **Description**: Neonatal brain MRI data from the ERC-funded dHCP consortium (King's College London, Imperial College London, University of Oxford)
- **Subjects**: 783 neonatal subjects, 886 datasets
- **Modalities**: T1w, T2w structural MRI, DWI (diffusion), rs-fMRI (resting-state)
- **Resolution**: 0.8mm isotropic T1w/T2w
- **Metadata**: Sex, age at birth, age at scan, birthweight, head circumference, radiology score
- **Download**: https://biomedia.github.io/dHCP-release-notes/
- **License**: Open access, requires application
- **Size**: ~500GB total

**Usage in project**: Thin-slice (0.8mm isotropic) ground truth. We simulate thick-slice (5.2mm) through-plane from this.

### 2. CBCP (Chinese Baby Connectome Project)

- **Description**: 10-institute collaboration in China, sponsored by STI-Major Projects
- **Subjects**: 362 infants aged 0-72 months
- **Modalities**: T1w (thick 0.4×0.4×5.2mm + thin 0.8×0.8×0.8mm), behavioral scores, EEG, fNIRS
- **Download**: Internal access only (contact: zhuzh2023@shanghaitech.edu.cn)
- **Usage in project**: Primary training data for T2T-Bridge (ISBI 2025 paper)

### 3. BCP (Baby Connectome Project)

- **Description**: NIH-funded longitudinal infant brain imaging study
- **Subjects**: 0-5 year olds, longitudinal scans
- **Download**: https://nda.nih.gov/ (requires NIMH Data Archive application)
- **Usage in project**: Supplementary training/testing data

---

## Project 2: Multimodal PET/MR Diagnosis

### 1. ADNI (Alzheimer's Disease Neuroimaging Initiative)

- **Description**: Longitudinal multicenter study for AD biomarkers
- **Subjects**: ~2000 subjects (CN, MCI, AD)
- **Modalities**:

| Modality | Approx. Subjects | Purpose |
|----------|-----------------|---------|
| T1w MRI | ~2000 | Brain structure, atrophy |
| FDG-PET | ~1500 | Glucose metabolism |
| Amyloid-PET (AV45) | ~1200 | Aβ plaque |
| Tau-PET (AV1451) | ~800 | Tau tangles |
| Clinical scores | ~1800 | MMSE, CDR, ADAS-Cog |

- **Download**: https://adni.loni.usc.edu/ (free application, ~2 weeks review)
- **Stages**:
  - ADNI-1: Baseline CN, MCI, AD (1.5T)
  - ADNI-GO: Additional MCI subjects
  - ADNI-2: 3T MRI, Amyloid-PET added
  - ADNI-3: Tau-PET added
- **License**: Research only, requires data use agreement
- **Size**: ~5TB total

**Usage in project**: Primary training data for cross-modal synthesis and AD diagnosis.

### 2. OASIS-3 (Open Access Series of Imaging Studies)

- **Description**: Longitudinal neuroimaging, clinical, cognitive dataset
- **Subjects**: ~1100 subjects (CN + AD)
- **Modalities**: T1w, T2w, FLAIR, PET
- **Download**: https://www.oasis-brains.org/ (free access)
- **License**: Open access
- **Size**: ~100GB

**Usage in project**: Supplementary data, particularly for MRI analysis.

### 3. UK Biobank

- **Description**: Large-scale population imaging + genetics
- **Subjects**: ~500,000 (imaging subset: ~100,000)
- **Modalities**: T1w, T2w, fMRI, DTI, SWI
- **Download**: https://www.ukbiobank.ac.uk/ (application required)
- **Size**: Imaging subset ~200TB

**Usage in project**: Large-scale pretraining of MRI encoder.

---

## Quick Download Commands

### dHCP (Project 1)

```bash
# Visit https://biomedia.github.io/dHCP-release-notes/
# Download the "Third Data Release" package via Globus/Xnat
# Extract to:
mkdir -p data/dhcp/raw
```

### ADNI (Project 2)

```bash
# 1. Apply at https://adni.loni.usc.edu/
# 2. After approval, use the IDA search tool:
#    - Modality: MRI → T1-weighted MPRAGE (ADNI-2/3)
#    - Modality: PET → FDG / AV45 / AV1451
# 3. Download via LONI IDA download manager
# 4. Extract to:
mkdir -p data/adni/raw
```

### OASIS-3 (Project 2)

```bash
# Direct download (no application needed):
wget https://www.oasis-brains.org/files/oasis_cross-sectional_disc1.tar.gz
mkdir -p data/oasis3 && tar -xzf oasis_cross-sectional_disc1.tar.gz -C data/oasis3
```

---

## Preprocessing

See project-specific preprocessing scripts:

| Project | Script |
|---------|--------|
| T2T-Bridge | `project1_ttsbridge/scripts/preprocess_dhcp.py` |
| Multimodal | `project2_multimodal/scripts/preprocess_adni.py` |

### Required External Tools

| Tool | Purpose | Install |
|------|---------|---------|
| dcm2niix | DICOM → NIfTI | `apt install dcm2niix` or https://github.com/rordenlab/dcm2niix |
| FSL | Registration (flirt) | https://fsl.fmrib.ox.ac.uk/ |
| ANTs | Nonlinear registration (SyN) | `apt install ants` or https://github.com/ANTsX/ANTs |
| SynthStrip | Brain extraction | `pip install synthstrip` |
| FreeSurfer | Cortical parcellation | https://surfer.nmr.mgh.harvard.edu/ |
| SimpleITK | NIfTI resampling | `pip install SimpleITK` |

### Minimal Setup (Python-only)

For quick experimentation without external tools:
```bash
pip install nibabel numpy scipy
python project1_ttsbridge/scripts/preprocess_dhcp.py --skip-skull-strip --skip-segmentation
```
