# Multimodal PET/MR Diagnosis with Variational Inference

> ICASSP 2025 · School of Biomedical Engineering, ShanghaiTech University  
> Yitian Tao, L. Wang, Q. Yang, Y. Liang, S. Liu et al.

## Overview

Develop a clinically feasible AI-based disease diagnosis model trained on **multimodal simultaneous functional PET/MR** data, with the key capability of **allowing single modality input during inference** while maintaining multimodal-level diagnostic accuracy.

**Core Challenge**: In clinical practice, not all modalities are available for every patient. The model must be robust to missing modalities at inference time.

## Core Idea

```
Training (multimodal):            Inference (single modality):
  FDG-PET ─┐                        FDG-PET ──→ Encoder_FDG ──→ z
  TAU-PET ─┼─→ Variational Fusion ──→ z ──→  Classifier (AD/MCI/NC)
  MRI     ─┘   (MoPoE VAE)           ├─→ Decoder_TAU (synthetic)
            └─→ Diagnosis Head       └─→ Decoder_MRI (synthetic)
```

The latent space `z` captures **shared modality-invariant features** — so even with only one modality input, the diagnosis head can leverage patterns learned from all modalities during training.

## Architecture

```
┌───────────────────────────────────────────────────────────┐
│  Modality-specific Encoders                                │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                 │
│  │ MRI Enc  │  │ FDG Enc  │  │ TAU Enc  │                 │
│  │ (3D CNN) │  │ (3D CNN) │  │ (3D CNN) │                 │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘                 │
│       │             │             │                        │
│       │   μ₁,σ₁     │   μ₂,σ₂     │   μ₃,σ₃               │
│       └──────┬──────┴──────┬──────┘                        │
│              │             │                               │
│         MoPoE Fusion: z ~ q(z|all available modalities)    │
│              │                                             │
│     ┌────────┼────────┐                                    │
│     │        │        │                                    │
│  AD/MCI/NC  Dec_MRI  Dec_FDG  Dec_TAU                      │
│  Classifier (synthesis for missing modalities)              │
└───────────────────────────────────────────────────────────┘
```

## Key Technical Components

### 1. MoPoE-VAE (Mixture-of-Products-of-Experts)

Fusion strategy for multiple modalities. Given modality-specific encoders that each produce (μᵢ, σᵢ):

```
q(z|X₁,...,Xₖ) ∝ Σ_{S⊆{1,...,k}} π_S · Π_{i∈S} q_i(z|X_i)

where π_S = 1/2ᵏ (uniform over all subsets)
```

This allows training with **arbitrary modality subsets** — the model learns to infer z from any combination of available inputs.

### 2. Missing Modality Training

During training, randomly drop modalities with probability p_drop (e.g., 0.3):
- Drop TAU-PET → model learns FDG→TAU relationship
- Drop FDG-PET → model learns TAU→FDG relationship
- Drop both PET → model learns MRI-only diagnosis
- Keep all → full multimodal performance (upper bound)

### 3. Cross-Modal Synthesis as Auxiliary Task

For missing modality dropout during training, the decoder for the dropped modality serves as cross-modal synthesis. Losses:

| Loss | Formula | Purpose |
|------|---------|---------|
| Reconstruction | L₂(xᵢ, Decᵢ(z)) | Fidelity for available modalities |
| Cross-modal | L₂(xⱼ, Decⱼ(z)) | Learn inter-modality mapping |
| KL | KL(q(z\|X)‖p(z)) | Regularize latent space |
| Classification | CE(y_pred, y_true) | Diagnosis accuracy |
| Contrastive | InfoNCE(z_i, z_j) | Align latent from different modality subsets |

### 4. Diagnosis Head

Multi-layer perceptron on z with:
- 3-class output: Normal (NC), Mild Cognitive Impairment (MCI), Alzheimer's Disease (AD)
- Focal loss to handle class imbalance
- MC-dropout for uncertainty estimation

---

## Experimental Plan

### Phase 1: Data Preparation (Week 1-2)

**Data Source**: ADNI (Alzheimer's Disease Neuroimaging Initiative)

**Required modalities per subject**:
| Modality | Purpose | Availability |
|----------|---------|--------------|
| T1w MRI | Structural brain anatomy | ~2000 subjects |
| FDG-PET | Glucose metabolism | ~1500 subjects |
| AV45/Amy-PET | Amyloid-β plaque | ~1200 subjects |
| AV1451/Tau-PET | Tau tangles | ~800 subjects |
| Clinical scores | MMSE, CDR, ADAS-Cog | ~1800 subjects |

**Preprocessing** (`scripts/preprocess_adni.py`):

1. DICOM→NIfTI conversion
2. PET-MRI co-registration (SPM/FSL flirt)
3. MNI152 spatial normalization (ANTs SyN)
4. Intensity normalization: PET → SUVR (cerebellar GM ref), MRI → z-score
5. Brain extraction (SynthStrip)
6. Patch extraction: 3D patches centered on hippocampus, PCC, temporal lobe
7. Train/val/test split: 70/10/20, stratified by diagnosis and age

**Output**:
```
data/adni/processed/
├── train/
│   ├── sub-001/
│   │   ├── t1w.nii.gz          # 1mm MNI space
│   │   ├── fdg_pet.nii.gz      # SUVR normalized
│   │   ├── tau_pet.nii.gz      # SUVR normalized
│   │   ├── clinical.csv         # MMSE, CDR, diagnosis
│   │   └── patches/
│   └── ...
├── val/
└── test/
```

### Phase 2: Model Implementation (Week 3-5)

**Files to implement**:

| File | Description |
|------|-------------|
| `models/encoders.py` | 3D ResNet-18/50 modality-specific encoders |
| `models/mopoe_vae.py` | MoPoE fusion: q(z\|Xsubset) computation |
| `models/decoders.py` | Modality-specific 3D decoders for reconstruction |
| `models/classifier.py` | Diagnosis head (MLP on z) with uncertainty |
| `models/losses.py` | Reconstruction + KL + Cross-modal + Classification |
| `models/fusion.py` | Cross-attention fusion alternative |
| `data/adni_dataset.py` | Dataset class with modality dropout |
| `data/transforms.py` | Augmentation: random flip, rotation, intensity |
| `train.py` | Training loop with modality dropout scheduler |
| `evaluate.py` | Test metrics across all modality subsets |

**Training Config** (`configs/default.yaml`):

```yaml
model:
  encoder: "resnet18_3d"     # 3D ResNet backbone
  latent_dim: 256            # Latent space dimension
  beta_kl: 0.001             # KL divergence weight (β-VAE)
  modalities: ["t1w", "fdg_pet", "tau_pet"]

training:
  batch_size: 32
  lr: 1e-4
  weight_decay: 1e-5
  max_epochs: 150
  warmup_epochs: 5

  modality_dropout:
    p_drop: 0.3              # Probability of dropping a modality
    schedule: "constant"     # or "increasing" (cosine ramp)

diagnosis:
  num_classes: 3             # NC, MCI, AD
  focal_gamma: 2.0

cross_modal:
  lambda_recon: 1.0
  lambda_cross: 0.5           # Cross-modal synthesis weight
  lambda_contrastive: 0.1
```

### Phase 3: Training & Evaluation (Week 6-8)

**Run training**:
```bash
python project2_multimodal/train.py \
    --config configs/default.yaml \
    --data-dir data/adni/processed/ \
    --name multimodal_vae_v1 \
    --gpus 2
```

**Evaluation protocol**: Test under ALL modality availability scenarios:

| During Inference | Available Modalities | Expected Performance |
|------------------|---------------------|---------------------|
| Full multimodal | T1w + FDG + TAU | Upper bound (best) |
| Only MRI | T1w | Lowest (target: close to full) |
| Only FDG-PET | FDG | Target: △ < 5% from full |
| Only TAU-PET | TAU | Target: △ < 5% from full |
| Any 2 of 3 | MRI+FDG, MRI+TAU, FDG+TAU | Near full |

**Metrics**:

| Category | Metric |
|----------|--------|
| Diagnosis | Accuracy, AUROC, F1 (per class), Sensitivity/Specificity |
| Image Synthesis | SSIM, PSNR, NMSE for synthetic TAU/FDG |
| Latent Quality | Silhouette score, cluster separability |
| Uncertainty | Expected Calibration Error (ECE) |

### Phase 4: Ablation Study (Week 8)

| Ablation | Research Question |
|----------|-------------------|
| Single VAE vs MoPoE | Is product-of-experts necessary? |
| w/o Cross-Modal Loss | Does cross-modal synthesis help diagnosis? |
| w/o Contrastive Loss | Does latent alignment matter? |
| p_drop ∈ {0.1, 0.3, 0.5, 0.7} | Optimal dropout rate for missing modality robustness |
| w/o Focal Loss | Effect on class-imbalanced AD dataset |
| CNN vs Transformer encoder | Architecture choice |
| latent_dim ∈ {64, 128, 256, 512} | Optimal latent capacity |

### Phase 5: Three-Network Fusion (Advanced, Week 9-10)

If uPMR 790 PET/MR synchronized data is available, extend to the full ICASSP 2025 pipeline:

```
BOLD-fMRI  → Hemodynamic Network   ─┐
FDG-PET    → Metabolic Network      ─┼→ Deeply Integrated Fusion → Diagnosis
ASL        → Perfusion Network      ─┘
```

- Each network is a modality-specific 3D CNN
- Fusion via cross-attention across the three latent representations
- Joint optimization for disease diagnosis using all three physiological perspectives

---

## References

1. Tao Y., Wang L. et al., "Revolutionizing Disease Diagnosis with simultaneous functional PET/MR and Deeply Integrated Brain Metabolic, Hemodynamic, and Perfusion Networks", ICASSP 2025.
2. Lee J. et al., "Synthesizing images of tau pathology from cross-modal neuroimaging using deep learning", Brain, 2024. [DOI](https://doi.org/10.1093/brain/awad346)
3. "Joint learning framework of cross-modal synthesis and diagnosis for Alzheimer's disease by mining underlying shared modality information", Medical Image Analysis, 2024.
4. "MC-RVAE: Multi-channel recurrent variational autoencoder for multimodal Alzheimer's disease progression modelling", NeuroImage, 2023.
5. Han K. et al., "Incomplete multi-modal disentanglement learning with application to Alzheimer's disease diagnosis", IEEE TMI, 2025.
6. Hamghalam M. et al., "Modality completion via Gaussian process prior variational autoencoders for multi-modal glioma segmentation", MICCAI 2021.
7. Kumar S. et al., "Normative Modeling using Multimodal Variational Autoencoders to Identify Abnormal Brain Structural Patterns in Alzheimer Disease", SPIE 2023.
8. Kumar S. et al., "Multimodal normative modeling in Alzheimer's Disease with introspective variational autoencoders", arXiv 2026.
9. Sikka A. et al., "MRI to PET Cross-Modality Translation using Globally and Locally Aware GAN (GLA-GAN)", arXiv 2021.

---

## Data

- **ADNI**: Alzheimer's Disease Neuroimaging Initiative. [Apply](https://adni.loni.usc.edu/)
- **OASIS-3**: Open Access Series of Imaging Studies. [Access](https://www.oasis-brains.org/)
- **UK Biobank**: Large-scale multimodal imaging + genetics. [Apply](https://www.ukbiobank.ac.uk/)

## Quick Start

```bash
# Install dependencies
pip install torch torchvision nibabel nilearn monai pandas scikit-learn

# Request ADNI data access → https://adni.loni.usc.edu/

# Preprocess
python project2_multimodal/scripts/preprocess_adni.py --data-dir data/adni/raw

# Train
python project2_multimodal/train.py --name multimodal_exp1

# Evaluate (tests all modality subsets)
python project2_multimodal/evaluate.py --ckpt checkpoints/multimodal_exp1/last.pt --test-all-subsets
```
