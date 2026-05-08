# T2T-Bridge: Direct Diffusion Bridge for Thick-to-Thin Slice Infant MRI Reconstruction

> ISBI 2025 · BID Lab, ShanghaiTech University  
> Zihao Zhu, Gaofeng Wu, Haowen Deng, Jungang Liu, Han Zhang

## Overview

Reconstruct high-resolution 3D thin-slice (0.8mm isotropic) infant brain MRI from clinically available 2D thick-slice (0.4×0.4×5.2mm) acquisitions. Uses a **Diffusion Bridge** model that directly connects thick-slice (degraded) and thin-slice (clean) distributions, avoiding the randomness of conventional diffusion models.

## Architecture

```
Training:
  Thick-slice x₀ ──→ Diffusion Bridge Module ──→ Predict Residual Δx
       │                    │
       │    Timestep t + Age CFG Embedding
       │                    │
       └── Segment ──→ Tissue Map ──→ Structure Consistency Loss

Inference (15-step):
  Thick-slice → Noise init → 15 Bridge steps → Thin-slice output
```

## Key Technical Components

### 1. Diffusion Bridge (I2SB)
- Directly bridges P_thick → P_thin (not Gaussian → target)
- Nonlinear diffusion process with analytical marginals
- β_t: 1e-4 → 3e-4 (linear schedule, 1000 training steps)

### 2. Residual Prediction
- Network predicts **Δx = x₁ − x_t** rather than x₁ directly
- More stable training, especially with high structural similarity between thick and thin
- `residual = denoiser(x_t, t_emb, x0_cond)`

### 3. Classifier-Free Age Guidance (CFG)
- Age embedding (sinusoidal encoding) injected alongside timestep
- Handles massive tissue contrast variability across 0-72 months
- During inference: `pred = (1+w) * pred_age — w * pred_uncond`

### 4. Structure Consistency Loss
- Tissue segmentation head outputs GM/WM/CSF probability maps
- Dice loss between predicted and GT tissue maps
- Enforces anatomical plausibility of generated images

## Comparison Results

| Method | SSIM | PSNR (dB) |
|--------|------|-----------|
| p2p GAN | 0.9309 | 28.00 |
| **T2T-Bridge (ours)** | **0.9573** | **33.38** |

Clinical validation: lesion-preserving reconstruction for 0-month-old neonate.

---

## Experimental Plan

### Phase 1: Data Pipeline (Week 1-2)

**Data Source**: dHCP (783 neonates, 886 scans, T1w 0.8mm isotropic)

**Preprocessing Pipeline** (matching CBCP pipeline):

1. `resample.py` — Resample to 0.4×0.4×0.8 mm in-plane, keep through-plane
2. `crop.py` — Remove neck region via intensity thresholding
3. `register.py` — Rigid + affine registration to MNI infant template
4. `skull_strip.py` — SynthStrip / ROBEX brain extraction
5. `simulate_thick.py` — Fourier low-pass filter in z-axis → downsample to 5.2mm
6. `segmentation.py` — FreeSurfer infantFS for tissue maps (GM/WM/CSF)

**Output**:
- `train/`: paired thick-slice + thin-slice + tissue_map
- `val/`: held-out subjects (stratified by age)
- `test/`: with clinical annotations if available

### Phase 2: Model Implementation (Week 3-5)

**Files to implement**:

| File | Description |
|------|-------------|
| `models/unet3d.py` | 3D U-Net backbone (down ×4, up ×4, skip connections) |
| `models/diffusion.py` | β schedule, forward process, bridge sampling (`q_sample`) |
| `models/t2t_bridge.py` | Main I2SB wrapper: training_step, validation_step |
| `models/age_cfg.py` | Sinusoidal age embedding + CFG inference |
| `models/seg_head.py` | Lightweight Conv3D decoder for tissue segmentation |
| `models/losses.py` | Diffusion MSE loss + Dice loss for tissue consistency |
| `train.py` | Training loop with gradient accumulation, EMA, checkpointing |
| `sample.py` | Inference: 15-step / 1-step bridge sampling |
| `evaluate.py` | SSIM, PSNR, LPIPS, Dice per tissue, inference time |

**Training Config** (`configs/default.yaml`):

```yaml
model:
  dim: 64                    # Base channel dimension
  dim_mults: [1, 2, 4, 8]   # Channel multipliers per level
  resnet_groups: 8           # GroupNorm groups
  image_size: [128, 128, 32] # H×W×D patches

diffusion:
  timesteps: 1000            # Training steps
  beta_start: 1e-4
  beta_end: 3e-4
  schedule: "linear"

training:
  batch_size: 16
  lr: 1e-4
  weight_decay: 1e-6
  ema_decay: 0.9999
  lambda_structure: 0.1      # Structure loss weight
  max_epochs: 200

inference:
  nfe: 15                    # Number of function evaluations
  cfg_scale: 1.5             # CFG guidance strength
```

### Phase 3: Training & Evaluation (Week 6-8)

**Run training**:
```bash
python project1_ttsbridge/train.py \
    --config configs/default.yaml \
    --data-dir data/dhcp/processed/ \
    --name t2t_bridge_v1 \
    --gpus 2
```

**Run inference**:
```bash
python project1_ttsbridge/sample.py \
    --ckpt checkpoints/t2t_bridge_v1/last.pt \
    --nfe 15 \
    --cfg-scale 1.5
```

**Evaluation metrics**:

| Category | Metric | Tool |
|----------|--------|------|
| Image Quality | PSNR, SSIM, LPIPS | torchmetrics / lpips |
| Tissue Segmentation | Dice_GM, Dice_WM, Dice_CSF | nibabel + numpy |
| Cortical Accuracy | Thickness error, surface distance | FreeSurfer |
| Efficiency | Inference time, GPU memory | torch.cuda |

### Phase 4: Ablation Study (Week 8)

| Ablation | Baseline | What to test |
|----------|----------|--------------|
| Age CFG | w/o age embedding | Does age conditioning help 0mo vs 72mo? |
| Structure Loss | w/o Dice loss | Does anatomical constraint matter? |
| Residual vs Direct | predict x₁ | Is residual prediction better? |
| NFE sweep | {1, 5, 10, 15, 20} | Optimal speed/quality tradeoff |
| Age generalization | train on 0-24mo | Does model generalize to 48-72mo? |

### Phase 5: Clinical Validation (Bonus)

- Test on real thick-slice data from uMR890 scanner
- Radiologist evaluation: lesion detectability, diagnostic confidence
- Compare with hospital's existing reconstruction pipeline

---

## References

1. Liu et al., "I2SB: Image-to-Image Schrodinger Bridge", ICML 2023. [arXiv:2302.05872](https://arxiv.org/abs/2302.05872) | [Code: NVlabs/I2SB](https://github.com/NVlabs/I2SB)
2. Zhu et al., "T2T-Bridge: Direct Diffusion Bridge Model for Thick-to-Thin Slice Infant MRI Reconstruction", ISBI 2025.
3. Guo et al., "Cas-DiffCom: Cascaded diffusion model for infant longitudinal super-resolution 3D medical image completion", arXiv 2024. [arXiv:2402.13776](https://arxiv.org/abs/2402.13776)
4. Wang et al., "Guided MRI Reconstruction via Schrodinger Bridge", arXiv 2024.

---

## Data

- **dHCP**: 3rd data release — 783 neonatal subjects, 886 datasets. [Access](https://biomedia.github.io/dHCP-release-notes/)
- **CBCP**: Chinese Baby Connectome Project — 362 infants (0-72 months). Internal access only.
- **BCP**: Baby Connectome Project — NIMH Data Archive. [Apply](https://nda.nih.gov/)

## Quick Start

```bash
# Install dependencies
pip install torch torchvision nibabel python-pptx lpips monai

# Download dHCP data (manual)
# → https://biomedia.github.io/dHCP-release-notes/

# Preprocess
python project1_ttsbridge/scripts/preprocess_dhcp.py --data-dir data/dhcp/raw

# Train
python project1_ttsbridge/train.py --name t2t_exp1

# Evaluate
python project1_ttsbridge/evaluate.py --ckpt checkpoints/t2t_exp1/last.pt
```
