# Research Report Template

---

# [Project Title]

**[Conference/Journal Name], [Year]**

**[Author Names]**
[Lab Name], School of Biomedical Engineering, ShanghaiTech University

---

## Abstract

[One paragraph summarizing the problem, method, key results, and significance.]

---

## 1. Introduction

### 1.1 Background

[Describe the clinical/scientific context and motivation.]

### 1.2 Problem Statement

[Formal definition of the problem being solved.]

### 1.3 Contributions

1. [Contribution 1]
2. [Contribution 2]
3. [Contribution 3]

---

## 2. Related Work

### 2.1 [Area A]
| Paper | Method | Key Difference from Our Work |
|-------|--------|------------------------------|
| ... | ... | ... |

### 2.2 [Area B]
[Discussion and comparison table.]

---

## 3. Method

### 3.1 Overview

```
[Architecture diagram — ASCII art or reference to Figure 1]
```

### 3.2 [Component A]

[Mathematical formulation, equations, explanation.]

### 3.3 [Component B]

### 3.4 Loss Function

L_total = α₁·L₁ + α₂·L₂ + ...

### 3.5 Implementation Details

| Parameter | Value |
|-----------|-------|
| Batch size | ... |
| Learning rate | ... |
| Optimizer | AdamW (β₁=0.9, β₂=0.999) |
| Epochs | ... |
| GPU | ... |

---

## 4. Experiments

### 4.1 Datasets

| Dataset | Subjects | Modalities | Resolution | Source |
|---------|----------|------------|------------|--------|
| ... | ... | ... | ... | ... |

### 4.2 Evaluation Metrics

| Metric | Formula | Purpose |
|--------|---------|---------|
| PSNR | ... | Image fidelity |
| SSIM | ... | Perceptual similarity |
| Dice | ... | Segmentation accuracy |
| ... | ... | ... |

### 4.3 Baseline Comparison

| Method | PSNR | SSIM | ... |
|--------|------|------|-----|
| Baseline A | ... | ... | ... |
| Baseline B | ... | ... | ... |
| **Ours** | **...** | **...** | **...** |

### 4.4 Ablation Study

| Ablation | PSNR | SSIM | Analysis |
|----------|------|------|----------|
| Full model | ... | ... | Best |
| w/o component A | ... | ... | Degrades by X% |
| w/o component B | ... | ... | Degrades by Y% |

### 4.5 Qualitative Results

[Description of visual comparisons — reference to Figures.]

---

## 5. Discussion

### 5.1 Key Findings

[Interpretation of results, what they mean clinically/scientifically.]

### 5.2 Limitations

1. [Limitation 1 — e.g., limited age range, single site data]
2. [Limitation 2]

### 5.3 Future Work

1. [Extend to other age groups / modalities / diseases]
2. [Improve inference speed / model efficiency]
3. [Prospective clinical validation]

---

## 6. Conclusion

[2-3 sentence summary of what was done and why it matters.]

---

## References

1. [Author et al., "Title", Venue, Year.]

---

## Appendix

### A. Training Details

[Additional hyperparameters, data augmentation details.]

### B. Additional Results

[Per-group breakdown, failure case analysis.]

### C. Code Availability

[GitHub link, license information.]
