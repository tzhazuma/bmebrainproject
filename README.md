# Brain MRI Research Projects

Two research projects in the BID Lab, School of Biomedical Engineering, ShanghaiTech University.

---

## Project 1: T2T-Bridge — Thick-to-Thin Slice Infant MRI Reconstruction

> ISBI 2025 | Zihao Zhu, Gaofeng Wu, Haowen Deng, Jungang Liu, Han Zhang

Direct Diffusion Bridge Model for reconstructing 3D thin-slice MRI from 2D thick-slice acquisitions.

- **Method**: Image-to-Image Schrodinger Bridge (I2SB) + Classifier-Free Age Guidance + Structure Consistency
- **Data**: Chinese Baby Connectome Project (CBCP), 362 infants aged 0-72 months
- **Key results**: SSIM 0.9573, PSNR 33.38 dB, 15-step inference
- **Clinical deployment**: uMR890 3.0T at Children's Hospital of Fudan University, Xiamen

[→ Project details](project1_ttsbridge/)

---

## Project 2: Multimodal PET/MR Diagnosis with Variational Inference

> ICASSP 2025 | Yitian Tao et al.

Cross-modal variational image synthesis and disease diagnosis using simultaneous PET/MR data.

- **Method**: Variational Inference with MoPoE fusion + Cross-modal synthesis + Missing-modality robust diagnosis
- **Data**: ADNI (MRI, FDG-PET, TAU-PET), uPMR 790 PET/MR scanner
- **Key goal**: Train on multimodal data, infer with single modality, maintain multimodal accuracy
- **Target**: Alzheimer's Disease diagnosis (AD/MCI/NC)

[→ Project details](project2_multimodal/)

---

## Directory Structure

```
bmebrainproject/
├── README.md
├── .gitignore
├── docs/                           # Papers, reports, plans
│   ├── isbi25_T2TBridge_v4.pptx
│   ├── T2T条件生成脑科学课题.pptx
│   └── Yitian课题介绍.pptx
├── project1_ttsbridge/             # T2T-Bridge
│   ├── README.md                   # Detailed implementation plan
│   ├── configs/
│   ├── data/
│   ├── models/
│   └── scripts/
└── project2_multimodal/            # Multimodal Diagnosis
    ├── README.md                   # Detailed implementation plan
    ├── configs/
    ├── data/
    ├── models/
    └── scripts/
```

---

## Key References

| Project | Paper | Venue |
|---------|-------|-------|
| P1 | I2SB: Image-to-Image Schrodinger Bridge (Liu et al.) | ICML 2023 |
| P1 | T2T-Bridge (Zhu et al.) | ISBI 2025 |
| P1 | Cas-DiffCom (Guo et al.) | arXiv 2024 |
| P2 | Revolutionizing Disease Diagnosis with simultaneous PET/MR (Tao et al.) | ICASSP 2025 |
| P2 | Synthesizing images of tau pathology (Lee et al.) | Brain 2024 |
| P2 | Joint learning of cross-modal synthesis and diagnosis | MedIA 2024 |
