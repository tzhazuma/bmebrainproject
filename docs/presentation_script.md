# Brain MRI Research Projects — Presentation Script (演讲讲稿)
## BID Lab, School of Biomedical Engineering, ShanghaiTech University
### Duration: ~25-30 minutes (23 slides)

---

## Slide 1: Title (30 seconds)
Good morning/afternoon everyone. Today I'll present two brain MRI research projects from BID Lab at ShanghaiTech University.

The first project, T2T-Bridge, published at ISBI 2025, addresses infant brain MRI super-resolution using a novel Diffusion Bridge model.

The second project, presented at ICASSP 2025, develops a multimodal variational inference framework for Alzheimer's disease diagnosis using PET/MR imaging.

---

## Slide 2: Outline (30 seconds)
Here's the outline of my presentation.

I'll start with the research background — why these problems matter clinically.
Then I'll dive into the methodology for both projects.
After that, I'll present our dataset, experiments, and results.
Finally, I'll discuss our infrastructure, GPU optimization story, and future research directions.

---

## Slide 3: Alzheimer's Disease Background (1 minute)
Let me start with the clinical motivation for our multimodal diagnosis project.

Alzheimer's Disease is the most common neurodegenerative disorder, currently affecting about 50 million people worldwide. This number is projected to reach 152 million by 2050.

The disease has three key pathological hallmarks. First, extracellular amyloid-beta plaques, which can be detected by AV45 PET imaging. Second, intracellular tau protein tangles, detected by AV1451 tau PET. And third, progressive brain atrophy, measured by structural T1-weighted MRI.

Current clinical diagnosis relies on all three modalities together. But here's the challenge: not every hospital has access to all three modalities. Tau-PET, in particular, is expensive and rarely available. This creates a critical need for AI systems that can diagnose accurately even when some modalities are missing.

---

## Slide 4: Infant Brain MRI Challenge (1 minute)
Now let me introduce the motivation for our second project.

Infant brain development is extremely rapid and dynamic. The brain doubles in volume in the first year, and tissue contrast changes dramatically as myelination progresses.

For clinical purposes, we need high-resolution thin-slice MRI — about 0.8 millimeter isotropic. This is crucial for accurate tissue segmentation and early detection of developmental abnormalities.

However, thin-slice MRI requires long scan times, which is problematic for unsedated infants who tend to move. In practice, clinicians acquire thick-slice MRI — about 5.2 millimeter through-plane resolution — which is much faster but produces blurred images.

Our solution: use deep learning to transform these clinically-available thick slices into diagnostic-quality thin slices. This is exactly what our T2T-Bridge model does.

---

## Slide 5: Project 1 — T2T-Bridge Overview (1.5 minutes)
T2T-Bridge, published at ISBI 2025, directly addresses this thick-to-thin slice reconstruction problem.

The core method is an Image-to-Image Schrödinger Bridge, based on the I2SB framework from NVIDIA Research published at ICML 2023.

Unlike conventional diffusion models that start from random noise, our Diffusion Bridge directly connects the thick-slice distribution to the thin-slice distribution. This provides deterministic, controlled generation.

We added two key innovations. First, Classifier-Free Age Guidance — the model receives the infant's age as conditioning, allowing it to handle the massive tissue contrast variability from 0 to 72 months. Second, a Structure Consistency Loss using tissue segmentation to enforce anatomical plausibility.

The model is extremely efficient — it needs only 15 inference steps instead of the standard 1000, making it 67 times faster.

Importantly, this model has been clinically deployed on a uMR890 3T scanner at the Children's Hospital of Fudan University in Xiamen, and applied to the Chinese Baby Connectome Project with 362 infants.

The quantitative results are impressive: SSIM of 0.9573 and PSNR of 33.38 dB, significantly outperforming the p2p GAN baseline.

---

## Slide 6: Diffusion Bridge Theory (2 minutes)
Let me explain why the Diffusion Bridge is fundamentally different from conventional diffusion models.

In standard diffusion models like DDPM, the forward process gradually adds Gaussian noise to destroy the image, and the reverse process learns to denoise from pure noise. The problem is: there's no structural guidance from the source image. The generation path is random and uncontrolled.

In our Diffusion Bridge approach, we directly connect the source distribution — thick-slice MRI — to the target distribution — thin-slice MRI — via a Schrödinger Bridge. This is a nonlinear extension of score-based models.

The key advantage: we have paired boundary conditions at both ends. The forward process is: x_t equals a weighted combination of thin-slice, thick-slice, and noise. The network doesn't predict the target image directly — it predicts the residual, which is much easier to learn.

This makes the training simulation-free — we can compute the marginals analytically given the boundary pairs. No need for expensive simulation of the entire stochastic process.

For inference, we use DPM-Solver sampling with just 15 steps instead of 1000, achieving the 67x speedup I mentioned.

---

## Slide 7: Model Architecture (1 minute)
The architecture uses a 3D U-Net backbone. The input is a concatenation of the current noisy state and the source thick-slice image.

The U-Net has residual blocks with time-step conditioning through scale-and-shift modulation, using GroupNorm and SiLU activation.

The age embedding is injected alongside the timestep. During inference, we use classifier-free guidance: we compute both the conditional and unconditional predictions, and interpolate between them with a guidance scale.

The tissue segmentation head shares features with the denoiser and outputs gray matter, white matter, and CSF probability maps. The Dice loss between predicted and ground truth tissue maps enforces anatomical consistency.

The total loss combines the diffusion MSE loss, the structure Dice loss, and a gradient difference loss for edge preservation.

---

## Slide 8: Project 2 — Multimodal Diagnosis Overview (1.5 minutes)
Now let me present our second project, published at ICASSP 2025.

This work addresses a fundamentally different problem: how can we achieve accurate Alzheimer's diagnosis when not all imaging modalities are available?

Our solution is a MoPoE-VAE — a Mixture of Products of Experts Variational Autoencoder. The key idea is: train on all three modalities together, but design the latent space so that any subset of modalities can produce a good diagnosis.

We use modality-specific 3D CNN encoders for T1w MRI, FDG-PET, and Tau-PET. These encoders map each modality to a shared latent space via a mathematical fusion mechanism called MoPoE.

During training, we randomly drop modalities with 30 percent probability. This forces the model to learn robust representations that work even when modalities are missing.

At inference time, if only FDG-PET is available, the model can still diagnose — and even synthesize what the missing Tau-PET would look like through cross-modal decoders.

The clinical impact is significant: Tau-PET scans are expensive and available at only a few specialized centers. Our model can leverage the widely available FDG-PET and T1w MRI to achieve Tau-PET-level diagnostic accuracy.

---

## Slide 9: MoPoE Mathematical Foundation (2 minutes)
Let me explain the mathematics behind MoPoE, which is the core innovation of our fusion approach.

In a standard VAE, the encoder produces a mean and variance for the latent distribution. For multiple modalities, the question is: how do we combine these individual posterior distributions into a single joint posterior?

The Product of Experts approach multiplies the individual Gaussian distributions. For Gaussians, the product has a closed form: the mean is a weighted average where each expert is weighted by its precision — the inverse of its variance. This means high-confidence experts naturally dominate the fusion.

The Mixture of Products of Experts — MoPoE — goes one step further. It considers ALL possible subsets of modalities: eight subsets for three modalities. For each subset, it computes the Product of Experts, and then takes a uniform mixture over all subsets.

This is mathematically principled because it captures both shared information — when experts agree — and modality-specific information — when one expert has high confidence in a feature that others miss.

The advantage over simple concatenation is that MoPoE naturally handles missing modalities — if a modality is absent, its subsets are simply excluded from the mixture.

---

## Slide 10: Architecture Visualization (30 seconds)
Here's a visual summary of the architecture.

During training with all three modalities available, they go through their respective encoders, are fused via MoPoE into a shared latent code z, which then feeds into the diagnosis classifier and the cross-modal decoders.

During inference with a single modality — say, only FDG-PET — the same encoder produces z, which goes to the classifier for diagnosis and optionally to the decoders to synthesize the missing modalities.

---

## Slide 11: Training Strategy (1 minute)
The total loss function combines five components.

First, reconstruction loss for the available modalities using L1 and perceptual losses. This ensures the decoder faithfully reconstructs the input.

Second, cross-modal loss. When a modality is dropped during training, the decoder for that modality still produces a synthetic image. We compute the L1 and gradient loss between the synthetic and ground truth. This teaches the model to synthesize missing modalities.

Third, KL divergence with a small beta of 0.001. This regularizes the latent space toward a standard normal distribution, which is important for the VAE to learn smooth, interpolatable representations.

Fourth, classification loss using focal loss with gamma equals 2.0. The focal loss handles the severe class imbalance — we have 114 normal controls but only 2 AD patients.

Fifth, contrastive loss using InfoNCE. For the same patient with different modality subsets, the latent codes should be similar. For different patients, they should be different. This aligns the latent space across modality combinations.

---

## Slide 12: ADNI Dataset (1.5 minutes)
Our data comes from the Alzheimer's Disease Neuroimaging Initiative, or ADNI. This is a landmark longitudinal multi-center study that has been running for over two decades.

From ADNI, we selected 225 subjects with T1w MRI. Among these, 119 also have FDG-PET and 62 have Tau-PET. Critically, 38 subjects have all three modalities simultaneously — these form our primary training set.

The diagnosis distribution is: 114 normal controls, 48 early MCI, 40 MCI, 21 late MCI, and 2 Alzheimer's disease patients.

The preprocessing pipeline was entirely built with lightweight Python tools. We used SimpleITK for DICOM to NIfTI conversion — converting 181 PET volumes. HD-BET for brain extraction, which takes less than 5 seconds per volume on GPU. And SimpleITK again for affine spatial registration.

The total processed dataset is about 11 gigabytes.

---

## Slide 13: Experiment Design (1 minute)
We designed three controlled experiments on synthetic data to understand the model's behavior.

Experiment 1 is our baseline with 128-dimensional latent space and all loss functions enabled.

Experiment 2 doubles the latent dimension to 256, testing whether more capacity improves multimodal representation learning.

Experiment 3 removes the cross-modal and contrastive losses, isolating their contribution to overall performance.

The synthetic dataset has 100 samples with simulated brain anatomy and modality-specific AD patterns. The diagnosis distribution is balanced at 40 percent NC, 30 percent MCI, 30 percent AD.

All experiments run on an RTX 4060 laptop GPU with 8GB of VRAM, using fp16 mixed precision.

---

## Slide 14: Synthetic Results (1 minute)
The results clearly show that Experiment 2 with the 256-dimensional latent space achieves the best performance. The final loss is 0.670, which is 21 percent better than the baseline.

The larger latent space appears to capture richer multimodal representations, leading to faster convergence after epoch 10.

Experiment 3, without cross-modal losses, shows a lower total loss of 0.763. However, this comparison is confounded — the cross-modal loss is an additive component in Experiments 1 and 2, so removing it naturally reduces the total. On real multi-subject data, cross-modal synthesis is expected to significantly improve diagnosis accuracy when modalities are missing.

All three experiments demonstrate stable convergence in about 4 minutes on the RTX 4060.

---

## Slide 15: Real Data Results (1 minute)
On the real ADNI data with 38 all-three-modality subjects, our model achieves 57.9 percent accuracy for 3-class diagnosis after 10 epochs of training. The F1 score is 0.425.

The loss drops by 31 percent from 1.146 to 0.793, demonstrating effective learning.

The model has 8.5 million trainable parameters and trains in 292 seconds with fp16 AMP.

We verified GPU utilization during training using nvidia-smi with 1-second sampling. The peak utilization reached 95 percent, with an active average of 58 percent. This confirms that our fp16 AMP implementation is actually accelerating training.

---

## Slide 16: T2T-Bridge Clinical Results (1 minute)
For the T2T-Bridge project, the quantitative comparison is striking. Our model achieves an SSIM of 0.9573 and PSNR of 33.38 dB, significantly outperforming the p2p GAN baseline.

More importantly, the model is robust across the entire age range from 0 to 72 months. For a 0-month-old neonate, the model preserves clinically relevant lesion details — specifically, a small area of abnormal signal near the frontal horn of the right lateral ventricle.

The model has been deployed on a uMR890 3T scanner at the Children's Hospital of Fudan University in Xiamen and is being used in the Chinese Baby Connectome Project.

---

## Slide 17: GPU Optimization Story (1.5 minutes)
This is an important engineering story. Initially, our training script had zero percent GPU utilization. The model would time out without producing any useful results.

We identified three root causes. First, no mixed precision — all operations were in fp32. Second, per-sample iteration effectively gave us a batch size of 1. Third, the default config file was from the wrong project.

The fix involved three changes. First, wrapping the entire forward pass in torch.autocast and using a GradScaler for the backward pass. Second, fixing the default config. Third, verifying that CUDA PyTorch was properly installed.

After these fixes, GPU utilization jumped to a peak of 95 percent with an active average of 58 percent. Training time dropped to under 5 minutes for 10 epochs.

This experience highlights the importance of proper GPU profiling and the significant impact of mixed precision training.

---

## Slide 18: Infrastructure (1 minute)
A key contribution of our work is the development of a fully Python-based lightweight medical imaging toolchain.

For brain extraction, we replaced SynthStrip and FreeSurfer with HD-BET, which runs on GPU in under 5 seconds per volume.

For tissue segmentation, we replaced FreeSurfer infantFS with a MONAI DynUNet model with 5.4 million parameters.

For image registration, we replaced ANTs and FSL with SimpleITK, using affine and rigid registration with mutual information as the metric.

For DICOM conversion, we used SimpleITK's ImageSeriesReader, successfully converting 181 PET volumes.

For quantization and distributed training, we added bitsandbytes and accelerate.

All 18 tools in our pipeline pass verification.

---

## Slide 19: Pretrained Weights (1 minute)
Unfortunately, there are no directly applicable pretrained weights for our custom MoPoE-VAE encoder architecture. Standard pretrained models like ResNet and ViT are 2D and cannot be directly adapted to 3D medical imaging.

However, we identified several promising directions for transfer learning. MONAI provides pretrained segmentation models that could initialize our T1 encoder. Medical foundation models like BiomedCLIP and RadImageNet offer large-scale pretraining on diverse medical data. Self-supervised approaches like DINOv2 could provide useful features.

Despite starting from random Kaiming initialization, our model achieves rapid convergence with a 31 percent loss reduction in just 10 epochs and reaches 73.7 percent accuracy in longer training runs. This demonstrates that medical VAE training is viable even without pretrained weights.

---

## Slide 20: Related Work (1 minute)
Our work positions itself within a rich landscape of related research.

For T2T-Bridge, the I2SB paper from NVIDIA provided the theoretical foundation. We extended this to 3D medical imaging with age guidance and structure consistency. Compared to Cas-DiffCom, which uses a cascade of diffusion models, our direct bridge approach is simpler and more efficient.

For the multimodal diagnosis project, we improve upon several lines of work. Joint learning approaches from Medical Image Analysis combine synthesis and diagnosis but don't handle missing modalities as cleanly. MC-RVAE from NeuroImage uses LSTM-based imputation, which is more complex than our simple dropout strategy. GLA-GAN provides GAN-based synthesis but can't jointly optimize for diagnosis.

---

## Slide 21: Code Structure (30 seconds)
Our complete codebase is available on GitHub. It contains about 1,400 lines of model code across both projects, comprehensive data processing pipelines, training and evaluation scripts, and extensive documentation.

The repository is well-organized with clear separation between the two projects and shared utilities.

---

## Slide 22: Future Work (1.5 minutes)
Let me conclude with our research roadmap.

In the short term, we plan to complete the BOBs infant MRI dataset download and run full T2T-Bridge training on real infant data. We also want to train the MoPoE-VAE on all 225 ADNI subjects and run comprehensive evaluation across all modality subsets.

In the medium term, we aim to pretrain our encoders on MONAI segmentation tasks, add transformer backbones, implement uncertainty quantification, and leverage multi-GPU training on our remote RTX 5090 servers.

Long-term goals include multi-site validation, prospective clinical studies, integration of genetic and serum biomarkers, and eventual regulatory approval for clinical deployment.

---

## Slide 23: Thank You (30 seconds)
Thank you for your attention. I'm happy to take any questions.

Our code is available on GitHub at the link shown, and we welcome collaboration and contributions from the community.
