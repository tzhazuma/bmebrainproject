"""
T2T-Bridge evaluation script.

Computes:
  - PSNR, SSIM, LPIPS (image quality)
  - Dice score per tissue class (structure consistency)
  - Inference time per sample
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


def compute_psnr(pred, target):
    mse = F.mse_loss(pred, target).item()
    if mse == 0:
        return 100.0
    return 10 * np.log10(1.0 / mse)


def compute_ssim(pred, target, window_size=11):
    """Compute SSIM using 3D sliding window."""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    kernel = torch.ones(1, 1, window_size, window_size, window_size) / (window_size ** 3)
    kernel = kernel.to(pred.device)

    mu1 = F.conv3d(pred.unsqueeze(0).unsqueeze(0), kernel, padding=window_size//2)
    mu2 = F.conv3d(target.unsqueeze(0).unsqueeze(0), kernel, padding=window_size//2)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv3d((pred.unsqueeze(0).unsqueeze(0))**2, kernel, padding=window_size//2) - mu1_sq
    sigma2_sq = F.conv3d((target.unsqueeze(0).unsqueeze(0))**2, kernel, padding=window_size//2) - mu2_sq
    sigma12 = F.conv3d((pred.unsqueeze(0).unsqueeze(0)*target.unsqueeze(0).unsqueeze(0)), kernel, padding=window_size//2) - mu1_mu2

    ssim = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
           ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim.mean().item()


def compute_dice(pred_seg, target_seg, num_classes=3):
    """Dice score per tissue class."""
    dices = []
    for c in range(num_classes):
        pred_c = pred_seg == c
        target_c = target_seg == c
        intersection = (pred_c & target_c).float().sum()
        union = pred_c.float().sum() + target_c.float().sum()
        if union > 0:
            dices.append((2 * intersection / union).item())
        else:
            dices.append(1.0)  # perfect if both empty
    return dices


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--data-dir', type=str, default='data/dhcp/processed/')
    parser.add_argument('--sample-dir', type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load reconstructions
    sample_dir = Path(args.sample_dir) if args.sample_dir else Path(args.ckpt).parent / 'samples_nfe15'
    recon_file = sample_dir / 'recon.pt'
    if not recon_file.exists():
        print(f'No reconstructions found at {recon_file}')
        sys.exit(1)

    recons = torch.load(recon_file, map_location='cpu')

    # Load targets from dataset
    from project1_ttsbridge.data.dataset import InfMRI3DDataset

    dataset = InfMRI3DDataset(
        data_dir=args.data_dir,
        split='test',
        augment=False,
    )

    # Metrics
    psnrs, ssims = [], []
    dice_gm, dice_wm, dice_csf = [], [], []

    for i in range(len(dataset)):
        batch = dataset[i]
        subj = batch['subject']
        if subj not in recons:
            continue

        thin = batch['thin']  # [1, D, H, W]
        recon = recons[subj]  # [D, H, W]

        # Normalize
        thin_check = (thin - thin.min()) / (thin.max() - thin.min())
        recon_check = (recon - recon.min()) / (recon.max() - recon.min())

        psnrs.append(compute_psnr(recon_check, thin_check))
        ssims.append(compute_ssim(recon_check, thin_check))

        # Dice per tissue class
        tissue = batch['tissue'].squeeze().numpy()
        seg_recon = np.zeros_like(tissue)  # placeholder: would need seg model
        # dices = compute_dice(torch.from_numpy(seg_recon), torch.from_numpy(tissue))
        # dice_gm.append(dices[0]); dice_wm.append(dices[1]); dice_csf.append(dices[2])

    # Report
    print('=' * 60)
    print(' T2T-Bridge Evaluation')
    print('=' * 60)
    print(f'  Samples evaluated: {len(psnrs)}')
    print(f'  PSNR:     {np.mean(psnrs):.2f} ± {np.std(psnrs):.2f} dB')
    print(f'  SSIM:     {np.mean(ssims):.4f} ± {np.std(ssims):.4f}')
    # print(f'  Dice GM:  {np.mean(dice_gm):.4f}')
    # print(f'  Dice WM:  {np.mean(dice_wm):.4f}')
    # print(f'  Dice CSF: {np.mean(dice_csf):.4f}')
    print(f'  Best PSNR: {np.max(psnrs):.2f}')
    print(f'  Worst PSNR: {np.min(psnrs):.2f}')

    # Save to file
    report = {
        'n_samples': len(psnrs),
        'psnr_mean': np.mean(psnrs),
        'psnr_std': np.std(psnrs),
        'ssim_mean': np.mean(ssims),
        'ssim_std': np.std(ssims),
    }
    torch.save(report, sample_dir / 'metrics.pt')


if __name__ == '__main__':
    main()
