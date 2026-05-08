"""
T2T-Bridge training script.

Usage:
    python train.py --config configs/default.yaml --name exp1 --gpus 2
"""

import os
import sys
import yaml
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from project1_ttsbridge.data.dataset import InfMRI3DDataset, SyntheticDataGenerator
from project1_ttsbridge.models.unet3d import UNet3D
from project1_ttsbridge.models.diffusion import DiffusionBridge
from project1_ttsbridge.models.losses import CombinedLoss
from project1_ttsbridge.models.seg_head import SegmentationHead


def parse_args():
    parser = argparse.ArgumentParser(description='T2T-Bridge Training')
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--data-dir', type=str, default='data/dhcp/processed/')
    parser.add_argument('--name', type=str, default='t2t_bridge_v1')
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--debug', action='store_true', help='Use synthetic data')
    return parser.parse_args()


def build_model(config, device):
    denoiser = UNet3D(
        in_channels=2,          # thick + x_t
        out_channels=1,         # residual
        model_channels=config['model']['dim'],
        channel_mult=config['model']['dim_mults'],
        num_res_blocks=config['model'].get('resnet_blocks', 2),
        dropout=config['model'].get('dropout', 0.0),
    ).to(device)

    bridge = DiffusionBridge(
        denoiser,
        timesteps=config['diffusion']['timesteps'],
        beta_start=config['diffusion']['beta_start'],
        beta_end=config['diffusion']['beta_end'],
        schedule=config['diffusion']['schedule'],
    ).to(device)

    seg_head = SegmentationHead(
        in_channels=config['model']['dim'],
        num_classes=3,
    ).to(device)

    return bridge, seg_head


def train():
    args = parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Build models
    bridge, seg_head = build_model(config, device)

    # Loss
    criterion = CombinedLoss(
        lambda_struct=config['training'].get('lambda_structure', 0.1),
    )

    # Optimizer
    params = list(bridge.parameters()) + list(seg_head.parameters())
    optimizer = AdamW(
        params,
        lr=config['training']['lr'],
        weight_decay=config['training'].get('weight_decay', 1e-6),
    )

    # Scheduler
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)

    # Dataset
    if args.debug:
        dataset = SyntheticDataGenerator(num_samples=100, size=(64, 64, 32))
    else:
        dataset = InfMRI3DDataset(
            data_dir=args.data_dir,
            split='train',
            patch_size=config['model'].get('image_size', [128, 128, 32]),
            augment=True,
        )

    dataloader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    # EMA
    ema_decay = config['training'].get('ema_decay', 0.9999)
    ema_model = torch.optim.swa_utils.AveragedModel(bridge.denoiser, avg_fn=lambda avg, p, n: ema_decay * avg + (1 - ema_decay) * p)

    # Checkpoint dir
    ckpt_dir = Path('checkpoints') / args.name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    max_epochs = config['training'].get('max_epochs', 200)
    global_step = 0
    start_epoch = 0

    # Resume if specified
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        bridge.load_state_dict(ckpt['bridge'])
        seg_head.load_state_dict(ckpt['seg_head'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        global_step = ckpt['global_step']
        print(f'Resumed from {args.resume} at epoch {start_epoch}')

    print(f'Starting training: {args.name}')
    print(f'  Epochs: {max_epochs}')
    print(f'  Batch size: {config["training"]["batch_size"]}')
    print(f'  LR: {config["training"]["lr"]}')

    for epoch in range(start_epoch, max_epochs):
        bridge.train()
        seg_head.train()
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for batch in dataloader:
            thin = batch['thin'].to(device)      # [B, 1, D, H, W]
            thick = batch['thick'].to(device)    # [B, 1, D, H, W]
            tissue = batch['tissue'].to(device)  # [B, 1, D, H, W]
            age = batch['age'].to(device)        # [B]

            # Training step: predict residual from x0 (thick) to x1 (thin)
            diffusion_loss, x_t, pred_r = bridge.training_step(thick, thin, age)

            # Structure consistency loss
            # Reconstruct approximate x1 from predicted residual
            # For I2SB, we can estimate x1_pred = x_t + pred_r
            x1_pred = x_t + pred_r

            # Get tissue prediction from x1_pred features
            # We use the denoiser intermediate features or a separate seg head
            # Simplified: pass x1_pred through seg_head
            pred_seg = seg_head(x1_pred)

            # One-hot encode tissue labels
            from project1_ttsbridge.models.seg_head import tissue_to_onehot
            target_seg = tissue_to_onehot(tissue, num_classes=3)

            # Combined loss
            loss, loss_dict = criterion(
                pred_r, (thin - x_t),    # target residual
                pred_seg=pred_seg,
                target_seg=target_seg,
            )

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

            # EMA
            ema_model.update_parameters(bridge.denoiser)

            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        elapsed = time.time() - t0
        print(f'Epoch {epoch:3d}/{max_epochs} | Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e} | Time: {elapsed:.1f}s')

        # Save checkpoint
        if (epoch + 1) % 10 == 0 or epoch == max_epochs - 1:
            ckpt_path = ckpt_dir / f'epoch_{epoch}.pt'
            torch.save({
                'epoch': epoch,
                'global_step': global_step,
                'bridge': bridge.state_dict(),
                'seg_head': seg_head.state_dict(),
                'ema_denoiser': ema_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': config,
            }, ckpt_path)
            latest_link = ckpt_dir / 'latest.pt'
            latest_link.unlink(missing_ok=True)
            latest_link.symlink_to(f'epoch_{epoch}.pt')
            print(f'  Saved: {ckpt_path}')

    print(f'Training complete: {args.name}')


if __name__ == '__main__':
    train()
