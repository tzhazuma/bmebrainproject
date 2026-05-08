# 🐍 Python or system PATH: 
# @utils: module: torch, numpy, nibabel
"""T2T-Bridge inference (sampling) script.

Usage:
    python sample.py --ckpt checkpoints/exp1/latest.pt --nfe 15
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from project1_ttsbridge.data.dataset import InfMRI3DDataset
from project1_ttsbridge.models.unet3d import UNet3D
from project1_ttsbridge.models.diffusion import DiffusionBridge


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--data-dir', type=str, default='data/dhcp/processed/')
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--nfe', type=int, default=15, help='NFE for sampling')
    parser.add_argument('--cfg-scale', type=float, default=1.5)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--use-fp16', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load checkpoint
    ckpt = torch.load(args.ckpt, map_location=device)
    config = ckpt['config']

    # Build model
    denoiser = UNet3D(
        in_channels=2,
        out_channels=1,
        model_channels=config['model']['dim'],
        channel_mult=config['model']['dim_mults'],
    ).to(device)

    denoiser.load_state_dict(ckpt.get('ema_denoiser', ckpt['bridge']['denoiser']))
    denoiser.eval()

    bridge = DiffusionBridge(denoiser, **config['diffusion']).to(device)

    # Dataset
    dataset = InfMRI3DDataset(
        data_dir=args.data_dir,
        split='test',
        patch_size=config['model'].get('image_size', [128, 128, 32]),
        augment=False,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
    )

    # Output
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.ckpt).parent / f'samples_nfe{args.nfe}'
    output_dir.mkdir(parents=True, exist_ok=True)

    all_recons = {}
    print(f'Sampling with NFE={args.nfe}, CFG scale={args.cfg_scale}')

    for batch_idx, batch in enumerate(dataloader):
        thick = batch['thick'].to(device)
        thin = batch['thin'].to(device)
        age = batch['age'].to(device)
        subjects = batch['subject']

        with torch.no_grad():
            if args.use_fp16:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    recon = bridge.sample(
                        thick, age=age,
                        nfe=args.nfe,
                        cfg_scale=args.cfg_scale,
                    )
            else:
                recon = bridge.sample(
                    thick, age=age,
                    nfe=args.nfe,
                    cfg_scale=args.cfg_scale,
                )

        recon = recon.squeeze(1).cpu()
        for i, subj in enumerate(subjects):
            all_recons[subj] = recon[i]

        if batch_idx % 5 == 0:
            print(f'  Batch {batch_idx}/{len(dataloader)}')

    torch.save(all_recons, output_dir / 'recon.pt')
    print(f'Saved {len(all_recons)} samples to {output_dir}')


if __name__ == '__main__':
    main()
