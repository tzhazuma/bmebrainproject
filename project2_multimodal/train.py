"""
Multimodal VAE training script.

Usage:
    python train.py --config configs/default.yaml --name multimodal_exp1
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from project2_multimodal.data.adni_dataset import ADNIMultimodalDataset, collate_fn
from project2_multimodal.models.encoders import ModalityEncoder, SimpleEncoder3D
from project2_multimodal.models.mopoe_vae import MoPoEVAE
from project2_multimodal.models.decoders import CrossModalDecoder
from project2_multimodal.models.classifier import DiagnosisHead, FocalLoss
from project2_multimodal.models.fusion import AdaptiveFusion
from project2_multimodal.models.losses import CombinedTrainingLoss


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--data-dir', type=str, default='data/adni/processed/')
    parser.add_argument('--name', type=str, default='multimodal_v1')
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--debug', action='store_true')
    return parser.parse_args()


def build_model(config):
    latent_dim = config['model']['latent_dim']
    modalities = config['model'].get('modalities', ['t1w', 'fdg_pet', 'tau_pet'])

    # Modality-specific encoders
    encoders = {}
    for mod in modalities:
        encoders[mod] = SimpleEncoder3D(in_channels=1, latent_dim=latent_dim)

    # MoPoE VAE
    vae = MoPoEVAE(
        modality_encoders=encoders,
        latent_dim=latent_dim,
        modalities=modalities,
        beta=config['training'].get('beta_kl', 0.001),
    )

    # Cross-modal decoders
    decoders = nn.ModuleDict()
    target_size = tuple(config['model'].get('patch_size', [64, 64, 32]))
    for mod in modalities:
        decoders[mod] = CrossModalDecoder(
            modality_name=mod,
            latent_dim=latent_dim,
            out_channels=1,
            target_size=target_size,
        )

    # Diagnosis classifier
    classifier = DiagnosisHead(
        latent_dim=latent_dim,
        num_classes=3,
        hidden_dims=config['model'].get('hidden_dims', [128, 64]),
        dropout=config['model'].get('dropout', 0.3),
    )

    # Fusion module
    fusion = AdaptiveFusion(
        latent_dim=latent_dim,
        num_modalities=len(modalities),
    )

    return vae, decoders, classifier, fusion


def train():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Build models
    vae, decoders, classifier, fusion = build_model(config)
    vae.to(device)
    decoders.to(device)
    classifier.to(device)
    fusion.to(device)

    # Loss
    criterion = CombinedTrainingLoss(
        lambda_recon=config['training'].get('lambda_recon', 1.0),
        lambda_cross=config['training'].get('lambda_cross', 0.5),
        lambda_cls=config['training'].get('lambda_cls', 1.0),
        lambda_contra=config['training'].get('lambda_contra', 0.1),
    )

    # Optimizer
    params = (
        list(vae.parameters()) +
        list(decoders.parameters()) +
        list(classifier.parameters()) +
        list(fusion.parameters())
    )
    optimizer = AdamW(params, lr=config['training']['lr'], weight_decay=config['training'].get('weight_decay', 1e-5))
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)

    # Dataset
    if args.debug:
        from project2_multimodal.data.synthetic_dataset import SyntheticMultimodalDataset
        patch_size = config['model'].get('patch_size', [32, 32, 16])
        dataset = SyntheticMultimodalDataset(num_samples=100, size=tuple(patch_size))
    else:
        dataset = ADNIMultimodalDataset(
            data_dir=args.data_dir,
            split='train',
            modalities=config['model'].get('modalities', ['t1w', 'fdg_pet', 'tau_pet']),
            patch_size=config['model'].get('patch_size', [64, 64, 32]),
            augment=True,
            modality_drop_prob=config['training'].get('modality_dropout', {}).get('p_drop', 0.3),
        )

    dataloader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=0 if args.debug else 4,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )

    # Checkpoint dir
    ckpt_dir = Path('checkpoints') / args.name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    max_epochs = config['training'].get('max_epochs', 150)
    global_step = 0
    start_epoch = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        vae.load_state_dict(ckpt['vae'])
        decoders.load_state_dict(ckpt['decoders'])
        classifier.load_state_dict(ckpt['classifier'])
        fusion.load_state_dict(ckpt['fusion'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        print(f'Resumed from epoch {start_epoch}')

    print(f'Training: {args.name}')
    for epoch in range(start_epoch, max_epochs):
        vae.train()
        decoders.train()
        classifier.train()
        fusion.train()

        epoch_loss = 0
        n_batches = 0
        t0 = time.time()

        for batch in dataloader:
            images_list = batch['images']
            labels = batch['label'].to(device)

            # For each item, encode and forward
            total_loss = 0
            for img_dict, label in zip(images_list, labels):
                # Move to device
                img_device = {k: v.to(device) for k, v in img_dict.items() if v is not None}

                # Available modalities
                avail = [k for k, v in img_device.items() if v is not None]

                # MoPoE encode
                z, mu, logvar, kl_loss = vae(img_device, avail)

                # Fuse for classification
                mus, logvars = vae.encode(img_device, avail)
                z_list = [mu for mu in mus]
                fused_z = fusion(z_list)

                # Classify
                logits = classifier(fused_z)
                focal_loss = FocalLoss(gamma=2.0)(logits, label.unsqueeze(0))

                # Reconstruct available modalities
                recon_outputs = {}
                for mod in avail:
                    recon_outputs[mod] = decoders[mod](z)

                # Cross-modal synthesis for dropped modalities
                cross_outputs = {}
                all_mods = config['model'].get('modalities', ['t1w', 'fdg_pet', 'tau_pet'])
                for mod in all_mods:
                    if mod not in avail:
                        cross_outputs[mod] = decoders[mod](z)

                # Combined loss
                outputs = {
                    'recon': recon_outputs,
                    'cross': cross_outputs,
                    'logits': logits,
                    'kl': kl_loss,
                    'z': z,
                }
                loss, loss_dict = criterion(outputs, img_device, label.unsqueeze(0))
                total_loss += loss

            total_loss = total_loss / len(images_list)

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

            epoch_loss += total_loss.item()
            n_batches += 1
            global_step += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        print(f'Epoch {epoch:3d} | Loss: {avg_loss:.4f} | Time: {time.time()-t0:.1f}s')

        # Checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'vae': vae.state_dict(),
                'decoders': decoders.state_dict(),
                'classifier': classifier.state_dict(),
                'fusion': fusion.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, ckpt_dir / f'epoch_{epoch}.pt')

    print(f'Training complete: {args.name}')


if __name__ == '__main__':
    train()
