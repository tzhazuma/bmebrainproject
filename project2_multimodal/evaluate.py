"""
Multimodal diagnosis evaluation.

Tests across all modality availability scenarios:
  - Full multimodal (upper bound)
  - Single modality: MRI-only, FDG-only, TAU-only
  - Pair combinations
"""

import argparse
import sys
import time
from pathlib import Path
from itertools import combinations

import torch
import yaml
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--data-dir', type=str, default='data/adni/processed/')
    parser.add_argument('--test-all-subsets', action='store_true',
                        help='Test all modality subset combinations')
    parser.add_argument('--batch-size', type=int, default=8)
    return parser.parse_args()


def test_subset(model_components, dataloader, available_mods, device):
    """Test with a specific modality subset."""
    vae, decoders, classifier, fusion = model_components
    vae.eval()
    classifier.eval()
    fusion.eval()

    all_preds, all_labels = [], []

    for batch in dataloader:
        images_list = batch['images']
        labels = batch['label']

        for img_dict, label in zip(images_list, labels):
            # Filter to only available modalities
            img_device = {}
            for mod in available_mods:
                if mod in img_dict and img_dict[mod] is not None:
                    img_device[mod] = img_dict[mod].to(device)

            if len(img_device) == 0:
                continue

            with torch.no_grad():
                mus, logvars = vae.encode(img_device, list(img_device.keys()))
                z_list = [mu for mu in mus]
                fused_z = fusion(z_list)
                logits = classifier(fused_z)

            all_preds.append(logits.argmax(dim=-1).item())
            all_labels.append(label.item() if isinstance(label, torch.Tensor) else label)

    if len(all_preds) == 0:
        return {'accuracy': 0, 'auc': 0, 'f1': 0, 'n': 0}

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'f1_weighted': f1_score(all_labels, all_preds, average='weighted'),
        'f1_per_class': f1_score(all_labels, all_preds, average=None).tolist(),
        'n_samples': len(all_preds),
    }

    # AUC (one-vs-rest)
    try:
        metrics['auc_ovr'] = roc_auc_score(
            np.eye(3)[all_labels],
            np.eye(3)[all_preds],
            multi_class='ovr',
            average='weighted',
        )
    except:
        metrics['auc_ovr'] = 0

    return metrics


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load checkpoint
    ckpt = torch.load(args.ckpt, map_location=device)
    config = None  # embedded in ckpt or load separately

    from project2_multimodal.train import build_model
    vae, decoders, classifier, fusion = build_model(config or {
        'model': {
            'latent_dim': 256,
            'modalities': ['t1w', 'fdg_pet', 'tau_pet'],
            'hidden_dims': [128, 64],
            'dropout': 0.3,
        },
        'training': {'beta_kl': 0.001},
    })

    vae.load_state_dict(ckpt['vae'])
    classifier.load_state_dict(ckpt['classifier'])
    decoders.load_state_dict(ckpt['decoders'])
    fusion.load_state_dict(ckpt['fusion'])

    vae.to(device)
    classifier.to(device)
    fusion.to(device)
    for d in decoders.values():
        d.to(device)

    model_components = (vae, decoders, classifier, fusion)

    all_modalities = ['t1w', 'fdg_pet', 'tau_pet']

    # Dataset
    from project2_multimodal.data.adni_dataset import ADNIMultimodalDataset
    dataset = ADNIMultimodalDataset(
        data_dir=args.data_dir,
        split='test',
        modalities=all_modalities,
        augment=False,
        modality_drop_prob=0,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Test all subsets
    print('=' * 60)
    print(' Multimodal Diagnosis Evaluation')
    print('=' * 60)

    all_results = {}

    if args.test_all_subsets:
        subsets = []
        for k in range(1, len(all_modalities) + 1):
            for combo in combinations(all_modalities, k):
                subsets.append(list(combo))
    else:
        subsets = [all_modalities]  # full only

    for mods in subsets:
        name = ' + '.join(mods) if len(mods) < 3 else 'All modalities'
        metrics = test_subset(model_components, dataloader, mods, device)
        all_results[name] = metrics
        print(f'  {name:25s} | Acc: {metrics["accuracy"]:.3f} | F1: {metrics["f1_weighted"]:.3f} | AUC: {metrics["auc_ovr"]:.3f} | N={metrics["n_samples"]}')

    # Summary
    print('-' * 60)
    print(' Drop from full:')
    full_acc = all_results.get('All modalities', {}).get('accuracy', 1.0)
    for name, result in all_results.items():
        if name != 'All modalities':
            delta = full_acc - result['accuracy']
            print(f'  {name}: ΔAcc = {delta:.4f} ({delta*100:.1f}%)')


if __name__ == '__main__':
    main()
