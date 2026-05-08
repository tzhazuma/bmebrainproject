"""
Training loss components for multimodal diagnosis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReconstructionLoss(nn.Module):
    """L1 + perceptual reconstruction loss."""

    def __init__(self, use_ssim=True):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.use_ssim = use_ssim

    def forward(self, pred, target):
        loss = self.l1(pred, target)
        return loss


class CrossModalLoss(nn.Module):
    """
    Cross-modal synthesis loss.

    When a modality is dropped during training, the decoder
    should still generate a reasonable image.
    """

    def __init__(self, lambda_l1=1.0, lambda_grad=0.5):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_grad = lambda_grad
        self.l1 = nn.SmoothL1Loss()

    def forward(self, pred, target):
        l1 = self.l1(pred, target)

        # Gradient loss for edges
        def gradient(x):
            dx = torch.abs(x[..., :-1] - x[..., 1:]).mean(dim=(-3, -2, -1))
            return dx.mean()

        grad_loss = 0
        for d in range(3):
            slices = [slice(None)] * 5
            slices[d + 2] = slice(0, -1)
            curr = pred[*slices]
            next_s = target[*slices]
            slices[d + 2] = slice(1, None)
            grad_loss += F.l1_loss(curr, next_s)

        return self.lambda_l1 * l1 + self.lambda_grad * grad_loss


class ContrastiveLatentLoss(nn.Module):
    """
    InfoNCE loss to align latent representations from different modality subsets.

    If subject i has potentials in subset A and subset B of modalities,
    z_i_A and z_i_B should be similar, and dissimilar to z_j for j ≠ i.
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, z1, z2):
        """
        Args:
            z1, z2: [B, D] latent from two modality subsets
        Returns:
            loss: scalar
        """
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)

        # Cosine similarity matrix
        logits = (z1 @ z2.T) / self.temperature  # [B, B]
        labels = torch.arange(logits.size(0), device=logits.device)

        loss = (self.criterion(logits, labels) + self.criterion(logits.T, labels)) / 2
        return loss


class CombinedTrainingLoss(nn.Module):
    """
    Total loss for multimodal VAE training:

    L = λ_recon * L_recon + λ_cross * L_cross + β * KL + λ_cls * L_ce + λ_contra * L_contra
    """

    def __init__(
        self,
        lambda_recon=1.0,
        lambda_cross=0.5,
        lambda_cls=1.0,
        lambda_contra=0.1,
    ):
        super().__init__()
        self.recon_loss = ReconstructionLoss()
        self.cross_loss = CrossModalLoss()
        self.contra_loss = ContrastiveLatentLoss()
        self.lambda_recon = lambda_recon
        self.lambda_cross = lambda_cross
        self.lambda_cls = lambda_cls
        self.lambda_contra = lambda_contra

    def forward(self, outputs, targets, labels=None):
        """
        Args:
            outputs: Dict with 'recon', 'cross', 'z', 'logits' keys
            targets: Dict of GT modality images
            labels: [B] diagnosis labels (0=NC, 1=MCI, 2=AD)
        """
        loss = 0.0
        loss_dict = {}

        # Reconstruction loss for available modalities
        recon_loss = 0
        for mod, pred in outputs['recon'].items():
            if mod in targets:
                recon_loss += self.recon_loss(pred, targets[mod])
        loss += self.lambda_recon * recon_loss
        loss_dict['recon'] = recon_loss.item() if isinstance(recon_loss, torch.Tensor) else recon_loss

        # Cross-modal synthesis loss
        cross_loss = 0
        if outputs.get('cross'):
            for mod, pred in outputs['cross'].items():
                if mod in targets:
                    cross_loss += self.cross_loss(pred, targets[mod])
            loss += self.lambda_cross * cross_loss
            loss_dict['cross'] = cross_loss.item() if isinstance(cross_loss, torch.Tensor) else cross_loss

        # Classification loss
        if labels is not None and 'logits' in outputs:
            cls_loss = F.cross_entropy(outputs['logits'], labels)
            loss += self.lambda_cls * cls_loss
            loss_dict['cls'] = cls_loss.item()

        # KL loss
        if 'kl' in outputs:
            loss += outputs['kl']
            loss_dict['kl'] = outputs['kl'].item() if isinstance(outputs['kl'], torch.Tensor) else outputs['kl']

        # Contrastive loss
        if 'z1' in outputs and 'z2' in outputs:
            contra_loss = self.contra_loss(outputs['z1'], outputs['z2'])
            loss += self.lambda_contra * contra_loss
            loss_dict['contra'] = contra_loss.item()

        loss_dict['total'] = loss.item()
        return loss, loss_dict
