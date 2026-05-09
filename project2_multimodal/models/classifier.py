"""Diagnosis head for AD/MCI/NC classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiagnosisHead(nn.Module):
    """
    MLP classifier on latent space for disease diagnosis.

    Args:
        latent_dim: Latent code dimension
        num_classes: 3 (NC, MCI, AD)
        hidden_dims: MLP hidden layer dimensions
        dropout: Dropout rate for MC-dropout uncertainty
    """

    def __init__(self, latent_dim=256, num_classes=3, hidden_dims=(128, 64), dropout=0.3):
        super().__init__()
        layers = []
        prev_dim = latent_dim

        for hdim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hdim))
            layers.append(nn.LayerNorm(hdim))          # LayerNorm works with batch_size=1
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            prev_dim = hdim

        layers.append(nn.Linear(prev_dim, num_classes))
        self.mlp = nn.Sequential(*layers)
        self.dropout = dropout

    def forward(self, z):
        """
        Args:
            z: [B, latent_dim]
        Returns:
            logits: [B, num_classes]
        """
        return self.mlp(z)

    def predict(self, z):
        """Return class probabilities."""
        return F.softmax(self(z), dim=-1)

    def mc_dropout_forward(self, z, n_samples=10):
        """
        Monte Carlo dropout forward passes for uncertainty estimation.

        Returns:
            mean_logits: [B, num_classes]
            var_logits: [B, num_classes]
        """
        self.train()  # Keep dropout active
        logits_samples = []
        for _ in range(n_samples):
            logits_samples.append(self(z))
        logits = torch.stack(logits_samples)  # [S, B, C]
        self.eval()
        return logits.mean(0), logits.var(0)


class FocalLoss(nn.Module):
    """
    Focal Loss for class-imbalanced AD diagnosis.

    FL(p_t) = -(1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_loss = (1 - p_t) ** self.gamma * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss
