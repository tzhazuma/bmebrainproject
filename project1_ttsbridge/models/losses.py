"""Loss functions for T2T-Bridge."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffusionLoss(nn.Module):
    """MSE loss between predicted and target residual in bridge process."""

    def __init__(self, reduction='mean'):
        super().__init__()
        self.mse = nn.MSELoss(reduction=reduction)

    def forward(self, pred_r, target_r):
        return self.mse(pred_r, target_r)


class DiceLoss(nn.Module):
    """
    Soft Dice loss for tissue segmentation (structure consistency).
    Works on multi-class one-hot encoded tensors.
    """

    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        """
        Args:
            pred: Predicted logits/probs [B, C, D, H, W]
            target: One-hot encoded GT [B, C, D, H, W]
        """
        pred = F.softmax(pred, dim=1)
        dims = (0, 2, 3, 4)  # sum over batch and spatial
        intersection = (pred * target).sum(dim=dims)
        union = pred.sum(dim=dims) + target.sum(dim=dims)
        dice = 2 * intersection / (union + self.smooth)
        return 1 - dice.mean()


class GradientDifferenceLoss(nn.Module):
    """
    Gradient difference loss to enforce edge consistency between
    generated and target images.
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        """
        Args:
            pred, target: [B, 1, D, H, W]
        """
        def gradients(x):
            dz = torch.abs(x[:, :, 1:, :, :] - x[:, :, :-1, :, :])
            dy = torch.abs(x[:, :, :, 1:, :] - x[:, :, :, :-1, :])
            dx = torch.abs(x[:, :, :, :, 1:] - x[:, :, :, :, :-1])
            return dz, dy, dx

        p_dz, p_dy, p_dx = gradients(pred)
        t_dz, t_dy, t_dx = gradients(target)

        return (
            F.l1_loss(p_dz, t_dz) +
            F.l1_loss(p_dy, t_dy) +
            F.l1_loss(p_dx, t_dx)
        )


class PerceptualLoss3D(nn.Module):
    """
    3D perceptual loss using a pretrained segmentation network's features.
    Placeholder — requires a pretrained 3D feature extractor.
    """

    def __init__(self, layers=None):
        super().__init__()
        self.layers = layers or ['relu1', 'relu2', 'relu3']

    def forward(self, pred, target):
        # TODO: Integrate pretrained 3D monai/prediction model
        return torch.tensor(0.0, device=pred.device)


class CombinedLoss(nn.Module):
    """
    Combined loss for T2T-Bridge training.

    L_total = L_diffusion + λ_struct * L_dice + λ_grad * L_gradient
    """

    def __init__(self, lambda_struct=0.1, lambda_grad=0.05):
        super().__init__()
        self.diffusion_loss = DiffusionLoss()
        self.dice_loss = DiceLoss()
        self.gradient_loss = GradientDifferenceLoss()
        self.lambda_struct = lambda_struct
        self.lambda_grad = lambda_grad

    def forward(self, pred_r, target_r, pred_seg=None, target_seg=None, pred_img=None, target_img=None):
        loss = self.diffusion_loss(pred_r, target_r)
        loss_dict = {'diffusion': loss.item()}

        if pred_seg is not None and target_seg is not None:
            seg_loss = self.dice_loss(pred_seg, target_seg)
            loss = loss + self.lambda_struct * seg_loss
            loss_dict['structure'] = seg_loss.item()

        if pred_img is not None and target_img is not None:
            grad_loss = self.gradient_loss(pred_img, target_img)
            loss = loss + self.lambda_grad * grad_loss
            loss_dict['gradient'] = grad_loss.item()

        loss_dict['total'] = loss.item()
        return loss, loss_dict
