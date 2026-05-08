"""Classifier-Free Age Guidance (CFG) module."""

import torch
import torch.nn as nn
import math


class AgeEmbedding(nn.Module):
    """
    Sinusoidal age embedding for CFG-based conditioning.

    Injects the infant's age (in months) into the diffusion process
    to handle tissue contrast variability across 0-72 months.
    """

    def __init__(self, dim, max_age=72, max_period=10000):
        super().__init__()
        self.dim = dim
        self.max_age = max_age
        self.register_buffer(
            'freqs',
            torch.exp(
                -math.log(max_period) *
                torch.arange(0, dim // 2, dtype=torch.float32) / (dim // 2)
            )
        )

    def forward(self, age):
        """
        Args:
            age: Age in months [B] or [B, 1], values in [0, max_age]
        Returns:
            embedding: [B, dim]
        """
        if age.dim() == 2:
            age = age.squeeze(-1)

        age = age.float() / self.max_age  # normalize to [0, 1]
        args = age.unsqueeze(-1) * self.freqs.unsqueeze(0).to(age.device)
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding


class AgeConditionalModule(nn.Module):
    """
    Wraps a module with CFG age injection.

    Usage:
        cond_pred = model(x, age)         # conditional
        uncond_pred = model(x, None)       # unconditional
        final_pred = uncond_pred + w * (cond_pred - uncond_pred)  # CFG
    """

    def __init__(self, base_model, age_emb_dim=256):
        super().__init__()
        self.base_model = base_model
        self.null_age = nn.Parameter(torch.randn(1, age_emb_dim))

    def forward(self, x, t, age=None):
        if age is None:
            return self.base_model(x, t, None)
        return self.base_model(x, t, age)
