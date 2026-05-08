"""
MoPoE-VAE (Mixture-of-Products-of-Experts) for multimodal fusion.

Given K modalities, the joint posterior is a mixture over all 2^K subsets:

    q_phi(z|X) = Σ_{S∈P({1..K})} π_S * Π_{i∈S} q_i(z|x_i)

where π_S = 1/2^K (uniform over subsets).

This enables:
  - Training with arbitrary modality subsets
  - Missing-modality robustness at inference time
  - Cross-modal synthesis via subset decoders
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import combinations


class MoPoEVAE(nn.Module):
    """
    Multimodal VAE with Mixture-of-Products-of-Experts fusion.

    Args:
        modality_encoders: Dict[str, ModalityEncoder]
        latent_dim: Latent space dimension
        modalities: List of modality names (e.g., ['t1w', 'fdg_pet', 'tau_pet'])
        beta: KL divergence weight (β-VAE)
        use_uniform_prior: Use π_S = 1/2^K or learnable subset weights
    """

    def __init__(
        self,
        modality_encoders,
        latent_dim=256,
        modalities=None,
        beta=0.001,
        use_uniform_prior=True,
    ):
        super().__init__()
        self.encoders = nn.ModuleDict(modality_encoders)
        self.latent_dim = latent_dim
        self.beta = beta
        self.use_uniform_prior = use_uniform_prior

        if modalities is None:
            modalities = list(modality_encoders.keys())
        self.modalities = modalities
        self.K = len(modalities)

        # Compute all non-empty subsets
        self.subsets = []
        for k in range(1, self.K + 1):
            for combo in combinations(range(self.K), k):
                self.subsets.append(list(combo))

        # Uniform prior weights
        if use_uniform_prior:
            self.register_buffer('subset_weights', torch.ones(len(self.subsets)) / len(self.subsets))
        else:
            self.subset_weights = nn.Parameter(torch.ones(len(self.subsets)) / len(self.subsets))

    def encode(self, inputs, available_modalities=None):
        """
        Encode each available modality into μᵢ, log σ²ᵢ.

        Args:
            inputs: Dict[str, Tensor] | Modality → [B, C, D, H, W]
            available_modalities: List[str] of available modalities (None = all)

        Returns:
            mus: List[Tensor] length = n_available, each [B, latent_dim]
            logvars: List[Tensor]
        """
        if available_modalities is None:
            available_modalities = [m for m in self.modalities if inputs.get(m) is not None]

        mus, logvars = [], []
        for mod in available_modalities:
            mu, logvar, _ = self.encoders[mod](inputs[mod])
            mus.append(mu)
            logvars.append(logvar)

        return mus, logvars

    def product_of_experts(self, mus, logvars):
        """
        Product of Experts: p(z) ∝ Π ᵢ N(μᵢ, σ²ᵢ).

        For Gaussians: μ_PoE = (Σ ᵢ μᵢ/σ²ᵢ) / (Σ ᵢ 1/σ²ᵢ)
                       σ²_PoE = 1 / (Σ ᵢ 1/σ²ᵢ)
        """
        precisions = [torch.exp(-lv) for lv in logvars]
        precision_sum = sum(precisions)

        mu_poe = sum(mu * prec for mu, prec in zip(mus, precisions)) / precision_sum
        logvar_poe = -torch.log(precision_sum)

        return mu_poe, logvar_poe

    def forward(self, inputs, available_modalities=None, n_samples=1):
        """
        Forward pass with MoPoE fusion.

        Args:
            inputs: Dict[str, Tensor] modality data
            available_modalities: subset for inference
            n_samples: Monte Carlo samples for training

        Returns:
            z: Latent samples [B, latent_dim]
            mu_posterior: MoPoE posterior mean [B, latent_dim]
            logvar_posterior: MoPoE posterior log-variance [B, latent_dim]
            kl_loss: KL divergence loss
        """
        all_mus, all_logvars = self.encode(inputs, None)  # get all individual posteriors

        # Build subset posteriors
        subset_mus, subset_logvars = [], []

        for subset in self.subsets:
            s_mus = [all_mus[i] for i in subset]
            s_logvars = [all_logvars[i] for i in subset]
            mu_poe, logvar_poe = self.product_of_experts(s_mus, s_logvars)
            subset_mus.append(mu_poe)
            subset_logvars.append(logvar_poe)

        # Mixture: q(z|x) = Σ π_S q_S(z|x)
        # MoPoE: sample a subset S, then sample z ~ q_S
        B = all_mus[0].shape[0]
        device = all_mus[0].device

        # Sample subset indices
        subset_idx = torch.multinomial(self.subset_weights, B, replacement=True)

        # Gather mu, logvar for selected subset
        mu = torch.stack([subset_mus[idx][b] for b, idx in enumerate(subset_idx)])
        logvar = torch.stack([subset_logvars[idx][b] for b, idx in enumerate(subset_idx)])

        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        # KL divergence: KL(q(z|x) || p(z))
        # For MoPoE, use the full mixture KL or the per-sample KL
        # Simplified: KL of the sampled component vs prior N(0, I)
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        kl_loss = self.beta * kl.mean()

        return z, mu, logvar, kl_loss
