"""
Diffusion Bridge process for T2T-Bridge.

Implements the Image-to-Image Schrodinger Bridge (I2SB) forward/reverse process.
Key difference from standard diffusion: bridges directly from x0 (thick) to x1 (thin),
not from random noise.
"""

import torch
import torch.nn as nn


class DiffusionBridge(nn.Module):
    """
    I2SB Diffusion Bridge.

    Args:
        denoiser: UNet3D model that predicts residual
        timesteps: Number of diffusion steps
        beta_start: Starting noise level
        beta_end: Ending noise level
        schedule: "linear" or "cosine"
    """

    def __init__(
        self,
        denoiser,
        timesteps=1000,
        beta_start=1e-4,
        beta_end=3e-4,
        schedule="linear",
    ):
        super().__init__()
        self.denoiser = denoiser
        self.timesteps = timesteps

        # Build noise schedule
        if schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, timesteps)
        elif schedule == "cosine":
            steps = timesteps + 1
            s = 0.008
            t = torch.linspace(0, timesteps, steps)
            alphas_cumprod = torch.cos((t / timesteps + s) / (1 + s) * torch.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
            betas = torch.clamp(betas, max=0.999)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

    def q_sample(self, x0, x1, t):
        """
        Forward diffusion (bridge from x0 to x1).

        In I2SB, the forward process is:
            x_t = (1 - α̅_t) · x1 + √(α̅_t · (1 - α̅_t)) · x0 + √(α̅_t) · ε

        Args:
            x0: Source (thick-slice) [B, C, D, H, W]
            x1: Target (thin-slice) [B, C, D, H, W]
            t: Timestep indices [B]

        Returns:
            x_t: Noised intermediate
            noise: The noise added
        """
        alpha_bar = self.alphas_cumprod[t]  # [B]
        alpha_bar = alpha_bar.view(-1, 1, 1, 1, 1)

        noise = torch.randn_like(x1)
        coef0 = (alpha_bar * (1 - alpha_bar)).sqrt()
        coef1 = (1 - alpha_bar)
        coef_eps = alpha_bar.sqrt()

        x_t = coef1 * x1 + coef0 * x0 + coef_eps * noise
        return x_t, noise

    def training_step(self, x0, x1, age=None):
        """
        Single training step: sample t, add noise, predict residual.

        Args:
            x0: Thick-slice [B, 1, D, H, W]
            x1: Thin-slice (GT) [B, 1, D, H, W]
            age: Age in months [B], optional for CFG

        Returns:
            loss: MSE between predicted and true residual
        """
        B = x0.shape[0]
        t = torch.randint(0, self.timesteps, (B,), device=x0.device)

        # Forward: x0 → x_t (bridge process)
        x_t, noise = self.q_sample(x0, x1, t)

        # Target: the residual that denoiser should predict
        target_r = x1 - x_t

        # Model input: concatenate [x_t, x0] along channel dim
        model_input = torch.cat([x_t, x0], dim=1)

        # Predict residual
        pred_r = self.denoiser(model_input, t, age)

        # MSE loss on residual
        loss = nn.functional.mse_loss(pred_r, target_r)

        return loss, x_t, pred_r

    @torch.no_grad()
    def sample(self, x0, age=None, nfe=15, cfg_scale=1.5):
        """
        Sample thin-slice from thick-slice via bridge reverse process.

        Uses DPM-Solver-like stepping with CFG.

        Args:
            x0: Thick-slice input [B, 1, D, H, W]
            age: Age in months [B], optional
            nfe: Number of function evaluations (default 15)
            cfg_scale: Classifier-free guidance scale (>1 for stronger conditioning)

        Returns:
            x1_pred: Predicted thin-slice [B, 1, D, H, W]
        """
        B = x0.shape[0]
        device = x0.device

        # Start from x0 (thick-slice) at t=0
        x_t = x0.clone()

        # Discretize into nfe steps
        step_indices = torch.linspace(0, self.timesteps - 1, nfe + 1, dtype=torch.long, device=device)

        for i in range(nfe):
            t_cur = step_indices[i]
            t_next = step_indices[i + 1]

            t = torch.full((B,), t_cur, device=device, dtype=torch.long)
            t_next_t = torch.full((B,), t_next, device=device, dtype=torch.long)

            model_input = torch.cat([x_t, x0], dim=1)

            # Classifier-free guidance
            if cfg_scale != 1.0 and age is not None:
                pred_cond = self.denoiser(model_input, t, age)
                pred_uncond = self.denoiser(model_input, t, None)
                pred_r = pred_uncond + cfg_scale * (pred_cond - pred_uncond)
            else:
                pred_r = self.denoiser(model_input, t, age)

            # Predicted x1 from current residual prediction
            alpha_bar = self.alphas_cumprod[t].view(-1, 1, 1, 1, 1)
            x1_from_pred = x_t + pred_r  # rough estimate

            # Step to next t using bridge dynamics
            alpha_bar_next = self.alphas_cumprod[t_next_t].view(-1, 1, 1, 1, 1)
            x_t = alpha_bar_next.sqrt() * (
                (1 - alpha_bar_next) * x1_from_pred + (alpha_bar_next * (1 - alpha_bar_next)).sqrt() * x0
            ) + (1 - alpha_bar_next.sqrt()) * x_t

        return x_t
