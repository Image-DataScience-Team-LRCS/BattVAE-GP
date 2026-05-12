# decoder.py
# Minimal, condition-free decoder for a vanilla VAE with:
#   x_hat(q) = baseline(q) + residual(q, z)
#
# Requirements implemented:
# 1) Condition-free: no FiLM/conditioning used
# 2) Baseline can be frozen later via freeze_baseline()
# 3) Only the first channel of q_cap_fourier is used (phase==0)
# 4) Reconstruction is baseline + residual

import math
from typing import Optional

import torch
import torch.nn as nn


class FourierFeatures(nn.Module):
    """
    Fixed Fourier features for a 1D coordinate q.

    This is NOT trainable; it expands q into [q, sin(2^k*pi*q), cos(2^k*pi*q)].
    Using fewer frequencies reduces the decoder's ability to fit everything from q alone.
    """
    def __init__(self, n_freq: int = 8, max_freq: float = 64.0):
        super().__init__()
        self.n_freq = int(n_freq)
        self.max_freq = float(max_freq)

        if self.n_freq > 0:
            # Log-spaced frequencies from 1 to max_freq
            freqs = torch.logspace(0, math.log10(self.max_freq), steps=self.n_freq)
            self.register_buffer("freqs", freqs, persistent=False)
        else:
            self.register_buffer("freqs", torch.empty(0), persistent=False)

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        # q: (B, N, 1)
        if self.n_freq == 0:
            return q

        # (B, N, n_freq)
        angles = math.pi * q * self.freqs.view(1, 1, -1)
        return torch.cat([q, torch.sin(angles), torch.cos(angles)], dim=-1)


class FFNBlock(nn.Module):
    """
    Transformer-style FFN block without attention:
      x <- x + MLP(LayerNorm(x))
    """
    def __init__(self, d_model: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(d_model * mlp_ratio)
        self.ln = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(self.ln(x))


class BaselineNet(nn.Module):
    """
    baseline(q) -> base curve (no z)
    Kept deliberately simple to represent the global mean trend.
    """
    def __init__(
        self,
        d_model: int,
        out_dim: int,
        n_layers: int = 2,
        n_fourier: int = 8,
        max_freq: float = 64.0,
        dropout: float = 0.0,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.ff = FourierFeatures(n_freq=n_fourier, max_freq=max_freq)
        q_feat_dim = 1 + 2 * n_fourier if n_fourier > 0 else 1

        self.in_proj = nn.Linear(q_feat_dim, d_model)
        self.blocks = nn.ModuleList([FFNBlock(d_model, mlp_ratio=mlp_ratio, dropout=dropout) for _ in range(n_layers)])
        self.head = nn.Linear(d_model, out_dim)

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        # q: (B, N, 1)
        h = self.in_proj(self.ff(q))
        for blk in self.blocks:
            h = blk(h)
        return self.head(h)  # (B, N, out_dim)


class ResidualNet(nn.Module):
    """
    residual(q, z, cond) -> residual curve
    """
    def __init__(
        self,
        z_dim: int,
        d_model: int,
        out_dim: int,
        cond_dim: int = 0,
        use_cond: bool = False,
        cond_init_scale: float = 0.25,
        n_layers: int = 2,
        n_fourier: int = 8,
        max_freq: float = 64.0,
        dropout: float = 0.0,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.ff = FourierFeatures(n_freq=n_fourier, max_freq=max_freq)
        q_feat_dim = 1 + 2 * n_fourier if n_fourier > 0 else 1

        self.q_proj = nn.Linear(q_feat_dim, d_model)
        self.z_proj = nn.Linear(z_dim, d_model)
        self.use_cond = bool(use_cond) and (cond_dim > 0)
        if self.use_cond:
            self.cond_proj = nn.Sequential(
                nn.Linear(cond_dim, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, d_model),
            )
            self.cond_scale = nn.Parameter(torch.tensor(float(cond_init_scale)))
        else:
            self.cond_proj = None
            self.cond_scale = None
        self.blocks = nn.ModuleList([FFNBlock(d_model, mlp_ratio=mlp_ratio, dropout=dropout) for _ in range(n_layers)])
        self.head = nn.Linear(d_model, out_dim)

    def forward(self, q: torch.Tensor, z: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        # q: (B, N, 1), z: (B, z_dim), cond: (B, cond_dim)
        hq = self.q_proj(self.ff(q))                 # (B, N, d)
        hz = self.z_proj(z).unsqueeze(1)             # (B, 1, d)
        h = hq + hz                                  # broadcast over N

        if self.use_cond:
            if cond is None:
                raise ValueError("Decoder residual conditioning is enabled but cond is None")
            hc = self.cond_proj(cond).unsqueeze(1)   # (B, 1, d)
            h = h + self.cond_scale * hc

        for blk in self.blocks:
            h = blk(h)

        return self.head(h)                          # (B, N, out_dim)


class Decoder(nn.Module):
    """
    Condition-free decoder: x_hat = baseline(q) + residual(q, z)

    Notes:
    - cond is accepted for interface compatibility but ignored.
    - Use freeze_baseline() at your chosen epoch to stop baseline updates.
    """
    def __init__(
        self,
        latent_dim: int,
        cond_dim: int,
        d_model: int,
        n_layers: int,
        n_fourier_baseline_decoder: int,
        n_fourier_residual_decoder: int,
        max_freq: float,
        dropout: float = 0.0,
        mlp_ratio: float = 2.0,
        n_vol: int = 2,
        n_drv: int = 4,
        use_cond_in_residual: bool = False,
        cond_init_scale: float = 0.01,
        out_activation: str = "none",  # "none" | "sigmoid" | "tanh"
    ):
        super().__init__()
        self.n_vol = int(n_vol)
        self.n_drv = int(n_drv)
        self.out_dim = self.n_vol + self.n_drv

        self.out_activation = out_activation.lower().strip()
        assert self.out_activation in {"none", "sigmoid", "tanh"}

        self.baseline = BaselineNet(
            d_model=d_model,
            out_dim=self.out_dim,
            n_layers=max(1, n_layers),
            n_fourier=n_fourier_baseline_decoder,
            max_freq=max_freq,
            dropout=dropout,
            mlp_ratio=mlp_ratio,
        )
        self.residual = ResidualNet(
            z_dim=latent_dim,
            d_model=d_model,
            out_dim=self.out_dim,
            cond_dim=cond_dim,
            use_cond=use_cond_in_residual,
            cond_init_scale=cond_init_scale,
            n_layers=max(1, n_layers),
            n_fourier=n_fourier_residual_decoder,
            max_freq=max_freq,
            dropout=dropout,
            mlp_ratio=mlp_ratio,
        )

        # Tracks whether baseline is frozen and whether to compute it under no_grad.
        self._baseline_frozen: bool = False
        self._baseline_no_grad: bool = True

    def freeze_baseline(self, use_no_grad: bool = True) -> None:
        """
        Freeze baseline parameters so only the residual network trains.
        Call this at your chosen epoch.

        use_no_grad=True also prevents autograd graph construction for baseline forward.
        """
        self._baseline_frozen = True
        self._baseline_no_grad = bool(use_no_grad)
        for p in self.baseline.parameters():
            p.requires_grad_(False)

    def unfreeze_baseline(self) -> None:
        """Enable baseline training again (mostly useful for debugging)."""
        self._baseline_frozen = False
        self._baseline_no_grad = False
        for p in self.baseline.parameters():
            p.requires_grad_(True)

    def forward(
        self,
        q_cap_fourier: torch.Tensor,
        z: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        q_cap_fourier: (B, N, K). Only q_cap_fourier[..., :1] is used.
        z:            (B, z_dim)
        returns:      (B, C, N) where C = n_vol + n_drv
        """
        q = q_cap_fourier[..., :1]  # requirement #3: phase==0 / first channel only

        if self._baseline_frozen and self._baseline_no_grad:
            with torch.no_grad():
                base = self.baseline(q)     # (B, N, C)
        else:
            base = self.baseline(q)

        res = self.residual(q, z, cond)     # (B, N, C)

        y = base + res                      # requirement #4: baseline + residual

        # Optional output activation (apply only to voltage channel(s), keep derivatives unconstrained)
        if self.out_activation != "none" and self.n_vol > 0:
            vol = y[..., : self.n_vol]
            drv = y[..., self.n_vol :]

            if self.out_activation == "sigmoid":
                vol = torch.sigmoid(vol)
            elif self.out_activation == "tanh":
                vol = torch.tanh(vol)

            y = torch.cat([vol, drv], dim=-1)

        return y.permute(0, 2, 1)           # (B, C, N)
