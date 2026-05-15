import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierFeatures(nn.Module):
    """
    Fixed Fourier features for q in [0,1].

    If n_freqs=0 -> returns q only (B,N,1).
    If n_freqs>=1 -> returns [q, sin(2*pi*f*q), cos(2*pi*f*q)] for f=1..n_freqs.
    """
    def __init__(self, n_freqs: int = 1, include_q: bool = True):
        super().__init__()
        self.n_freqs = int(n_freqs)
        self.include_q = bool(include_q)
        if self.n_freqs > 0:
            freqs = 2.0 * math.pi * torch.arange(1, self.n_freqs + 1).float()  # (F,)
            self.register_buffer("freqs", freqs)
        else:
            self.register_buffer("freqs", torch.empty(0))

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        """
        q: (B,N) expected in [0,1]
        returns: (B,N, 1 + 2*n_freqs) if include_q else (B,N, 2*n_freqs)
        """
        q_ = q.unsqueeze(-1)  # (B,N,1)
        feats = []
        if self.include_q:
            feats.append(q_)
        if self.n_freqs > 0:
            ang = q_ * self.freqs  # (B,N,F)
            feats.append(torch.sin(ang))
            feats.append(torch.cos(ang))
        return torch.cat(feats, dim=-1) if len(feats) > 1 else feats[0]


class MaskedMeanPool(nn.Module):
    """Simple masked mean pooling over tokens."""
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = float(eps)

    def forward(self, x: torch.Tensor, mask_valid: torch.Tensor) -> torch.Tensor:
        """
        x: (B,N,D)
        mask_valid: (B,N) bool, True=valid token
        returns: (B,D)
        """
        m = mask_valid.unsqueeze(-1).to(x.dtype)  # (B,N,1)
        num = (x * m).sum(dim=1)                  # (B,D)
        den = m.sum(dim=1).clamp_min(self.eps)    # (B,1)
        return num / den


class Encoder(nn.Module):
    """
    Simple, consistent encoder:
      - LayerNorm everywhere
      - GELU activation
      - PyTorch TransformerEncoderLayer with norm_first=True, batch_first=True
      - Masked mean pooling (no learned pooling)
      - Optional, *gated* conditioning injection AFTER pooling (default off)

    Inputs:
      x_input: (B,C,N) with q_cap at channel 1 and voltage channels at 2..3.
      token_valid_mask: (B,N) bool True=valid, False=pad
      cond (optional): (B,cond_dim)

    Returns:
      mu: (B,latent_dim)
      logvar: (B,latent_dim)
      coord_feats: (B,N,F) where F = 1 + 2*n_fourier (used features of q)
    """
    def __init__(
        self,
        d_model: int = 2,
        n_layers: int = 2,
        n_heads: int = 4,
        d_ff_mult: float = 2.0,
        dropout: float = 0.0,
        n_fourier: int = 1,
        latent_dim: int = 16,
    ):
        super().__init__()
        d_model = int(d_model)
        n_layers = int(n_layers)
        n_heads = int(n_heads)
        latent_dim = int(latent_dim)

        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads}).")

        self.d_model = d_model
        self.latent_dim = latent_dim

        # q features
        self.ff = FourierFeatures(n_freqs=int(n_fourier), include_q=True)
        ff_dim = 1 + 2 * int(n_fourier)

        # Separate small projections for voltage vs derivatives (your scaling rationale is valid)
        # Keep them simple: Linear -> GELU -> Linear
        d_vol = max(4, d_model // 8)          # small
        d_drv = max(8, d_model // 4)          # slightly larger
        self.vol_proj = nn.Sequential(
            nn.Linear(2, d_vol),
            nn.GELU(),
            nn.Linear(d_vol, d_vol),
        )
        self.drv_proj = nn.Sequential(
            nn.Linear(4, d_drv),
            nn.GELU(),
            nn.Linear(d_drv, d_drv),
        )

        # Token assembly currently uses:
        #   voltage embedding + derivative embedding + Fourier(q_cap)
        in_dim = d_vol + d_drv + ff_dim 
        self.in_proj = nn.Linear(in_dim, d_model)

        # Transformer encoder (tested implementation)
        ff_dim_tr = int(d_ff_mult * d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim_tr,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.norm_final = nn.LayerNorm(d_model)
        self.pool = MaskedMeanPool()


        self.mu = nn.Linear(d_model, latent_dim)
        self.logvar = nn.Linear(d_model, latent_dim)

    def forward(self, x_input: torch.Tensor, token_valid_mask: torch.Tensor, cond: torch.Tensor | None = None):
        if x_input.dim() != 3:
            raise ValueError(f"x_input must be (B,C,N), got {tuple(x_input.shape)}")
        if token_valid_mask.dim() != 2:
            raise ValueError(f"token_valid_mask must be (B,N), got {tuple(token_valid_mask.shape)}")

        B, C, N = x_input.shape
        if token_valid_mask.shape != (B, N):
            raise ValueError(f"token_valid_mask shape {tuple(token_valid_mask.shape)} doesn't match (B,N)=({B},{N})")

        # ----- select channels -----
        q_cap    = x_input[:, 1, :]  # (B,N) in [0,1]
        V_ch     = x_input[:, 2, :]
        V_dis    = x_input[:, 3, :]

        dV_ch    = x_input[:, 4, :]
        dV_dis   = x_input[:, 5, :]

        # d2V_ch   = x_input[:, 6, :]
        # d2V_dis  = x_input[:, 7, :]

        dQdV_ch  = x_input[:, 8, :]
        dQdV_dis = x_input[:, 9, :]

        # h_raw = x_input[:, 10, :] 
        # ----- coordinate features -----
        coord_feats = self.ff(q_cap)  # (B,N,F)

        # ----- token feature assembly -----
        # v_tok = V_ch.unsqueeze(-1)  # (B,N,1)
        v_tok = torch.stack([V_ch, V_dis], dim=-1)  # (B,N,2)
        d_tok = torch.stack([dV_ch, dV_dis, dQdV_ch, dQdV_dis], dim=-1)  # (B,N,4)
        
        vol_emb = self.vol_proj(v_tok)  # (B,N,d_vol)
        drv_emb = self.drv_proj(d_tok)  # (B,N,d_drv)

        tok = torch.cat([vol_emb, drv_emb, coord_feats], dim=-1)  # (B,N,in_dim)

        # zero-out padded tokens before projection (stability)
        tok = tok * token_valid_mask.unsqueeze(-1).to(tok.dtype)

        h = self.in_proj(tok)  # (B,N,d_model)

        # PyTorch transformer expects key_padding_mask True for PAD
        key_padding_mask = ~token_valid_mask  # (B,N)

        h = self.encoder(h, src_key_padding_mask=key_padding_mask)  # (B,N,d_model)
        h = self.norm_final(h)

        pooled = self.pool(h, token_valid_mask)  # (B,d_model)

        mu = self.mu(pooled)
        logvar = self.logvar(pooled)
        return mu, logvar, coord_feats
