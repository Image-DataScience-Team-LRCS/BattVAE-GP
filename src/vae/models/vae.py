# models/vae.py
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Optional

from src.vae.models.encoder import Encoder
from src.vae.models.decoder import Decoder

from src.common.logger.logging import get_logger
from src.common.utils.config_schema import FullConfig

logger = get_logger(__name__)


# ----------------------------- VAE module -----------------------------

class VAE(nn.Module):
    """
    Encoder(x, mask, cond?) -> (mu, logvar, q_fourier)
    z = reparameterize(mu, logvar)
    Decoder(q_fourier, z, cond?) -> (V_ch_norm, V_dis_norm)

    Shapes:
      x: (B, C, N) with C >= 4 (encoder uses channels 1..3)
      mask: (B, N) bool (True = valid)
      cond: (B, cond_dim) or None (if FiLM enabled)
      q_fourier: (B, N, 1 + 2*n_fourier); z: (B, latent_dim)
    """
    def __init__(self, config: FullConfig) -> None:
        super().__init__()
        hp = config.HYPER_PARAMETERS

        # knobs (prefer reading from config if present)
        d_model   = int(hp.d_model)
        n_heads   = int(hp.num_heads)
        n_enc     = int(hp.num_transformer_encoder_layers)
        n_dec     = int(hp.num_transformer_decoder_layers)
        n_fourier = int(hp.n_fourier_encoder)
        latent    = int(hp.latent_dim)
        cond_dim  = int(hp.conditional_vector_dim)
        use_film  = bool(getattr(hp, "use_film_in_encoder", cond_dim > 0))

        # --- Encoder / Decoder ---
        self.encoder = Encoder(
            d_model=d_model,
            n_layers=n_enc,
            n_heads=n_heads,
            d_ff_mult=4,
            n_fourier=n_fourier,
            # use_film_in_encoder=use_film,
            cond_dim=cond_dim,
            latent_dim=latent,
        )
        self.decoder = Decoder(
            latent_dim=latent,
            cond_dim=cond_dim,
            d_model=d_model,
            n_layers=n_dec,
            max_freq=hp.max_freq,
            n_fourier_baseline_decoder=int(hp.n_fourier_baseline_decoder),
            n_fourier_residual_decoder=int(hp.n_fourier_residual_decoder),
            # use_cond_in_residual=(cond_dim > 0),
        )


        # SOH predictor
        self.soh_predictor = nn.Sequential(
            nn.Linear(latent , 32),
            nn.LayerNorm(32),
            nn.ReLU(inplace=True),
            # nn.Dropout(self.dropout),
            nn.Linear(32, 16),
            nn.LayerNorm(16),
            nn.ReLU(inplace=True),
            # nn.Dropout(self.dropout),
            nn.Linear(16, 1),
        )

        # expose a couple of flags
        self.cond_dim = cond_dim
        self.use_film_in_encoder = use_film

        logger.info(f"VAE initialized with {sum(p.numel() for p in self.parameters() if p.requires_grad)} trainable parameters")


    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self,
        x: torch.Tensor,                         # (B, C, N)
        mask: torch.Tensor,                      # (B, N) bool
        conditional_vector: Optional[torch.Tensor] = None,
    ):
        assert x.dim() == 3, f"x must be (B,C,N), got {tuple(x.shape)}"
        B, C, N = x.shape

        mask = mask.to(torch.bool)
        if self.cond_dim > 0:
            if conditional_vector is None:
                raise ValueError(
                    f"conditional_vector is required because conditional_vector_dim={self.cond_dim}."
                )
            if conditional_vector.dim() != 2 or int(conditional_vector.size(1)) != int(self.cond_dim):
                raise ValueError(
                    f"conditional_vector must be (B,{self.cond_dim}), got {tuple(conditional_vector.shape)}."
                )


        # ------------- encode/reparameterization/decode -------------
        mu, logvar, q_fourier = self.encoder(x, mask, conditional_vector)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decoder(q_fourier, z, conditional_vector)  

        soh_pred = self.soh_predictor(mu)

        # soh_pred = torch.full(soh_pred.shape, -1.0, dtype=soh_pred.dtype, device=soh_pred.device)
        return reconstruction, mu, logvar, z, soh_pred

# ------------------------ construction / viz ------------------------

def build_model(config: FullConfig) -> VAE:
    return VAE(config)

def plot_architecture(model: nn.Module, config: FullConfig) -> None:
    from torchviz import make_dot
    try:
        path = Path(config.PATHS.visualization) / "VAE_architecture"
        hp = config.HYPER_PARAMETERS

        B = 2
        C = int(hp.input_channel)
        N = int(hp.input_seq_len)

        x = torch.randn(B, C, N)
        mask = torch.ones(B, N, dtype=torch.bool)
        cond_dim = int(getattr(hp, "conditional_vector_dim", 0))
        cond = torch.zeros(B, cond_dim) if cond_dim > 0 else None

        model.eval()
        out, mu, logvar, z, _ = model(x, mask, cond)

        graph = make_dot((out, mu, logvar), params=dict(model.named_parameters()),
                         show_attrs=True, show_saved=True)
        graph.render(path, format="pdf", cleanup=True)
        logger.info("Architecture saved as VAE_architecture.pdf")
    except Exception as e:
        logger.info(f"plot_architecture error: {e}")
