# models/vae.py
import torch
import torch.nn as nn

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

        # --- Encoder / Decoder ---
        self.encoder = Encoder(
            d_model=d_model,
            n_layers=n_enc,
            n_heads=n_heads,
            d_ff_mult=4,
            n_fourier=n_fourier,
            latent_dim=latent,
        )
        self.decoder = Decoder(
            latent_dim=latent,
            d_model=d_model,
            n_layers=n_dec,
            max_freq=hp.max_freq,
            n_fourier_baseline_decoder=int(hp.n_fourier_baseline_decoder),
            n_fourier_residual_decoder=int(hp.n_fourier_residual_decoder),
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
    ):
        assert x.dim() == 3, f"x must be (B,C,N), got {tuple(x.shape)}"
        B, C, N = x.shape

        mask = mask.to(torch.bool)

        # ------------- encode/reparameterization/decode -------------
        mu, logvar, q_fourier = self.encoder(x, mask)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decoder(q_fourier, z)  

        soh_pred = self.soh_predictor(mu)

        return reconstruction, mu, logvar, z, soh_pred

# ------------------------ construction / viz ------------------------

def build_model(config: FullConfig) -> VAE:
    return VAE(config)
