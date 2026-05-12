from __future__ import annotations
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.vae.training.physics_heads import PhysicsHead


class LossFactory(nn.Module):
    """
    Orchestrates losses:
      - Direct reconstruction on normalized decoder heads
      - KL with free-bits
      - Physics-head supervision on dV/dQ, dQ/dV and H_raw
      - Optional small L1 in volts
    """

    def __init__(
        self,
        # voltage_norm_stats: Dict[str, Any],
        norm_cfg: Dict[str, Any],
        enable_physics_head: bool = True,
        edge_trim_frac: float = 0.02,
        huber_beta: float = 1.0,
        free_bits_delta: float = 0.1,
        w_recon: float = 10.0,
        w_recon_deriv: float = 2.0,
        w_recon_hyst: float = 2.0,
        w_dv_dq: float = 0.00,
        w_d2v_dq2: float = 0.00,
        w_dq_dv: float = 0.00,
        w_volt_phys: float = 1.0,
        huber_beta_per_channel: Optional[list[float]] = None,
        enable_channel_weighting: bool = True,
        channel_weights: Optional[list[float]] = None,
        channel_weight_ema: float = 0.05,
        channel_weight_power: float = 0.5,
        channel_weight_clip_min: float = 0.50,
        channel_weight_clip_max: float = 2.50,
        latent_dependence_cfg: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.enable_physics_head = bool(enable_physics_head)
        self.phys = PhysicsHead(norm_cfg, edge_trim_frac=edge_trim_frac) if self.enable_physics_head else None
        self.huber_beta = float(huber_beta)
        self.free_bits_delta = float(free_bits_delta)
        self.w_recon = float(w_recon)
        # self.w_kl = float(w_kl)
        self.w_recon_deriv = float(w_recon_deriv)
        self.w_recon_hyst = float(w_recon_hyst)
        self.w_dv_dq = float(w_dv_dq)
        self.w_d2v_dq2 = float(w_d2v_dq2)
        self.w_dq_dv = float(w_dq_dv)
        self.w_volt_phys = float(w_volt_phys)

        if huber_beta_per_channel is not None:
            beta_ch = torch.tensor(huber_beta_per_channel, dtype=torch.float32)
            if beta_ch.ndim != 1 or beta_ch.numel() == 0:
                raise ValueError("huber_beta_per_channel must be a non-empty 1D list.")
            self.register_buffer("huber_beta_per_channel", beta_ch)
        else:
            self.register_buffer("huber_beta_per_channel", torch.empty(0, dtype=torch.float32))

        self.enable_channel_weighting = bool(enable_channel_weighting)
        self.channel_weight_ema = float(channel_weight_ema)
        self.channel_weight_power = float(channel_weight_power)
        self.channel_weight_clip_min = float(channel_weight_clip_min)
        self.channel_weight_clip_max = float(channel_weight_clip_max)

        if channel_weights is not None:
            manual = torch.tensor(channel_weights, dtype=torch.float32)
            if manual.ndim != 1 or manual.numel() == 0:
                raise ValueError("channel_weights must be a non-empty 1D list.")
            self.register_buffer("manual_channel_weights", manual)
        else:
            self.register_buffer("manual_channel_weights", torch.empty(0, dtype=torch.float32))

        # EMA of per-channel reconstruction error for adaptive inverse-loss weighting.
        self.register_buffer("running_channel_loss", torch.empty(0, dtype=torch.float32))
        latent_cfg = dict(latent_dependence_cfg or {})
        self.latent_dependence_enabled = bool(latent_cfg.get("enabled", False))
        self.latent_dependence_weight = float(latent_cfg.get("weight", 0.0))
        self.cycle_latent_index = int(latent_cfg.get("cycle_latent_index", 0))
        self.crate_latent_index = int(latent_cfg.get("crate_latent_index", 1))
        self.swap_penalty_weight = float(latent_cfg.get("swap_penalty_weight", 0.25))
        self.min_abs_correlation = float(latent_cfg.get("min_abs_correlation", 0.0))

    # ----------------- helpers -----------------

    @staticmethod
    def _make_voltage_mask(mask: torch.Tensor, B: int, N: int, dtype, channel_idx: Optional[list[int]] = None) -> torch.Tensor:
        """Return (B,2,N) float mask for [V_ch,V_dis] from (B,N) or (B,C,N)."""
        if mask.dim() == 3:
            C = mask.size(1)
            return mask[:, channel_idx, :].to(dtype)  # V_ch, V_dis + derivatives
        raise ValueError(f"Unsupported mask shape {tuple(mask.shape)}")
    
    @staticmethod
    def _normalize_gp_prior_shapes(
        prior_mean: torch.Tensor, prior_var: torch.Tensor, B: int, D: int, device, dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make GP prior stats shape (B,D) and on the right device/dtype.
        Handles (B,D), (D,B), (B,), (D,), or scalars (fallback).
        """
        pm, pv = prior_mean, prior_var

        # Scalars => fallback to zeros/ones (handled by caller)
        if not torch.is_tensor(pm) or not torch.is_tensor(pv):
            return None, None  # signal to fallback

        pm = pm.to(device=device, dtype=dtype, non_blocking=True)
        pv = pv.to(device=device, dtype=dtype, non_blocking=True)

        # Flatten 1D shapes
        if pm.dim() == 1 and pm.numel() == D:  # (D,)
            pm = pm.unsqueeze(0).expand(B, D)
            pv = pv if pv.dim() == 2 else pv.unsqueeze(0).expand(B, D)
        elif pm.dim() == 1 and pm.numel() == B:  # (B,)
            pm = pm.unsqueeze(1).expand(B, D)
            pv = pv if pv.dim() == 2 else pv.unsqueeze(1).expand(B, D)
        elif pm.dim() == 0:
            return None, None

        # If transposed (D,B), fix it
        if pm.dim() == 2 and pm.shape == (D, B):
            pm = pm.transpose(0, 1).contiguous()
            pv = pv.transpose(0, 1).contiguous()

        # Final sanity: broadcast or clamp
        if pm.shape != (B, D):
            # Try broadcast if one dimension is 1
            if pm.shape[0] == 1 and pm.shape[1] == D:
                pm = pm.expand(B, D)
                pv = pv.expand(B, D)
            elif pm.shape[1] == 1 and pm.shape[0] == B:
                pm = pm.expand(B, D)
                pv = pv.expand(B, D)
            else:
                raise RuntimeError(f"Cannot normalize GP prior shapes: got mean {tuple(pm.shape)}, expected (B,D) with B={B}, D={D}")

        return pm, pv

    @staticmethod
    def _kl_per_dim_with_gp_prior(
        mu: torch.Tensor, logvar: torch.Tensor,
        prior_mean: Optional[torch.Tensor], prior_var: Optional[torch.Tensor],
        free_bits_delta: float = 0.0, eps: float = 1e-8
    ) -> torch.Tensor:
        """
        KL per latent dimension. If prior_mean/var provided (as tensors),
        compute KL(q||N(prior_mean, prior_var)); else KL(q||N(0,1)).
        """
        B, D = mu.shape
        if torch.is_tensor(prior_mean) and torch.is_tensor(prior_var):
            pm, pv = LossFactory._normalize_gp_prior_shapes(prior_mean, prior_var, B, D, mu.device, mu.dtype)
        else:
            pm, pv = None, None

        if pm is not None and pv is not None:
            pv = pv.clamp_min(eps)
            # KL(q || p) where q = N(mu, diag(exp(logvar))), p = N(pm, diag(pv))
            kl_bd = 0.5 * (
                (logvar.exp() / pv) +
                (mu - pm).pow(2) / pv -
                1.0 + pv.log() - logvar
            )
        else:
            # KL(q || N(0, I))
            kl_bd = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())

        # if free_bits_delta > 0:
        #     kl = torch.maximum(kl, free_bits_delta * torch.ones_like(kl))  # free bits per-dim
        # return kl  # (B,D)
    
        # --- batch-mean per dimension, then apply free-bits floor per dim ---
        kl_dim_mean = kl_bd.mean(dim=0)  # (D,) for logging
        if free_bits_delta > 0:
            floor = torch.full_like(kl_dim_mean, free_bits_delta)
            kl_dim_capped = torch.maximum(kl_dim_mean, floor)  # (D,)
            kl_loss = kl_dim_capped.sum()  # scalar
        else:
            kl_loss = kl_dim_mean.sum()

        return kl_loss, kl_dim_mean


    def _masked_huber_1d(self, a: torch.Tensor, b: torch.Tensor, mask_1d: torch.Tensor) -> torch.Tensor:
        """Huber over time with (B,N) mask -> mean over batch of masked means."""
        per = F.smooth_l1_loss(a, b, beta=self.huber_beta, reduction="none")  # (B,N)
        m = mask_1d.to(per.dtype)
        num = (per * m).sum(dim=1)                 # (B,)
        den = m.sum(dim=1).clamp_min(1.0)          # (B,)
        return (num / den).mean()

    @staticmethod
    def _masked_l1_1d(a: torch.Tensor, b: torch.Tensor, mask_1d: torch.Tensor) -> torch.Tensor:
        """L1 over time with (B,N) mask -> mean over batch of masked means."""
        per = torch.abs(a - b)
        m = mask_1d.to(per.dtype)
        num = (per * m).sum(dim=1)
        den = m.sum(dim=1).clamp_min(1.0)
        return (num / den).mean()

    def _resolve_huber_beta(
        self,
        n_channels: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Return Huber beta broadcast shape:
          - scalar beta -> (1,1,1)
          - per-channel beta -> (1,C,1)
        """
        if self.huber_beta_per_channel.numel() == 0:
            beta = torch.tensor(float(self.huber_beta), dtype=dtype, device=device).view(1, 1, 1)
            return beta.clamp_min(1e-8)

        if int(self.huber_beta_per_channel.numel()) != int(n_channels):
            raise ValueError(
                f"huber_beta_per_channel has {int(self.huber_beta_per_channel.numel())} values, "
                f"but reconstruction currently has {n_channels} channels."
            )
        beta = self.huber_beta_per_channel.to(device=device, dtype=dtype).view(1, n_channels, 1)
        return beta.clamp_min(1e-8)

    def _compute_channel_weights(
        self,
        per_channel_loss: torch.Tensor,   # (B, C_sel)
        training: bool,
    ) -> torch.Tensor:
        """
        Return normalized channel weights of shape (C_sel,).
        - Manual mode: uses channel_weights passed in ctor.
        - Adaptive mode: inverse-loss weights from EMA of channel errors.
        """
        C = int(per_channel_loss.size(1))
        dtype = per_channel_loss.dtype
        device = per_channel_loss.device

        if not self.enable_channel_weighting:
            return torch.ones(C, dtype=dtype, device=device)

        if self.manual_channel_weights.numel() > 0:
            if int(self.manual_channel_weights.numel()) != C:
                raise ValueError(
                    f"manual channel_weights has {int(self.manual_channel_weights.numel())} values, "
                    f"but current reconstruction has {C} channels."
                )
            w = self.manual_channel_weights.to(device=device, dtype=dtype)
            return w / w.mean().clamp_min(1e-8)

        batch_channel_loss = per_channel_loss.detach().mean(dim=0).clamp_min(1e-8)  # (C,)

        if self.running_channel_loss.numel() != C:
            self.running_channel_loss = batch_channel_loss.to(dtype=torch.float32)
        elif training:
            ema = self.channel_weight_ema
            self.running_channel_loss = (
                (1.0 - ema) * self.running_channel_loss + ema * batch_channel_loss.to(dtype=torch.float32)
            )

        stats = self.running_channel_loss.to(device=device, dtype=dtype).clamp_min(1e-8)
        med = torch.median(stats)
        # Emphasize harder channels: larger running loss -> larger weight.
        w = (stats / med).pow(self.channel_weight_power)
        w = torch.clamp(w, min=self.channel_weight_clip_min, max=self.channel_weight_clip_max)
        w = w / w.mean().clamp_min(1e-8)
        return w

    @staticmethod
    def _safe_centered(x: torch.Tensor) -> torch.Tensor:
        return x - x.mean()

    @staticmethod
    def _safe_abs_corr(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        x_c = LossFactory._safe_centered(x.float())
        y_c = LossFactory._safe_centered(y.float())
        x_std = torch.sqrt(torch.mean(x_c.pow(2))).clamp_min(eps)
        y_std = torch.sqrt(torch.mean(y_c.pow(2))).clamp_min(eps)
        return torch.mean((x_c / x_std) * (y_c / y_std)).abs()

    def _latent_dependence_loss(
        self,
        mu: torch.Tensor,
        norm_cycle_numbers: Optional[torch.Tensor],
        charging_rate: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        zero = mu.new_tensor(0.0)
        if (
            not self.latent_dependence_enabled
            or self.latent_dependence_weight <= 0.0
            or norm_cycle_numbers is None
            or charging_rate is None
        ):
            return zero, {
                "cycle_corr": zero,
                "crate_corr": zero,
                "cycle_swap_corr": zero,
                "crate_swap_corr": zero,
            }

        latent_dim = int(mu.size(1))
        if self.cycle_latent_index >= latent_dim or self.crate_latent_index >= latent_dim:
            raise ValueError(
                f"Latent dependence indices ({self.cycle_latent_index}, {self.crate_latent_index}) "
                f"require latent_dim>{max(self.cycle_latent_index, self.crate_latent_index)}, got {latent_dim}."
            )

        z_cycle = mu[:, self.cycle_latent_index]
        z_crate = mu[:, self.crate_latent_index]
        cycle_target = norm_cycle_numbers.to(device=mu.device, dtype=mu.dtype)
        crate_target = charging_rate.to(device=mu.device, dtype=mu.dtype)

        cycle_corr = self._safe_abs_corr(z_cycle, cycle_target)
        crate_corr = self._safe_abs_corr(z_crate, crate_target)
        cycle_swap_corr = self._safe_abs_corr(z_cycle, crate_target)
        crate_swap_corr = self._safe_abs_corr(z_crate, cycle_target)

        cycle_term = (1.0 - cycle_corr).clamp_min(0.0)
        crate_term = (1.0 - crate_corr).clamp_min(0.0)
        if self.min_abs_correlation > 0.0:
            cycle_term = (self.min_abs_correlation - cycle_corr).clamp_min(0.0)
            crate_term = (self.min_abs_correlation - crate_corr).clamp_min(0.0)

        swap_term = self.swap_penalty_weight * (cycle_swap_corr + crate_swap_corr)
        loss = self.latent_dependence_weight * (cycle_term + crate_term + swap_term)
        return loss, {
            "cycle_corr": cycle_corr.detach(),
            "crate_corr": crate_corr.detach(),
            "cycle_swap_corr": cycle_swap_corr.detach(),
            "crate_swap_corr": crate_swap_corr.detach(),
        }

    # ----------------- forward -----------------

    def forward(
        self,
        reconstruction: torch.Tensor,   # (B,6,N) normalized decoder heads
        input_tensor: torch.Tensor,     # (B,12,N) normalized; [2]=V_ch, [3]=V_dis
        mu: torch.Tensor,               # (B,D)
        logvar: torch.Tensor,           # (B,D)
        z: torch.Tensor,                # (B,D) (unused; API compat)
        mask: torch.Tensor,             # (B,N) or (B,C,N)
        prior_mean: Optional[torch.Tensor] = None,  # unused
        prior_var: Optional[torch.Tensor] = None,   # unused
        training: Optional[bool] = None,            # unused
        q0_Ah: Optional[torch.Tensor] = None,       # (B,)  
        norm_cycle_numbers: Optional[torch.Tensor] = None,
        charging_rate: Optional[torch.Tensor] = None,
    ):
        """
        Returns (recon_loss, kl_loss, kl_per_dim_mean, aux_dict)
        where aux_dict includes:
          - recon_loss_per_sample: (B,) mean over selected channels per sample
          - recon_loss_per_sample_per_channel: (B,C_sel) per-sample per-channel
        """
        B, C, N = input_tensor.shape
        assert C >= 4, "input_tensor must include q_cap, V_ch, V_dis in channels [1..3]"
        if self.enable_physics_head:
            if q0_Ah is None:
                raise ValueError("q0_Ah is required when physics_head.enabled is true.")

            q_cap = input_tensor[:, 1, :]
            Vch_true = input_tensor[:, 2, :]
            Vds_true = input_tensor[:, 3, :]
            Vch_pred = reconstruction[:, 0, :]
            Vds_pred = reconstruction[:, 1, :]

            packs = self.phys.build_targets(
                Vch_pred_norm=Vch_pred,
                Vds_pred_norm=Vds_pred,
                Vch_true_norm=Vch_true.detach(),
                Vds_true_norm=Vds_true.detach(),
                q_cap=q_cap,
                q0_Ah=q0_Ah,
                mask_tokens=mask,
            )
            pred, true, m = packs["pred"], packs["true"], packs["masks"]

            target = torch.stack(
                [
                    true["Vch_norm"],
                    true["Vdis_norm"],
                    true["dVdQ_ch_norm"],
                    true["dVdQ_dis_norm"],
                    true["dQdV_ch_norm"],
                    true["dQdV_dis_norm"],
                ],
                dim=1,
            )
            m_volt = torch.stack(
                [
                    m["V_ch"],
                    m["V_dis"],
                    m["dVdQ_ch"],
                    m["dVdQ_dis"],
                    m["dQdV_ch"],
                    m["dQdV_dis"],
                ],
                dim=1,
            ).to(reconstruction.dtype)
        else:
            if C < 10:
                raise ValueError(
                    "input_tensor must include V_ch/V_dis, dVdQ_ch/dis, and dQdV_ch/dis channels "
                    "when physics_head.enabled is false."
                )
            target = torch.stack(
                [
                    input_tensor[:, 2, :],
                    input_tensor[:, 3, :],
                    input_tensor[:, 4, :],
                    input_tensor[:, 5, :],
                    input_tensor[:, 8, :],
                    input_tensor[:, 9, :],
                ],
                dim=1,
            )
            if mask.dim() == 3:
                m_volt = torch.stack(
                    [
                        mask[:, 2, :],
                        mask[:, 3, :],
                        mask[:, 4, :],
                        mask[:, 5, :],
                        mask[:, 8, :],
                        mask[:, 9, :],
                    ],
                    dim=1,
                ).to(reconstruction.dtype)
            elif mask.dim() == 2:
                m_volt = mask.unsqueeze(1).expand(-1, 6, -1).to(reconstruction.dtype)
            else:
                raise ValueError(f"Unsupported mask shape {tuple(mask.shape)}")
        assert reconstruction.shape == target.shape == (B, 6, N), (
            f"reconstruction and target must have the same shape (B,6,N), "
            f"got recon {tuple(reconstruction.shape)}, target {tuple(target.shape)}"
        )


        diff = target - reconstruction
        
        ### MSE
        masked_sq_err = diff.pow(2) * m_volt  # (B,C,N) masked squared error
        num = masked_sq_err.sum(dim=(1, 2))            # (B,)
        den = m_volt.sum(dim=(1, 2))                   # (B,)
        assert torch.all(den > 0), "No valid voltage tokens in at least one sample."
        recon_loss_per_sample_unweighted = num / den   # (B,)

        # Per-sample, per-channel reconstruction loss (B, C_sel).
        num_ch = masked_sq_err.sum(dim=2)              # (B,C_sel)
        den_ch = m_volt.sum(dim=2).clamp_min(1.0)      # (B,C_sel)
        recon_loss_per_sample_per_channel = num_ch / den_ch   # (B,C_sel)

        alpha = torch.tensor([0.5, 0.5, 1.8, 1.8, 1.8, 1.8], device=reconstruction.device, dtype=reconstruction.dtype)
        # alpha = torch.tensor([0.5, 0.5], device=reconstruction.device, dtype=reconstruction.dtype)

        alpha = alpha / alpha.mean().clamp_min(1e-8)  # (C_sel,) normalized to mean 1.0
        # print(recon_loss_per_sample_per_channel.dim(), alpha.dim())

        recon_loss_per_sample_per_channel_weighted = recon_loss_per_sample_per_channel * alpha.unsqueeze(0)  # (B,C_sel)

        recon_loss_per_sample = recon_loss_per_sample_per_channel_weighted.mean(dim=1)  # (B,)

        ## Huber        
        # abs_diff = torch.abs(diff)
        # beta = self._resolve_huber_beta(
        #     n_channels=target.size(1),
        #     dtype=reconstruction.dtype,
        #     device=reconstruction.device,
        # )  # (1,1,1) or (1,C_sel,1)

        # quadratic = 0.5 * (abs_diff.pow(2)) / beta
        # linear = abs_diff - 0.5 * beta

        # hubber = torch.where(abs_diff <= beta, quadratic, linear)
        # masked_hubber = hubber * m_volt

        # # Total per-sample reconstruction loss (unweighted; averaged over channels and tokens)
        # num = masked_hubber.sum(dim=(1, 2))            # (B,)
        # den = m_volt.sum(dim=(1, 2))                   # (B,)
        # assert torch.all(den > 0), "No valid voltage tokens in at least one sample."
        # recon_loss_per_sample_unweighted = num / den   # (B,)

        # Per-sample, per-channel reconstruction loss (B, C_sel).
        # num_ch = masked_hubber.sum(dim=2)              # (B,C_sel)
        # den_ch = m_volt.sum(dim=2).clamp_min(1.0)      # (B,C_sel)
        # recon_loss_per_sample_per_channel = num_ch / den_ch

        # Channel-across weighting:
        # recon(sample) = weighted mean over channels of channel losses.
        # use_training = bool(training) if training is not None else self.training
        # channel_weights = self._compute_channel_weights(
        #     recon_loss_per_sample_per_channel,
        #     training=use_training,
        # )  # (C_sel,)
        # ch_w_den = channel_weights.sum().clamp_min(1e-8)
        # recon_loss_per_sample = (
        #     (recon_loss_per_sample_per_channel * channel_weights.unsqueeze(0)).sum(dim=1) / ch_w_den
        # )  # (B,)
        # recon_loss = recon_loss_per_sample.mean()

        # # Relative reconstruction loss: normalize by per-sample target energy.
        # # This reduces bias toward high-amplitude (typically early-cycle) samples.
        # target_energy_per_channel = torch.sqrt(
        #     ((target.pow(2) * m_volt).sum(dim=2) / den_ch).clamp_min(0.0)
        # )  # (B,C_sel)
        # target_energy = (
        #     (target_energy_per_channel * channel_weights.unsqueeze(0)).sum(dim=1) / ch_w_den
        # ).clamp_min(1e-3)  # (B,)
        # recon_loss_per_sample_relative = recon_loss_per_sample / target_energy

        # Scalar used for optimization: mean reconstruction loss over all samples and channels.
        recon_loss = recon_loss_per_sample.mean()

        # 3) KL with free-bits
        # kl_per_dim = self._kl_freebits(mu, logvar, self.free_bits_delta)  # (B,D)
        # kl_loss = kl_per_dim.sum(dim=1).mean()

        # --- KL to GP prior if available; else to N(0,1) ---
        kl_loss, kl_per_dim = self._kl_per_dim_with_gp_prior(mu, logvar, prior_mean, prior_var, free_bits_delta=self.free_bits_delta)
        latent_dependence_loss, latent_dependence_stats = self._latent_dependence_loss(
            mu=mu,
            norm_cycle_numbers=norm_cycle_numbers,
            charging_rate=charging_rate,
        )
        # kl_loss = kl_per_dim.sum(dim=1).mean()
        # kl_per_dim_mean = kl_per_dim.mean(dim=0)

        # 4) Physics-head losses.
        # d2V/dQ2 is intentionally excluded.
        if self.enable_physics_head:
            dv_dq_loss = 0.5 * (
                self._masked_huber_1d(pred["dVdQ_ch_norm"], true["dVdQ_ch_norm"], m["dVdQ_ch"]) +
                self._masked_huber_1d(pred["dVdQ_dis_norm"], true["dVdQ_dis_norm"], m["dVdQ_dis"])
            )
            dq_dv_loss = 0.5 * (
                self._masked_huber_1d(pred["dQdV_ch_norm"], true["dQdV_ch_norm"], m["dQdV_ch"]) +
                self._masked_huber_1d(pred["dQdV_dis_norm"], true["dQdV_dis_norm"], m["dQdV_dis"])
            )
            recon_hyst = self._masked_huber_1d(pred["H_raw_norm"], true["H_raw_norm"], m["H_raw"])

            if self.w_volt_phys > 0:
                volt_ch = self._masked_l1_1d(pred["Vch_V"], true["Vch_V"], m["V_ch"])
                volt_ds = self._masked_l1_1d(pred["Vdis_V"], true["Vdis_V"], m["V_dis"])
                volt_phys = 0.5 * (volt_ch + volt_ds)
            else:
                volt_phys = reconstruction.new_tensor(0.0)
        else:
            dv_dq_loss = reconstruction.new_tensor(0.0)
            dq_dv_loss = reconstruction.new_tensor(0.0)
            recon_hyst = reconstruction.new_tensor(0.0)
            volt_phys = reconstruction.new_tensor(0.0)

        dv_dq_weight = self.w_dv_dq if self.w_dv_dq > 0 else self.w_recon_deriv
        dq_dv_weight = self.w_dq_dv if self.w_dq_dv > 0 else self.w_recon_deriv

        total = (
            recon_loss
            + dv_dq_weight * dv_dq_loss
            + dq_dv_weight * dq_dv_loss
            + self.w_recon_hyst * recon_hyst
            + self.w_volt_phys * volt_phys
        )

        aux = {
            "total": total,
            "recon_loss_per_sample": recon_loss_per_sample,  # (B,)
            "recon_loss_per_sample_unweighted": recon_loss_per_sample_unweighted,  # (B,)
            "recon_loss_per_sample_per_channel": recon_loss_per_sample_per_channel,  # (B,C_sel)
            "latent_dependence_loss": latent_dependence_loss.detach(),
            **latent_dependence_stats,
            "dv_dq": dv_dq_loss.detach(),
            "dq_dv": dq_dv_loss.detach(),
            "volt_phys": volt_phys.detach(),
            "recon_hyst": recon_hyst.detach(),
            # "recon_loss_per_sample_relative": recon_loss_per_sample_relative,  # (B,)
            # "target_energy_per_sample": target_energy,  # (B,)
            # "channel_weights": channel_weights.detach(),  # (C_sel,)
        }
        
        # print("Reconstruction Loss:", recon_loss.item())
        # print("KL Loss:", kl_loss.item())
        # print("dv/dq Loss:", dv_dq_loss.item())
        # print("d2v/dq2 Loss:", d2v_dq2_loss.item())
        # print("dq/dv Loss:", dq_dv_loss.item())
        # print("Volt Phys Loss:", volt_phys.item())
        # print("kl_per_dim:", kl_per_dim)
        
        return total, kl_loss, kl_per_dim, aux
