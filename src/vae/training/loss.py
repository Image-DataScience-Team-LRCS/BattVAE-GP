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
        norm_cfg: Dict[str, Any],
        norm_stats: Optional[Dict[str, Any]] = None,
        enable_physics_head: bool = True,
        edge_trim_frac: float = 0.02,
        free_bits_delta: float = 0.05,
        w_recon: float = 1.0,
        w_recon_deriv: float = 1.0,
        w_recon_hyst: float = 0.25,
        w_dv_dq: float = 0.25,
        w_d2v_dq2: float = 0.00,
        w_dq_dv: float = 0.05,
        w_volt_phys: float = 0.10,
    ):
        super().__init__()
        self.enable_physics_head = bool(enable_physics_head)
        self.phys = (
            PhysicsHead(
                norm_cfg=norm_cfg,
                norm_stats=norm_stats,
                edge_trim_frac=edge_trim_frac,
            )
            if self.enable_physics_head
            else None
        )
        self.free_bits_delta = float(free_bits_delta)
        self.w_recon = float(w_recon)
        # self.w_kl = float(w_kl)
        self.w_recon_deriv = float(w_recon_deriv)
        self.w_recon_hyst = float(w_recon_hyst)
        self.w_dv_dq = float(w_dv_dq)
        self.w_d2v_dq2 = float(w_d2v_dq2)
        self.w_dq_dv = float(w_dq_dv)
        self.w_volt_phys = float(w_volt_phys)




    # ----------------- helpers -----------------

    @staticmethod
    def _make_voltage_mask(mask: torch.Tensor, B: int, N: int, dtype, channel_idx: Optional[list[int]] = None) -> torch.Tensor:
        """Return (B,2,N) float mask for [V_ch,V_dis] from (B,N) or (B,C,N)."""
        if mask.dim() == 3:
            C = mask.size(1)
            return mask[:, channel_idx, :].to(dtype)  # V_ch, V_dis + derivatives
        raise ValueError(f"Unsupported mask shape {tuple(mask.shape)}")
    


    @staticmethod
    def _kl_per_dim(
        mu: torch.Tensor, 
        logvar: torch.Tensor,
        free_bits_delta: float = 0.0, 
    ) -> torch.Tensor:


        kl_bd = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())

        kl_dim_mean = kl_bd.mean(dim=0)  # (D,) for logging
        if free_bits_delta > 0:
            floor = torch.full_like(kl_dim_mean, free_bits_delta)
            kl_dim_capped = torch.maximum(kl_dim_mean, floor)  # (D,)
            kl_loss = kl_dim_capped.sum()  # scalar
        else:
            kl_loss = kl_dim_mean.sum()

        return kl_loss, kl_dim_mean

    @staticmethod
    def _masked_huber_1d(pred, target, mask, delta=1.0):
        mask = mask.to(pred.dtype)
        loss = F.huber_loss(pred, target, delta=delta, reduction="none")
        return (loss * mask).sum() / mask.sum().clamp_min(1.0)


    @staticmethod
    def _masked_l1_1d(pred, target, mask):
        mask = mask.to(pred.dtype)
        loss = torch.abs(pred - target)
        return (loss * mask).sum() / mask.sum().clamp_min(1.0)


    @staticmethod
    def _masked_mse_1d(pred, target, mask):
        mask = mask.to(pred.dtype)
        loss = (pred - target).pow(2)
        return (loss * mask).sum() / mask.sum().clamp_min(1.0)



    # ----------------- forward -----------------

    def forward(
        self,
        reconstruction: torch.Tensor,   # (B,6,N) normalized decoder heads
        input_tensor: torch.Tensor,     # (B,12,N) normalized; [2]=V_ch, [3]=V_dis
        mu: torch.Tensor,               # (B,D)
        logvar: torch.Tensor,           # (B,D)
        z: torch.Tensor,                # (B,D) (unused; API compat)
        mask: torch.Tensor,             # (B,N) or (B,C,N)
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
        assert C >= 11, "input_tensor must include q_cap, V_ch, V_dis and H_raw in channels [1..10]"
        if mask.dim() != 3 or mask.size(1) <= 10:
            raise ValueError("mask must be channel-aware with an H_raw mask at channel 10 for hysteresis loss.")
        
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

        alpha = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], device=reconstruction.device, dtype=reconstruction.dtype)

        alpha = alpha / alpha.mean().clamp_min(1e-8)  # (C_sel,) normalized to mean 1.0

        recon_loss_per_sample_per_channel_weighted = recon_loss_per_sample_per_channel * alpha.unsqueeze(0)  # (B,C_sel)

        recon_loss_per_sample = recon_loss_per_sample_per_channel_weighted.mean(dim=1)  # (B,)


        recon_loss = recon_loss_per_sample.mean()


        kl_loss, kl_per_dim = self._kl_per_dim(mu, logvar, free_bits_delta=self.free_bits_delta)


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
            h_target = input_tensor[:, 10, :]
            h_mask = mask[:, 10, :].bool() & m["H_raw"]
            recon_hyst = self._masked_mse_1d(pred["H_raw_norm"], h_target, h_mask)

            if self.w_volt_phys > 0:
                volt_ch = self._masked_l1_1d(pred["Vch_V"], true["Vch_V"], m["V_ch"])
                volt_ds = self._masked_l1_1d(pred["Vdis_V"], true["Vdis_V"], m["V_dis"])
                volt_phys = 0.5 * (volt_ch + volt_ds)
            else:
                volt_phys = reconstruction.new_tensor(0.0)
        else:
            dv_dq_loss = reconstruction.new_tensor(0.0)
            dq_dv_loss = reconstruction.new_tensor(0.0)
            volt_phys = reconstruction.new_tensor(0.0)

            h_pred = reconstruction[:, 0, :] - reconstruction[:, 1, :]
            h_target = input_tensor[:, 10, :]
            h_mask = mask[:, 10, :].bool()
            recon_hyst = self._masked_mse_1d(h_pred, h_target, h_mask)

        dv_dq_weight = self.w_dv_dq if self.w_dv_dq > 0 else self.w_recon_deriv
        dq_dv_weight = self.w_dq_dv if self.w_dq_dv > 0 else self.w_recon_deriv

        total = (
            self.w_recon * recon_loss
            + dv_dq_weight * dv_dq_loss
            + dq_dv_weight * dq_dv_loss
            + self.w_volt_phys * volt_phys
            + self.w_recon_hyst * recon_hyst
        )

        aux = {
            "total": total,
            "recon_loss_per_sample": recon_loss_per_sample,  # (B,)
            "recon_loss_per_sample_unweighted": recon_loss_per_sample_unweighted,  # (B,)
            "recon_loss_per_sample_per_channel": recon_loss_per_sample_per_channel,  # (B,C_sel)
            "dv_dq": dv_dq_loss.detach(),
            "dq_dv": dq_dv_loss.detach(),
            "volt_phys": volt_phys.detach(),
            "recon_hyst": recon_hyst.detach(),
        }

        
        return total, kl_loss, kl_per_dim, aux
