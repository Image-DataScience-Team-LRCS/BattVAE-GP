from __future__ import annotations
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn.functional as F

Tensor = torch.Tensor


class PhysicsHead:
    """
    Channel-aware physics head.

    Input mask layout (B,C,N) if provided (aligned with preprocessing):
      0: q_global, 1: q_cap, 2: V_ch, 3: V_dis,
      4: dVdQ_ch, 5: dVdQ_dis, 6: d2VdQ2_ch, 7: d2VdQ2_dis,
      8: dQdV_ch, 9: dQdV_dis, 10: H_raw, 11: H_corr (optional)

    Produces physical tensors (Volts, V/Ah, V/Ah², Ah/V) and
    normalized counterparts based on `norm_cfg`.
    """

    def __init__(
        self,
        # voltage_norm_stats: Dict[str, Any],
        norm_cfg: Dict[str, Any],
        edge_trim_frac: float = 0.05,
        eps: float = 1e-6,
    ):
        # self.vstats = dict(voltage_norm_stats)
        self.cfg = dict(norm_cfg)
        self.edge = float(edge_trim_frac)
        self.eps = float(eps)

        mode = self.cfg["voltage"].get("mode", "phys_minmax")
        if mode not in ("phys_minmax", "none", "zscore", "robust_zscore"):
            raise ValueError(f"Unsupported voltage_norm_stats.mode: {mode}")
        if mode == "phys_minmax" and not {"v_min", "v_max"} <= self.cfg["voltage"].keys():
            raise ValueError("phys_minmax requires v_min and v_max")

    # --------------------------- Public API ---------------------------

    def build_targets(
        self,
        Vch_pred_norm: Tensor,  # (B,N)
        Vds_pred_norm: Tensor,  # (B,N)
        Vch_true_norm: Tensor,  # (B,N)
        Vds_true_norm: Tensor,  # (B,N)
        q_cap: Tensor,          # (B,N) normalized capacity axis
        q0_Ah: Tensor,          # (B,) absolute capacity
        mask_tokens: Tensor,    # (B,N) or (B,C,N) mask; True=valid
    ) -> Dict[str, Dict[str, Tensor]]:
        # Resolve channel-aware masks (all -> (B,N) bool)
        masks = self._make_channel_masks(mask_tokens)

        # Denormalize voltage to physical volts
        Vch_pred_V = self._denorm_voltage(Vch_pred_norm)
        Vds_pred_V = self._denorm_voltage(Vds_pred_norm)
        Vch_true_V = self._denorm_voltage(Vch_true_norm)
        Vds_true_V = self._denorm_voltage(Vds_true_norm)

        # ΔQ per step (Ah)
        dQ = self._delta_Q(q0_Ah, q_cap)  # (B,1)

        # Derivatives per branch with interior trim and channel masks
        pred_ch = self._derivs_for_branch(Vch_pred_V, dQ, masks["V_ch"])
        pred_ds = self._derivs_for_branch(Vds_pred_V, dQ, masks["V_dis"])
        true_ch = self._derivs_for_branch(Vch_true_V, dQ, masks["V_ch"])
        true_ds = self._derivs_for_branch(Vds_true_V, dQ, masks["V_dis"])

        # Merge to dicts with explicit keys
        H_pred_V = Vch_pred_V - Vds_pred_V
        H_true_V = Vch_true_V - Vds_true_V
        pred_phys = {
            "Vch_V": Vch_pred_V, "Vdis_V": Vds_pred_V,
            "dVdQ_ch": pred_ch["dVdQ"], "dVdQ_dis": pred_ds["dVdQ"],
            "d2VdQ2_ch": pred_ch["d2VdQ2"], "d2VdQ2_dis": pred_ds["d2VdQ2"],
            "dQdV_ch": pred_ch["dQdV"], "dQdV_dis": pred_ds["dQdV"],
            "H_raw_V": H_pred_V,
        }
        true_phys = {
            "Vch_V": Vch_true_V, "Vdis_V": Vds_true_V,
            "dVdQ_ch": true_ch["dVdQ"], "dVdQ_dis": true_ds["dVdQ"],
            "d2VdQ2_ch": true_ch["d2VdQ2"], "d2VdQ2_dis": true_ds["d2VdQ2"],
            "dQdV_ch": true_ch["dQdV"], "dQdV_dis": true_ds["dQdV"],
            "H_raw_V": H_true_V,
        }

        # Apply robust normalization (voltage & derivs)
        pred_norm = self._apply_norm_cfg(pred_phys, masks["V_ch"], masks["V_dis"])
        true_norm = self._apply_norm_cfg(true_phys, masks["V_ch"], masks["V_dis"])

        final_masks = {
            "V_ch": masks["V_ch"],
            "V_dis": masks["V_dis"],
            "dVdQ_ch": masks["dVdQ_ch"], "dVdQ_dis": masks["dVdQ_dis"],
            "d2VdQ2_ch": masks["d2VdQ2_ch"], "d2VdQ2_dis": masks["d2VdQ2_dis"],
            "dQdV_ch": masks["dQdV_ch"], "dQdV_dis": masks["dQdV_dis"],
            "H_raw": (masks["V_ch"] & masks["V_dis"]) if isinstance(masks["V_ch"], torch.Tensor) else masks["V_ch"],
        }

        return {
            "pred": {**pred_phys, **pred_norm},
            "true": {**true_phys, **true_norm},
            "masks": final_masks,
        }

    # ---------------------- Voltage de/normalization ------------------

    def _denorm_voltage(self, v_norm: Tensor) -> Tensor:
        mode = self.cfg["voltage"].get("mode", "phys_minmax")
        if mode == "none":
            return v_norm
        if mode in ("zscore", "robust_zscore"):
            raise NotImplementedError("Voltage zscore de-normalization not configured.")
        vmin, vmax = float(self.cfg["voltage"]["v_min"]), float(self.cfg["voltage"]["v_max"])
        target_range = str(self.cfg["voltage"].get("target_range", "zero_to_one")).strip().lower()
        if target_range == "minus_one_to_one":
            v_norm = (v_norm + 1.0) / 2.0
        return v_norm * (vmax - vmin) + vmin

    # ------------------------------ Masks -----------------------------

    def _make_channel_masks(self, mask_tokens: Tensor) -> Dict[str, Tensor]:
        """
        Return per-channel (B,N) bool masks. If (B,N) provided, reuse for all.
        If (B,C,N), use indices:
          2: V_ch, 3: V_dis, 4: dVdQ_ch, 5: dVdQ_dis, 6: d2VdQ2_ch, 7: d2VdQ2_dis, 8: dQdV_ch, 9: dQdV_dis, 10: H_raw
        Missing channels fallback to the corresponding voltage mask.
        """
        if mask_tokens.dim() == 2:
            m = mask_tokens.bool()
            return dict(
                V_ch=m, V_dis=m,
                dVdQ_ch=m, dVdQ_dis=m,
                dQdV_ch=m, dQdV_dis=m,
                d2VdQ2_ch=m, d2VdQ2_dis=m,
            )
        if mask_tokens.dim() != 3:
            raise ValueError(f"Unsupported mask shape {tuple(mask_tokens.shape)}")

        B, C, N = mask_tokens.shape
        M = mask_tokens.bool()
        any_token = M.any(dim=1)

        def get(i: int, default: Tensor) -> Tensor:
            return M[:, i, :] if i < C else default

        V_ch  = get(2, any_token)
        V_dis = get(3, any_token)

        dVdQ_ch    = get(4, V_ch)
        dVdQ_dis   = get(5, V_dis)
        d2VdQ2_ch  = get(6, V_ch)
        d2VdQ2_dis = get(7, V_dis)
        dQdV_ch    = get(8, V_ch)
        dQdV_dis   = get(9, V_dis)
        H_raw      = get(10, V_ch & V_dis)

        return dict(
            V_ch=V_ch, V_dis=V_dis,
            dVdQ_ch=dVdQ_ch, dVdQ_dis=dVdQ_dis,
            d2VdQ2_ch=d2VdQ2_ch, d2VdQ2_dis=d2VdQ2_dis,
            dQdV_ch=dQdV_ch, dQdV_dis=dQdV_dis,
            H_raw=H_raw,
        )

    # --------------------- Q grid & derivatives -----------------------

    def _delta_Q(self, q0_Ah: Tensor, q_cap: Tensor) -> Tensor:
        B, N = q_cap.shape
        diffs = q_cap[:, 1:] - q_cap[:, :-1]               # (B, N-1)
        step = diffs.mean(dim=1, keepdim=True).clamp_min(self.eps)  # (B,1)
        dQ = (q0_Ah.view(B, 1) * step).clamp_min(self.eps)          # (B,1)
        return dQ

    def _derivs_for_branch(self, V: Tensor, dQ: Tensor, mask_branch: Tensor) -> Dict[str, Tensor]:
        B, N = V.shape
        assert mask_branch.shape == (B, N)

        k = max(1, int(N * self.edge))
        interior = torch.ones(B, N, dtype=torch.bool, device=V.device)
        if N > 2:
            interior[:, :k] = False
            interior[:, -k:] = False
        interior = interior & mask_branch

        def cdiff(x: Tensor, d: Tensor) -> Tensor:
            xp = torch.roll(x, -1, 1)
            xm = torch.roll(x, +1, 1)
            dx = (xp - xm) / (2.0 * d)
            dx[:, 0]  = (x[:, 1]  - x[:, 0])  / d.squeeze(1)
            dx[:, -1] = (x[:, -1] - x[:, -2]) / d.squeeze(1)
            return dx

        dVdQ   = cdiff(V, dQ)             # V/Ah
        d2VdQ2 = cdiff(dVdQ, dQ)          # V/Ah^2
        dQdV   = 1.0 / dVdQ.clamp(min=self.eps)  # Ah/V

        m = interior.to(V.dtype)
        return {
            "dVdQ": dVdQ * m,
            "d2VdQ2": d2VdQ2 * m,
            "dQdV": dQdV * m,
            "mask_interior": interior,
        }

    # ------------------ Robust normalization for loss -----------------

    def _masked_stats_robust(self, x: Tensor, mask: Tensor, interior_only: bool, trim_frac: float) -> Tuple[Tensor, Tensor, Tensor]:
        B, N = x.shape
        use = mask.clone()
        if interior_only and N > 2:
            k = max(1, int(N * trim_frac))
            use[:, :k] = False
            use[:, -k:] = False
        empty = (use.sum(dim=1) == 0)
        use[empty] = mask[empty]

        med = torch.zeros(B, device=x.device, dtype=x.dtype)
        sig = torch.ones(B, device=x.device, dtype=x.dtype)
        for b in range(B):
            xb = x[b][use[b]]
            if xb.numel() >= 3:
                m = xb.median()
                mad = (xb - m).abs().median()
                med[b] = m
                sig[b] = torch.clamp(1.4826 * mad, min=self.eps)
        return med, sig, use

    def _robust_z(
        self, x: Tensor, mask: Tensor,
        interior_only: bool, trim_frac: float,
        clip_k: Optional[float] = None, pclip: Optional[float] = None
    ) -> Tensor:
        med, sig, use = self._masked_stats_robust(x, mask, interior_only, trim_frac)
        z = (x - med.unsqueeze(1)) / (sig.unsqueeze(1) + self.eps)
        if pclip is not None:
            out = []
            low_p = (100 - pclip) / 200.0
            high_p = 1.0 - low_p
            for b in range(z.size(0)):
                zb = z[b][use[b]]
                if zb.numel() > 0:
                    lo = torch.quantile(zb, low_p)
                    hi = torch.quantile(zb, high_p)
                    out.append(torch.clamp(z[b], lo, hi))
                else:
                    out.append(z[b])
            z = torch.stack(out, dim=0)
        if clip_k is not None:
            z = torch.clamp(z, -float(clip_k), float(clip_k))
        return z

    def _apply_norm_cfg(self, phys: Dict[str, Tensor], mask_ch: Tensor, mask_dis: Tensor) -> Dict[str, Tensor]:
        """
        Produce normalized channels for loss/metrics:
          - Vch_norm, Vdis_norm
          - dVdQ_ch_norm, dVdQ_dis_norm
          - d2VdQ2_ch_norm, d2VdQ2_dis_norm
          - dQdV_ch_norm, dQdV_dis_norm
          - H_raw_norm
        """
        cfg = self.cfg
        eps = float(cfg.get("epsilon", 1e-6))
        interior_only = bool(cfg.get("interior_only", True))
        trim_frac = float(cfg.get("interior_trim_frac", 0.05))

        out: Dict[str, Tensor] = {}

        # Voltage
        vcfg = cfg.get("voltage", {"mode": "none"})
        mode = vcfg.get("mode", "none")
        if mode == "phys_minmax":
            vmin, vmax = float(vcfg["v_min"]), float(vcfg["v_max"])
            out["Vch_norm"]  = (phys["Vch_V"]  - vmin) / max(vmax - vmin, eps)
            out["Vdis_norm"] = (phys["Vdis_V"] - vmin) / max(vmax - vmin, eps)
            target_range = str(vcfg.get("target_range", "zero_to_one")).strip().lower()
            if target_range == "minus_one_to_one":
                out["Vch_norm"] = 2.0 * out["Vch_norm"] - 1.0
                out["Vdis_norm"] = 2.0 * out["Vdis_norm"] - 1.0
        elif mode in ("zscore", "robust_zscore"):
            out["Vch_norm"]  = self._robust_z(phys["Vch_V"],  mask_ch, interior_only, trim_frac)
            out["Vdis_norm"] = self._robust_z(phys["Vdis_V"], mask_dis, interior_only, trim_frac)
        elif mode == "none":
            out["Vch_norm"], out["Vdis_norm"] = phys["Vch_V"], phys["Vdis_V"]
        else:
            raise ValueError(f"Unsupported voltage norm mode: {mode}")

        # dV/dQ
        d1 = cfg.get("dvdq", {"mode": "robust_zscore"})
        out["dVdQ_ch_norm"]  = self._robust_z(phys["dVdQ_ch"],  mask_ch, interior_only, trim_frac, clip_k=d1.get("clip_k"))
        out["dVdQ_dis_norm"] = self._robust_z(phys["dVdQ_dis"], mask_dis, interior_only, trim_frac, clip_k=d1.get("clip_k"))

        # d2V/dQ2
        d2 = cfg.get("d2vdq2", {"mode": "robust_zscore"})
        out["d2VdQ2_ch_norm"]  = self._robust_z(phys["d2VdQ2_ch"],  mask_ch, interior_only, trim_frac,
                                                clip_k=d2.get("clip_k"), pclip=d2.get("pclip"))
        out["d2VdQ2_dis_norm"] = self._robust_z(phys["d2VdQ2_dis"], mask_dis, interior_only, trim_frac,
                                                clip_k=d2.get("clip_k"), pclip=d2.get("pclip"))

        # dQ/dV (new): robust z-score by default unless you set a different rule under "dqdv"
        d3 = cfg.get("dqdv", {"mode": "robust_zscore"})
        out["dQdV_ch_norm"]  = self._robust_z(phys["dQdV_ch"],  mask_ch, interior_only, trim_frac, clip_k=d3.get("clip_k", 5.0))
        out["dQdV_dis_norm"] = self._robust_z(phys["dQdV_dis"], mask_dis, interior_only, trim_frac, clip_k=d3.get("clip_k", 5.0))

        # Hysteresis H_raw normalization
        hcfg = cfg.get("hyst", {"mode": "none"})
        hmode = hcfg.get("mode", "none")
        if hmode == "phys_minmax":
            if mode != "phys_minmax":
                raise ValueError("NORMALIZATION.hyst.mode=phys_minmax requires NORMALIZATION.voltage.mode=phys_minmax")
            scale = (vmax - vmin)
            if target_range == "minus_one_to_one":
                scale = 2.0 * scale
            out["H_raw_norm"] = phys["H_raw_V"] / scale
        elif hmode == "none":
            out["H_raw_norm"] = phys["H_raw_V"]
        else:
            out["H_raw_norm"] = self._robust_z(phys["H_raw_V"], mask_ch & mask_dis, interior_only, trim_frac, clip_k=hcfg.get("clip_k", 5.0))

        return out
