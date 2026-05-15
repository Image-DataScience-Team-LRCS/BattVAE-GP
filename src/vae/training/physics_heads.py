from __future__ import annotations

from typing import Dict, Any, Optional, Tuple

import torch

Tensor = torch.Tensor


class PhysicsHead:
    """
    Channel-aware physics head.

    Expected input mask layout if mask_tokens has shape (B, C, N):

      0: q_global
      1: q_cap
      2: V_ch
      3: V_dis
      4: dVdQ_ch
      5: dVdQ_dis
      6: d2VdQ2_ch
      7: d2VdQ2_dis
      8: dQdV_ch
      9: dQdV_dis
      10: H_raw
      11: H_corr optional

    Main purpose:
      - Convert predicted and true normalized voltages into physical volts.
      - Compute dV/dQ, dQ/dV and hysteresis from voltage.
      - Normalize those derived quantities consistently for training losses.
      - Return masks that exclude physically unreliable edge regions.
    """

    def __init__(
        self,
        norm_cfg: Dict[str, Any],
        norm_stats: Optional[Dict[str, Any]] = None,
        edge_trim_frac: float = 0.03,
        eps: float = 1e-6,
    ):
        self.cfg = dict(norm_cfg)
        self.norm_stats = norm_stats or {}
        self.edge = float(edge_trim_frac)
        self.eps = float(eps)

        if "voltage" not in self.cfg:
            raise ValueError("norm_cfg must contain a 'voltage' section.")

        mode = self.cfg["voltage"].get("mode", "phys_minmax")
        if mode not in {"phys_minmax", "none", "zscore", "robust_zscore"}:
            raise ValueError(f"Unsupported voltage normalization mode: {mode}")

        if mode == "phys_minmax":
            if not {"v_min", "v_max"} <= self.cfg["voltage"].keys():
                raise ValueError("voltage.mode='phys_minmax' requires 'v_min' and 'v_max'.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_targets(
        self,
        Vch_pred_norm: Tensor,   # (B, N)
        Vds_pred_norm: Tensor,   # (B, N)
        Vch_true_norm: Tensor,   # (B, N)
        Vds_true_norm: Tensor,   # (B, N)
        q_cap: Tensor,           # (B, N), normalized capacity coordinate
        q0_Ah: Tensor,           # (B,), absolute reference capacity
        mask_tokens: Tensor,     # (B, N) or (B, C, N)
    ) -> Dict[str, Dict[str, Tensor]]:
        """
        Returns:
          {
            "pred":  physical + normalized tensors from predicted voltage,
            "true":  physical + normalized tensors from true voltage,
            "masks": channel-specific masks for loss computation
          }
        """
        self._check_shapes(
            Vch_pred_norm,
            Vds_pred_norm,
            Vch_true_norm,
            Vds_true_norm,
            q_cap,
            q0_Ah,
        )

        masks = self._make_channel_masks(mask_tokens)

        # Denormalize voltage to physical volts.
        Vch_pred_V = self._denorm_voltage(Vch_pred_norm)
        Vds_pred_V = self._denorm_voltage(Vds_pred_norm)
        Vch_true_V = self._denorm_voltage(Vch_true_norm)
        Vds_true_V = self._denorm_voltage(Vds_true_norm)

        # Capacity spacing in Ah.
        dQ = self._delta_Q(q0_Ah, q_cap)

        # Derivatives from voltage.
        pred_ch = self._derivs_for_branch(Vch_pred_V, dQ, masks["V_ch"], masks["dVdQ_ch"], masks["dQdV_ch"])
        pred_ds = self._derivs_for_branch(Vds_pred_V, dQ, masks["V_dis"], masks["dVdQ_dis"], masks["dQdV_dis"])

        true_ch = self._derivs_for_branch(Vch_true_V, dQ, masks["V_ch"], masks["dVdQ_ch"], masks["dQdV_ch"])
        true_ds = self._derivs_for_branch(Vds_true_V, dQ, masks["V_dis"], masks["dVdQ_dis"], masks["dQdV_dis"])

        H_pred_V = Vch_pred_V - Vds_pred_V
        H_true_V = Vch_true_V - Vds_true_V

        H_mask = masks["H_raw"] & masks["V_ch"] & masks["V_dis"]

        pred_phys = {
            "Vch_V": Vch_pred_V,
            "Vdis_V": Vds_pred_V,
            "dVdQ_ch": pred_ch["dVdQ"],
            "dVdQ_dis": pred_ds["dVdQ"],
            "d2VdQ2_ch": pred_ch["d2VdQ2"],
            "d2VdQ2_dis": pred_ds["d2VdQ2"],
            "dQdV_ch": pred_ch["dQdV"],
            "dQdV_dis": pred_ds["dQdV"],
            "H_raw_V": H_pred_V,
        }

        true_phys = {
            "Vch_V": Vch_true_V,
            "Vdis_V": Vds_true_V,
            "dVdQ_ch": true_ch["dVdQ"],
            "dVdQ_dis": true_ds["dVdQ"],
            "d2VdQ2_ch": true_ch["d2VdQ2"],
            "d2VdQ2_dis": true_ds["d2VdQ2"],
            "dQdV_ch": true_ch["dQdV"],
            "dQdV_dis": true_ds["dQdV"],
            "H_raw_V": H_true_V,
        }

        final_masks = {
            "V_ch": masks["V_ch"],
            "V_dis": masks["V_dis"],

            # Use derivative-specific masks with edge regions removed.
            "dVdQ_ch": pred_ch["mask_dVdQ"],
            "dVdQ_dis": pred_ds["mask_dVdQ"],
            "d2VdQ2_ch": pred_ch["mask_d2VdQ2"],
            "d2VdQ2_dis": pred_ds["mask_d2VdQ2"],
            "dQdV_ch": pred_ch["mask_dQdV"],
            "dQdV_dis": pred_ds["mask_dQdV"],

            "H_raw": H_mask,
        }

        pred_norm = self._apply_norm_cfg(pred_phys, final_masks)
        true_norm = self._apply_norm_cfg(true_phys, final_masks)

        return {
            "pred": {**pred_phys, **pred_norm},
            "true": {**true_phys, **true_norm},
            "masks": final_masks,
        }

    # ------------------------------------------------------------------
    # Shape checks
    # ------------------------------------------------------------------

    @staticmethod
    def _check_shapes(
        Vch_pred_norm: Tensor,
        Vds_pred_norm: Tensor,
        Vch_true_norm: Tensor,
        Vds_true_norm: Tensor,
        q_cap: Tensor,
        q0_Ah: Tensor,
    ) -> None:
        if Vch_pred_norm.ndim != 2:
            raise ValueError(f"Expected Vch_pred_norm shape (B,N), got {tuple(Vch_pred_norm.shape)}")

        expected = Vch_pred_norm.shape
        for name, x in {
            "Vds_pred_norm": Vds_pred_norm,
            "Vch_true_norm": Vch_true_norm,
            "Vds_true_norm": Vds_true_norm,
            "q_cap": q_cap,
        }.items():
            if x.shape != expected:
                raise ValueError(f"{name} must have shape {tuple(expected)}, got {tuple(x.shape)}")

        if q0_Ah.ndim != 1 or q0_Ah.shape[0] != expected[0]:
            raise ValueError(f"q0_Ah must have shape (B,), got {tuple(q0_Ah.shape)}")

    # ------------------------------------------------------------------
    # Voltage normalization / denormalization
    # ------------------------------------------------------------------

    def _denorm_voltage(self, v_norm: Tensor) -> Tensor:
        """
        Convert normalized voltage to physical voltage.

        For voltage.mode='phys_minmax':
          target_range='zero_to_one'     : v_norm in [0, 1]
          target_range='minus_one_to_one': v_norm in [-1, 1]

        Important:
          This function does not clamp v_norm. The decoder should already
          bound voltage output using sigmoid/tanh when appropriate.
        """
        vcfg = self.cfg["voltage"]
        mode = vcfg.get("mode", "phys_minmax")

        if mode == "none":
            return v_norm

        if mode in {"zscore", "robust_zscore"}:
            raise NotImplementedError(
                "Voltage z-score de-normalization requires stored mean/std or median/scale. "
                "Use phys_minmax for the current physics head."
            )

        vmin = float(vcfg["v_min"])
        vmax = float(vcfg["v_max"])
        scale = max(vmax - vmin, self.eps)

        target_range = str(vcfg.get("target_range", "zero_to_one")).lower().strip()

        if target_range == "zero_to_one":
            v01 = v_norm
        elif target_range == "minus_one_to_one":
            v01 = 0.5 * (v_norm + 1.0)
        else:
            raise ValueError(f"Unsupported voltage target_range: {target_range}")

        return v01 * scale + vmin

    def _norm_voltage(self, v_phys: Tensor) -> Tensor:
        vcfg = self.cfg["voltage"]
        mode = vcfg.get("mode", "phys_minmax")

        if mode == "none":
            return v_phys

        if mode in {"zscore", "robust_zscore"}:
            raise NotImplementedError(
                "Voltage z-score normalization requires stored mean/std or median/scale. "
                "Use phys_minmax for the current physics head."
            )

        vmin = float(vcfg["v_min"])
        vmax = float(vcfg["v_max"])
        scale = max(vmax - vmin, self.eps)

        v01 = (v_phys - vmin) / scale

        target_range = str(vcfg.get("target_range", "zero_to_one")).lower().strip()

        if target_range == "zero_to_one":
            return v01
        if target_range == "minus_one_to_one":
            return 2.0 * v01 - 1.0

        raise ValueError(f"Unsupported voltage target_range: {target_range}")

    # ------------------------------------------------------------------
    # Masks
    # ------------------------------------------------------------------

    def _make_channel_masks(self, mask_tokens: Tensor) -> Dict[str, Tensor]:
        """
        Convert a token mask into channel-specific boolean masks.

        If mask_tokens is (B,N), the same mask is reused for all channels.
        If mask_tokens is (B,C,N), channel-aware masks are extracted.
        """
        if mask_tokens.ndim == 2:
            m = mask_tokens.bool()
            return {
                "V_ch": m,
                "V_dis": m,
                "dVdQ_ch": m,
                "dVdQ_dis": m,
                "d2VdQ2_ch": m,
                "d2VdQ2_dis": m,
                "dQdV_ch": m,
                "dQdV_dis": m,
                "H_raw": m,
            }

        if mask_tokens.ndim != 3:
            raise ValueError(f"Unsupported mask_tokens shape: {tuple(mask_tokens.shape)}")

        _, C, _ = mask_tokens.shape
        M = mask_tokens.bool()
        any_token = M.any(dim=1)

        def get(i: int, default: Tensor) -> Tensor:
            return M[:, i, :] if i < C else default

        V_ch = get(2, any_token)
        V_dis = get(3, any_token)

        return {
            "V_ch": V_ch,
            "V_dis": V_dis,
            "dVdQ_ch": get(4, V_ch),
            "dVdQ_dis": get(5, V_dis),
            "d2VdQ2_ch": get(6, V_ch),
            "d2VdQ2_dis": get(7, V_dis),
            "dQdV_ch": get(8, V_ch),
            "dQdV_dis": get(9, V_dis),
            "H_raw": get(10, V_ch & V_dis),
        }

    def _interior_mask(self, mask: Tensor) -> Tensor:
        """
        Remove edge regions where finite-difference derivatives are least reliable.
        """
        B, N = mask.shape
        interior = mask.clone().bool()

        if N > 2:
            k = max(1, int(round(N * self.edge)))
            interior[:, :k] = False
            interior[:, -k:] = False

        return interior

    # ------------------------------------------------------------------
    # Capacity step and derivatives
    # ------------------------------------------------------------------

    def _delta_Q(self, q0_Ah: Tensor, q_cap: Tensor) -> Tensor:
        """
        Estimate the capacity step dQ in Ah.

        q_cap is normalized, usually uniform from 0 to 1.
        q0_Ah converts normalized capacity step into physical Ah.
        """
        B, N = q_cap.shape

        if N < 2:
            raise ValueError("q_cap must contain at least two grid points.")

        diffs = q_cap[:, 1:] - q_cap[:, :-1]

        # Keep only positive finite differences. This avoids pathological
        # behavior if padded or invalid regions contain repeated q values.
        positive = torch.where(diffs > self.eps, diffs, torch.nan)

        step = torch.nanmedian(positive, dim=1, keepdim=True).values

        # Fallback if nanmedian fails because all diffs are invalid.
        fallback = diffs.abs().mean(dim=1, keepdim=True).clamp_min(self.eps)
        step = torch.where(torch.isfinite(step), step, fallback)

        return (q0_Ah.view(B, 1) * step).clamp_min(self.eps)

    def _central_diff(self, x: Tensor, d: Tensor) -> Tensor:
        """
        Central finite difference along the N dimension.

        x: (B,N)
        d: (B,1)
        """
        B, N = x.shape

        if N < 2:
            raise ValueError("Need at least two points for finite differences.")

        if N == 2:
            dx = (x[:, 1:2] - x[:, 0:1]) / d
            return dx.expand(B, N)

        xp = torch.roll(x, shifts=-1, dims=1)
        xm = torch.roll(x, shifts=1, dims=1)

        dx = (xp - xm) / (2.0 * d)

        dx[:, 0] = (x[:, 1] - x[:, 0]) / d.squeeze(1)
        dx[:, -1] = (x[:, -1] - x[:, -2]) / d.squeeze(1)

        return dx

    def _safe_inverse_dvdq(self, dVdQ: Tensor) -> Tensor:
        """
        Compute dQ/dV safely.

        Default behavior uses dQ/dV magnitude:
            dQdV = 1 / |dV/dQ|

        This avoids the old error:
            1 / dVdQ.clamp(min=eps)

        which turns negative slopes into huge positive spikes.
        """
        dqdv_cfg = self.cfg.get("dqdv", {})
        inverse_mode = str(dqdv_cfg.get("inverse_mode", "magnitude")).lower().strip()

        if inverse_mode == "magnitude":
            denom = dVdQ.abs().clamp_min(self.eps)
            return 1.0 / denom

        if inverse_mode == "signed":
            sign = torch.where(dVdQ >= 0, torch.ones_like(dVdQ), -torch.ones_like(dVdQ))
            denom = sign * dVdQ.abs().clamp_min(self.eps)
            return 1.0 / denom

        raise ValueError(f"Unsupported dqdv.inverse_mode: {inverse_mode}")

    def _derivs_for_branch(
        self,
        V: Tensor,
        dQ: Tensor,
        voltage_mask: Tensor,
        dvdq_mask: Tensor,
        dqdv_mask: Tensor,
    ) -> Dict[str, Tensor]:
        """
        Compute derivatives and return derivative-specific masks.

        Values outside valid derivative masks are zeroed, and the same masks
        are returned so losses do not learn artificial edge values.
        """
        B, N = V.shape

        if voltage_mask.shape != (B, N):
            raise ValueError(f"voltage_mask must have shape {(B, N)}, got {tuple(voltage_mask.shape)}")

        interior = self._interior_mask(voltage_mask)

        mask_dVdQ = interior & dvdq_mask.bool()
        mask_d2VdQ2 = torch.zeros_like(interior, dtype=torch.bool)
        mask_dQdV = interior & dqdv_mask.bool()

        dVdQ = self._central_diff(V, dQ)
        # d2V/dQ2 is not used by the current pipeline; keep the output key
        # shape-stable but skip the extra finite-difference computation.
        d2VdQ2 = torch.zeros_like(dVdQ)
        dQdV = self._safe_inverse_dvdq(dVdQ)

        dVdQ = torch.where(mask_dVdQ, dVdQ, torch.zeros_like(dVdQ))
        d2VdQ2 = torch.where(mask_d2VdQ2, d2VdQ2, torch.zeros_like(d2VdQ2))
        dQdV = torch.where(mask_dQdV, dQdV, torch.zeros_like(dQdV))

        return {
            "dVdQ": dVdQ,
            "d2VdQ2": d2VdQ2,
            "dQdV": dQdV,
            "mask_dVdQ": mask_dVdQ,
            "mask_d2VdQ2": mask_d2VdQ2,
            "mask_dQdV": mask_dQdV,
        }

    # ------------------------------------------------------------------
    # Robust normalization
    # ------------------------------------------------------------------

    def _masked_stats_robust(
        self,
        x: Tensor,
        mask: Tensor,
        trim_frac: float,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Per-sample robust median and MAD-based scale.
        """
        B, N = x.shape
        use = mask.clone().bool()

        if N > 2 and trim_frac > 0:
            k = max(1, int(round(N * trim_frac)))
            use[:, :k] = False
            use[:, -k:] = False

        # If trimming removed everything for a sample, fall back to original mask.
        empty = use.sum(dim=1) == 0
        if empty.any():
            use[empty] = mask[empty].bool()

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
        self,
        x: Tensor,
        mask: Tensor,
        clip_k: Optional[float] = None,
        pclip: Optional[float] = None,
    ) -> Tensor:
        """
        Robust z-score normalization using per-sample median and MAD.
        """
        trim_frac = float(self.cfg.get("interior_trim_frac", self.edge))

        med, sig, use = self._masked_stats_robust(x, mask, trim_frac=trim_frac)
        z = (x - med.unsqueeze(1)) / (sig.unsqueeze(1) + self.eps)

        if pclip is not None:
            z_clipped = []
            low_p = (100.0 - float(pclip)) / 200.0
            high_p = 1.0 - low_p

            for b in range(z.shape[0]):
                zb = z[b][use[b]]
                if zb.numel() > 0:
                    lo = torch.quantile(zb, low_p)
                    hi = torch.quantile(zb, high_p)
                    z_clipped.append(torch.clamp(z[b], lo, hi))
                else:
                    z_clipped.append(z[b])

            z = torch.stack(z_clipped, dim=0)

        if clip_k is not None:
            z = torch.clamp(z, -float(clip_k), float(clip_k))

        return z

    def _global_stats_entry(self, family: str) -> Optional[Dict[str, Any]]:
        block = self.norm_stats.get(family)
        if not isinstance(block, dict) or not block:
            return None

        if "mode" in block:
            return block

        for key in ("None", "none", "global"):
            stats = block.get(key)
            if isinstance(stats, dict):
                return stats

        if len(block) == 1:
            stats = next(iter(block.values()))
            if isinstance(stats, dict):
                return stats

        raise ValueError(
            f"PhysicsHead requires global frozen stats for '{family}', "
            "but received multiple scoped entries. Set NORMALIZATION.scope: global "
            "or pass the sample scope into the physics head."
        )

    def _normalize_with_global_stats(
        self,
        x: Tensor,
        stats: Dict[str, Any],
        default_clip: Optional[float],
    ) -> Tensor:
        mode = str(stats.get("mode", "none")).lower().strip()

        if mode in {"none", "raw"}:
            return x

        if mode in {"zscore", "robust_zscore"}:
            hi = stats.get("hi")
            if hi is not None:
                x = torch.clamp(x, -float(hi), float(hi))

            mu = x.new_tensor(float(stats["mu"]))
            sigma = x.new_tensor(max(float(stats["sigma"]), self.eps))
            z = (x - mu) / (sigma + self.eps)

        elif mode == "asinh_robust":
            tau = x.new_tensor(max(float(stats["tau"]), self.eps))
            mu = x.new_tensor(float(stats["mu"]))
            sigma = x.new_tensor(max(float(stats["sigma"]), self.eps))
            z = (torch.asinh(x / (tau + self.eps)) - mu) / (sigma + self.eps)

        elif mode in {"phys_scale", "phys_minmax"}:
            scale = float(stats.get("scale", 0.0))
            if scale <= 0.0:
                scale = float(stats["v_max"]) - float(stats["v_min"])
                target_range = str(stats.get("target_range", "zero_to_one")).lower().strip()
                if target_range == "minus_one_to_one":
                    scale *= 2.0
            z = x / max(scale, self.eps)

        else:
            raise ValueError(f"Unsupported frozen normalization mode for physics head: {mode}")

        clip_k = stats.get("clip_k", default_clip)
        if clip_k is not None:
            z = torch.clamp(z, -float(clip_k), float(clip_k))

        return z

    # ------------------------------------------------------------------
    # Apply normalization config
    # ------------------------------------------------------------------

    def _apply_norm_cfg(
        self,
        phys: Dict[str, Tensor],
        masks: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        """
        Produce normalized tensors used by LossFactory.

        Output keys:
          Vch_norm, Vdis_norm
          dVdQ_ch_norm, dVdQ_dis_norm
          d2VdQ2_ch_norm, d2VdQ2_dis_norm
          dQdV_ch_norm, dQdV_dis_norm
          H_raw_norm
        """
        out: Dict[str, Tensor] = {}

        # Voltage.
        out["Vch_norm"] = self._norm_voltage(phys["Vch_V"])
        out["Vdis_norm"] = self._norm_voltage(phys["Vdis_V"])

        # dV/dQ.
        dvdq_cfg = self.cfg.get("dvdq", {"mode": "robust_zscore"})
        out["dVdQ_ch_norm"] = self._normalize_nonvoltage(
            phys["dVdQ_ch"],
            masks["dVdQ_ch"],
            dvdq_cfg,
            default_clip=None,
            family="dvdq",
        )
        out["dVdQ_dis_norm"] = self._normalize_nonvoltage(
            phys["dVdQ_dis"],
            masks["dVdQ_dis"],
            dvdq_cfg,
            default_clip=None,
            family="dvdq",
        )

        # d2V/dQ2 is intentionally not computed or normalized in this pipeline.
        out["d2VdQ2_ch_norm"] = torch.zeros_like(phys["d2VdQ2_ch"])
        out["d2VdQ2_dis_norm"] = torch.zeros_like(phys["d2VdQ2_dis"])

        # dQ/dV.
        dqdv_cfg = self.cfg.get("dqdv", {"mode": "robust_zscore", "clip_k": 5.0})
        out["dQdV_ch_norm"] = self._normalize_nonvoltage(
            phys["dQdV_ch"],
            masks["dQdV_ch"],
            dqdv_cfg,
            default_clip=5.0,
            family="dqdv",
        )
        out["dQdV_dis_norm"] = self._normalize_nonvoltage(
            phys["dQdV_dis"],
            masks["dQdV_dis"],
            dqdv_cfg,
            default_clip=5.0,
            family="dqdv",
        )

        # Hysteresis.
        hyst_cfg = self.cfg.get("hyst", {"mode": "phys_scale"})
        hmode = str(hyst_cfg.get("mode", "phys_scale")).lower().strip()

        if hmode in {"none", "raw"}:
            out["H_raw_norm"] = phys["H_raw_V"]

        elif hmode in {"phys_scale", "phys_minmax"}:
            vcfg = self.cfg["voltage"]
            if vcfg.get("mode", "phys_minmax") != "phys_minmax":
                raise ValueError("hyst.mode='phys_scale' requires voltage.mode='phys_minmax'.")

            vmin = float(vcfg["v_min"])
            vmax = float(vcfg["v_max"])
            scale = max(vmax - vmin, self.eps)

            target_range = str(vcfg.get("target_range", "zero_to_one")).lower().strip()
            if target_range == "minus_one_to_one":
                scale = 2.0 * scale

            out["H_raw_norm"] = phys["H_raw_V"] / scale

        elif hmode == "robust_zscore":
            out["H_raw_norm"] = self._normalize_nonvoltage(
                phys["H_raw_V"],
                masks["H_raw"],
                hyst_cfg,
                default_clip=5.0,
                family="hyst",
            )

        else:
            raise ValueError(f"Unsupported hyst.mode: {hmode}")

        return out

    def _normalize_nonvoltage(
        self,
        x: Tensor,
        mask: Tensor,
        cfg: Dict[str, Any],
        default_clip: Optional[float],
        family: Optional[str] = None,
    ) -> Tensor:
        mode = str(cfg.get("mode", "robust_zscore")).lower().strip()

        if mode in {"none", "raw"}:
            return x

        stats = self._global_stats_entry(family or "")
        if stats is not None:
            return self._normalize_with_global_stats(x, stats, default_clip)

        raise ValueError(
            f"Missing frozen global normalization stats for '{family}'. "
            "Pass train_norm_stats into LossFactory/PhysicsHead so physics-head "
            "targets reuse the train-set statistics."
        )
