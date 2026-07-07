from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

from src.common.logger.logging import get_logger
from src.common.utils.config_schema import FullConfig

logger = get_logger(__name__)


def _robust_location_scale(x: np.ndarray) -> Tuple[float, float]:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0, 1.0
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    if mad > 0:
        sigma = 1.4826 * mad
    else:
        sigma = float(np.std(x) + 1e-12)
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 1.0
    return med, sigma


def _interp_valid_range_on_grid(Q: np.ndarray, grid: np.ndarray) -> np.ndarray:
    if Q.size < 2:
        return np.zeros_like(grid, dtype=bool)
    qmin = float(np.min(Q))
    qmax = float(np.max(Q))
    if not np.isfinite(qmin) or not np.isfinite(qmax) or qmax <= qmin:
        return np.zeros_like(grid, dtype=bool)
    return (grid >= qmin) & (grid <= qmax)


def _interp_values(
    Q: np.ndarray,
    V: np.ndarray,
    grid: np.ndarray,
    pad_value: float,
) -> Tuple[np.ndarray, np.ndarray]:
    out = np.full_like(grid, pad_value, dtype=np.float32)

    if Q.size < 2 or V.size != Q.size:
        return out, np.zeros_like(grid, dtype=bool)

    order = np.argsort(Q)
    Qs = np.asarray(Q[order], dtype=float)
    Vs = np.asarray(V[order], dtype=float)

    Qs_unique, unique_idx = np.unique(Qs, return_index=True)
    Vs_unique = Vs[unique_idx]
    if Qs_unique.size < 2:
        return out, np.zeros_like(grid, dtype=bool)

    valid = _interp_valid_range_on_grid(Qs_unique, grid)
    if not valid.any():
        return out, valid

    try:
        interp = PchipInterpolator(Qs_unique, Vs_unique, extrapolate=False)
        out[valid] = interp(grid[valid]).astype(np.float32)
    except Exception:
        out[valid] = np.interp(grid[valid], Qs_unique, Vs_unique).astype(np.float32)

    return out, valid


def _pchip_values_and_derivatives(
    Q: np.ndarray,
    V: np.ndarray,
    grid: np.ndarray,
    pad_value: float,
    want_second: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray]:
    Vg = np.full_like(grid, pad_value, dtype=np.float32)
    d1 = np.full_like(grid, pad_value, dtype=np.float32)
    d2 = np.full_like(grid, pad_value, dtype=np.float32) if want_second else None

    if Q.size < 2 or V.size != Q.size or np.allclose(Q.max(), Q.min()):
        return Vg, d1, d2, np.zeros_like(grid, dtype=bool)

    order = np.argsort(Q)
    Qs = np.asarray(Q[order], dtype=float)
    Vs = np.asarray(V[order], dtype=float)
    Qs_unique, unique_idx = np.unique(Qs, return_index=True)
    Vs_unique = Vs[unique_idx]
    if Qs_unique.size < 2:
        return Vg, d1, d2, np.zeros_like(grid, dtype=bool)

    valid = _interp_valid_range_on_grid(Qs_unique, grid)
    if not valid.any():
        return Vg, d1, d2, valid

    try:
        p = PchipInterpolator(Qs_unique, Vs_unique, extrapolate=False)
        Vg[valid] = p(grid[valid]).astype(np.float32)
        d1p = p.derivative(1)
        d1[valid] = d1p(grid[valid]).astype(np.float32)
        if want_second:
            d2p = p.derivative(2)
            d2[valid] = d2p(grid[valid]).astype(np.float32)
    except Exception:
        Vg[valid] = np.interp(grid[valid], Qs_unique, Vs_unique).astype(np.float32)
        idx = np.flatnonzero(valid)
        if idx.size >= 3:
            g = grid[idx].astype(np.float32)
            v = Vg[idx].astype(np.float32)
            dv = np.gradient(v, g, edge_order=1)
            d1[idx] = dv.astype(np.float32)
            if want_second and idx.size >= 5:
                d2[idx] = np.gradient(dv, g, edge_order=1).astype(np.float32)

    return Vg, d1, d2, valid


def _finite_difference_on_valid_grid(
    values: np.ndarray,
    grid: np.ndarray,
    valid: np.ndarray,
    pad_value: float,
    want_second: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    d1 = np.full_like(values, pad_value, dtype=np.float32)
    d2 = np.full_like(values, pad_value, dtype=np.float32) if want_second else None

    idx = np.flatnonzero(valid)
    if idx.size < 2:
        return d1, d2

    g = grid[idx].astype(np.float32)
    v = values[idx].astype(np.float32)
    dv = np.gradient(v, g, edge_order=1).astype(np.float32)
    d1[idx] = dv

    if want_second:
        if idx.size >= 3:
            d2[idx] = np.gradient(dv, g, edge_order=1).astype(np.float32)
        else:
            d2[idx] = 0.0

    return d1, d2


def _assert_same_branch_masks(charge_mask: np.ndarray, discharge_mask: np.ndarray) -> None:
    if np.array_equal(charge_mask, discharge_mask):
        return

    diff_idx = np.flatnonzero(charge_mask != discharge_mask)
    first_idx = int(diff_idx[0]) if diff_idx.size else -1
    raise AssertionError(
        "Charge and discharge masks must be identical. "
        f"First mismatch at grid index {first_idx}."
    )


class DataProcessor:
    """
    Capacity-grid preprocessing with a fixed nominal capacity.

    Output channels:
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
      11: mask
    """

    def __init__(self, config: FullConfig) -> None:
        self.config = config
        self.hyper_parameters = config.HYPER_PARAMETERS
        self.paths = config.PATHS

        self.groupby_cols = ["source_dataset", "Cycle"]
        self.cycle_id_col = "global_index"

        self.data: Optional[pd.DataFrame] = None
        self.total_cycles: Optional[int] = None
        self.max_cycle_length: Optional[int] = None

        self.n_interp_points = int(self.hyper_parameters.input_seq_len)
        self.padding_value = float(self.hyper_parameters.padding_value)
        self.i_eps = float(getattr(self.hyper_parameters, "i_eps", 0.02))
        self.edge_trim_frac = float(getattr(self.hyper_parameters, "edge_trim_frac", 0.02))
        self.dqdv_eps_mult = float(getattr(self.hyper_parameters, "dqdv_eps_mult", 3.0))
        self.q_max_ah = 5.0
        self.v_dis_fill_value = float(
            getattr(getattr(config, "NORMALIZATION", None), "voltage", {}).get("v_min", 2.5)
        )

    def prepare_cycle_data(self, df: pd.DataFrame) -> int:
        if df is None:
            raise ValueError("DataFrame is None")

        required = {"source_dataset", "Cycle", "Time", "Voltage", "Current"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        df = df.sort_values(self.groupby_cols + ["Time"]).reset_index(drop=True)
        self.data = df
        self.total_cycles = int(df.groupby(self.groupby_cols).ngroups)
        self.max_cycle_length = int(df.groupby(self.groupby_cols).size().max())

        logger.info("Prepared %s cycles across %s dataset(s)", self.total_cycles, df["source_dataset"].nunique())
        return self.max_cycle_length

    def _integrate_capacity(self) -> None:
        assert self.data is not None
        df = self.data.copy()

        grp = df.groupby(self.groupby_cols, sort=False)
        dt = df["Time"].diff()
        new_group = grp.ngroup().diff().fillna(1) != 0
        dt[new_group] = 0.0
        df["dt"] = dt.values
        df["cap_inc"] = (df["Current"] * df["dt"]) / 3600.0
        df["cum_cap"] = grp["cap_inc"].cumsum()

        self.data = df

    def _apply_normalization(
        self,
        features: np.ndarray,
        masks: np.ndarray,
        labels: np.ndarray,
        norm_cfg: Optional[dict],
        feature_names: List[str],
        train_selector: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict[str, dict]]:
        if norm_cfg is None:
            return features, {}

        num_cycles, _, _ = features.shape
        name_to_idx = {n: i for i, n in enumerate(feature_names)}
        masks_bool = masks.astype(bool)
        norm_stats: Dict[str, dict] = {}

        idx_vch = name_to_idx.get("V_ch")
        idx_vdis = name_to_idx.get("V_dis")
        idx_dv_ch = name_to_idx.get("dVdQ_ch")
        idx_dv_dis = name_to_idx.get("dVdQ_dis")
        idx_d2_ch = name_to_idx.get("d2VdQ2_ch")
        idx_d2_dis = name_to_idx.get("d2VdQ2_dis")
        idx_dqdv_ch = name_to_idx.get("dQdV_ch")
        idx_dqdv_dis = name_to_idx.get("dQdV_dis")
        idx_h_raw = name_to_idx.get("H_raw")

        fit_on_raw = str(norm_cfg.get("fit_on", "train_only")).strip().lower()
        fit_on_aliases = {
            "train_only": "train_only",
            "train": "train_only",
            "train_split": "train_only",
            "all": "all_loaded",
            "all_loaded": "all_loaded",
            "loaded_split": "all_loaded",
        }
        fit_on = fit_on_aliases.get(fit_on_raw)
        if fit_on is None:
            raise ValueError(
                "NORMALIZATION.fit_on must be one of "
                "{train_only, train, train_split, all_loaded, loaded_split, all}"
            )
        scope = norm_cfg.get("scope", "global")
        if fit_on == "all_loaded":
            train_selector = np.ones((num_cycles,), dtype=bool)
        else:
            if train_selector is None:
                logger.info(
                    "NORMALIZATION.fit_on=%s requested, but no train_selector was provided; "
                    "fitting on all cycles in the currently loaded split only.",
                    fit_on_raw,
                )
                train_selector = np.ones((num_cycles,), dtype=bool)
            else:
                train_selector = train_selector.astype(bool)

        def _iter_scopes():
            if scope == "per_dataset":
                return sorted(set(labels[:, 0].astype(int).tolist()))
            return [None]

        def _fit_family(
            channels: Tuple[Optional[int], ...],
            family: str,
            cfg: dict,
            allow_asinh: bool = False,
            pclip: Optional[float] = None,
        ) -> None:
            mode = cfg.get("mode", "none")
            if mode == "none":
                norm_stats[family] = {"mode": "none"}
                return
            clip_k = float(cfg.get("clip_k", 5.0))

            for scope_idx in _iter_scopes():
                cyc_mask = (labels[:, 0].astype(int) == scope_idx) if scope_idx is not None else np.ones((num_cycles,), dtype=bool)
                cyc_mask &= train_selector
                vals = []
                for ch in channels:
                    if ch is None:
                        continue
                    m = masks_bool[:, ch, :]
                    vals.append(features[cyc_mask, ch, :][m[cyc_mask]])
                vals = [v for v in vals if v.size > 0]
                v = np.concatenate(vals) if vals else np.array([], dtype=np.float32)
                if v.size == 0:
                    continue

                stats_entry: Dict[str, float | str] = {"mode": mode, "clip_k": clip_k}
                if pclip is not None:
                    hi = float(np.percentile(np.abs(v), pclip))
                    v = np.clip(v, -hi, hi)
                    stats_entry["pclip"] = float(pclip)
                    stats_entry["hi"] = hi

                sel = (labels[:, 0].astype(int) == scope_idx) if scope_idx is not None else slice(None)
                if allow_asinh and mode == "asinh_robust":
                    _, sig = _robust_location_scale(v)
                    tau_cfg = cfg.get("tau", "auto")
                    tau = float(sig if tau_cfg == "auto" else tau_cfg)
                    v_tr = np.arcsinh(v / (tau + 1e-12))
                    mu, sig2 = _robust_location_scale(v_tr)
                    for ch in channels:
                        if ch is None:
                            continue
                        X = features[sel, ch, :]
                        m = masks_bool[sel, ch, :]
                        X[m] = np.arcsinh(X[m] / (tau + 1e-12))
                        X[m] = (X[m] - mu) / (sig2 + 1e-12)
                        np.clip(X, -clip_k, clip_k, out=X)
                        features[sel, ch, :] = X
                    stats_entry.update({"tau": tau, "mu": mu, "sigma": sig2})
                else:
                    if mode == "robust_zscore":
                        mu, sig = _robust_location_scale(v)
                    elif mode == "zscore":
                        mu, sig = float(np.mean(v)), float(np.std(v) + 1e-12)
                    else:
                        continue
                    for ch in channels:
                        if ch is None:
                            continue
                        X = features[sel, ch, :]
                        m = masks_bool[sel, ch, :]
                        if pclip is not None:
                            hi_f = float(stats_entry["hi"])
                            X[m] = np.clip(X[m], -hi_f, hi_f)
                        X[m] = (X[m] - mu) / (sig + 1e-12)
                        np.clip(X, -clip_k, clip_k, out=X)
                        features[sel, ch, :] = X
                    stats_entry.update({"mu": mu, "sigma": sig})

                norm_stats.setdefault(family, {})[str(scope_idx)] = stats_entry

        v_cfg = norm_cfg.get("voltage", {"mode": "none"})
        v_mode = v_cfg.get("mode", "none")
        h_cfg = norm_cfg.get("hyst", {"mode": "none"})
        h_mode = h_cfg.get("mode", "none")
        if v_mode == "phys_minmax":
            vmin = float(v_cfg.get("v_min"))
            vmax = float(v_cfg.get("v_max"))
            if not (np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin):
                raise ValueError("phys_minmax requires finite v_min < v_max")
            target_range = str(v_cfg.get("target_range", "zero_to_one")).strip().lower()
            if target_range not in {"zero_to_one", "minus_one_to_one"}:
                raise ValueError("NORMALIZATION.voltage.target_range must be one of {zero_to_one, minus_one_to_one}")
            for ch in (idx_vch, idx_vdis):
                if ch is None:
                    continue
                X = features[:, ch, :]
                m = masks_bool[:, ch, :]
                X[m] = (X[m] - vmin) / (vmax - vmin)
                if target_range == "minus_one_to_one":
                    X[m] = 2.0 * X[m] - 1.0
                    np.clip(X, -1.0, 1.0, out=X)
                else:
                    np.clip(X, 0.0, 1.0, out=X)
                features[:, ch, :] = X
            norm_stats["voltage"] = {
                "mode": "phys_minmax",
                "target_range": target_range,
                "v_min": vmin,
                "v_max": vmax,
            }
            if h_mode == "phys_minmax":
                if idx_h_raw is None:
                    raise ValueError("H_raw channel is required for NORMALIZATION.hyst.mode=phys_minmax")
                scale = float(vmax - vmin)
                h_scale = 2.0 * scale if target_range == "minus_one_to_one" else scale
                X = features[:, idx_h_raw, :]
                m = masks_bool[:, idx_h_raw, :]
                X[m] = X[m] / h_scale
                features[:, idx_h_raw, :] = X
                norm_stats["hyst"] = {
                    "mode": "phys_minmax",
                    "target_range": target_range,
                    "v_min": vmin,
                    "v_max": vmax,
                    "scale": h_scale,
                }
            else:
                _fit_family((idx_h_raw,), "hyst", h_cfg)
        else:
            _fit_family((idx_vch, idx_vdis), "voltage", v_cfg)
            _fit_family((idx_h_raw,), "hyst", h_cfg)

        _fit_family((idx_dv_ch, idx_dv_dis), "dvdq", norm_cfg.get("dvdq", {"mode": "none"}))
        _fit_family(
            (idx_d2_ch, idx_d2_dis),
            "d2vdq2",
            norm_cfg.get("d2vdq2", {"mode": "none"}),
            pclip=float(norm_cfg.get("d2vdq2", {}).get("pclip", 99.5)) if norm_cfg.get("d2vdq2", {}).get("mode", "none") in ("zscore", "robust_zscore") else None,
        )
        _fit_family(
            (idx_dqdv_ch, idx_dqdv_dis),
            "dqdv",
            norm_cfg.get("dqdv", {"mode": "none"}),
            allow_asinh=True,
        )

        for family in ("dvdq", "d2vdq2", "dqdv", "hyst"):
            norm_stats.setdefault(family, {"mode": "none"})

        return features, norm_stats

    def _apply_existing_normalization(
        self,
        features: np.ndarray,
        masks: np.ndarray,
        labels: np.ndarray,
        feature_names: List[str],
        norm_stats: Dict[str, dict],
    ) -> np.ndarray:
        if not norm_stats:
            return features

        name_to_idx = {n: i for i, n in enumerate(feature_names)}
        masks_bool = masks.astype(bool)

        idx_vch = name_to_idx.get("V_ch")
        idx_vdis = name_to_idx.get("V_dis")
        idx_dv_ch = name_to_idx.get("dVdQ_ch")
        idx_dv_dis = name_to_idx.get("dVdQ_dis")
        idx_d2_ch = name_to_idx.get("d2VdQ2_ch")
        idx_d2_dis = name_to_idx.get("d2VdQ2_dis")
        idx_dqdv_ch = name_to_idx.get("dQdV_ch")
        idx_dqdv_dis = name_to_idx.get("dQdV_dis")
        idx_h_raw = name_to_idx.get("H_raw")

        def _iter_family_entries(family: str):
            block = norm_stats.get(family, {})
            if not isinstance(block, dict) or not block:
                return []
            if "mode" in block:
                return [(slice(None), block)]

            entries = []
            for scope_key, stats in block.items():
                if not isinstance(stats, dict):
                    continue
                if scope_key == "None":
                    sel = slice(None)
                else:
                    try:
                        scope_idx = int(scope_key)
                    except (TypeError, ValueError):
                        continue
                    sel = labels[:, 0].astype(int) == scope_idx
                entries.append((sel, stats))
            return entries

        def _apply_affine(channels: Tuple[Optional[int], ...], family: str) -> None:
            for sel, stats in _iter_family_entries(family):
                mode = stats.get("mode", "none")
                if mode not in ("zscore", "robust_zscore"):
                    continue
                mu = float(stats["mu"])
                sig = float(stats["sigma"])
                clip_k = float(stats.get("clip_k", 5.0))
                hi = stats.get("hi")
                for ch in channels:
                    if ch is None:
                        continue
                    X = features[sel, ch, :]
                    m = masks_bool[sel, ch, :]
                    if hi is not None:
                        hi_f = float(hi)
                        X[m] = np.clip(X[m], -hi_f, hi_f)
                    X[m] = (X[m] - mu) / (sig + 1e-12)
                    np.clip(X, -clip_k, clip_k, out=X)
                    features[sel, ch, :] = X

        for _, stats in _iter_family_entries("voltage"):
            mode = stats.get("mode", "none")
            if mode == "phys_minmax":
                vmin = float(stats["v_min"])
                vmax = float(stats["v_max"])
                target_range = str(stats.get("target_range", "zero_to_one")).strip().lower()
                for ch in (idx_vch, idx_vdis):
                    if ch is None:
                        continue
                    X = features[:, ch, :]
                    m = masks_bool[:, ch, :]
                    X[m] = (X[m] - vmin) / (vmax - vmin)
                    if target_range == "minus_one_to_one":
                        X[m] = 2.0 * X[m] - 1.0
                        np.clip(X, -1.0, 1.0, out=X)
                    else:
                        np.clip(X, 0.0, 1.0, out=X)
                    features[:, ch, :] = X
            elif mode in ("zscore", "robust_zscore"):
                _apply_affine((idx_vch, idx_vdis), "voltage")
            break

        h_applied = False
        for _, stats in _iter_family_entries("hyst"):
            mode = stats.get("mode", "none")
            if mode == "phys_minmax":
                if idx_h_raw is None:
                    break
                scale = float(stats.get("scale", float(stats["v_max"]) - float(stats["v_min"])))
                X = features[:, idx_h_raw, :]
                m = masks_bool[:, idx_h_raw, :]
                X[m] = X[m] / scale
                features[:, idx_h_raw, :] = X
                h_applied = True
            elif mode in ("zscore", "robust_zscore"):
                _apply_affine((idx_h_raw,), "hyst")
                h_applied = True
            elif mode == "none":
                h_applied = True
            break

        _apply_affine((idx_dv_ch, idx_dv_dis), "dvdq")
        _apply_affine((idx_d2_ch, idx_d2_dis), "d2vdq2")
        if not h_applied:
            _apply_affine((idx_h_raw,), "hyst")

        for sel, stats in _iter_family_entries("dqdv"):
            mode = stats.get("mode", "none")
            if mode == "asinh_robust":
                tau = float(stats["tau"])
                mu = float(stats["mu"])
                sig = float(stats["sigma"])
                clip_k = float(stats.get("clip_k", 5.0))
                for ch in (idx_dqdv_ch, idx_dqdv_dis):
                    if ch is None:
                        continue
                    X = features[sel, ch, :]
                    m = masks_bool[sel, ch, :]
                    X[m] = np.arcsinh(X[m] / (tau + 1e-12))
                    X[m] = (X[m] - mu) / (sig + 1e-12)
                    np.clip(X, -clip_k, clip_k, out=X)
                    features[sel, ch, :] = X
            elif mode in ("zscore", "robust_zscore"):
                _apply_affine((idx_dqdv_ch, idx_dqdv_dis), "dqdv")

        return features

    def create_capacity_features_and_masks(
        self,
        return_derivatives: bool = True,
        compute_second_deriv: bool = True,
        compute_dqdv: bool = True,
        compute_hysteresis: bool = True,
        normalize: bool = False,
        norm_cfg: Optional[dict] = None,
        train_selector: Optional[np.ndarray] = None,
        r_ohmic_per_cycle: Optional[Dict[Tuple[str, int], float]] = None,
        save_debug: bool = False,
        frozen_norm_stats: Optional[Dict[str, dict]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], List[str], Dict[str, np.ndarray]]:
        del r_ohmic_per_cycle

        assert self.data is not None and self.total_cycles is not None
        raw_capacity_col = "Discharge capacity [A.h]"
        use_raw_capacity = raw_capacity_col in self.data.columns
        if not use_raw_capacity and ("cum_cap" not in self.data.columns or "dt" not in self.data.columns):
            logger.info(
                "Raw '%s' column is unavailable; integrating capacity from Current and Time columns.",
                raw_capacity_col,
            )
            self._integrate_capacity()

        df = self.data
        N = int(self.n_interp_points)
        if N <= 1:
            raise ValueError("n_interp_points must be >= 2")

        q_cap_grid = np.linspace(0.0, 1.0, N, dtype=np.float32)
        grid_abs = q_cap_grid.astype(np.float64) * self.q_max_ah

        feature_names = [
            "q_global",
            "q_cap",
            "V_ch",
            "V_dis",
            "dVdQ_ch",
            "dVdQ_dis",
            "d2VdQ2_ch",
            "d2VdQ2_dis",
            "dQdV_ch",
            "dQdV_dis",
            "H_raw",
            "mask",
        ]
        label_names = ["dataset_id", "Cycle number", "Normalised cycle number", "SOH", "q_cap_max", "charging rate", "q0_Ah"]
        C = len(feature_names)

        features = np.full((self.total_cycles, C, N), self.padding_value, dtype=np.float32)
        masks = np.zeros((self.total_cycles, C, N), dtype=np.float32)
        labels = np.zeros((self.total_cycles, 7), dtype=np.float32)

        ds_unique = df["source_dataset"].unique().tolist()
        ds_to_idx = {ds: idx for idx, ds in enumerate(ds_unique)}
        if use_raw_capacity:
            logger.info(
                "Using raw '%s' values as the capacity coordinate and q_cap_max source with q_max_ah=%.3f.",
                raw_capacity_col,
                self.q_max_ah,
            )
        else:
            logger.warning(
                "Raw '%s' column is unavailable; falling back to integrated Current*dt capacity.",
                raw_capacity_col,
            )

        for k, ((ds, cyc), g) in enumerate(df.groupby(self.groupby_cols, sort=False)):
            curr = g["Current"].to_numpy(dtype=float)
            volt = g["Voltage"].to_numpy(dtype=float)
            raw_capacity = (
                g[raw_capacity_col].to_numpy(dtype=float) if use_raw_capacity else None
            )

            charge_raw = curr > self.i_eps
            discharge_raw = curr < -self.i_eps

            if raw_capacity is not None:
                capacity_coord = raw_capacity
                finite_capacity = np.isfinite(capacity_coord)
                charge_mask = charge_raw & finite_capacity
                discharge_mask = discharge_raw & finite_capacity

                q_ch = capacity_coord[charge_mask]
                q_dis = capacity_coord[discharge_mask]
                v_ch = volt[charge_mask]
                v_dis = volt[discharge_mask]

                finite_cycle_capacity = capacity_coord[finite_capacity]
                charge_end_ah = (
                    min(float(np.max(finite_cycle_capacity)), self.q_max_ah)
                    if finite_cycle_capacity.size
                    else 0.0
                )

                q_ch_soc = np.clip(charge_end_ah - q_ch, 0.0, self.q_max_ah)
                q_dis_soc = np.clip(charge_end_ah - q_dis, 0.0, self.q_max_ah)
            else:
                q_cycle = g["cum_cap"].to_numpy(dtype=float) - float(g["cum_cap"].iloc[0])

                def _segment_capacity(Q: np.ndarray, mask: np.ndarray, charge: bool) -> np.ndarray:
                    if not mask.any():
                        return np.array([], dtype=float)
                    qseg = Q[mask]
                    out = (qseg - qseg[0]) if charge else -(qseg - qseg[0])
                    return np.maximum(out, 0.0)

                q_ch = _segment_capacity(q_cycle, charge_raw, charge=True)
                q_dis = _segment_capacity(q_cycle, discharge_raw, charge=False)
                v_ch = volt[charge_raw]
                v_dis = volt[discharge_raw]

                charge_end_ah = min(float(q_ch.max()), self.q_max_ah) if q_ch.size else 0.0
                if charge_end_ah <= 0.0 and q_dis.size:
                    charge_end_ah = min(float(q_dis.max()), self.q_max_ah)

                q_ch_soc = np.clip(q_ch, 0.0, self.q_max_ah)
                q_dis_soc = np.clip(charge_end_ah - q_dis, 0.0, self.q_max_ah)

            want_second = bool(compute_second_deriv)
            vch_grid, dVdQ_ch_g, d2VdQ2_ch_g, valid_ch = _pchip_values_and_derivatives(
                q_ch_soc, v_ch, grid_abs, self.padding_value, want_second
            )
            vdis_grid, dVdQ_dis_g, d2VdQ2_dis_g, valid_dis = _pchip_values_and_derivatives(
                q_dis_soc, v_dis, grid_abs, self.padding_value, want_second
            )

            common_valid = valid_ch & valid_dis
            charge_missing_inside = np.zeros_like(common_valid, dtype=bool)
            discharge_missing_inside = np.zeros_like(common_valid, dtype=bool)
            if not common_valid.any():
                if valid_ch.any():
                    common_valid = valid_ch.copy()
                    discharge_missing_inside = common_valid & (~valid_dis)
                elif valid_dis.any():
                    common_valid = valid_dis.copy()
                    charge_missing_inside = common_valid & (~valid_ch)

            if charge_missing_inside.any():
                vch_grid[charge_missing_inside] = self.v_dis_fill_value
                valid_ch = common_valid.copy()
                dVdQ_ch_g, d2VdQ2_ch_g = _finite_difference_on_valid_grid(
                    values=vch_grid,
                    grid=grid_abs,
                    valid=valid_ch,
                    pad_value=self.padding_value,
                    want_second=want_second,
                )

            if discharge_missing_inside.any():
                vdis_grid[discharge_missing_inside] = self.v_dis_fill_value
                valid_dis = common_valid.copy()

            # If discharge has already reached the left-tail fill value at some
            # q > 0, keep it at that value for the entire left tail down to q = 0.
            fill_idx = np.flatnonzero(common_valid & (vdis_grid <= self.v_dis_fill_value))
            if fill_idx.size:
                fill_stop = int(fill_idx.max()) + 1
                vdis_grid[:fill_stop] = self.v_dis_fill_value

            if discharge_missing_inside.any() or fill_idx.size:
                dVdQ_dis_g, d2VdQ2_dis_g = _finite_difference_on_valid_grid(
                    values=vdis_grid,
                    grid=grid_abs,
                    valid=valid_dis,
                    pad_value=self.padding_value,
                    want_second=want_second,
                )

            valid_ch = common_valid.copy()
            valid_dis = common_valid.copy()

            mtrim = max(1, int(self.edge_trim_frac * N))
            interior_ch = common_valid.copy()
            interior_dis = valid_dis.copy()
            if interior_ch.any():
                i0, i1 = np.flatnonzero(interior_ch)[0], np.flatnonzero(interior_ch)[-1]
                interior_ch[: i0 + mtrim] = False
                interior_ch[max(i0, i1 - mtrim + 1): i1 + 1] = False
            if interior_dis.any():
                j0, j1 = np.flatnonzero(interior_dis)[0], np.flatnonzero(interior_dis)[-1]
                interior_dis[: j0 + mtrim] = False
                interior_dis[max(j0, j1 - mtrim + 1): j1 + 1] = False

            dqdv_interior_ch = interior_ch.copy()
            dqdv_interior_dis = interior_dis.copy()

            def _raw_recip_pos(dv: np.ndarray, interior: np.ndarray) -> np.ndarray:
                out = np.full_like(dv, self.padding_value, dtype=np.float32)
                if interior.any():
                    abs_dv = np.abs(dv[interior]).astype(np.float32)
                    recip = np.full_like(abs_dv, np.inf, dtype=np.float32)
                    np.divide(1.0, abs_dv, out=recip, where=abs_dv > 0.0)
                    out[interior] = recip
                return out

            if compute_dqdv:
                dQdV_ch_g = _raw_recip_pos(dVdQ_ch_g, dqdv_interior_ch)
                dQdV_dis_g = _raw_recip_pos(dVdQ_dis_g, dqdv_interior_dis)
            else:
                dQdV_ch_g = np.full_like(vch_grid, self.padding_value, dtype=np.float32)
                dQdV_dis_g = np.full_like(vdis_grid, self.padding_value, dtype=np.float32)

            q_global_grid = q_cap_grid.copy()
            mask_channel = common_valid.astype(np.float32)
            hysteresis_grid = np.full_like(vch_grid, self.padding_value, dtype=np.float32)
            if compute_hysteresis:
                hysteresis_grid[common_valid] = (
                    vch_grid[common_valid] - vdis_grid[common_valid]
                ).astype(np.float32)

            features[k, 0, :] = q_global_grid
            features[k, 1, :] = q_cap_grid
            features[k, 2, :] = vch_grid
            features[k, 3, :] = vdis_grid
            if return_derivatives:
                features[k, 4, :] = np.where(interior_ch, dVdQ_ch_g, self.padding_value).astype(np.float32)
                features[k, 5, :] = np.where(interior_dis, dVdQ_dis_g, self.padding_value).astype(np.float32)
            if compute_second_deriv:
                if d2VdQ2_ch_g is None:
                    d2VdQ2_ch_g = np.full_like(vch_grid, self.padding_value, dtype=np.float32)
                if d2VdQ2_dis_g is None:
                    d2VdQ2_dis_g = np.full_like(vdis_grid, self.padding_value, dtype=np.float32)
                features[k, 6, :] = np.where(interior_ch, d2VdQ2_ch_g, self.padding_value).astype(np.float32)
                features[k, 7, :] = np.where(interior_dis, d2VdQ2_dis_g, self.padding_value).astype(np.float32)
            if compute_dqdv:
                features[k, 8, :] = dQdV_ch_g
                features[k, 9, :] = dQdV_dis_g
            features[k, 10, :] = hysteresis_grid
            features[k, 11, :] = mask_channel

            masks[k, 0, :] = 1.0
            masks[k, 1, :] = 1.0
            masks[k, 2, common_valid] = 1.0
            masks[k, 3, common_valid] = 1.0
            if return_derivatives:
                masks[k, 4, interior_ch] = 1.0
                masks[k, 5, interior_dis] = 1.0
            if compute_second_deriv:
                masks[k, 6, interior_ch] = 1.0
                masks[k, 7, interior_dis] = 1.0
            if compute_dqdv:
                masks[k, 8, dqdv_interior_ch] = 1.0
                masks[k, 9, dqdv_interior_dis] = 1.0
            if compute_hysteresis:
                masks[k, 10, common_valid] = 1.0
            masks[k, 11, :] = 1.0

            if raw_capacity is not None:
                finite_raw_capacity = raw_capacity[np.isfinite(raw_capacity)]
                achieved_ah = (
                    float(np.max(finite_raw_capacity)) if finite_raw_capacity.size else 0.0
                )
            else:
                achieved_ah = 0.0
                if q_ch.size:
                    achieved_ah = max(achieved_ah, float(np.max(q_ch)))
                if q_dis.size:
                    achieved_ah = max(achieved_ah, float(np.max(q_dis)))
            q_cap_max = float(np.clip(achieved_ah / self.q_max_ah, 0.0, 1.0))

            labels[k, 0] = ds_to_idx[ds]
            labels[k, 1] = float(cyc)
            cycle_max = float(g["cycle_max"].iloc[0]) if "cycle_max" in g.columns else np.nan
            labels[k, 2] = float(cyc / cycle_max) if np.isfinite(cycle_max) and cycle_max > 0 else np.nan
            labels[k, 3] = q_cap_max
            labels[k, 4] = q_cap_max
            labels[k, 5] = float(g["charging_rate"].iloc[0]) if "charging_rate" in g.columns else np.nan
            labels[k, 6] = 1.0

        norm_stats: Dict[str, dict] = {}
        if normalize:
            if frozen_norm_stats is None:
                features, norm_stats = self._apply_normalization(
                    features=features,
                    masks=masks,
                    labels=labels,
                    norm_cfg=norm_cfg,
                    feature_names=feature_names,
                    train_selector=train_selector,
                )
            else:
                features = self._apply_existing_normalization(
                    features=features,
                    masks=masks,
                    labels=labels,
                    feature_names=feature_names,
                    norm_stats=frozen_norm_stats,
                )
                norm_stats = frozen_norm_stats

        token_masks = masks[:, 2, :].copy()
        _assert_same_branch_masks(masks[:, 2, :] > 0.5, masks[:, 3, :] > 0.5)
        extras = {
            "feature_names": np.array(feature_names),
            "label_names": np.array(label_names),
            "ds_to_idx": ds_to_idx,
            "norm_stats": norm_stats,
        }

        if save_debug:
            outdir = Path(self.paths.predicted_data) / "capacity_preproc_simple"
            outdir.mkdir(parents=True, exist_ok=True)
            np.save(outdir / "features.npy", features)
            np.save(outdir / "masks.npy", masks)
            np.save(outdir / "token_masks.npy", token_masks)
            np.save(outdir / "labels.npy", labels)

        logger.info(
            "Built simplified capacity features: features.shape=%s, masks.shape=%s",
            features.shape,
            masks.shape,
        )
        return features, masks, labels, token_masks, feature_names, label_names, extras
