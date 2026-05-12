import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D
from pathlib import Path
from typing import List, Optional



def plot_capacity_multichannel(
    features: np.ndarray,
    masks: np.ndarray,
    labels: np.ndarray,
    feature_names: List[str],
    source_datasets: Optional[List] = None,   # ints or strings ok (e.g., [0] or ["dataset_0"])
    cycles: Optional[List] = None,            # raw cycle numbers to plot (e.g., [1, 10, 50])
    save_dir: Optional[Path] = None,
    dpi: int = 300,
    combine_datasets: bool = True,
    color_mode: str = "categorical",   # "categorical" or "cycle"
    cmap_name: str = "viridis",        # used when color_mode="cycle"
):
    """
    Multichannel visualization for calibration checks.

    Panels (if available in `features`):
      (1) V_ch & V_dis vs q_cap
      (2) Hysteresis (H_raw solid, H_corr dotted) + median ΔH annotation
      (3) dV/dQ (charge & discharge)
      (4) dQ/dV (ICA; charge & discharge)
      (5) d2V/dQ2 (charge & discharge)
      (6) Coverage indicator (q_cap_max; vertical lines)

    Labels columns: [dataset_id, Cycle, norm_cycle, SOH, q_cap_max, charging_rate, q0_Ah].
    """
    assert features.ndim == 3 and features.shape == masks.shape, "features/masks must be (B, C, N)"
    B, C, N = features.shape

    # ---- channel indices (by name) ----
    name_to_idx = {n: i for i, n in enumerate(feature_names)}

    IDX_QCAP  = name_to_idx.get("q_cap", 1)
    IDX_VCH   = name_to_idx.get("V_ch", 2)
    IDX_VDCH  = name_to_idx.get("V_dis", 3)
    IDX_DV_CH = name_to_idx.get("dVdQ_ch", 4)
    IDX_DV_DH = name_to_idx.get("dVdQ_dis", 5)

    IDX_D2V_CH  = name_to_idx.get("d2VdQ2_ch", None)
    IDX_D2V_DH  = name_to_idx.get("d2VdQ2_dis", None)
    IDX_DQDV_CH = name_to_idx.get("dQdV_ch", None)
    IDX_DQDV_DH = name_to_idx.get("dQdV_dis", None)
    IDX_HRAW    = name_to_idx.get("H_raw", None)
    IDX_HCORR   = name_to_idx.get("H_corr", None)

    q = features[0, IDX_QCAP, :]

    # ---- dataset & cycle selection helpers ----
    all_ds = np.unique(labels[:, 0].astype(int)).tolist()

    def _parse_ds(x):
        if isinstance(x, (int, np.integer)): return int(x)
        if isinstance(x, str):
            s = x.strip().lower()
            if s.startswith("dataset_"): s = s.split("dataset_")[-1]
            try: return int(s)
            except: return None
        return None

    if not source_datasets:
        ds_indices = all_ds
    else:
        parsed = [_parse_ds(x) for x in source_datasets]
        ds_indices = [d for d in parsed if d is not None and d in all_ds] or all_ds

    def _rows_for_dataset(ds_idx):
        sel = labels[:, 0].astype(int) == ds_idx
        rows = np.nonzero(sel)[0]
        cyc_nums = labels[sel, 1].astype(int)
        return rows, cyc_nums

    def _parse_cycles(cyc_list, available):
        if not cyc_list: return available.tolist()
        want = []
        for c in cyc_list:
            try: want.append(int(c))
            except: pass
        avset = set(available.tolist())
        return [c for c in want if c in avset]

    # ---- colors ----
    default_colors = plt.rcParams.get("axes.prop_cycle", None)
    base_colors = default_colors.by_key().get("color", None) if default_colors else None
    if not base_colors:
        base_colors = [f"C{i}" for i in range(10)]

    def _plot_rows(
        rows_pick,
        title,
        save_name,
        color_mode: str = "categorical",  # "categorical" or "cycle"
        cmap_name: str = "viridis",
    ):
        fig, axs = plt.subplots(3, 2, figsize=(13, 10), sharex=True, dpi=dpi)
        (ax_v, ax_h), (ax_dv, ax_dqdv), (ax_d2v, ax_cov) = axs
        fig.subplots_adjust(hspace=0.28, wspace=0.22, right=0.83)

        # Segment legend (solid=charge, dashed=discharge)
        seg_handles = [
            Line2D([0], [0], color="black", lw=1.6, ls="-"),
            Line2D([0], [0], color="black", lw=1.6, ls="--"),
        ]
        seg_labels = ["Charge (solid)", "Discharge (dashed)"]

        # Cycle numbers for this selection
        cyc_nums = [int(labels[r, 1]) for r in rows_pick]
        cyc_min, cyc_max = min(cyc_nums), max(cyc_nums)

        # Colormap
        if color_mode == "cycle":
            cmap = cm.get_cmap(cmap_name)
        else:
            cmap = cm.get_cmap(cmap_name, len(rows_pick))

        cyc_handles, cyc_labels = [], []
        deltas_all = []

        for j, r in enumerate(rows_pick):
            cyc_no = int(labels[r, 1])

            if color_mode == "cycle":
                # Normalize cycle number to [0,1] for colormap
                if cyc_max > cyc_min:
                    t = (cyc_no - cyc_min) / (cyc_max - cyc_min)
                else:
                    t = 0.5
                color = cmap(t)
            else:
                # One discrete color per curve (current behaviour)
                color = cmap(j)

            qmax = float(labels[r, 4]) if labels.shape[1] >= 5 else 1.0

            mVch  = masks[r, IDX_VCH,  :].astype(bool)
            mVdh  = masks[r, IDX_VDCH, :].astype(bool)
            mdVch = masks[r, IDX_DV_CH, :].astype(bool)
            mdVdh = masks[r, IDX_DV_DH, :].astype(bool)

            # (1) Voltage
            if np.any(mVch):
                ax_v.plot(q[mVch], features[r, IDX_VCH, mVch],
                        color=color, ls="-", lw=1.6)
            if np.any(mVdh):
                ax_v.plot(q[mVdh], features[r, IDX_VDCH, mVdh],
                        color=color, ls="--", lw=1.6)

            # (2) Hysteresis (H_raw solid, H_corr dotted)
            if IDX_HRAW is not None:
                mHr = masks[r, IDX_HRAW, :].astype(bool)
                if np.any(mHr):
                    ax_h.plot(q[mHr], features[r, IDX_HRAW, mHr],
                            color=color, lw=1.6, ls="-")
            if IDX_HCORR is not None:
                mHc = masks[r, IDX_HCORR, :].astype(bool)
                if np.any(mHc):
                    ax_h.plot(q[mHc], features[r, IDX_HCORR, mHc],
                            color=color, lw=1.6, ls=":")

            # collect ΔH only where BOTH are valid
            if (IDX_HRAW is not None) and (IDX_HCORR is not None):
                mBoth = masks[r, IDX_HRAW, :].astype(bool) & masks[r, IDX_HCORR, :].astype(bool)
                if np.any(mBoth):
                    d = features[r, IDX_HRAW, mBoth] - features[r, IDX_HCORR, mBoth]
                    if d.size:
                        deltas_all.append(np.median(np.abs(d)))

            # (3) dV/dQ
            if np.any(mdVch):
                ax_dv.plot(q[mdVch], features[r, IDX_DV_CH, mdVch],
                        color=color, ls="-", lw=1.2)
            if np.any(mdVdh):
                ax_dv.plot(q[mdVdh], features[r, IDX_DV_DH, mdVdh],
                        color=color, ls="--", lw=1.2)

            # (4) ICA: dQ/dV
            if IDX_DQDV_CH is not None:
                m = masks[r, IDX_DQDV_CH, :].astype(bool)
                if np.any(m):
                    ax_dqdv.plot(q[m], features[r, IDX_DQDV_CH, m],
                                color=color, ls="-", lw=1.0)
            if IDX_DQDV_DH is not None:
                m = masks[r, IDX_DQDV_DH, :].astype(bool)
                if np.any(m):
                    ax_dqdv.plot(q[m], features[r, IDX_DQDV_DH, m],
                                color=color, ls="--", lw=1.0)

            # (5) d2V/dQ2
            if IDX_D2V_CH is not None:
                m = masks[r, IDX_D2V_CH, :].astype(bool)
                if np.any(m):
                    ax_d2v.plot(q[m], features[r, IDX_D2V_CH, m],
                                color=color, ls="-", lw=1.0)
            if IDX_D2V_DH is not None:
                m = masks[r, IDX_D2V_DH, :].astype(bool)
                if np.any(m):
                    ax_d2v.plot(q[m], features[r, IDX_D2V_DH, m],
                                color=color, ls="--", lw=1.0)

            # (6) Coverage marker
            ax_cov.axvline(qmax, color=color, lw=1.6, alpha=0.9)

            cyc_handles.append(Line2D([0], [0], color=color, lw=2.0))
            cyc_labels.append(f"Cycle {cyc_no}")

        # axes setup
        ax_v.set_title("Voltage vs q_cap"); ax_v.set_ylabel("Voltage [V]"); ax_v.grid(True, ls=":", lw=0.6)
        ax_h.set_title("Hysteresis"); ax_h.set_ylabel("V_ch − V_dis [V]"); ax_h.grid(True, ls=":", lw=0.6)
        ax_dv.set_title("dV/dQ"); ax_dv.set_ylabel("V/Ah"); ax_dv.grid(True, ls=":", lw=0.6)
        ax_dqdv.set_title("ICA: dQ/dV"); ax_dqdv.set_ylabel("Ah/V"); ax_dqdv.grid(True, ls=":", lw=0.6)
        ax_d2v.set_title("d²V/dQ²"); ax_d2v.set_ylabel("1/Ah²"); ax_d2v.grid(True, ls=":", lw=0.6)
        ax_cov.set_title("Coverage (q_cap_max) and x-axis"); ax_cov.set_ylabel("—"); ax_cov.grid(True, ls=":", lw=0.6)
        for ax in (ax_dv, ax_dqdv, ax_d2v, ax_cov):
            ax.set_xlabel("q_cap")

        # segment legend
        fig.legend(seg_handles, seg_labels, title="Segments",
                loc="center left", bbox_to_anchor=(0.86, 0.70),
                frameon=True, fontsize=9, title_fontsize=9)

        # cycle legend / colorbar
        if color_mode == "categorical":
            # existing behaviour: explicit legend per cycle
            ncol = 1 if len(cyc_handles) <= 12 else 2 if len(cyc_handles) <= 30 else 3
            fig.legend(cyc_handles, cyc_labels, title="Cycles (color)",
                    loc="center left", bbox_to_anchor=(0.86, 0.30),
                    frameon=True, ncol=ncol, fontsize=8, title_fontsize=9)
        else:
            # trend view: colorbar encodes cycle number
            norm = plt.Normalize(vmin=cyc_min, vmax=cyc_max)
            sm = cm.ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=axs.ravel().tolist(), fraction=0.02, pad=0.02)
            cbar.set_label("Cycle number")

        # hysteresis legend (style legend independent of color)
        hyst_handles = [
            Line2D([0], [0], color="black", lw=1.8, ls="-"),
            Line2D([0], [0], color="black", lw=1.8, ls=":")
        ]
        hyst_labels = ["H_raw (uncorrected)", "H_corr (IR-corrected)"]
        ax_h.legend(hyst_handles, hyst_labels, loc="best",
                    frameon=True, fontsize=9, title="Curves", title_fontsize=9)

        # median ΔH annotation
        if deltas_all:
            delta_med = float(np.median(np.asarray(deltas_all)))
            ax_h.text(
                0.02, 0.96, f"median ΔH ≈ {delta_med:.2f} V",
                transform=ax_h.transAxes, ha="left", va="top",
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", alpha=0.9),
            )

        fig.suptitle(title)

        if save_dir is not None:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            fig.savefig(Path(save_dir) / save_name, bbox_inches="tight", dpi=dpi)
        plt.close(fig)


    # ---- build row indices to plot ----
    if combine_datasets:
        rows_all = []
        for ds_idx in ds_indices:
            rows, cyc_nums = _rows_for_dataset(ds_idx)
            chosen = _parse_cycles(cycles, cyc_nums)
            if not chosen: continue
            lookup = {c: r for c, r in zip(cyc_nums.tolist(), rows.tolist())}
            rows_all.extend([lookup[c] for c in chosen])
        if not rows_all: return
        title = f"Overlay of {len(rows_all)} cycle(s) across {len(ds_indices)} dataset(s)"
        _plot_rows(rows_all, title, save_name="ALL_DATASETS_multichannel.png", color_mode=color_mode, cmap_name=cmap_name)
    else:
        for ds_idx in ds_indices:
            rows, cyc_nums = _rows_for_dataset(ds_idx)
            chosen = _parse_cycles(cycles, cyc_nums)
            if not chosen: continue
            lookup = {c: r for c, r in zip(cyc_nums.tolist(), rows.tolist())}
            rows_pick = [lookup[c] for c in chosen]
            title = f"dataset_{ds_idx} — {len(rows_pick)} cycle(s)"
            _plot_rows(rows_pick, title, save_name=f"dataset_{ds_idx}_multichannel.png")


def plot_capacity_multichannel_paper(
    features: np.ndarray,
    masks: np.ndarray,
    labels: np.ndarray,
    feature_names: List[str],
    source_datasets: Optional[List] = None,
    cycles: Optional[List] = None,
    save_dir: Optional[Path] = None,
    dpi: int = 600,
    combine_datasets: bool = True,
    charge_cmap_name: str = "viridis",
    discharge_cmap_name: str = "rainbow",
):
    """
    Publication-style 2x2 overlay layout:
    top-left: V_ch and V_dis vs q_cap
    top-right: dV/dQ
    bottom-left: d²V/dQ²
    bottom-right: dQ/dV

    Charge curves use `charge_cmap_name`; discharge curves use
    `discharge_cmap_name`. Both colormaps encode cycle number.
    """
    assert features.ndim == 3 and features.shape == masks.shape, "features/masks must be (B, C, N)"

    name_to_idx = {n: i for i, n in enumerate(feature_names)}
    idx_qcap = name_to_idx.get("q_cap", 1)
    q = features[0, idx_qcap, :]

    panel_specs = [
        ("Voltage [V]", name_to_idx.get("V_ch"), name_to_idx.get("V_dis")),
        ("dV/dQ [V/Ah]", name_to_idx.get("dVdQ_ch"), name_to_idx.get("dVdQ_dis")),
        ("d²V/dQ² [1/Ah²]", name_to_idx.get("d2VdQ2_ch"), name_to_idx.get("d2VdQ2_dis")),
        ("dQ/dV [Ah/V]", name_to_idx.get("dQdV_ch"), name_to_idx.get("dQdV_dis")),
    ]

    all_ds = np.unique(labels[:, 0].astype(int)).tolist()

    def _parse_ds(value):
        if isinstance(value, (int, np.integer)):
            return int(value)
        if isinstance(value, str):
            text = value.strip().lower()
            if text.startswith("dataset_"):
                text = text.split("dataset_")[-1]
            try:
                return int(text)
            except ValueError:
                return None
        return None

    if not source_datasets:
        ds_indices = all_ds
    else:
        parsed = [_parse_ds(value) for value in source_datasets]
        ds_indices = [ds_idx for ds_idx in parsed if ds_idx is not None and ds_idx in all_ds] or all_ds

    def _rows_for_dataset(ds_idx):
        selected = labels[:, 0].astype(int) == ds_idx
        rows = np.nonzero(selected)[0]
        cyc_nums = labels[selected, 1].astype(int)
        return rows, cyc_nums

    def _dataset_label(ds_idx):
        selected = labels[:, 0].astype(int) == ds_idx
        if not np.any(selected):
            return f"dataset_{ds_idx}"

        charging_rate = float(np.median(labels[selected, 5]))
        return f"dataset_{ds_idx} ({charging_rate:.2f}C)"

    def _parse_cycles(cyc_list, available):
        if not cyc_list:
            return available.tolist()
        selected = []
        for cyc in cyc_list:
            try:
                selected.append(int(cyc))
            except (TypeError, ValueError):
                continue
        available_set = set(available.tolist())
        return [cyc for cyc in selected if cyc in available_set]

    def _style_axis(ax):
        ax.grid(False)
        ax.minorticks_on()
        ax.tick_params(
            axis="both",
            which="major",
            direction="in",
            top=True,
            right=True,
            length=4,
            width=0.9,
            pad=3,
        )
        ax.tick_params(
            axis="both",
            which="minor",
            direction="in",
            top=True,
            right=True,
            length=2.5,
            width=0.7,
        )
        for spine in ax.spines.values():
            spine.set_linewidth(0.9)

    def _plot_rows(rows_pick, save_name):
        if not rows_pick:
            return

        cycle_numbers = [int(labels[row, 1]) for row in rows_pick]
        cyc_min = min(cycle_numbers)
        cyc_max = max(cycle_numbers)
        charge_cmap = cm.get_cmap(charge_cmap_name)
        discharge_cmap = cm.get_cmap(discharge_cmap_name)

        style = {
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }

        with plt.rc_context(style):
            fig, axs = plt.subplots(2, 2, figsize=(11.5, 8.6), dpi=dpi, sharex=True)
            fig.subplots_adjust(left=0.10, right=0.86, bottom=0.10, top=0.90, hspace=0.20, wspace=0.20)

            ax_v, ax_dv = axs[0]
            ax_d2v, ax_dqdv = axs[1]

            for ax in axs.ravel():
                _style_axis(ax)

            # ax_v.set_title("Voltage vs q_cap")
            # ax_dv.set_title("dV/dQ vs q_cap")
            # ax_d2v.set_title("d²V/dQ² vs q_cap")
            # ax_dqdv.set_title("dQ/dV vs q_cap")

            ax_v.set_ylabel("Normalized Voltage")
            ax_dv.set_ylabel("Normalized dV/dQ")
            ax_d2v.set_ylabel("Normalized d²V/dQ²")
            ax_dqdv.set_ylabel("Normalized dQ/dV")
            ax_d2v.set_xlabel("Normalized capacity")
            ax_dqdv.set_xlabel("Normalized capacity")

            for row_idx, cyc_no in zip(rows_pick, cycle_numbers):
                if cyc_max > cyc_min:
                    t = (cyc_no - cyc_min) / (cyc_max - cyc_min)
                else:
                    t = 0.5

                charge_color = charge_cmap(t)
                discharge_color = discharge_cmap(t)

                for ax, _, idx_charge, idx_discharge in (
                    (ax_v, "Voltage [V]", panel_specs[0][1], panel_specs[0][2]),
                    (ax_dv, "dV/dQ [V/Ah]", panel_specs[1][1], panel_specs[1][2]),
                    (ax_d2v, "d²V/dQ² [1/Ah²]", panel_specs[2][1], panel_specs[2][2]),
                    (ax_dqdv, "dQ/dV [Ah/V]", panel_specs[3][1], panel_specs[3][2]),
                ):
                    if idx_charge is not None:
                        mask_charge = masks[row_idx, idx_charge, :].astype(bool)
                        if np.any(mask_charge):
                            ax.plot(
                                q[mask_charge],
                                features[row_idx, idx_charge, mask_charge],
                                color=charge_color,
                                lw=1.4,
                                ls="-",
                                alpha=0.95,
                            )

                    if idx_discharge is not None:
                        mask_discharge = masks[row_idx, idx_discharge, :].astype(bool)
                        if np.any(mask_discharge):
                            ax.plot(
                                q[mask_discharge],
                                features[row_idx, idx_discharge, mask_discharge],
                                color=discharge_color,
                                lw=1.4,
                                ls="-",
                                alpha=0.95,
                            )

            segment_handles = [
                Line2D([0], [0], color="black", lw=1.8, ls="-"),
                Line2D([0], [0], color="black", lw=1.8, ls="--"),
            ]
            # fig.legend(
            #     segment_handles,
            #     ["Charge", "Discharge"],
            #     loc="upper center",
            #     bbox_to_anchor=(0.43, 0.985),
            #     ncol=2,
            #     frameon=False,
            #     handlelength=2.6,
            # )

            norm = plt.Normalize(vmin=cyc_min, vmax=cyc_max)
            charge_sm = cm.ScalarMappable(norm=norm, cmap=charge_cmap)
            discharge_sm = cm.ScalarMappable(norm=norm, cmap=discharge_cmap)
            charge_sm.set_array([])
            discharge_sm.set_array([])

            cax_charge = fig.add_axes([0.89, 0.56, 0.022, 0.24])
            cax_discharge = fig.add_axes([0.89, 0.18, 0.022, 0.24])
            cbar_charge = fig.colorbar(charge_sm, cax=cax_charge)
            cbar_discharge = fig.colorbar(discharge_sm, cax=cax_discharge)
            cbar_charge.set_label("Charge cycle")
            cbar_discharge.set_label("Discharge cycle")
            ds_in_plot = np.unique(labels[rows_pick, 0].astype(int))
            if ds_in_plot.size == 1:
                fig.suptitle(_dataset_label(int(ds_in_plot[0])))

            if save_dir is not None:
                Path(save_dir).mkdir(parents=True, exist_ok=True)
                output_path = Path(save_dir) / save_name
                fig.savefig(output_path, bbox_inches="tight", dpi=dpi)
            plt.close(fig)

    if combine_datasets:
        rows_all = []
        for ds_idx in ds_indices:
            rows, cyc_nums = _rows_for_dataset(ds_idx)
            chosen = _parse_cycles(cycles, cyc_nums)
            if not chosen:
                continue
            lookup = {cyc: row for cyc, row in zip(cyc_nums.tolist(), rows.tolist())}
            rows_all.extend([lookup[cyc] for cyc in chosen])
        if not rows_all:
            return
        _plot_rows(rows_all, save_name="ALL_DATASETS_multichannel_paper.png")
    else:
        for ds_idx in ds_indices:
            rows, cyc_nums = _rows_for_dataset(ds_idx)
            chosen = _parse_cycles(cycles, cyc_nums)
            if not chosen:
                continue
            lookup = {cyc: row for cyc, row in zip(cyc_nums.tolist(), rows.tolist())}
            rows_pick = [lookup[cyc] for cyc in chosen]
            _plot_rows(rows_pick, save_name=f"dataset_{ds_idx}_multichannel_paper.png")





