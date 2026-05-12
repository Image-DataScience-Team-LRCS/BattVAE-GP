import re
import os
import torch
from typing import List, Tuple, Any, Optional
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def load_data(
    file_path: Path, padding_value=-9999.0
) -> Tuple[
    List[int], List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]
]:
    cycles = []
    voltages = []
    currents = []
    dv_dt_values = []
    di_dt_values = []
    # Regular expression to capture the cycle, voltage array, current array, and dv_dt, di_dt arrays
    line_pattern = re.compile(
        r"(\d+),\s*\[([^\]]+)\],\s*\[([^\]]+)\],\s*\[([^\]]+)\],\s*\[([^\]]+)\]"
    )

    with open(file_path, "r") as file:
        for line in file:
            # Strip the line of any extra spaces
            line = line.strip()

            # Match the line to the pattern
            match = line_pattern.match(line)
            if match:
                cycle = int(match.group(1))
                # Create masks for non-padding values
                # Replace padding values with zeros instead of masking
                voltage_values = np.fromstring(match.group(2), sep=",")
                voltage_values[voltage_values == padding_value] = 0

                current_values = np.fromstring(match.group(3), sep=",")
                current_values[current_values == padding_value] = 0

                dv_dt = np.fromstring(match.group(4), sep=",")
                dv_dt[dv_dt == padding_value] = 0

                di_dt = np.fromstring(match.group(5), sep=",")
                di_dt[di_dt == padding_value] = 0

                cycles.append(cycle)
                voltages.append(voltage_values)
                currents.append(current_values)
                dv_dt_values.append(dv_dt)
                di_dt_values.append(di_dt)

    return cycles, voltages, currents, dv_dt_values, di_dt_values


def animate_reconstructed_data(
    original_data_path: Path, predicted_data_path: Path, steps: int, save_path: Path
) -> None:
    """
    Create an animation showing voltage over cycles.
    Args:
        original_data (pd.DataFrame): DataFrame with 'Cycle' and 'Voltage' columns for original data.
        predicted_data (pd.DataFrame): DataFrame with 'Cycle' and 'Voltage' columns for predicted data.
        save_path (str, optional): Path to save the animation (e.g., 'animation.mp4').
    """

    # Load data
    orig_cycles, orig_voltages, orig_currents, orig_dv_dt, orig_di_dt = load_data(
        original_data_path
    )
    pred_cycles, pred_voltages, pred_currents, pred_dv_dt, pred_di_dt = load_data(
        predicted_data_path
    )

    # Create dictionaries for easy cycle lookup
    orig_data = dict(
        zip(orig_cycles, zip(orig_voltages, orig_currents, orig_dv_dt, orig_di_dt))
    )
    pred_data = dict(
        zip(pred_cycles, zip(pred_voltages, pred_currents, pred_dv_dt, pred_di_dt))
    )

    # Find common cycles
    common_cycles = sorted(set(orig_cycles) & set(pred_cycles))

    common_cycles = common_cycles[::steps]

    for cycle in common_cycles:
        idx = orig_cycles.index(cycle)
        orig_data[cycle] = [
            np.array(orig_voltages[idx]),
            np.array(orig_currents[idx]),
            np.array(orig_dv_dt[idx]),
            np.array(orig_di_dt[idx]),
        ]

        idx = pred_cycles.index(cycle)
        pred_data[cycle] = [
            np.array(pred_voltages[idx]),
            np.array(pred_currents[idx]),
            np.array(pred_dv_dt[idx]),
            np.array(pred_di_dt[idx]),
        ]

    # Memory cleanup
    del orig_voltages, orig_currents, orig_dv_dt, orig_di_dt
    del pred_voltages, pred_currents, pred_dv_dt, pred_di_dt

    # Set style
    # plt.style.use('seaborn-darkgrid')

    # Set up the figure and axes
    fig, axes = plt.subplots(2, 2, figsize=(20, 15), dpi=100)
    fig.set_facecolor("white")

    # Add spacing between subplots
    plt.subplots_adjust(hspace=0.4, wspace=0.3, top=0.85)
    ax1, ax2, ax3, ax4 = axes.flatten()

    # Define plotting parameters
    plot_params = {
        "orig_color": "#FF4B4B",
        "pred_color": "#4B4BFF",
        "linewidth": 3.0,
        "fontsize": {"title": 16, "axis": 14, "legend": 12},
        "alpha": 0.3,
    }

    # Create plot lines with enhanced styling
    lines = []
    for ax, title, ylabel in zip(
        [ax1, ax2, ax3, ax4],
        ["Voltage vs. Time", "Current vs. Time", "dV/dt vs. Time", "dI/dt vs. Time"],
        ["Voltage", "Current", "dV/dt", "dI/dt"],
    ):
        # Original data line
        (l1,) = ax.plot(
            [],
            [],
            color=plot_params["orig_color"],
            linewidth=plot_params["linewidth"],
            label="Original",
        )
        # Predicted data line
        (l2,) = ax.plot(
            [],
            [],
            color=plot_params["pred_color"],
            linewidth=plot_params["linewidth"],
            linestyle="--",
            label="Predicted",
        )

        ax.set_title(
            title, fontsize=plot_params["fontsize"]["title"], pad=15, fontweight="bold"
        )
        ax.set_xlabel("Time Index", fontsize=plot_params["fontsize"]["axis"])
        ax.set_ylabel(ylabel, fontsize=plot_params["fontsize"]["axis"])
        ax.legend(
            fontsize=plot_params["fontsize"]["legend"],
            framealpha=0.9,
            loc="upper right",
        )
        ax.grid(True, alpha=plot_params["alpha"])
        ax.tick_params(labelsize=12)

        lines.extend([l1, l2])

    # Create the title as a Text object instead of using fig.text
    title = fig.suptitle(
        "Battery Cycle Analysis", fontsize=20, fontweight="bold", y=0.95
    )

    def update(frame: int) -> Any:
        cycle = common_cycles[frame]
        data_orig = orig_data[cycle]
        data_pred = pred_data[cycle]

        # Update all lines efficiently
        for i, (orig, pred) in enumerate(zip(data_orig, data_pred)):
            x = np.arange(len(orig))
            lines[i * 2].set_data(x, orig)
            lines[i * 2 + 1].set_data(x, pred)

        # Update cycle text with current cycle number
        title.set_text(f"Battery Cycle Analysis - Cycle: {cycle}")
        return lines + [title]

    # Set axis limits
    first_cycle = common_cycles[0]
    first_data = orig_data[first_cycle]
    for ax, data in zip(axes.flatten(), first_data):
        ax.set_xlim(0, len(data))
        data_range = np.ptp(data)
        data_min, data_max = np.min(data), np.max(data)
        ax.set_ylim(data_min - 0.1 * data_range, data_max + 0.1 * data_range)

    # Create and save animation with memory optimization
    ani = FuncAnimation(
        fig, update, frames=len(common_cycles), interval=200
    )  # Increased interval

    # Save with optimized settings
    ani.save(
        save_path,
        writer="pillow",
        fps=5,  # Reduced fps
        dpi=100,
        savefig_kwargs={"facecolor": "white"},
    )

    plt.close("all")  # Ensure all figures are closed


def _plot_recon_error_df(df: pd.DataFrame, save_path: Path, title_suffix: str = "") -> None:
    """Helper to plot reconstruction error vs cycle for a given DataFrame."""
    plt.figure(figsize=(16, 10))
    scatter = plt.scatter(
        df["Cycle"],
        df["Reconstruction Error"],
        alpha=0.5,
        c=df["Reconstruction Error"],
        cmap="viridis",
        s=10,
        label="Reconstruction Error",
    )

    if len(df) >= 2:
        z = np.polyfit(df["Cycle"], df["Reconstruction Error"], 1)
        p = np.poly1d(z)
        plt.plot(
            df["Cycle"],
            p(df["Cycle"]),
            "r--",
            alpha=0.8,
            label=f"Trend (slope: {z[0]:.2e})",
        )

    mean_error = np.mean(df["Reconstruction Error"])
    std_error = np.std(df["Reconstruction Error"])
    plt.axhline(
        y=mean_error, color="g", linestyle="--", label=f"Mean: {mean_error:.2e}"
    )
    plt.fill_between(
        df["Cycle"],
        mean_error - std_error,
        mean_error + std_error,
        alpha=0.2,
        color="g",
        label=f"±1 STD: {std_error:.2e}",
    )

    plt.colorbar(scatter, label="MSE Magnitude")

    textstr = f"Mean: {mean_error:.4e}\nSTD: {std_error:.4e}"
    plt.gca().text(
        0.98,
        0.98,
        textstr,
        transform=plt.gca().transAxes,
        fontsize=14,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8),
    )

    title = "Reconstruction Error per Cycle"
    if title_suffix:
        title += f" ({title_suffix})"
    plt.title(title)
    plt.xlabel("Cycle Number")
    plt.ylabel("Reconstruction Error")
    plt.grid(True, alpha=0.3)

    plt.savefig(save_path, dpi=300)
    plt.close()


def compute_reconstruction_error_per_channel_df(
    latent_space_path: Path,
    channel_names: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build a long-form DataFrame with reconstruction error per cycle and per channel.

    Expects ``latent_space.pth`` to contain:
      - ``cycle_numbers``: (B,)
      - ``reconstruction_error_per_cycle_per_channel``: (B, C)
    """
    data = torch.load(latent_space_path, map_location="cpu")

    required_keys = {"cycle_numbers", "reconstruction_error_per_cycle_per_channel"}
    missing = required_keys - set(data.keys())
    if missing:
        raise KeyError(
            "latent_space file must contain keys "
            f"{sorted(required_keys)}, missing {sorted(missing)}"
        )

    cycle_numbers = np.asarray(data["cycle_numbers"], dtype=float)
    recon_err_ch = np.asarray(data["reconstruction_error_per_cycle_per_channel"], dtype=float)

    if recon_err_ch.ndim == 1:
        recon_err_ch = recon_err_ch[:, None]
    if recon_err_ch.ndim != 2:
        raise ValueError(
            "Expected reconstruction_error_per_cycle_per_channel to have shape (B, C), "
            f"got {recon_err_ch.shape}"
        )

    n_samples, n_channels = recon_err_ch.shape
    if cycle_numbers.shape[0] != n_samples:
        raise ValueError(
            "Mismatch between cycle_numbers and channel-wise reconstruction errors: "
            f"{cycle_numbers.shape[0]} vs {n_samples}"
        )

    if channel_names is None:
        if n_channels == 4:
            channel_names = ["V_ch", "dV/dQ_ch", "d2V/dQ2_ch", "dQ/dV_ch"]
        elif n_channels == 8:
            channel_names = [
                "V_ch",
                "V_dis",
                "dV/dQ_ch",
                "dV/dQ_dis",
                "d2V/dQ2_ch",
                "d2V/dQ2_dis",
                "dQ/dV_ch",
                "dQ/dV_dis",
            ]
        elif n_channels == 9:
            channel_names = [
                "V_ch",
                "V_dis",
                "dV/dQ_ch",
                "dV/dQ_dis",
                "d2V/dQ2_ch",
                "d2V/dQ2_dis",
                "dQ/dV_ch",
                "dQ/dV_dis",
                "H_raw",
            ]
        else:
            channel_names = [f"Channel {i + 1}" for i in range(n_channels)]
    elif len(channel_names) != n_channels:
        raise ValueError(
            f"channel_names length ({len(channel_names)}) does not match "
            f"number of channels ({n_channels})"
        )

    frames = []
    for ch_idx, ch_name in enumerate(channel_names):
        df_ch = pd.DataFrame(
            {
                "Cycle": cycle_numbers,
                "Reconstruction Error": recon_err_ch[:, ch_idx],
                "Channel": ch_name,
            }
        )
        frames.append(df_ch)

    df = pd.concat(frames, axis=0, ignore_index=True)
    df = df.sort_values(["Channel", "Cycle"], kind="mergesort").reset_index(drop=True)
    return df, channel_names


def plot_reconstructed_error(latent_space_path: Path, save_path: Path) -> None:
    """
    Plot reconstruction error from latent space data.

    If ``charging_rate`` is available in the latent file, one plot per
    C‑rate is saved, plus the global plot.
    """
    data = torch.load(latent_space_path)

    cycle_numbers = np.array(data["cycle_numbers"])
    reconstruction_errors = np.array(data["reconstruction_error_per_cycle"])

    has_rate = "charging_rate" in data
    if has_rate:
        rates = np.array(data["charging_rate"], dtype=float)

    df_global = pd.DataFrame(
        {"Cycle": cycle_numbers, "Reconstruction Error": reconstruction_errors}
    )
    _plot_recon_error_df(df_global, save_path)

    if has_rate:
        unique_rates = np.unique(rates)
        for r in unique_rates:
            mask = rates == r
            df_r = pd.DataFrame(
                {
                    "Cycle": cycle_numbers[mask],
                    "Reconstruction Error": reconstruction_errors[mask],
                }
            )
            out_path = save_path.with_name(
                f"{save_path.stem}_c{r:.2f}C{save_path.suffix}"
            )
            _plot_recon_error_df(df_r, out_path, title_suffix=f"{r:.2f}C")


def plot_reconstructed_error_per_channel(
    latent_space_path: Path,
    save_path: Path,
    channel_names: Optional[List[str]] = None,
) -> None:
    """
    Plot reconstruction error vs cycle as vertical subplots (one subplot per channel).

    Each subplot includes:
      - scatter points
      - linear trend line
      - mean line and +-1 std band
    """
    df, ordered_channel_names = compute_reconstruction_error_per_channel_df(
        latent_space_path=latent_space_path,
        channel_names=channel_names,
    )

    n_channels = len(ordered_channel_names)
    fig_height = max(4.0, 3.2 * n_channels)
    fig, axes = plt.subplots(
        n_channels,
        1,
        figsize=(16, fig_height),
        dpi=160,
        sharex=True,
    )

    if n_channels == 1:
        axes = [axes]

    for ax, ch_name in zip(axes, ordered_channel_names):
        df_ch = df[df["Channel"] == ch_name].dropna(subset=["Cycle", "Reconstruction Error"])
        if df_ch.empty:
            ax.set_title(f"{ch_name} (no data)")
            ax.set_ylabel("Error")
            ax.grid(True, alpha=0.3)
            continue

        x = df_ch["Cycle"].to_numpy(dtype=float)
        y = df_ch["Reconstruction Error"].to_numpy(dtype=float)

        order = np.argsort(x)
        x = x[order]
        y = y[order]

        ax.scatter(
            x,
            y,
            alpha=0.45,
            s=10,
            color="#1f77b4",
            label="Reconstruction Error",
        )

        if len(x) >= 2 and np.unique(x).size >= 2:
            coeff = np.polyfit(x, y, 1)
            trend = np.poly1d(coeff)
            ax.plot(
                x,
                trend(x),
                "r--",
                alpha=0.9,
                linewidth=1.6,
                label=f"Trend (slope: {coeff[0]:.2e})",
            )

        mean_error = float(np.mean(y))
        std_error = float(np.std(y))
        ax.axhline(
            y=mean_error,
            color="g",
            linestyle="--",
            linewidth=1.2,
            label=f"Mean: {mean_error:.2e}",
        )
        ax.fill_between(
            x,
            mean_error - std_error,
            mean_error + std_error,
            alpha=0.2,
            color="g",
            label=f"±1 STD: {std_error:.2e}",
        )

        ax.text(
            0.99,
            0.98,
            f"Mean: {mean_error:.4e}\nSTD: {std_error:.4e}",
            transform=ax.transAxes,
            fontsize=10,
            va="top",
            ha="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

        ax.set_title(f"{ch_name}")
        ax.set_ylabel("Error")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=9)

    axes[-1].set_xlabel("Cycle Number")
    fig.suptitle("Reconstruction Error per Cycle by Channel", fontsize=14, y=0.995)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.985])
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_reconstruction_vs_input_multichannel(
    latent_space_path: Path,
    cycles: List[int],
    save_dir: Path,
    padding_value: float = -9999.0,
) -> None:
    """
    For each requested cycle, plot input vs reconstructed multichannel data
    and save one PNG per cycle in ``save_dir``.

    Panels are inferred from the available decoder heads.

    Expects ``latent_space.pth`` (from ``extract_and_save_latent_space``) to contain:
      - ``inputs``: (B, C_in, N) with at least 11 channels
      - ``reconstructions``: (B, C_out, N) decoder heads
      - ``cycle_numbers``: (B,) cycle indices
    """
    import torch

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    data = torch.load(latent_space_path, map_location="cpu")
    required_keys = {"inputs", "reconstructions", "cycle_numbers"}
    missing = required_keys - set(data.keys())
    if missing:
        raise KeyError(
            f"latent_space file must contain keys {sorted(required_keys)}, missing {sorted(missing)}"
        )

    inputs = data["inputs"].detach().cpu().numpy()  # (B, C_in, N)
    recon_all = data["reconstructions"].detach().cpu().numpy()  # (B, 9, N)
    cycle_numbers = data["cycle_numbers"].detach().cpu().numpy().astype(int)  # (B,)

    if inputs.shape[1] < 9:
        raise ValueError(
            f"Expected at least 9 input channels, got {inputs.shape[1]}"
        )

    # Input channel indices
    CH_QCAP = 1
    CH_V_CH = 2
    CH_V_DIS = 3
    CH_DVDQ_CH = 4
    CH_DVDQ_DIS = 5
    CH_D2V_CH = 6
    CH_D2V_DIS = 7
    CH_DQDV_CH = 8
    CH_DQDV_DIS = 9
    CH_H_RAW = 10

    channel_specs_by_count = {
        4: [
            ("V_ch", CH_V_CH, 0, "Voltage"),
            ("dVdQ_ch", CH_DVDQ_CH, 1, "dV/dQ"),
            ("d2VdQ2_ch", CH_D2V_CH, 2, "d²V/dQ²"),
            ("dQdV_ch", CH_DQDV_CH, 3, "dQ/dV"),
        ],
        8: [
            ("V_ch", CH_V_CH, 0, "Voltage"),
            ("V_dis", CH_V_DIS, 1, "Voltage"),
            ("dVdQ_ch", CH_DVDQ_CH, 2, "dV/dQ"),
            ("dVdQ_dis", CH_DVDQ_DIS, 3, "dV/dQ"),
            ("d2VdQ2_ch", CH_D2V_CH, 4, "d²V/dQ²"),
            ("d2VdQ2_dis", CH_D2V_DIS, 5, "d²V/dQ²"),
            ("dQdV_ch", CH_DQDV_CH, 6, "dQ/dV"),
            ("dQdV_dis", CH_DQDV_DIS, 7, "dQ/dV"),
        ],
        9: [
            ("V_ch", CH_V_CH, 0, "Voltage"),
            ("V_dis", CH_V_DIS, 1, "Voltage"),
            ("dVdQ_ch", CH_DVDQ_CH, 2, "dV/dQ"),
            ("dVdQ_dis", CH_DVDQ_DIS, 3, "dV/dQ"),
            ("d2VdQ2_ch", CH_D2V_CH, 4, "d²V/dQ²"),
            ("d2VdQ2_dis", CH_D2V_DIS, 5, "d²V/dQ²"),
            ("dQdV_ch", CH_DQDV_CH, 6, "dQ/dV"),
            ("dQdV_dis", CH_DQDV_DIS, 7, "dQ/dV"),
            ("H_raw", CH_H_RAW, 8, "H_raw"),
        ],
    }
    channels = channel_specs_by_count.get(recon_all.shape[1])
    if channels is None:
        max_known = min(recon_all.shape[1], 9)
        channels = channel_specs_by_count[9][:max_known]

    for cyc in cycles:
        # Find first matching sample for this cycle
        matches = (cycle_numbers == int(cyc)).nonzero()[0]
        if matches.size == 0:
            # Skip silently if cycle is not present
            continue

        idx = int(matches[0])

        q_cap = inputs[idx, CH_QCAP, :]
        if np.all(q_cap == padding_value):
            # Nothing meaningful to plot for this cycle
            continue

        n_panels = len(channels)
        n_cols = min(3, n_panels)
        n_rows = int(np.ceil(n_panels / n_cols))
        fig, axs = plt.subplots(
            n_rows,
            n_cols,
            figsize=(5 * n_cols, 3.5 * n_rows),
            sharex=True,
            dpi=140,
        )
        axs = np.atleast_1d(axs).reshape(n_rows, n_cols)
        fig.suptitle(
            f"Cycle {int(cyc)} – Input vs Reconstruction ({recon_all.shape[1]} heads)",
            fontsize=14,
        )

        for i, (name, ch_in, ch_out, ylabel) in enumerate(channels):
            r, c = divmod(i, n_cols)
            ax = axs[r, c]

            y_true = inputs[idx, ch_in, :]
            y_pred = recon_all[idx, ch_out, :]

            # Use input padding to define validity
            valid = (q_cap != padding_value) & (y_true != padding_value)
            if not valid.any():
                ax.axis("off")
                continue

            x = q_cap[valid]
            ax.plot(
                x,
                y_true[valid],
                label="Input",
                color="#1f77b4",
                linewidth=1.4,
            )
            ax.plot(
                x,
                y_pred[valid],
                label="Reconstructed",
                color="#ff7f0e",
                linestyle="--",
                linewidth=1.4,
            )
            ax.set_title(name)
            ax.set_xlabel("q_cap")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7)

        for j in range(len(channels), n_rows * n_cols):
            r, c = divmod(j, n_cols)
            axs[r, c].axis("off")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        out_path = save_dir / f"recon_vs_input_cycle_{int(cyc)}.png"
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
