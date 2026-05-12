from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pandas as pd
import torch

from src.vae.inference.checkpoints import load_best_model_checkpoint
from src.vae.inference.inference import build_reconstruction_loss
from src.vae.analysis.plot_soh import plot_soh_evolution
from src.common.logger.logging import get_logger
from src.vae.preprocessing.processor import preprocess_main
from src.common.utils.conditioning import build_condition_vector
from src.common.utils.config_schema import DatasetConfig, FullConfig
logger = get_logger(__name__)

RECON_CHANNEL_IDX = [2, 3, 4, 5, 8, 9]
RECON_CHANNEL_NAMES = ["V_ch", "V_dis", "dVdQ_ch", "dVdQ_dis", "dQdV_ch", "dQdV_dis"]
SOH_UNCERTAINTY_SAMPLES = 512
SOH_UNCERTAINTY_LEVELS = (0.90, 0.95)


def _crate_tag(dataset_cfg: DatasetConfig) -> str:
    if dataset_cfg.id:
        return str(dataset_cfg.id)
    return f"{float(dataset_cfg.charging_rate):.2f}C"


def _denormalize_voltage(values: torch.Tensor, config: FullConfig) -> torch.Tensor:
    voltage_cfg = config.NORMALIZATION.voltage
    mode = str(voltage_cfg.get("mode", "none")).strip().lower()
    if mode == "none":
        return values
    if mode != "phys_minmax":
        raise NotImplementedError(
            f"Voltage denormalization is not implemented for mode '{mode}'."
        )

    v_min = float(voltage_cfg["v_min"])
    v_max = float(voltage_cfg["v_max"])
    target_range = str(voltage_cfg.get("target_range", "zero_to_one")).strip().lower()
    if target_range == "minus_one_to_one":
        values = (values + 1.0) / 2.0
    return values * (v_max - v_min) + v_min


def _load_latent_points(
    dataset_cfg: DatasetConfig,
    latent_dim: int,
) -> Tuple[pd.DataFrame, Path, List[str], str]:
    if not dataset_cfg.interpolation_latent_path:
        raise ValueError("Dataset is missing interpolation_latent_path in config.")

    latent_path = Path(dataset_cfg.interpolation_latent_path or "")
    if not latent_path.exists():
        raise FileNotFoundError(
            f"Interpolation latent CSV not found: {latent_path}"
        )

    latent_df = pd.read_csv(latent_path)
    z_cols = [f"z{i + 1}" for i in range(latent_dim)]
    required_cols = {"Cycle", *z_cols}
    missing = required_cols - set(latent_df.columns)
    if missing:
        raise ValueError(
            f"Interpolation CSV {latent_path} is missing columns: {sorted(missing)}"
        )

    rate_col = next(
        (col for col in ("C-rate", "charging_rate", "ChargingRate", "c_rate") if col in latent_df.columns),
        None,
    )
    if rate_col is None:
        rate_col = "C-rate"
        latent_df[rate_col] = float(dataset_cfg.charging_rate)

    latent_df = latent_df.sort_values("Cycle").reset_index(drop=True)
    return latent_df, latent_path, z_cols, rate_col


def _prepare_reference_loader(
    config: FullConfig,
    dataset_name: str,
    dataset_cfg: DatasetConfig,
    train_norm_stats: dict,
):
    dataset_path = Path(dataset_cfg.path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Interpolation reference dataset not found: {dataset_path}")

    dataframe = pd.read_csv(dataset_path)
    dataset_metadata = dataset_cfg.model_dump(exclude={"path", "interpolation_latent_path"})
    for key, value in dataset_metadata.items():
        if key not in dataframe.columns:
            dataframe[key] = value

    dataframe["source_dataset"] = dataset_name
    dataframe["global_index"] = dataframe.Cycle.astype(str)
    dataframe["Current"] = -1 * dataframe["Current"]
    _, _, extras = preprocess_main(
        dataframe,
        config,
        frozen_norm_stats=train_norm_stats or None,
    )
    full_loader = extras.get("full_loader")
    if full_loader is None:
        raise RuntimeError("Preprocessing did not return a full inference loader.")
    return full_loader


def _collect_reference_by_cycle(data_loader) -> Dict[int, Dict[str, torch.Tensor]]:
    reference_by_cycle: Dict[int, Dict[str, torch.Tensor]] = {}
    for inputs, masks, token_masks, labels in data_loader:
        for idx in range(inputs.size(0)):
            cycle = int(labels[idx, 1].item())
            if cycle in reference_by_cycle:
                continue
            reference_by_cycle[cycle] = {
                "inputs": inputs[idx : idx + 1].clone(),
                "masks": masks[idx : idx + 1].clone(),
                "token_masks": token_masks[idx : idx + 1].clone(),
                "labels": labels[idx : idx + 1].clone(),
            }
    return reference_by_cycle


def _build_reconstruction_rows(
    cycle: int,
    charging_rate: float,
    q_cap: torch.Tensor,
    reconstruction: torch.Tensor,
    config: FullConfig,
    inputs: torch.Tensor | None = None,
    masks: torch.Tensor | None = None,
) -> List[dict]:
    q_cap_np = q_cap.squeeze(0).cpu().numpy()
    recon_np = reconstruction.squeeze(0).cpu().numpy()

    inputs_np = inputs.squeeze(0).cpu().numpy() if inputs is not None else None
    masks_np = masks.squeeze(0).cpu().numpy() > 0.5 if masks is not None else None

    rows: List[dict] = []
    for q_idx, q_value in enumerate(q_cap_np):
        row = {
            "Cycle": cycle,
            "charging_rate": charging_rate,
            "q_index": q_idx,
            "q_cap": float(q_value),
        }
        for out_idx, (name, in_idx) in enumerate(zip(RECON_CHANNEL_NAMES, RECON_CHANNEL_IDX)):
            target_value = None
            if inputs_np is not None and masks_np is not None and masks_np[in_idx, q_idx]:
                target_value = float(inputs_np[in_idx, q_idx])

            row[f"{name}_target_norm"] = target_value
            row[f"{name}_reconstruction_norm"] = float(recon_np[out_idx, q_idx])

            if name in {"V_ch", "V_dis"}:
                row[f"{name}_target_V"] = (
                    None
                    if target_value is None
                    else float(
                        _denormalize_voltage(
                            torch.tensor(target_value, dtype=torch.float32),
                            config=config,
                        ).item()
                    )
                )
                row[f"{name}_reconstruction_V"] = float(
                    _denormalize_voltage(
                        torch.tensor(recon_np[out_idx, q_idx], dtype=torch.float32),
                        config=config,
                    ).item()
                )
        rows.append(row)
    return rows


def _stack_tensors(tensors: List[torch.Tensor]) -> torch.Tensor:
    if not tensors:
        return torch.empty(0)
    return torch.cat(tensors, dim=0)


def _covariance_column_name(columns: set[str], first_col: str, second_col: str) -> str | None:
    candidates = (
        f"{first_col}_{second_col}_cov",
        f"{second_col}_{first_col}_cov",
        f"cov_{first_col}_{second_col}",
        f"cov_{second_col}_{first_col}",
        f"covar_{first_col}_{second_col}",
        f"covar_{second_col}_{first_col}",
    )
    return next((candidate for candidate in candidates if candidate in columns), None)


def _has_latent_uncertainty_columns(latent_df: pd.DataFrame, z_cols: List[str]) -> bool:
    return all(f"{col}_std" in latent_df.columns for col in z_cols)


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _build_latent_covariance_matrix(
    row: dict,
    z_cols: List[str],
    columns: set[str],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    latent_dim = len(z_cols)
    covariance = torch.zeros((latent_dim, latent_dim), dtype=dtype, device=device)
    std_values = []
    for idx, col in enumerate(z_cols):
        std = max(_finite_float(row[f"{col}_std"]), 0.0)
        std_values.append(std)
        covariance[idx, idx] = std * std

    for first_idx in range(latent_dim):
        for second_idx in range(first_idx + 1, latent_dim):
            cov_col = _covariance_column_name(
                columns,
                z_cols[first_idx],
                z_cols[second_idx],
            )
            cov_value = 0.0 if cov_col is None else _finite_float(row[cov_col])
            max_abs_cov = std_values[first_idx] * std_values[second_idx]
            if max_abs_cov <= 0.0:
                cov_value = 0.0
            elif abs(cov_value) > max_abs_cov:
                cov_value = 0.999 * max_abs_cov * (1.0 if cov_value > 0.0 else -1.0)
            covariance[first_idx, second_idx] = cov_value
            covariance[second_idx, first_idx] = cov_value

    jitter = torch.eye(latent_dim, dtype=dtype, device=device) * 1e-8
    covariance = covariance + jitter
    try:
        torch.linalg.cholesky(covariance)
        return covariance
    except RuntimeError:
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        eigenvalues = torch.clamp(eigenvalues, min=1e-8)
        return (eigenvectors * eigenvalues.unsqueeze(0)) @ eigenvectors.T


def _predict_soh_uncertainty_from_latent_gp(
    model: torch.nn.Module,
    row: dict,
    z_cols: List[str],
    columns: set[str],
    sample_count: int,
    dtype: torch.dtype,
    device: torch.device,
    generator: torch.Generator | None,
) -> dict[str, float] | None:
    if sample_count <= 1 or not all(f"{col}_std" in columns for col in z_cols):
        return None

    mean = torch.tensor(
        [[float(row[col]) for col in z_cols]],
        dtype=dtype,
        device=device,
    )
    covariance = _build_latent_covariance_matrix(
        row=row,
        z_cols=z_cols,
        columns=columns,
        dtype=dtype,
        device=device,
    )
    cholesky = torch.linalg.cholesky(covariance)
    try:
        eps = torch.randn(
            sample_count,
            len(z_cols),
            dtype=dtype,
            device=device,
            generator=generator,
        )
    except RuntimeError:
        eps = torch.randn(sample_count, len(z_cols), dtype=dtype, device=device)
    latent_samples = mean + eps @ cholesky.T
    soh_samples = model.soh_predictor(latent_samples).squeeze(-1)

    result = {
        "soh_predicted_mc_mean": float(soh_samples.mean().item()),
        "soh_predicted_mc_std": float(soh_samples.std(unbiased=False).item()),
        "soh_uncertainty_samples": int(sample_count),
    }
    for level in SOH_UNCERTAINTY_LEVELS:
        tail = (1.0 - level) / 2.0
        suffix = str(int(level * 100))
        quantiles = torch.quantile(
            soh_samples,
            torch.tensor([tail, 1.0 - tail], dtype=dtype, device=device),
        )
        result[f"soh_predicted_lower_{suffix}"] = float(quantiles[0].item())
        result[f"soh_predicted_upper_{suffix}"] = float(quantiles[1].item())
    return result


def _make_generator(device: torch.device, seed: int) -> torch.Generator | None:
    try:
        generator = torch.Generator(device=device)
    except (RuntimeError, TypeError):
        try:
            generator = torch.Generator()
        except RuntimeError:
            return None
    generator.manual_seed(int(seed))
    return generator


def _plot_reconstructed_voltage_by_cycle(
    cycle_numbers: List[int],
    q_grids: List[torch.Tensor],
    reconstructions: List[torch.Tensor],
    save_path: Path,
    charge_cmap_name: str = "viridis",
    discharge_cmap_name: str = "rainbow",
    dpi: int = 300,
) -> None:
    if not cycle_numbers:
        return

    q_cap_all = _stack_tensors(q_grids).numpy()
    recon_all = _stack_tensors(reconstructions)
    vch_recon_all = recon_all[:, 0, :].numpy()
    vdis_recon_all = recon_all[:, 1, :].numpy()

    cyc_min = min(cycle_numbers)
    cyc_max = max(cycle_numbers)
    charge_cmap = cm.get_cmap(charge_cmap_name)
    discharge_cmap = cm.get_cmap(discharge_cmap_name)
    norm = plt.Normalize(vmin=cyc_min, vmax=cyc_max)

    fig, ax = plt.subplots(figsize=(7.4, 5.2), dpi=dpi)
    for idx, cycle in enumerate(cycle_numbers):
        ax.plot(
            q_cap_all[idx],
            vch_recon_all[idx],
            color=charge_cmap(norm(cycle)),
            lw=1.0,
            alpha=0.7,
            solid_capstyle="round",
        )
        ax.plot(
            q_cap_all[idx],
            vdis_recon_all[idx],
            color=discharge_cmap(norm(cycle)),
            lw=1.0,
            alpha=0.7,
            solid_capstyle="round",
        )

    ax.set_xlabel("Normalized capacity", fontsize=18)
    ax.set_ylabel("Normalized voltage", fontsize=18)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(False)
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.tick_params(axis="both", which="major", labelsize=16, direction="out", length=4, width=0.8)

    sm_charge = cm.ScalarMappable(norm=norm, cmap=charge_cmap)
    sm_charge.set_array([])
    cbar_charge = fig.colorbar(sm_charge, ax=ax, pad=0.02, fraction=0.046)
    cbar_charge.set_label("Cycle number", fontsize=18)
    cbar_charge.ax.text(0.5, -0.06, "Charge", transform=cbar_charge.ax.transAxes, rotation=90, ha="center", va="top", fontsize=16)
    cbar_charge.ax.tick_params(labelsize=16)

    sm_discharge = cm.ScalarMappable(norm=norm, cmap=discharge_cmap)
    sm_discharge.set_array([])
    cbar_discharge = fig.colorbar(sm_discharge, ax=ax, pad=0.10, fraction=0.046)
    cbar_discharge.set_label("")
    cbar_discharge.ax.text(0.5, -0.06, "Discharge", transform=cbar_discharge.ax.transAxes, rotation=90, ha="center", va="top", fontsize=16)
    cbar_discharge.set_ticks([])

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)


def _decode_all_latent_cycles(
    model: torch.nn.Module,
    latent_df: pd.DataFrame,
    z_cols: List[str],
    rate_col: str,
    dataset_cfg: DatasetConfig,
    config: FullConfig,
    device: torch.device,
    reference_by_cycle: Dict[int, Dict[str, torch.Tensor]],
) -> Dict[int, Dict[str, Any]]:
    seq_len = int(config.HYPER_PARAMETERS.input_seq_len)
    if seq_len <= 0:
        raise ValueError("HYPER_PARAMETERS.input_seq_len must be set for decoder-only interpolation.")

    model_dtype = next(model.parameters()).dtype
    q_template = torch.linspace(
        0.0,
        1.0,
        seq_len,
        dtype=model_dtype,
        device=device,
    ).unsqueeze(0)

    decoded_by_cycle: Dict[int, Dict[str, Any]] = {}
    cycle_max = max(int(dataset_cfg.cycle_max), 1)
    latent_columns = set(latent_df.columns)
    should_sample_soh = _has_latent_uncertainty_columns(latent_df, z_cols)
    soh_generator = _make_generator(device, int(config.GENERAL.seed))

    with torch.no_grad():
        for row in latent_df.to_dict(orient="records"):
            cycle = int(row["Cycle"])
            charging_rate_value = float(row[rate_col])

            sample = reference_by_cycle.get(cycle)
            if sample is not None:
                labels = sample["labels"].to(device)
                norm_nominal_capacity = labels[:, 6]
                norm_cycle_numbers = labels[:, 2]
            else:
                norm_nominal_capacity = torch.tensor([1.0], dtype=model_dtype, device=device)
                norm_cycle_numbers = torch.tensor(
                    [cycle / cycle_max],
                    dtype=model_dtype,
                    device=device,
                )

            charging_rate = torch.tensor(
                [charging_rate_value],
                dtype=model_dtype,
                device=device,
            )
            cond_vec = build_condition_vector(
                norm_cycle_numbers=norm_cycle_numbers,
                charging_rate=charging_rate,
                norm_nominal_capacity=norm_nominal_capacity,
                cond_dim=config.HYPER_PARAMETERS.conditional_vector_dim,
                device=device,
            )
            z = torch.tensor(
                [[float(row[col]) for col in z_cols]],
                dtype=model_dtype,
                device=device,
            )
            reconstruction = model.decoder(q_template.unsqueeze(-1), z, cond_vec).detach().cpu()
            soh_predicted = model.soh_predictor(z).detach().cpu()
            soh_uncertainty = (
                _predict_soh_uncertainty_from_latent_gp(
                    model=model,
                    row=row,
                    z_cols=z_cols,
                    columns=latent_columns,
                    sample_count=SOH_UNCERTAINTY_SAMPLES,
                    dtype=model_dtype,
                    device=device,
                    generator=soh_generator,
                )
                if should_sample_soh
                else None
            )

            decoded_by_cycle[cycle] = {
                "q_cap": q_template.detach().cpu(),
                "reconstruction": reconstruction,
                "latent": z.detach().cpu(),
                "charging_rate": charging_rate_value,
                "soh_predicted": soh_predicted,
            }
            if soh_uncertainty is not None:
                decoded_by_cycle[cycle]["soh_uncertainty"] = soh_uncertainty

    return decoded_by_cycle


def run_interpolation_inference(
    model: torch.nn.Module,
    config: FullConfig,
    device: torch.device,
    train_norm_stats: dict,
    filename: str | None = None,
) -> None:
    load_best_model_checkpoint(
        model=model,
        config=config,
        device=device,
        filename=filename,
    )
    model.eval()
    loss_fn = build_reconstruction_loss(config)
    dataset_names = [
        name
        for name, dataset_cfg in config.GP_INTERPOLATION_DATASETS.items()
        if dataset_cfg.interpolation_latent_path
    ]
    if not dataset_names:
        raise ValueError(
            "No interpolation datasets with interpolation_latent_path were found in config.GP_INTERPOLATION_DATASETS."
        )

    requested_dataset = config.GENERAL.interpolation_dataset
    if requested_dataset:
        if requested_dataset not in config.GP_INTERPOLATION_DATASETS:
            raise KeyError(
                f"Interpolation dataset '{requested_dataset}' not found in config.GP_INTERPOLATION_DATASETS."
            )
        dataset_names = [requested_dataset]

    for dataset_name in dataset_names:
        dataset_cfg = config.GP_INTERPOLATION_DATASETS[dataset_name]
        latent_df, latent_path, z_cols, rate_col = _load_latent_points(
            dataset_cfg=dataset_cfg,
            latent_dim=config.HYPER_PARAMETERS.latent_dim,
        )

        logger.info("Preparing reference data for interpolation dataset %s", dataset_name)
        reference_loader = _prepare_reference_loader(
            config=config,
            dataset_name=dataset_name,
            dataset_cfg=dataset_cfg,
            train_norm_stats=train_norm_stats,
        )
        reference_by_cycle = _collect_reference_by_cycle(reference_loader)
        if not reference_by_cycle:
            raise RuntimeError(f"No reference cycles available for interpolation dataset {dataset_name}.")

        decoded_by_cycle = _decode_all_latent_cycles(
            model=model,
            latent_df=latent_df,
            z_cols=z_cols,
            rate_col=rate_col,
            dataset_cfg=dataset_cfg,
            config=config,
            device=device,
            reference_by_cycle=reference_by_cycle,
        )

        summary_rows: List[dict] = []
        reconstruction_rows: List[dict] = []
        soh_rows: List[dict] = []
        all_cycle_numbers: List[int] = []
        all_q_grids: List[torch.Tensor] = []
        all_recons: List[torch.Tensor] = []
        all_latents: List[torch.Tensor] = []
        matched_cycles: List[int] = []
        skipped_cycles: List[int] = []

        collected_inputs: List[torch.Tensor] = []
        collected_masks: List[torch.Tensor] = []
        collected_recons: List[torch.Tensor] = []
        collected_labels: List[torch.Tensor] = []
        collected_latents: List[torch.Tensor] = []
        collected_weighted_mse: List[torch.Tensor] = []
        collected_unweighted_mse: List[torch.Tensor] = []
        collected_channel_mse: List[torch.Tensor] = []

        for cycle, decoded in decoded_by_cycle.items():
            sample = reference_by_cycle.get(cycle)
            all_cycle_numbers.append(cycle)
            all_q_grids.append(decoded["q_cap"])
            all_recons.append(decoded["reconstruction"])
            all_latents.append(decoded["latent"])
            reconstruction_rows.extend(
                _build_reconstruction_rows(
                    cycle=cycle,
                    charging_rate=float(decoded["charging_rate"]),
                    q_cap=decoded["q_cap"],
                    reconstruction=decoded["reconstruction"],
                    config=config,
                    inputs=None if sample is None else sample["inputs"],
                    masks=None if sample is None else sample["masks"],
                )
            )
            soh_row = {
                "Cycle": cycle,
                "cycle_numbers": cycle,
                "soh_predicted": float(decoded["soh_predicted"].squeeze().item()),
            }
            soh_uncertainty = decoded.get("soh_uncertainty")
            if soh_uncertainty is not None:
                soh_row.update(soh_uncertainty)
            if sample is not None:
                soh_computed = float(sample["labels"][0, 3].item())
                soh_row["soh_computed"] = soh_computed
                soh_row["abs_error"] = abs(soh_computed - soh_row["soh_predicted"])
            soh_rows.append(soh_row)

        with torch.no_grad():
            for row in latent_df.to_dict(orient="records"):
                cycle = int(row["Cycle"])
                sample = reference_by_cycle.get(cycle)
                if sample is None:
                    skipped_cycles.append(cycle)
                    continue

                inputs = sample["inputs"].to(device)
                masks = sample["masks"].to(device)
                labels = sample["labels"].to(device)
                norm_nominal_capacity = labels[:, 6]
                decoded = decoded_by_cycle[cycle]
                reconstruction = decoded["reconstruction"].to(device)
                z = decoded["latent"].to(device)
                q0_Ah = norm_nominal_capacity * 5.0
                dummy_stats = torch.zeros_like(z)
                _, _, _, aux = loss_fn(
                    reconstruction,
                    inputs,
                    dummy_stats,
                    dummy_stats,
                    z,
                    masks,
                    training=False,
                    q0_Ah=q0_Ah,
                )

                mse_weighted = aux["recon_loss_per_sample"].detach().cpu()
                mse_unweighted = aux["recon_loss_per_sample_unweighted"].detach().cpu()
                mse_per_channel = aux["recon_loss_per_sample_per_channel"].detach().cpu()

                matched_cycles.append(cycle)
                collected_inputs.append(sample["inputs"])
                collected_masks.append(sample["masks"])
                collected_labels.append(sample["labels"])
                collected_recons.append(decoded["reconstruction"])
                collected_latents.append(decoded["latent"])
                collected_weighted_mse.append(mse_weighted)
                collected_unweighted_mse.append(mse_unweighted)
                collected_channel_mse.append(mse_per_channel)

                summary_row = {
                    "Cycle": cycle,
                    "charging_rate": float(decoded["charging_rate"]),
                    "mse_weighted": float(mse_weighted.item()),
                    "mse": float(mse_unweighted.item()),
                }
                for col in z_cols:
                    summary_row[col] = float(row[col])
                for ch_name, ch_mse in zip(RECON_CHANNEL_NAMES, mse_per_channel.squeeze(0).tolist()):
                    summary_row[f"mse_{ch_name}"] = float(ch_mse)
                summary_rows.append(summary_row)

        if not summary_rows:
            raise RuntimeError(
                "No overlapping cycles were found between the interpolation latent CSV and the preprocessed reference dataset."
            )

        interpolation_root = Path(
            config.PATHS.vae_interpolation_dir or Path(config.PATHS.predicted_data) / "interpolation"
        )
        output_dir = interpolation_root / dataset_name
        output_dir.mkdir(parents=True, exist_ok=True)

        summary_df = pd.DataFrame(summary_rows).sort_values("Cycle")
        recon_df = pd.DataFrame(reconstruction_rows)
        soh_df = pd.DataFrame(soh_rows).sort_values("Cycle")
        if not recon_df.empty:
            recon_df = recon_df.sort_values(["Cycle", "q_index"])

        summary_path = output_dir / "gp_interpolation_metrics.csv"
        recon_path = output_dir / "gp_interpolation_reconstructions.csv"
        soh_path = output_dir / f"interpolated_soh_{_crate_tag(dataset_cfg)}.csv"
        soh_plot_path = output_dir / f"interpolated_soh_{_crate_tag(dataset_cfg)}.png"
        bundle_path = output_dir / "gp_interpolation_bundle.pth"
        report_path = output_dir / "summary.txt"
        voltage_plot_path = output_dir / "gp_interpolated_voltage_by_cycle.png"

        summary_df.to_csv(summary_path, index=False)
        recon_df.to_csv(recon_path, index=False)
        soh_df.to_csv(soh_path, index=False)
        if {"cycle_numbers", "soh_computed", "soh_predicted", "abs_error"}.issubset(soh_df.columns):
            plot_soh_evolution(soh_path, soh_plot_path)

        bundle = {
            "dataset_name": dataset_name,
            "latent_source_path": str(latent_path),
            "latent_columns": z_cols,
            "channel_names": RECON_CHANNEL_NAMES,
            "all_cycle_numbers": torch.tensor(all_cycle_numbers, dtype=torch.long),
            "all_q_cap": _stack_tensors(all_q_grids),
            "all_reconstructions": _stack_tensors(all_recons),
            "all_reconstructions_voltage_V": _denormalize_voltage(
                _stack_tensors(all_recons)[:, :2, :],
                config=config,
            ),
            "all_latents": _stack_tensors(all_latents),
            "cycle_numbers": torch.tensor(matched_cycles, dtype=torch.long),
            "latents": _stack_tensors(collected_latents),
            "inputs": _stack_tensors(collected_inputs),
            "masks": _stack_tensors(collected_masks),
            "labels": _stack_tensors(collected_labels),
            "reconstructions": _stack_tensors(collected_recons),
            "reconstructions_voltage_V": _denormalize_voltage(
                _stack_tensors(collected_recons)[:, :2, :],
                config=config,
            ),
            "mse_per_cycle": _stack_tensors(collected_unweighted_mse),
            "weighted_mse_per_cycle": _stack_tensors(collected_weighted_mse),
            "mse_per_cycle_per_channel": _stack_tensors(collected_channel_mse),
            "skipped_cycles": torch.tensor(skipped_cycles, dtype=torch.long),
        }
        torch.save(bundle, bundle_path)

        with open(report_path, "w") as report_file:
            report_file.write(f"Dataset: {dataset_name}\n")
            report_file.write(f"Latent source: {latent_path}\n")
            report_file.write(f"Matched cycles: {len(matched_cycles)}\n")
            report_file.write(f"Skipped cycles: {len(skipped_cycles)}\n")
            report_file.write(f"Mean MSE: {summary_df['mse'].mean():.6f}\n")
            report_file.write(f"Mean weighted MSE: {summary_df['mse_weighted'].mean():.6f}\n")

        _plot_reconstructed_voltage_by_cycle(
            cycle_numbers=all_cycle_numbers,
            q_grids=all_q_grids,
            reconstructions=[reconstruction[:, :2, :] for reconstruction in all_recons],
            save_path=voltage_plot_path,
        )

        logger.info("Saved GP interpolation metrics to %s", summary_path)
        logger.info("Saved GP interpolation reconstructions to %s", recon_path)
        logger.info("Saved interpolated SOH predictions to %s", soh_path)
        if soh_plot_path.exists():
            logger.info("Saved interpolated SOH plot to %s", soh_plot_path)
        logger.info("Saved GP interpolation tensor bundle to %s", bundle_path)
        logger.info("Saved GP-interpolated voltage plot to %s", voltage_plot_path)
