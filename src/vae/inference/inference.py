import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from src.common.logger.logging import get_logger
from src.vae.training.loss import LossFactory
from src.common.utils.utils import save_reconstructed_data
from src.common.utils.config_schema import FullConfig
from typing import Any, List, Tuple
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, r2_score


logger = get_logger(__name__)


def build_reconstruction_loss(
    config: FullConfig,
    train_norm_stats: dict[str, Any] | None = None,
) -> LossFactory:
    norm_cfg = config.NORMALIZATION.model_dump()
    return LossFactory(
        norm_cfg=norm_cfg,
        norm_stats=train_norm_stats,
        enable_physics_head=config.HYPER_PARAMETERS.physics_head.enabled,
    )


def extract_and_save_latent_space(
    vae: torch.nn.Module,
    data_loader: DataLoader,
    config: FullConfig,
    device: torch.device,
    train_norm_stats: dict[str, Any] | None = None,
) -> Path:
    """Extract, reconstruct and save VAE outputs"""
    logger.info("Extracting latent space and reconstructions...")
    vae.eval()
    hyper_params = config.HYPER_PARAMETERS

    # Initialize collectors
    representations: dict[str, List] = {
        "latent_space_mu": [],
        "latent_space_logvar": [],
        "cycle_numbers": [],
        "soh_computed": [],
        "soh_predicted": [],
        "charging_rate": [],
        "reconstructions": [],  # 2-channel voltages [V_ch, V_dis]
        "inputs": [],  # Store original inputs for comparison
        "reconstruction_error_per_cycle": [],
        "reconstruction_error_per_cycle_abs": [],
        "reconstruction_error_per_cycle_per_channel": [],
    }

    vae_loss = LossFactory(
        config.NORMALIZATION.model_dump(),
        norm_stats=train_norm_stats,
        enable_physics_head=config.HYPER_PARAMETERS.physics_head.enabled,
    )
    with torch.no_grad():

        progress_bar = tqdm(
            enumerate(data_loader),
            total=len(data_loader),
            desc="Extracting latent space",
            colour="white",
            unit="batch",
            disable=not config.GENERAL.show_progress,
        )
        for batch_idx, (inputs, masks, token_masks, labels) in progress_bar:

            inputs = inputs.to(device)
            masks = masks.to(device)
            labels = labels.to(device)
            token_masks = token_masks.to(device)

            cycle_numbers         = labels[:, 1]
            norm_cycle_numbers    = labels[:, 2]
            soh_computed          = labels[:, 3]
            charging_rate         = labels[:, 5]
            norm_nominal_capacity = labels[:, 6]

            # Use all incident channels for the encoder (it will drop q_global/H_corr internally)
            inputs = inputs

            # Get all VAE outputs
            (
                reconstruction,
                mu,
                logvar,
                z,
                soh_predicted,
            ) = vae(inputs, token_masks)


            # qo_Ah = torch.full((inputs.shape[0],), 3.902778, dtype=inputs.dtype, device=inputs.device)
            q0_Ah = norm_nominal_capacity*5.000

            _, _, _, aux = vae_loss(
                reconstruction, inputs, mu, logvar, z, masks,
                training=False, q0_Ah=q0_Ah
            )
            cycle_mse_abs = aux["recon_loss_per_sample"]                       # (B,)
            cycle_mse = aux.get("recon_loss_per_sample_relative", cycle_mse_abs)  # (B,)
            cycle_mse_per_ch = aux.get("recon_loss_per_sample_per_channel")  # (B,C_sel) if present

            # soh_predicted = torch.full((inputs.shape[0],), 0.00001, dtype=inputs.dtype, device=inputs.device)

            representations["cycle_numbers"].extend(cycle_numbers.tolist())
            representations["soh_computed"].extend(soh_computed.tolist())
            representations["soh_predicted"].extend(soh_predicted.squeeze().cpu().tolist())
            representations["charging_rate"].extend(charging_rate.tolist())
            representations["latent_space_mu"].extend(mu.cpu().tolist())
            representations["latent_space_logvar"].extend(logvar.cpu().tolist())
            representations["reconstructions"].extend(reconstruction.cpu().tolist())
            representations["inputs"].extend(inputs.cpu().tolist())
            representations["reconstruction_error_per_cycle"].extend(cycle_mse.cpu().tolist())
            representations["reconstruction_error_per_cycle_abs"].extend(cycle_mse_abs.cpu().tolist())
            if cycle_mse_per_ch is not None:
                representations["reconstruction_error_per_cycle_per_channel"].extend(
                    cycle_mse_per_ch.cpu().tolist()
                )
            
            # Update progress bar
            progress_bar.set_postfix({"Batch": batch_idx + 1,})

    # Convert to tensors and sort by cycle number
    tensors = {k: torch.tensor(v) for k, v in representations.items()}
    sorted_indices = torch.argsort(tensors["cycle_numbers"])
    for key in tensors:
        tensors[key] = tensors[key][sorted_indices]

    # Save tensors
    output_dir = Path(config.PATHS.latent_space_save)
    torch.save(tensors, output_dir / "latent_space.pth")

    # Save CSV with latent space
    df_data = zip(
        tensors["latent_space_mu"].numpy(),
        tensors["cycle_numbers"].numpy(),
        tensors["soh_computed"].numpy(),
        tensors["soh_predicted"].numpy(),
    )

    latent_dim = tensors["latent_space_mu"].shape[1]
    headers = [f"z{i+1}" for i in range(latent_dim)] + [
        "Cycle",
        "SOH_computed",
        "SOH_predicted",
    ]

    with open(output_dir / "latent_space.csv", "w") as f:
        f.write(",".join(headers) + "\n")
        for latent, cycle, soh_computed, soh_pred in df_data:
            latent_str = ",".join(f"{x:.6f}" for x in latent)
            f.write(f"{latent_str},{int(cycle)},{soh_computed},{soh_pred}\n")

    #save_reconstructed_data(
    #    reconstructed_data=torch.tensor(representations["reconstructions"]),
    #    cycle_numbers=torch.tensor(representations["cycle_numbers"]),
    #    batch_idx=0,
    #    save_dir=config.PATHS.predicted_data,
    #    epoch=None,
    #)

    # Convert to DataFrame and save compact per-purpose exports.
    df = pd.DataFrame(representations)
    df = df.sort_values("cycle_numbers")

    # Calculate prediction metrics

    mse = mean_squared_error(df["soh_computed"], df["soh_predicted"])
    r2 = r2_score(df["soh_computed"], df["soh_predicted"])

    # Add metrics as additional columns
    df["abs_error"] = abs(df["soh_computed"] - df["soh_predicted"])
    df["squared_error"] = (df["soh_computed"] - df["soh_predicted"]) ** 2

    unique_rates = sorted({float(rate) for rate in df["charging_rate"].tolist()})
    if len(unique_rates) == 1:
        rate_tag = str(unique_rates[0]).replace(".", "p")
        soh_filename = f"soh_predictions_{rate_tag}.csv"
        reconstruction_filename = f"reconstructions_{rate_tag}.csv"
    else:
        soh_filename = "soh_predictions.csv"
        reconstruction_filename = "reconstructions.csv"

    soh_dir = Path(config.PATHS.predicted_data) / "SOH"
    soh_dir.mkdir(parents=True, exist_ok=True)
    soh_csv_path = soh_dir / soh_filename
    soh_df = df[["cycle_numbers", "soh_predicted", "soh_computed", "abs_error"]].copy()
    soh_df.to_csv(soh_csv_path, index=False)

    reconstruction_dir = Path(config.PATHS.predicted_data) / "Reconstruction"
    reconstruction_dir.mkdir(parents=True, exist_ok=True)
    reconstruction_csv_path = reconstruction_dir / reconstruction_filename
    reconstruction_df = df[["cycle_numbers", "reconstructions"]].copy()
    reconstruction_df.to_csv(reconstruction_csv_path, index=False)

    # Save summary metrics
    with open("artifacts/metrics/soh_metrics.txt", "w") as f:
       f.write(f"MSE: {mse:.6f}\n")
       f.write(f"R2 Score: {r2:.6f}\n")
       f.write(f"Mean Absolute Error: {df['abs_error'].mean():.6f}\n")

    logger.info(f"SOH predictions saved to {soh_csv_path}")
    logger.info(f"Reconstructions saved to {reconstruction_csv_path}")
    logger.info(f"MSE: {mse:.6f}")
    logger.info(f"R2 Score: {r2:.6f}")

    logger.info(f"Saved latent representations and reconstructions to {output_dir}")
    return soh_csv_path
