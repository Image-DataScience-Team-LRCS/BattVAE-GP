import torch
from pathlib import Path
from src.vae.inference.checkpoints import load_best_model_checkpoint
from src.vae.inference.inference import extract_and_save_latent_space
from src.common.logger.logging import setup_logging, get_logger
from torch.utils.data import DataLoader
from src.common.utils.utils import merge_dataloaders
from src.common.utils.config_schema import FullConfig
from typing import Tuple
from src.vae.analysis.plot_latent_distances import analyze_latent_distances
from src.vae.analysis.plot_soh import plot_soh_evolution
from src.vae.analysis.plot_metrics import plot_all_metrics, plot_kl_per_dimension
from src.vae.analysis.plot_latent_manifold import (
    visualize_latent_space,
    plot_latent_space_distributions,
    animate_latent_space,
)
from src.vae.analysis.plot_reconstructed_data import (
    animate_reconstructed_data,
    plot_reconstructed_error,
    plot_reconstructed_error_per_channel,
    plot_reconstruction_vs_input_multichannel,
)
from src.vae.analysis.correlation_analysis import LatentDependencyAnalyzer
import sys

logger = get_logger(__name__)


def run_inference(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    config: FullConfig,
    device: torch.device,
    filename: str = None,
) -> Path:
    """Run complete inference pipeline"""
    logger.info("🔍 Starting inference pipeline...")

    try:
        checkpoint = load_best_model_checkpoint(
            model=model,
            config=config,
            device=device,
            filename=filename,
        )
    except Exception as e:
        logger.error(f"�� Failed to load best model: {str(e)}", exc_info=True)
        sys.exit(1)


    if train_loader.batch_size == None:
        train_loader.batch_size = config.HYPER_PARAMETERS.batch_size

    if val_loader is None:
        logger.info("Using requested dataset loader only for inference.")
        merged_loader = train_loader
    else:
        logger.info("Merging datasets...")
        merged_loader = merge_dataloaders(train_loader, val_loader, train_loader.batch_size)

    train_norm_stats = dict(checkpoint.get("train_norm_stats", {}) or {})

    # Extract latent space
    logger.info("Extracting latent space...")
    soh_csv_path = extract_and_save_latent_space(
        model,
        merged_loader,
        config,
        device,
        train_norm_stats=train_norm_stats,
    )
    logger.info("Latent space saved! ✨")
    return soh_csv_path


def visualize_results(config: FullConfig, soh_csv_path: Path | None = None) -> None:
    """Generate all visualizations"""
    paths = config.PATHS
    latent_file_path = Path(paths.latent_space_save) / "latent_space.pth"

    # Plot metrics
    logger.info("Plotting training metrics...")
    plot_all_metrics(Path(paths.metrics) / "metrics.csv")
    plot_kl_per_dimension(Path(paths.metrics) / "metrics.csv")
    logger.info("✓ Plotted training metrics")

    # Latent space visualizations
    logger.info("Generating latent space visualizations...")
    visualize_latent_space(latent_file_path)
    plot_latent_space_distributions(latent_file_path)
    logger.info("✓ Generated latent space visualizations")

    # Correlation analysis
    logger.info("Analyzing latent dependencies...")
    latent_analyzer = LatentDependencyAnalyzer(
        latent_file_path,
        vis_range=None,
    )
    results = latent_analyzer.run_all()
    logger.info(50 * "=")
    logger.info("Detrended Correlation Matrix:")
    logger.info(results["correlation"])

    logger.info("Mutual Information Matrix:")
    logger.info(results["mutual_information"])

    logger.info("R² Predictability (each dim from others):")
    logger.info(results["r2_predictability"])
    logger.info(50 * "=")
    logger.info("✓ Analyzed latent dependencies")

    # Additional analyses
    logger.info("Analyzing latent distances...")
    analyze_latent_distances(
        Path(paths.latent_space_save) / "latent_space.csv",
        Path(paths.visualization) / "latent_distances.png",
    )
    logger.info("✓ Analyzed latent distances")

    # Plot reconstructed error
    logger.info("Plotting reconstruction error...")
    plot_reconstructed_error(
        latent_file_path, Path(paths.visualization) / "reconstruction_error.png"
    )
    plot_reconstructed_error_per_channel(
        latent_file_path,
        Path(paths.visualization) / "reconstruction_error_per_channel.png",
    )
    logger.info("✓ Plotted reconstruction error")

    # Plot reconstructed vs input multichannel voltage for selected cycles
    logger.info("Plotting reconstructed vs input voltage for selected cycles...")
    cycles_to_plot = [i for i in range(10, 100, 50)]  # adjust this list as needed
    plot_reconstruction_vs_input_multichannel(
        latent_space_path=latent_file_path,
        cycles=cycles_to_plot,
        save_dir=Path(paths.visualization),
        padding_value=config.HYPER_PARAMETERS.padding_value,
    )
    logger.info("✓ Plotted reconstructed vs input voltage")

    # SOH evolution
    plot_soh_evolution(
        soh_csv_path or Path(paths.predicted_data) / "soh" / "soh_predictions.csv",
        Path(paths.visualization) / "soh_evolution.png",
    )
    logger.info("✓ Plotted SOH evolution")

    # # Animate latent space
    # logger.info("Animating latent space evolution...")
    # animate_latent_space(
    #     latent_file_path,
    #     Path(paths.visualization) / 'LatentSpaceEvolution.gif'
    # )
    # logger.info("✓ Animated latent space evolution")

    # Animate reconstructed data
    # logger.info("Animating reconstructed data...")
    # animate_reconstructed_data(
    #     original_data_path=Path(paths.predicted_data) / 'original_normalized_data.csv',
    #     predicted_data_path=Path(paths.predicted_data) / 'reconstructed_final.csv',
    #     steps=10,
    #     save_path=Path(paths.visualization) / 'ReconstructedData.gif'
    # )
    # logger.info("✓ Animated reconstructed data")

    logger.info("Visualizations completed successfully!")
