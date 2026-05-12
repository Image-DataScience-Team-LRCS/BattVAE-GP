import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import List, Tuple
import pandas as pd
from src.common.logger.logging import get_logger

# Setup logging
logger = get_logger(__name__)


def compare_curves(raw_data: pd.DataFrame, cycle_num: List[int] = [0, 1]) -> None:
    """
    Compare voltage and current curves for specified cycles

    Args:
        raw_data (pd.DataFrame): DataFrame containing Voltage and Current data
        cycle_num (List[int]): List of cycle numbers to plot
    """
    # Create figure and subplots
    fig, axs = plt.subplots(2, 1, figsize=(12, 8))

    try:
        for cycle in cycle_num:
            # Filter data for the specified cycle
            if cycle not in raw_data["Cycle"].values:
                print(f"Cycle {cycle} not found in data.")
                continue
            cycle_data = raw_data[raw_data["Cycle"] == cycle]
            voltage = cycle_data["Voltage"].values
            current = cycle_data["Current"].values

            # Plot voltage in top subplot
            axs[0].plot(voltage, label=f"Cycle {cycle}")
            axs[0].set_title("Voltage Comparison")
            axs[0].set_xlabel("Time Step")
            axs[0].set_ylabel("Voltage (V)")
            axs[0].grid(True, alpha=0.3)
            axs[0].legend()

            # Plot current in bottom subplot
            axs[1].plot(current, label=f"Cycle {cycle}")
            axs[1].set_title("Current Comparison")
            axs[1].set_xlabel("Time Step")
            axs[1].set_ylabel("Current (A)")
            axs[1].grid(True, alpha=0.3)
            axs[1].legend()

        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(
            "artifacts/visualizations/voltage_current_comparison_cycle.png",
            dpi=300,
            bbox_inches="tight",
        )
        logger.info(
            f"Plot saved to: artifacts/visualizations/voltage_current_comparison_cycle.png"
        )
        plt.close()
    except Exception as e:
        logger.error(f"Error plotting curves: {str(e)}", exc_info=True)


def plot_feature_curves(
    raw_data: pd.DataFrame,
    normalized_data: np.ndarray,
    cycle_num: int = 10,
    name: str = "voltage_current_comparison_cycle",
) -> None:
    """
    Plot raw and normalized voltage and current curves for a specific cycle.

    Args:
        raw_data (pd.DataFrame): Original data containing 'Voltage' and 'Current'.
        normalized_data (torch.Tensor or np.array): Normalized data.
        cycle_num (int): Cycle number to visualize.
    """
    # Filtrer les données pour le cycle donné
    cycle_data = raw_data[raw_data["Cycle"] == cycle_num]
    raw_voltage = cycle_data["Voltage"].values
    raw_current = cycle_data["Current"].values

    mask = normalized_data != -9999.0
    normalized_data = normalized_data * mask

    # Normalisation min-max
    # min_vol, max_vol = raw_voltage.min(), raw_voltage.max()
    # min_cur, max_cur = raw_current.min(), raw_current.max()

    # normalized_voltage_raw = (raw_voltage - min_vol) / (max_vol - min_vol)
    # normalized_current_raw = (raw_current - min_cur) / (max_cur - min_cur)

    # Extraire les données normalisées fournies
    cycle_idx = cycle_num - 1  # Assuming cycles start at 1
    if isinstance(normalized_data, torch.Tensor):
        normalized_voltage = normalized_data[cycle_idx, 0].detach().cpu().numpy()
        normalized_current = normalized_data[cycle_idx, 1].detach().cpu().numpy()
        normalized_voltage_derivative = (
            normalized_data[cycle_idx, 2].detach().cpu().numpy()
        )
        normalized_current_derivative = (
            normalized_data[cycle_idx, 3].detach().cpu().numpy()
        )
        print(normalized_voltage_derivative)

    else:
        normalized_voltage = normalized_data[cycle_idx, 0]
        normalized_current = normalized_data[cycle_idx, 1]
        normalized_voltage_derivative = normalized_data[cycle_idx, 2]
        normalized_current_derivative = normalized_data[cycle_idx, 3]
        # print(normalized_voltage_derivative)

    # Assurer la même longueur
    min_length = min(len(raw_voltage), len(normalized_voltage))

    raw_voltage = raw_voltage[:min_length]
    raw_current = raw_current[:min_length]

    normalized_voltage = normalized_voltage[:min_length]
    normalized_current = normalized_current[:min_length]

    normalized_voltage_derivative = normalized_voltage_derivative[:min_length]
    normalized_current_derivative = normalized_current_derivative[:min_length]

    # Axe X (nombre de points de temps)
    x_points = np.arange(min_length)

    # Création des subplots
    fig, axs = plt.subplots(3, 2, figsize=(12, 8))

    # Raw Voltage
    axs[0, 0].plot(
        x_points, raw_voltage, color="blue", alpha=0.7, marker="o", markersize=2
    )
    axs[0, 0].set_title(f"Raw Voltage - Cycle {cycle_num}")
    axs[0, 0].set_xlabel("Time Step")
    axs[0, 0].set_ylabel("Voltage (V)")
    axs[0, 0].grid(True, alpha=0.3)

    # Raw Current
    axs[0, 1].plot(
        x_points, raw_current, color="green", alpha=0.7, marker="o", markersize=2
    )
    axs[0, 1].set_title(f"Raw Current - Cycle {cycle_num}")
    axs[0, 1].set_xlabel("Time Step")
    axs[0, 1].set_ylabel("Current (A)")
    axs[0, 1].grid(True, alpha=0.3)

    # Normalized Voltage
    # axs[1, 0].plot(x_points, normalized_voltage_raw, color='red', alpha=0.7, linestyle='--', marker='x', markersize=2, label='Min-Max')
    axs[1, 0].plot(
        x_points,
        normalized_voltage,
        color="purple",
        alpha=0.7,
        linestyle="-",
        marker="x",
        markersize=2,
        label="Normalized Data",
    )
    axs[1, 0].set_title(f"Normalized Voltage - Cycle {cycle_num}")
    axs[1, 0].set_xlabel("Time Step")
    axs[1, 0].set_ylabel("Normalized Voltage")
    axs[1, 0].grid(True, alpha=0.3)

    # Normalized Current
    # axs[1, 1].plot(x_points, normalized_current_raw, color='orange', alpha=0.7, linestyle='--', marker='x', markersize=2, label='Min-Max')
    axs[1, 1].plot(
        x_points,
        normalized_current,
        color="brown",
        alpha=0.7,
        linestyle="-",
        marker="x",
        markersize=2,
        label="Normalized Data",
    )
    axs[1, 1].set_title(f"Normalized Current - Cycle {cycle_num}")
    axs[1, 1].set_xlabel("Time Step")
    axs[1, 1].set_ylabel("Normalized Current")
    axs[1, 1].grid(True, alpha=0.3)

    # Normalized Voltage derivative
    # axs[1, 0].plot(x_points, normalized_voltage_raw, color='red', alpha=0.7, linestyle='--', marker='x', markersize=2, label='Min-Max')
    axs[2, 0].plot(
        x_points,
        normalized_voltage_derivative,
        color="purple",
        alpha=0.7,
        linestyle="-",
        marker="x",
        markersize=2,
        label="Normalized Data",
    )
    axs[2, 0].set_title(f"Normalized Voltage derivate - Cycle {cycle_num}")
    axs[2, 0].set_xlabel("Time Step")
    axs[2, 0].set_ylabel("Normalized Voltage derivative")
    axs[2, 0].grid(True, alpha=0.3)

    # Normalized Current derivative
    # axs[1, 1].plot(x_points, normalized_current_raw, color='orange', alpha=0.7, linestyle='--', marker='x', markersize=2, label='Min-Max')
    axs[2, 1].plot(
        x_points,
        normalized_current_derivative,
        color="brown",
        alpha=0.7,
        linestyle="-",
        marker="x",
        markersize=2,
        label="Normalized Data",
    )
    axs[2, 1].set_title(f"Normalized Current derivative - Cycle {cycle_num}")
    axs[2, 1].set_xlabel("Time Step")
    axs[2, 1].set_ylabel("Normalized Current derivative")
    axs[2, 1].grid(True, alpha=0.3)

    # Ajuster l'espace entre les subplots
    plt.tight_layout()

    # Sauvegarde
    save_path = f"artifacts/visualizations/{name}_{cycle_num}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
