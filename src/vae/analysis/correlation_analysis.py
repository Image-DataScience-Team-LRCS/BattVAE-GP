import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import r2_score
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional, Tuple
from pathlib import Path
import torch
import os


class LatentDependencyAnalyzer:
    def __init__(
        self,
        latent_file: Path,
        vis_range: Optional[Tuple[int, int]] = None,
        save_dir="artifacts/visualizations/",
    ) -> None:
        """
        :param latent_space: np.ndarray of shape (N, D) where N = # time steps, D = # latent dims
        :param save_dir: Directory to save visualizations
        """

        self.latent_file = latent_file
        self.vis_range = vis_range
        self.data = torch.load(latent_file, weights_only=True)
        self.latent_space = self.data["latent_space_mu"].numpy()
        self.cycle_numbers = np.array(self.data["cycle_numbers"])
        self.save_dir = save_dir

        # Sort by cycle number
        sort_indices = np.argsort(self.cycle_numbers)
        self.cycle_numbers = self.cycle_numbers[sort_indices]
        self.latent_space = self.latent_space[sort_indices]

        self.N, self.D = self.latent_space.shape
        self.latent_detrended = self._detrend_latents()
        self.feature_names = [f"z{i}" for i in range(self.D)]

    def _detrend_latents(self) -> np.ndarray:
        t = np.arange(self.N).reshape(-1, 1)
        detrended = np.zeros_like(self.latent_space)
        for j in range(self.D):
            model = LinearRegression().fit(t, self.latent_space[:, j])
            trend = model.predict(t)
            detrended[:, j] = self.latent_space[:, j] - trend
        return detrended

    def compute_correlation(self) -> np.ndarray:
        df = pd.DataFrame(self.latent_detrended, columns=self.feature_names)
        corr_matrix = df.corr()
        plt.figure(figsize=(8, 6))
        sns.heatmap(np.abs(corr_matrix), annot=True, cmap="coolwarm", vmin=0, vmax=1)
        plt.title("Heatmap of Detrended Latent Correlations")
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/detrended_correlation_heatmap.png", dpi=300)
        plt.close()
        return corr_matrix

    def compute_mutual_information(self) -> np.ndarray:
        mi_matrix = np.zeros((self.D, self.D))
        df = pd.DataFrame(self.latent_detrended, columns=self.feature_names)
        for i in range(self.D):
            for j in range(self.D):
                if i != j:
                    mi = mutual_info_regression(
                        df[[self.feature_names[i]]], df[self.feature_names[j]]
                    )[0]
                    mi_matrix[i, j] = mi

        plt.figure(figsize=(8, 6))
        sns.heatmap(mi_matrix, annot=True, cmap="YlGnBu")
        plt.title("Mutual Information Between Detrended Latent Dimensions")
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/mutual_information_heatmap.png", dpi=300)
        plt.close()
        return mi_matrix

    def compute_r2_predictability(self) -> np.ndarray:
        r2_matrix = np.zeros((self.D, self.D))
        for target in range(self.D):
            predictors = [i for i in range(self.D) if i != target]
            X = self.latent_detrended[:, predictors]
            y = self.latent_detrended[:, target]
            model = LinearRegression().fit(X, y)
            y_pred = model.predict(X)
            r2 = r2_score(y, y_pred)
            # Store in diagonal (self-predictability using others)
            r2_matrix[target, target] = r2

        # Plot heatmap (diagonal only has values)
        display_matrix = np.zeros_like(r2_matrix)
        for i in range(self.D):
            display_matrix[i, i] = r2_matrix[i, i]

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            display_matrix,
            annot=True,
            cmap="Purples",
            vmin=0,
            vmax=1,
            xticklabels=self.feature_names,
            yticklabels=self.feature_names,
        )
        plt.title("R² Predictability of Each Latent from Others")
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/r2_predictability_heatmap.png", dpi=300)
        plt.close()

        return r2_matrix

    def run_all(self) -> dict:
        print("Running full latent dependency analysis...")
        corr = self.compute_correlation()
        mi = self.compute_mutual_information()
        r2 = self.compute_r2_predictability()
        return {"correlation": corr, "mutual_information": mi, "r2_predictability": r2}
