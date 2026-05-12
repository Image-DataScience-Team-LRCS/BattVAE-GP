from __future__ import annotations

import logging
from dataclasses import asdict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gpytorch
import numpy as np
import torch
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import (
    Kernel,
    LinearKernel,
    MaternKernel,
    RBFKernel,
    RQKernel,
    ScaleKernel,
)
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.models import ApproximateGP
from gpytorch.mlls import VariationalELBO
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from src.common.logger.logging import get_logger
from rich.logging import RichHandler
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm


logger = get_logger(__name__)


def _log_info_file_only(message: str, *args: Any) -> None:
    logger.info(message, *args)


def _resolve_early_stopping_config(training_config: dict[str, Any]) -> tuple[bool, str, int, float]:
    early_stopping_cfg = dict(training_config.get("early_stopping", {}) or {})
    enabled = bool(early_stopping_cfg.get("enabled", False))
    monitor = str(early_stopping_cfg.get("monitor", "val_nll")).lower()
    patience = int(early_stopping_cfg.get("patience", training_config.get("patience", 10)))
    min_delta = float(early_stopping_cfg.get("min_delta", training_config.get("min_delta", 1e-4)))
    if monitor not in {"val_nll", "train_loss"}:
        raise ValueError(f"Unsupported early stopping monitor: {monitor}")
    return enabled, monitor, patience, min_delta


@dataclass
class PosteriorPrediction:
    mean: np.ndarray
    variance: np.ndarray

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(np.maximum(self.variance, 1e-12))


@dataclass
class GPTrainingSummary:
    best_epoch: int
    best_validation_nll: float | None
    best_test_nll: float | None
    final_train_loss: float
    epochs_completed: int
    final_learning_rate: float


def resolve_torch_device(runtime_config: dict[str, Any]) -> torch.device:
    requested = str(runtime_config.get("device", "auto")).lower()
    use_cuda = bool(runtime_config.get("use_cuda", True))

    if requested == "auto":
        if use_cuda and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for the GP experiment, but CUDA is not available.")

    return torch.device(requested)


def resolve_torch_dtype(runtime_config: dict[str, Any]) -> torch.dtype:
    name = str(runtime_config.get("dtype", "float32")).lower()
    if name == "float32":
        return torch.float32
    if name == "float64":
        return torch.float64
    raise ValueError(f"Unsupported torch dtype: {name}")


def set_random_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_inducing_points(
    x: np.ndarray,
    groups: np.ndarray | None,
    cycles: np.ndarray | None,
    inducing_points: int,
    inducing_method: str,
) -> np.ndarray:
    total = x.shape[0]
    count = min(int(inducing_points), total)
    if count <= 0:
        raise ValueError("inducing_points must be greater than zero.")
    if count == total:
        return x.copy()

    if inducing_method == "uniform" or groups is None or cycles is None:
        indices = np.linspace(0, total - 1, count, dtype=int)
        return x[indices]

    selected: list[int] = []
    unique_groups = list(dict.fromkeys(groups.tolist()))
    base = max(1, count // max(len(unique_groups), 1))
    extras = max(0, count - base * len(unique_groups))

    for group_index, group_name in enumerate(unique_groups):
        group_rows = np.where(groups == group_name)[0]
        ordered_rows = group_rows[np.argsort(cycles[group_rows])]
        quota = min(len(ordered_rows), base + (1 if group_index < extras else 0))
        if quota <= 0:
            continue
        local_idx = np.linspace(0, len(ordered_rows) - 1, quota, dtype=int)
        selected.extend(ordered_rows[local_idx].tolist())

    if len(selected) < count:
        existing = set(selected)
        filler = [
            idx
            for idx in np.linspace(0, total - 1, count, dtype=int).tolist()
            if idx not in existing
        ]
        selected.extend(filler[: count - len(selected)])

    return x[np.array(selected[:count], dtype=int)]


def _iter_leaf_kernels(kernel: Kernel) -> list[Kernel]:
    if hasattr(kernel, "base_kernel"):
        return _iter_leaf_kernels(kernel.base_kernel)
    if hasattr(kernel, "kernels"):
        leaves: list[Kernel] = []
        for part in kernel.kernels:
            leaves.extend(_iter_leaf_kernels(part))
        return leaves
    return [kernel]


def _build_component_kernel(name: str, input_dim: int) -> Kernel:
    kernel_name = name.strip().lower()
    if kernel_name == "rbf":
        return RBFKernel(ard_num_dims=input_dim)
    if kernel_name == "matern12":
        return MaternKernel(nu=0.5, ard_num_dims=input_dim)
    if kernel_name == "matern32":
        return MaternKernel(nu=1.5, ard_num_dims=input_dim)
    if kernel_name == "matern52":
        return MaternKernel(nu=2.5, ard_num_dims=input_dim)
    if kernel_name == "rq":
        return RQKernel(ard_num_dims=input_dim)
    if kernel_name == "linear":
        return LinearKernel(ard_num_dims=input_dim)
    raise ValueError(f"Unsupported kernel component: {name}")


def build_covariance_kernel(kernel_spec: str, input_dim: int) -> ScaleKernel:
    parts = [part.strip() for part in str(kernel_spec).split("+") if part.strip()]
    if not parts:
        raise ValueError("Kernel specification is empty.")

    kernel = _build_component_kernel(parts[0], input_dim)
    for part in parts[1:]:
        kernel = kernel + _build_component_kernel(part, input_dim)
    return ScaleKernel(kernel)


class VariationalSOHGPModel(ApproximateGP):
    def __init__(
        self,
        inducing_points: torch.Tensor,
        kernel_spec: str,
        input_dim: int,
        lengthscale: float | list[float],
        outputscale: float,
        rq_alpha: float,
    ) -> None:
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )
        super().__init__(variational_strategy)

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = build_covariance_kernel(kernel_spec, input_dim)

        self.mean_module.initialize(constant=0.0)
        self.covar_module.initialize(outputscale=float(outputscale))

        if np.isscalar(lengthscale):
            init_lengthscale = float(lengthscale)
        else:
            init_lengthscale = np.asarray(lengthscale, dtype=float).reshape(1, -1)

        for leaf in _iter_leaf_kernels(self.covar_module):
            # LinearKernel does not support lengthscale initialization.
            if hasattr(leaf, "lengthscale") and not isinstance(leaf, LinearKernel):
                leaf.initialize(lengthscale=init_lengthscale)
            if isinstance(leaf, RQKernel):
                leaf.initialize(alpha=float(rq_alpha))

    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class SparseGaussianProcessRegressor:
    def __init__(
        self,
        kernel: str = "rq",
        lengthscale: float | list[float] = 1.0,
        variance: float = 1.0,
        noise_variance: float = 1e-3,
        rq_alpha: float = 1.0,
        inducing_points: int = 256,
        inducing_method: str = "per_crate_cycle",
        jitter: float = 1e-6,
        random_seed: int = 42,
        training_config: dict[str, Any] | None = None,
        runtime_config: dict[str, Any] | None = None,
    ) -> None:
        self.kernel_spec = str(kernel)
        self.lengthscale = lengthscale
        self.variance = float(variance)
        self.noise_variance = float(noise_variance)
        self.rq_alpha = float(rq_alpha)
        self.inducing_points = int(inducing_points)
        self.inducing_method = str(inducing_method)
        self.jitter = float(jitter)
        self.random_seed = int(random_seed)
        self.training_config = dict(training_config or {})
        self.runtime_config = dict(runtime_config or {})

        self.device = resolve_torch_device(self.runtime_config)
        self.dtype = resolve_torch_dtype(self.runtime_config)

        self.model_: VariationalSOHGPModel | None = None
        self.likelihood_: GaussianLikelihood | None = None
        self.mll_: VariationalELBO | None = None
        self.training_summary_: GPTrainingSummary | None = None
        self.training_objective_: float | None = None
        self.inducing_locations_: np.ndarray | None = None
        self.training_history_: list[dict[str, Any]] = []

    def _to_tensor(self, array: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(array, dtype=self.dtype)

    def _predict_distribution(self, x: np.ndarray, include_noise: bool = True) -> tuple[np.ndarray, np.ndarray]:
        if self.model_ is None or self.likelihood_ is None:
            raise RuntimeError("Model must be fitted before prediction.")

        x_tensor = self._to_tensor(np.asarray(x, dtype=np.float32))
        dataset = TensorDataset(x_tensor)
        batch_size = int(self.training_config.get("eval_batch_size", 4096))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        mean_parts: list[np.ndarray] = []
        var_parts: list[np.ndarray] = []

        self.model_.eval()
        self.likelihood_.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            for (x_batch_cpu,) in loader:
                x_batch = x_batch_cpu.to(self.device)
                posterior = self.model_(x_batch)
                if include_noise:
                    posterior = self.likelihood_(posterior)
                mean_parts.append(posterior.mean.detach().cpu().numpy())
                var_parts.append(posterior.variance.detach().cpu().numpy())

        mean = np.concatenate(mean_parts, axis=0)
        variance = np.concatenate(var_parts, axis=0)
        return mean, np.maximum(variance, 1e-12)

    def _validation_nll(self, x: np.ndarray, y: np.ndarray) -> float:
        mean, variance = self._predict_distribution(x, include_noise=True)
        residual = y.reshape(-1) - mean.reshape(-1)
        return float(np.mean(0.5 * (np.log(2.0 * np.pi * variance) + (residual * residual) / variance)))

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray | None = None,
        cycles: np.ndarray | None = None,
        val_x: np.ndarray | None = None,
        val_y: np.ndarray | None = None,
        test_x: np.ndarray | None = None,
        test_y: np.ndarray | None = None,
        run_name: str | None = None,
        epochs_override: int | None = None,
    ) -> "SparseGaussianProcessRegressor":
        set_random_seed(self.random_seed)
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1)

        logger.info(
            "Starting GP fit%s: device=%s dtype=%s kernel=%s inducing_points=%d train_rows=%d",
            f" [{run_name}]" if run_name else "",
            self.device,
            self.dtype,
            self.kernel_spec,
            self.inducing_points,
            len(x),
        )

        self.inducing_locations_ = select_inducing_points(
            x=x,
            groups=groups,
            cycles=cycles,
            inducing_points=self.inducing_points,
            inducing_method=self.inducing_method,
        )
        logger.info(
            "Selected %d inducing points using method=%s",
            len(self.inducing_locations_),
            self.inducing_method,
        )

        train_x = self._to_tensor(x)
        train_y = self._to_tensor(y)
        inducing_tensor = self._to_tensor(self.inducing_locations_).to(self.device)

        self.model_ = VariationalSOHGPModel(
            inducing_points=inducing_tensor,
            kernel_spec=self.kernel_spec,
            input_dim=x.shape[1],
            lengthscale=self.lengthscale,
            outputscale=self.variance,
            rq_alpha=self.rq_alpha,
        ).to(self.device, self.dtype)
        self.likelihood_ = GaussianLikelihood(
            noise_constraint=gpytorch.constraints.GreaterThan(max(self.jitter, 1e-8))
        ).to(self.device, self.dtype)
        self.likelihood_.initialize(noise=float(self.noise_variance))
        self.mll_ = VariationalELBO(self.likelihood_, self.model_, num_data=train_x.size(0))

        optimizer = torch.optim.Adam(
            [
                {"params": self.model_.parameters()},
                {"params": self.likelihood_.parameters()},
            ],
            lr=float(self.training_config.get("learning_rate", 0.01)),
            weight_decay=float(self.training_config.get("weight_decay", 0.0)),
        )
        scheduler_config = dict(self.training_config.get("scheduler", {}) or {})
        scheduler_name = str(scheduler_config.get("name", "none")).lower()
        scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None = None
        if scheduler_name == "reduce_on_plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=str(scheduler_config.get("mode", "min")),
                factor=float(scheduler_config.get("factor", 0.5)),
                patience=int(scheduler_config.get("patience", 3)),
                threshold=float(scheduler_config.get("threshold", 1e-4)),
                threshold_mode=str(scheduler_config.get("threshold_mode", "rel")),
                cooldown=int(scheduler_config.get("cooldown", 0)),
                min_lr=float(scheduler_config.get("min_lr", 1e-5)),
            )
            logger.info(
                "Enabled ReduceLROnPlateau scheduler: factor=%.3f patience=%d threshold=%.6f min_lr=%.6g",
                float(scheduler_config.get("factor", 0.5)),
                int(scheduler_config.get("patience", 3)),
                float(scheduler_config.get("threshold", 1e-4)),
                float(scheduler_config.get("min_lr", 1e-5)),
            )
        elif scheduler_name not in {"", "none"}:
            raise ValueError(f"Unsupported scheduler: {scheduler_name}")

        train_loader = DataLoader(
            TensorDataset(train_x, train_y),
            batch_size=int(self.training_config.get("batch_size", 1024)),
            shuffle=True,
        )

        best_model_state = deepcopy(self.model_.state_dict())
        best_likelihood_state = deepcopy(self.likelihood_.state_dict())
        best_validation_nll = float("inf")
        best_epoch = 0
        epochs_without_improvement = 0
        best_test_nll: float | None = None

        epochs = int(epochs_override or self.training_config.get("epochs", 80))
        early_stopping_enabled, early_stopping_monitor, patience, min_delta = _resolve_early_stopping_config(
            self.training_config
        )
        log_every_epochs = int(self.training_config.get("log_every_epochs", 1))
        show_progress = bool(self.training_config.get("show_progress", True))
        final_train_loss = float("inf")
        current_learning_rate = float(optimizer.param_groups[0]["lr"])

        val_x_array = None if val_x is None else np.asarray(val_x, dtype=np.float32)
        val_y_array = None if val_y is None else np.asarray(val_y, dtype=np.float32).reshape(-1)
        test_x_array = None if test_x is None else np.asarray(test_x, dtype=np.float32)
        test_y_array = None if test_y is None else np.asarray(test_y, dtype=np.float32).reshape(-1)
        self.training_history_ = []

        if test_x_array is not None and test_y_array is not None:
            logger.info("Test-set NLL will be logged each epoch for monitoring only; it is not used for model selection.")

        epoch_iterator = tqdm(
            range(1, epochs + 1),
            desc=f"GP epochs{f' [{run_name}]' if run_name else ''}",
            leave=False,
            disable=not show_progress,
        )
        for epoch in epoch_iterator:
            self.model_.train()
            self.likelihood_.train()
            batch_losses: list[float] = []

            for x_batch_cpu, y_batch_cpu in train_loader:
                x_batch = x_batch_cpu.to(self.device)
                y_batch = y_batch_cpu.to(self.device)

                optimizer.zero_grad()
                output = self.model_(x_batch)
                loss = -self.mll_(output, y_batch)
                loss.backward()
                optimizer.step()
                batch_losses.append(float(loss.detach().cpu().item()))

            final_train_loss = float(np.mean(batch_losses))
            self.training_objective_ = final_train_loss
            validation_nll: float | None = None
            test_nll: float | None = None

            if (val_x_array is None or val_y_array is None) and early_stopping_monitor != "train_loss":
                best_epoch = epoch
                best_validation_nll = final_train_loss
                best_model_state = deepcopy(self.model_.state_dict())
                best_likelihood_state = deepcopy(self.likelihood_.state_dict())
            elif val_x_array is not None and val_y_array is not None:
                validation_nll = self._validation_nll(val_x_array, val_y_array)
                if validation_nll < (best_validation_nll - min_delta):
                    best_validation_nll = validation_nll
                    best_epoch = epoch
                    epochs_without_improvement = 0
                    best_model_state = deepcopy(self.model_.state_dict())
                    best_likelihood_state = deepcopy(self.likelihood_.state_dict())
                else:
                    epochs_without_improvement += 1

            if early_stopping_enabled and early_stopping_monitor == "train_loss":
                if final_train_loss < (best_validation_nll - min_delta):
                    best_validation_nll = final_train_loss
                    best_epoch = epoch
                    epochs_without_improvement = 0
                    best_model_state = deepcopy(self.model_.state_dict())
                    best_likelihood_state = deepcopy(self.likelihood_.state_dict())
                else:
                    epochs_without_improvement += 1

            if test_x_array is not None and test_y_array is not None:
                test_nll = self._validation_nll(test_x_array, test_y_array)
                if validation_nll is None:
                    best_test_nll = test_nll
                elif best_test_nll is None or epoch == best_epoch:
                    best_test_nll = test_nll

            scheduler_metric = final_train_loss if validation_nll is None else validation_nll
            if scheduler is not None:
                scheduler.step(scheduler_metric)
            current_learning_rate = float(optimizer.param_groups[0]["lr"])

            self.training_history_.append(
                {
                    "run_name": run_name or "",
                    "epoch": epoch,
                    "train_loss": final_train_loss,
                    "val_nll": validation_nll,
                    "test_nll": test_nll,
                    "learning_rate": current_learning_rate,
                    "best_epoch_so_far": best_epoch,
                    "best_val_nll_so_far": None if val_x_array is None else best_validation_nll,
                    "best_test_nll_so_far": best_test_nll,
                    "epochs_without_improvement": epochs_without_improvement,
                }
            )

            postfix: dict[str, str] = {"train": f"{final_train_loss:.4f}", "lr": f"{current_learning_rate:.2e}"}
            if validation_nll is not None:
                postfix["val"] = f"{validation_nll:.4f}"
            if test_nll is not None:
                postfix["test"] = f"{test_nll:.4f}"
            epoch_iterator.set_postfix(postfix)

            if epoch == 1 or epoch % log_every_epochs == 0 or epoch == epochs:
                _log_info_file_only(
                    "Epoch %d/%d%s: train_loss=%.6f%s%s%s",
                    epoch,
                    epochs,
                    f" [{run_name}]" if run_name else "",
                    final_train_loss,
                    f" lr={current_learning_rate:.6g}",
                    "" if validation_nll is None else f" val_nll={validation_nll:.6f}",
                    "" if test_nll is None else f" test_nll={test_nll:.6f}",
                )

            should_stop_on_val = (
                early_stopping_enabled
                and early_stopping_monitor == "val_nll"
                and val_x_array is not None
                and val_y_array is not None
            )
            should_stop_on_train = early_stopping_enabled and early_stopping_monitor == "train_loss"
            if (should_stop_on_val or should_stop_on_train) and epochs_without_improvement >= patience:
                logger.info(
                    "Early stopping triggered at epoch %d%s: best_epoch=%d monitor=%s best_score=%.6f",
                    epoch,
                    f" [{run_name}]" if run_name else "",
                    best_epoch,
                    early_stopping_monitor,
                    best_validation_nll,
                )
                break

        self.model_.load_state_dict(best_model_state)
        self.likelihood_.load_state_dict(best_likelihood_state)
        self.training_summary_ = GPTrainingSummary(
            best_epoch=best_epoch,
            best_validation_nll=None if val_x_array is None else best_validation_nll,
            best_test_nll=best_test_nll,
            final_train_loss=final_train_loss,
            epochs_completed=epoch,
            final_learning_rate=current_learning_rate,
        )
        self.training_objective_ = final_train_loss
        logger.info(
            "Finished GP fit%s: epochs_completed=%d best_epoch=%d final_train_loss=%.6f final_lr=%.6g%s%s",
            f" [{run_name}]" if run_name else "",
            epoch,
            best_epoch,
            final_train_loss,
            current_learning_rate,
            "" if self.training_summary_.best_validation_nll is None else f" best_val_nll={self.training_summary_.best_validation_nll:.6f}",
            "" if self.training_summary_.best_test_nll is None else f" monitored_test_nll={self.training_summary_.best_test_nll:.6f}",
        )
        return self

    def save_artifact(self, path: str | Path, metadata: dict[str, Any] | None = None) -> Path:
        if self.model_ is None or self.likelihood_ is None or self.inducing_locations_ is None:
            raise RuntimeError("Model must be fitted before saving.")

        save_path = Path(path).resolve()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "kernel_spec": self.kernel_spec,
            "lengthscale": self.lengthscale,
            "variance": self.variance,
            "noise_variance": self.noise_variance,
            "rq_alpha": self.rq_alpha,
            "inducing_points": self.inducing_points,
            "inducing_method": self.inducing_method,
            "jitter": self.jitter,
            "random_seed": self.random_seed,
            "training_config": self.training_config,
            "runtime_config": self.runtime_config,
            "inducing_locations": np.asarray(self.inducing_locations_, dtype=np.float32),
            "model_state_dict": self.model_.state_dict(),
            "likelihood_state_dict": self.likelihood_.state_dict(),
            "training_summary": None if self.training_summary_ is None else asdict(self.training_summary_),
            "metadata": dict(metadata or {}),
        }
        torch.save(payload, save_path)
        best_epoch = None if self.training_summary_ is None else self.training_summary_.best_epoch
        epochs_completed = None if self.training_summary_ is None else self.training_summary_.epochs_completed
        logger.info(
            "✅ Finished: Saving GP model artifact to %s (best_epoch=%s epochs_completed=%s)",
            save_path,
            best_epoch,
            epochs_completed,
        )
        return save_path

    @classmethod
    def load_artifact(
        cls,
        path: str | Path,
        map_location: str | torch.device | None = None,
    ) -> tuple["SparseGaussianProcessRegressor", dict[str, Any]]:
        artifact_path = Path(path).resolve()
        payload = torch.load(artifact_path, map_location=map_location, weights_only=False)
        model = cls(
            kernel=str(payload["kernel_spec"]),
            lengthscale=payload["lengthscale"],
            variance=float(payload["variance"]),
            noise_variance=float(payload["noise_variance"]),
            rq_alpha=float(payload["rq_alpha"]),
            inducing_points=int(payload["inducing_points"]),
            inducing_method=str(payload["inducing_method"]),
            jitter=float(payload["jitter"]),
            random_seed=int(payload["random_seed"]),
            training_config=dict(payload.get("training_config", {}) or {}),
            runtime_config=dict(payload.get("runtime_config", {}) or {}),
        )

        inducing_locations = np.asarray(payload["inducing_locations"], dtype=np.float32)
        inducing_tensor = model._to_tensor(inducing_locations).to(model.device)
        model.model_ = VariationalSOHGPModel(
            inducing_points=inducing_tensor,
            kernel_spec=model.kernel_spec,
            input_dim=inducing_locations.shape[1],
            lengthscale=model.lengthscale,
            outputscale=model.variance,
            rq_alpha=model.rq_alpha,
        ).to(model.device, model.dtype)
        model.likelihood_ = GaussianLikelihood(
            noise_constraint=gpytorch.constraints.GreaterThan(max(model.jitter, 1e-8))
        ).to(model.device, model.dtype)
        model.model_.load_state_dict(payload["model_state_dict"])
        model.likelihood_.load_state_dict(payload["likelihood_state_dict"])
        model.inducing_locations_ = inducing_locations

        training_summary = payload.get("training_summary")
        if training_summary is not None:
            model.training_summary_ = GPTrainingSummary(**training_summary)

        logger.info("Loaded GP model artifact from %s", artifact_path)
        return model, dict(payload.get("metadata", {}) or {})

    def predict(self, x: np.ndarray, include_noise: bool = True) -> PosteriorPrediction:
        mean, variance = self._predict_distribution(x, include_noise=include_noise)
        return PosteriorPrediction(mean=mean, variance=variance)
