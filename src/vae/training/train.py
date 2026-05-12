import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.vae.training.loss import LossFactory
from src.common.utils.utils import save_metrics, save_reconstructed_data
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.common.logger.logging import get_logger
from pathlib import Path
from tqdm import tqdm

from src.common.utils.config_schema import FullConfig
import gpytorch
from gpytorch.kernels import Kernel
from typing import List, Optional, Tuple, Dict, Any
from src.vae.models.gp_prior import log_gp_hypers
from src.common.utils.conditioning import build_condition_vector

logger = get_logger(__name__)


class Trainer:
    """
    Trainer class for VAE with GP prior.
    Handles training and validation using a variational GP prior in latent space.
    """

    def __init__(
        self,
        model: nn.Module,
        config: FullConfig,
        device: torch.device,
        train_norm_stats: Optional[Dict[str, Any]] = None,
        gp_model: Optional[nn.Module] = None,
        likelihood: Optional[gpytorch.likelihoods.GaussianLikelihood] = None,
        mll: Optional[gpytorch.mlls.ExactMarginalLogLikelihood] = None,
    ) -> None:
        """Initialize the trainer with configuration"""
        self.config = config
        self.device = device
        self.model = model.to(self.device)
        self.hyper_params = self.config.HYPER_PARAMETERS
        self.show_progress = self.config.GENERAL.show_progress
        self.metrics: dict[str, float] = []

        norm_cfg = config.NORMALIZATION.model_dump()
        latent_dependence_cfg = config.HYPER_PARAMETERS.latent_dependence.model_dump()
        self.loss_fn = LossFactory(
            norm_cfg=norm_cfg,
            enable_physics_head=config.HYPER_PARAMETERS.physics_head.enabled,
            latent_dependence_cfg=latent_dependence_cfg,
        )

        # Initialize basic components
        self.best_train_loss = float("inf")
        self.model_save_path = config.PATHS.model_save
        self.metrics_save_path = config.PATHS.metrics
        self.train_norm_stats = train_norm_stats or {}

        # Setup GP components if enabled
        self.use_gp = self.hyper_params.gp_schedule.enable
        if self.use_gp:
            self.gp_model = gp_model.to(device)
            self.likelihood = likelihood.to(device)
            self.mll = mll.to(device)

        self._initialize_optimizers()

        # Early stopping setup
        self.early_stopping_counter = 0
        self.best_val_loss = float("inf")
        self.early_stopping_patience = self.hyper_params.early_stopping.patience
        self.min_delta = self.hyper_params.early_stopping.min_delta


    def _build_cond_vec(
        self,
        norm_cycle_numbers: torch.Tensor,
        charging_rate: torch.Tensor,
        norm_nominal_capacity: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        return build_condition_vector(
            norm_cycle_numbers=norm_cycle_numbers,
            charging_rate=charging_rate,
            norm_nominal_capacity=norm_nominal_capacity,
            cond_dim=int(getattr(self.hyper_params, "conditional_vector_dim", 0)),
            device=self.device,
        )


    def _initialize_optimizers(self) -> None:
        """Initialize optimizers and schedulers"""
        # VAE optimizer
        lr_schedule = self.hyper_params.lr_schedule

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=lr_schedule.learning_rate,
            weight_decay=lr_schedule.weight_decay,
        )

        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=lr_schedule.decay_factor,
            patience=lr_schedule.patience,
            min_lr=lr_schedule.min_lr,
        )


        # GP optimizer setup if GP components are available
        if self.use_gp:
            gp_schedule = self.hyper_params.gp_schedule
            gp_params = [
                {"params": self.gp_model.mean_module.parameters()},
                {"params": self.gp_model.covar_module.parameters()},
                {"params": self.likelihood.parameters()},
            ]
            self.gp_optimizer = optim.Adam(
                gp_params,
                lr=gp_schedule.learning_rate,
                weight_decay=gp_schedule.weight_decay,
            )
            self._initialize_gp_prior()

    def _initialize_gp_prior(
        self,
        lengthscale: float | tuple[float, float] = 0.2,
        outputscale: float | None = 1.0,
        noise: float | None = 1e-3,
    ) -> None:
        """
        Initialize the batched GP prior (Option B) in a version-safe, kernel-agnostic way.
    
        lengthscale:
            - float -> same value for both input dims (cycle, rate)
            - (l_cycle, l_rate) -> ARD values for 2 input dims
        outputscale:
            - if not None, set covar_module.outputscale to this value
        noise:
            - if not None, set GaussianLikelihood noise per task to this value
        """
    
        gp = self.gp_model          # BatchedGPModel
        lk = self.likelihood        # GaussianLikelihood(batch_shape=[D])
    
        device = next(gp.parameters()).device
        dtype  = next(gp.parameters()).dtype
    
        # ---- build a [.., 2] tensor for ARD or a scalar for isotropic
        if isinstance(lengthscale, tuple) or isinstance(lengthscale, list):
            assert len(lengthscale) == 2, "ARD needs two values: (l_cycle, l_rate)"
            ls_base = torch.tensor(lengthscale, device=device, dtype=dtype)    # [2]
        else:
            ls_base = torch.tensor([float(lengthscale), float(lengthscale)],
                                   device=device, dtype=dtype)                 # [2]
    
        # Expand to batch shape if needed (for kernels with shape [D, 1, 2])
        def _shape_like_lengthscale_param(param: torch.Tensor) -> torch.Tensor:
            """
            Return ls tensor broadcastable to `param` shape.
            Common shapes:
              - [D, 1, 2] : batch_shape=[D], ARD=2
              - [1, 1, 2] : no batch, ARD=2
              - [D, 1, 1] : isotropic (we'll pass scalar)
            """
            # If last dim is 2, we provide ARD vector; else use scalar
            if param.dim() >= 1 and param.shape[-1] == 2:
                # make [1,1,2] and broadcast to param
                ls = ls_base.view(1, 1, 2)
                # if there is a batch dim D, expand it
                if param.dim() >= 3 and param.shape[0] > 1:
                    ls = ls.expand(param.shape[0], 1, 2)
                return ls.to(device=device, dtype=dtype)
            else:
                # isotropic: single scalar
                return torch.tensor(float(ls_base.mean().item()), device=device, dtype=dtype)
    
        @torch.no_grad()
        def _init_kernels_recursively(k: Kernel) -> None:
            """
            Walk Sum/Product kernels and initialize any sub-kernel that has .lengthscale.
            """
            # Compound kernels in gpytorch typically expose `.kernels`
            subks = getattr(k, "kernels", None)
            if subks is not None and isinstance(subks, (list, tuple)):
                for child in subks:
                    _init_kernels_recursively(child)
                return
    
            # Plain kernel: if it has a lengthscale tensor, set it
            if hasattr(k, "lengthscale") and k.lengthscale is not None:
                ls_param = k.lengthscale  # this is a Parameter
                ls_value = _shape_like_lengthscale_param(ls_param)
                # honor lower bound constraint
                try:
                    lb = getattr(k, "lengthscale_constraint", None)
                    if lb is not None:
                        min_val = float(getattr(lb, "lower_bound", torch.tensor(0.0, device=device)))
                        if isinstance(ls_value, torch.Tensor):
                            ls_value = ls_value.clamp_min(min_val + 1e-6)
                        else:
                            ls_value = max(float(ls_value), min_val + 1e-6)
                except Exception:
                    pass
                ls_param.copy_(ls_value)
    
        # ---- initialize base kernel(s)
        base = gp.covar_module.base_kernel
        _init_kernels_recursively(base)
    
        # ---- outputscale (covariance amplitude)
        if outputscale is not None:
            with torch.no_grad():
                gp.covar_module.outputscale.fill_(float(outputscale))
    
        # ---- likelihood noise per task (batch_shape=[D])
        if noise is not None:
            with torch.no_grad():
                target = torch.as_tensor(float(noise), device=device, dtype=dtype)
                # GaussianLikelihood w/ batch_shape=[D] supports vector noise; setting a scalar broadcasts
                lk.noise = target
    
        # ---- small summary
        try:
            # Try to read a representative lengthscale back
            rep_ls = None
            if hasattr(base, "lengthscale") and base.lengthscale is not None:
                rep_ls = base.lengthscale
            elif hasattr(base, "kernels"):
                for child in base.kernels:
                    if hasattr(child, "lengthscale") and child.lengthscale is not None:
                        rep_ls = child.lengthscale
                        break
            rep_ls_str = f"{tuple(rep_ls.shape)}" if rep_ls is not None else "n/a"
            print(f"[GP init] lengthscale set (example param shape {rep_ls_str}), "
                  f"outputscale={gp.covar_module.outputscale.detach().mean().item():.4f}, "
                  f"noise≈{lk.noise.detach().mean().item():.4e}")
        except Exception:
            pass

    def _adaptive_gp_init(self, gp_input: torch.Tensor, z: torch.Tensor) -> None:
        """Adaptively initialize GP parameters based on actual data characteristics"""
        if hasattr(self, '_gp_initialized_with_data'):
            return
        
        with torch.no_grad():
            # Estimate lengthscale from input range
            input_range = gp_input.max(dim=0)[0] - gp_input.min(dim=0)[0]
            # Use 10-20% of input range as initial lengthscale
            lengthscale = input_range.mean().item() * 0.15
            lengthscale = max(lengthscale, 0.01)  # Minimum lengthscale
            
            # Estimate output scale from target variance
            z_var = torch.var(z, dim=0).mean().item()
            outputscale = max(z_var, 1e-4)  # Minimum output scale
            
            # Update GP parameters
            self.gp_model.covar_module.outputscale.fill_(outputscale)
            
            # Handle composite kernel (linear_kernel + rbf_kernel)
            base_kernel = self.gp_model.covar_module.base_kernel
            
            if hasattr(base_kernel, 'kernels'):  # AdditiveKernel
                for kernel in base_kernel.kernels:
                    # Only set lengthscale for kernels that have it (RBF, Matern, etc.)
                    if hasattr(kernel, 'lengthscale') and kernel.lengthscale is not None:
                        kernel.lengthscale.fill_(lengthscale)
            elif hasattr(base_kernel, 'lengthscale') and base_kernel.lengthscale is not None:
                # Single kernel case
                base_kernel.lengthscale.fill_(lengthscale)
            
            # Initialize mean to approximate mean of latent variables
            z_mean = torch.mean(z, dim=0).mean().item()
            self.gp_model.mean_module.constant.fill_(z_mean)

            self._gp_initialized_with_data = True

    def train(
        self, train_loader: DataLoader, val_loader: DataLoader
    ) -> Tuple[nn.Module, List[float]]:
        """Train the VAE model"""

        start_epoch = 0
        try:
            if self.config.GENERAL.resume_training:
                start_epoch = self.load_checkpoint("best_model.pth")
                logger.info(f"🔄 Resuming training from epoch {start_epoch}")
        except Exception as e:
            logger.error(f"⚠️ Failed to resume training: {str(e)}", exc_info=True)

        if start_epoch >= self.hyper_params.epochs:
            logger.info(f"✅ Training already completed up to epoch {start_epoch}.")
            return self.model, self.metrics

        if self.use_gp:
            logger.info(f"MLL num data: {len(train_loader.dataset)}")
            self.mll.num_data = compute_elbo_num_data(train_loader)
        # torch.autograd.set_detect_anomaly(True)

        # self._build_joint_rate_cycle_tables(
        #     train_loader,
        #     n_cycle_bins=20,
        #     binning="uniform",
        #     early_boost=self.early_cycle_boost_initial,
        # )

        monitor_metric = self.hyper_params.early_stopping.monitor_metric
        best_monitor_value = float("inf")
        for epoch in range(start_epoch, self.hyper_params.epochs):
            epoch_num = epoch + 1
            warmup_active = epoch_num <= self.hyper_params.early_stopping.warmup_epochs

            prefix = "🔥 Warmup" if warmup_active else "🔬 Epoch"
            logger.info(
                f"{prefix} {epoch_num}/{self.hyper_params.epochs} "
                f"with lr={self.optimizer.param_groups[0]['lr']:.4e}"
            )

            # Training
            train_loss = self._train_epoch(train_loader, epoch)

            # Validation
            val_loss = None
            if val_loader is not None:
                val_loss = self._validate_epoch(val_loader, epoch)

            # Select monitor value
            if monitor_metric == "train_loss":
                monitor_value = train_loss
            elif monitor_metric == "val_loss":
                if val_loss is None:
                    raise ValueError(
                        "monitor_metric='val_loss' but no val_loader was provided."
                    )
                monitor_value = val_loss
            else:
                raise ValueError(
                    f"Unsupported monitor_metric: {monitor_metric}. "
                    f"Use 'train_loss' or 'val_loss'."
                )

            # Scheduler step
            self.scheduler.step(monitor_value)

            # GP hyperparameter logging
            if self.use_gp:
                self._log_gp_hyperparameters(epoch)

            # Early stopping
            if not warmup_active:
                if monitor_value < (best_monitor_value - self.min_delta):
                    best_monitor_value = monitor_value
                    self.best_val_loss = monitor_value  # kept for compatibility
                    self.early_stopping_counter = 0
                    filename = "best_model.pth"
                    self.save_checkpoint(filename)
                else:
                    self.early_stopping_counter += 1

                if self.early_stopping_counter >= self.early_stopping_patience:
                    logger.info(
                        f"⏹️ Early stopping triggered after {epoch_num} epochs "
                        f"based on {monitor_metric}={monitor_value:.6f}"
                    )
                    break
            else:
                logger.info(
                    f"🔥 Warmup epoch {epoch_num}/"
                    f"{self.hyper_params.early_stopping.warmup_epochs}: "
                    f"skipping early stopping"
                )

                if monitor_value < (best_monitor_value - self.min_delta):
                    best_monitor_value = monitor_value
                    self.best_val_loss = monitor_value  # kept for compatibility
                    filename = f"best_model_epoch{epoch_num}.pth"
                    self.save_checkpoint(filename)

        
        save_metrics(self.metrics, Path(self.config.PATHS.metrics))
        logger.info(f"📈 Metrics saved to {self.metrics_save_path}")
        return self.model, filename

    def _log_gp_hyperparameters(self, epoch: int) -> None:
        """
        Log key GP prior hyperparameters for monitoring convergence.

        Logs (per epoch):
          - kernel components and outputscale stats (via log_gp_hypers)
          - likelihood noise statistics across tasks
        """
        try:
            log_gp_hypers(
                self.gp_model,
                prefix=f"[GP][epoch {epoch+1}]",
                logger=logger,
            )

            with torch.no_grad():
                noise_vals = None
                if hasattr(self.likelihood, "task_noises"):
                    noise_vals = self.likelihood.task_noises.detach().cpu()
                elif hasattr(self.likelihood, "noise"):
                    noise_vals = self.likelihood.noise.detach().flatten().cpu()

                if noise_vals is not None and noise_vals.numel() > 0:
                    logger.info(
                        f"[GP][epoch {epoch+1}] noise "
                        f"mean={noise_vals.mean().item():.4e} "
                        f"min={noise_vals.min().item():.4e} "
                        f"max={noise_vals.max().item():.4e}"
                    )
        except Exception as e:
            logger.warning(f"[GP] Failed to log GP hyperparameters: {e}")

    def _freeze_baseline_if_needed(self, epoch: int) -> None:
        freeze_epoch = self.hyper_params.freeze_baseline_epoch
        if epoch < freeze_epoch:
            return
        if getattr(self, "_baseline_frozen_done", False):
            return

        # --- locate decoder ---
        # adjust if your model structure differs
        dec = None
        if hasattr(self.model, "decoder"):
            dec = self.model.decoder
        elif hasattr(self.model, "vae") and hasattr(self.model.vae, "decoder"):
            dec = self.model.vae.decoder
        else:
            raise RuntimeError("Could not find decoder (expected self.model.decoder or self.model.vae.decoder)")

        # --- freeze baseline weights only ---
        # IMPORTANT: do not wrap baseline forward in no_grad if you later use autograd to compute dV/dQ
        if hasattr(dec, "freeze_baseline"):
            dec.freeze_baseline(use_no_grad=False)  # weights frozen, forward still differentiable w.r.t q
        else:
            # fallback: freeze by module name(s); change 'baseline' to your actual attribute
            if not hasattr(dec, "baseline"):
                raise RuntimeError("Decoder has no freeze_baseline() and no .baseline attribute to freeze.")
            for p in dec.baseline.parameters():
                p.requires_grad_(False)

        # --- rebuild optimizer (Adam) using only trainable params ---
        old_opt = self.optimizer
        current_lr = old_opt.param_groups[0]["lr"]
        wd = old_opt.param_groups[0].get("weight_decay", 0.0)

        self.optimizer = optim.Adam(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=current_lr,
            weight_decay=wd,
        )

        # --- rebuild ReduceLROnPlateau scheduler because it references optimizer instance ---
        lr_schedule = self.hyper_params.lr_schedule
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=lr_schedule.decay_factor,
            patience=lr_schedule.patience,
            min_lr=1e-5,
        )

        self._baseline_frozen_done = True



    def _get_kl_weight(self, epoch: int) -> float:
        """Calculate KL weight with annealing"""
        # Validate parameters
        annealing_config = self.hyper_params.kl_annealing
        warmup_epochs = annealing_config.warmup_epochs
        start_weight = annealing_config.start_factor
        target_weight = annealing_config.target_factor

        # Parameter validation
        if warmup_epochs <= 0:
            logger.error(f"warmup_epochs must be positive, got {warmup_epochs}")
            sys.exit(1)
        if start_weight < 0 or target_weight < 0:
            logger.error(f"Weights must be non-negative, got {start_weight} and {target_weight}")
            sys.exit(1)

        # Calculate weight with smooth transition
        if epoch < warmup_epochs:
            # Cosine annealing for smoother transition
            progress = min(epoch / warmup_epochs, 1.0)
            # progress = 0.5 * (1 - torch.cos(torch.tensor(progress * torch.pi))) ## Cosine annealing
            weight = start_weight + (target_weight - start_weight) * progress
        else:
            weight = target_weight

        return float(weight)


    def _current_early_cycle_boost(self, epoch: int) -> float:
        """
        Epoch-dependent early-cycle boost:
        - keep full boost for initial hold fraction
        - linearly decay to floor by decay_end fraction
        - stay at floor afterwards
        """
        total_epochs = max(1, int(self.hyper_params.epochs))
        hold_ep = int(round(self.early_cycle_boost_hold_frac * total_epochs))
        decay_end_ep = int(round(self.early_cycle_boost_decay_end_frac * total_epochs))
        decay_end_ep = max(decay_end_ep, hold_ep + 1)

        if epoch < hold_ep:
            return float(self.early_cycle_boost_initial)
        if epoch >= decay_end_ep:
            return float(self.early_cycle_boost_min)

        progress = float(epoch - hold_ep) / float(decay_end_ep - hold_ep)
        progress = min(max(progress, 0.0), 1.0)
        return float(
            self.early_cycle_boost_initial
            + (self.early_cycle_boost_min - self.early_cycle_boost_initial) * progress
        )

    def _joint_rate_cycle_weights(
        self,
        charging_rate: torch.Tensor,
        norm_cycle_values: torch.Tensor,
        epoch: int,
    ) -> torch.Tensor:
        """
        Return per-sample weights from joint (rate, cycle_bin) counts with optional
        early-cycle phase boost.
        """
        if not hasattr(self, "rate_cycle_counts"):
            return torch.ones_like(norm_cycle_values, dtype=torch.float32)

        rate = torch.round(charging_rate.float() * 1000) / 1000
        rate_levels = self.rate_levels.to(rate.device)
        rate_id = torch.argmin((rate[:, None] - rate_levels[None, :]).abs(), dim=1)  # [B]

        cycle_bin = self._cycle_to_bin_ids(norm_cycle_values, device=norm_cycle_values.device)
        counts = self.rate_cycle_counts.to(rate.device)[rate_id, cycle_bin].float()
        inv = self.rate_cycle_scale.to(rate.device) / counts
        beta_t = self._current_early_cycle_boost(epoch)
        centers = self.cycle_bin_centers.to(rate.device)[cycle_bin]
        phase = 1.0 + beta_t * (1.0 - centers)
        return inv * phase

    def _per_cycle_weights(
        self,
        norm_cycle_values: torch.Tensor,
        epoch: int,
    ) -> torch.Tensor:
        """
        Continuous per-cycle weighting on top of joint (rate, cycle-bin) weighting.
        Larger weights for early cycles:
          w_cycle = 1 + gamma_t * (1 - norm_cycle)^power
        where gamma_t follows the same early-boost schedule.
        """
        x = torch.nan_to_num(
            norm_cycle_values.float(),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        ).clamp(0.0, 1.0)
        gamma_t = 1.0 #self._current_early_cycle_boost(epoch)
        # return 1.0 + gamma_t * torch.pow(1.0 - x, self.per_cycle_weight_power)
        return 1.0 + gamma_t * torch.pow(1.0 - x, 10.0)
    


    def _train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """Run one training epoch"""
        self.model.train()
        if self.use_gp:
            self.gp_model.train()
            self.likelihood.train()

        num_batches = len(train_loader)
        kl_weight = self._get_kl_weight(epoch)
        gp_scale = float(self.hyper_params.gp_schedule.scale_factor)

        gp_scale  = gp_scale * min(1.0, (epoch+1) / 10.0)

        soh_scale = float(self.hyper_params.soh_factor)
        soh_scale = soh_scale * min(1.0, (epoch+1) / 10.0)

        epoch_losses = {
            "total": 0.0,
            "reconstruction": 0.0,
            "kl": 0.0,
            "gp_loss": 0.0,
            "soh_loss": 0.0,
            "latent_dependence": 0.0,
        }
        latent_stats = {
            "cycle_corr": 0.0,
            "crate_corr": 0.0,
            "cycle_swap_corr": 0.0,
            "crate_swap_corr": 0.0,
        }
        kl_per_dim_batches = []

        self._freeze_baseline_if_needed(epoch)

        progress_bar = tqdm(
            enumerate(train_loader),
            total=num_batches,
            desc="⚽️ Training",
            leave=True,
            colour="blue",
            disable=not self.show_progress,
        )

        for batch_idx, (x_batch, mask_batch, token_mask_batch, label_batch) in progress_bar:
            x_batch = x_batch.to(self.device)
            label_batch = label_batch.to(self.device)
            mask = mask_batch.to(self.device)
            token_mask = token_mask_batch.to(self.device)

            # Add a little noise to the input data
            # noise = torch.randn_like(x_batch) * 0.001
            x_batch_noised = x_batch

            # Add numerical stability checks
            if torch.isnan(x_batch).any() or torch.isinf(x_batch).any():
                logger.error("Exiting: Input contains nan or inf values")
                logger.error(f"X_batch Nan: {torch.isnan(x_batch).any()}")
                logger.error(f"X_batch Inf: {torch.isinf(x_batch).any()}")
                sys.exit(1)

            cycle_numbers         = label_batch[:, 1]
            norm_cycle_numbers    = label_batch[:, 2]
            soh_values            = label_batch[:, 3]
            charging_rate         = label_batch[:, 5]
            norm_nominal_capacity = label_batch[:, 6]



            cond_vec = self._build_cond_vec(
                norm_cycle_numbers=norm_cycle_numbers,
                charging_rate=charging_rate,
                norm_nominal_capacity=norm_nominal_capacity,
            )

            gp_input = torch.cat([norm_cycle_numbers.unsqueeze(-1), charging_rate.unsqueeze(-1)], dim=1).to(self.device).contiguous()
            assert gp_input.dim() == 2 and gp_input.size(-1) == 2, f"gp_input must be [B,2], got {tuple(gp_input.shape)}"

            # VAE forward pass and loss
            self.optimizer.zero_grad()
            if self.use_gp:
                self.gp_optimizer.zero_grad()

            reconstruction, mu, logvar, z, soh_pred = self.model(x_batch_noised, token_mask, cond_vec)

            # Check VAE outputs
            if (
                torch.isnan(reconstruction).any()
                or torch.isnan(mu).any()
                or torch.isnan(logvar).any()
            ):
                print(cycle_numbers)
                logger.error("Exiting: VAE output contains nan values")
                logger.error(f"Reconstruction NaN: {torch.isnan(reconstruction).any()}")
                logger.error(f"Mu NaN: {torch.isnan(reconstruction).any()}")
                logger.error(f"Logvar NaN: {torch.isnan(reconstruction).any()}")
                sys.exit(1)        
                
            # Adaptive GP initialization on first batch
            if self.use_gp and not hasattr(self, '_gp_initialized_with_data'):
                self._adaptive_gp_init(gp_input.detach(), z.detach())
                logger.info("🔧 GP model initialized with actual data statistics")


            prior_mean, prior_var, gp_loss_val = 0.0, 0.0, 0.0
            if self.use_gp:
                gp_posterior = self.gp_model(gp_input)  # batched MVN
                
                # Infer B and D safely
                B = int(gp_input.size(0))
                D = int(z.size(1))  # safest ground truth for latent dim
                
                pm = gp_posterior.mean
                pv = gp_posterior.variance
                
                # Normalize to [B, D]
                if pm.ndim != 2:
                    raise RuntimeError(f"Expected 2D GP mean, got {tuple(pm.shape)}")
                if pm.shape == (D, B):            # model returned [D, B]
                    prior_mean = pm.transpose(0, 1).contiguous()
                    prior_var  = pv.transpose(0, 1).contiguous()
                elif pm.shape == (B, D):          # already [B, D]
                    prior_mean = pm.contiguous()
                    prior_var  = pv.contiguous()
                else:
                    raise RuntimeError(
                        f"Unexpected GP mean shape {tuple(pm.shape)}; expected (D,B)=({D},{B}) or (B,D)=({B},{D})"
                )
                
                # Detach before feeding into KL term (we’re using the detached-mean VARIATIONAL prior route)
                prior_mean = prior_mean.detach()
                prior_var  = prior_var.detach()
                
                gp_loss_val = self._compute_gp_loss(gp_input, z, epoch)  # this uses [D,B] internally already

            recon_loss, kl_loss, kl_per_dim, aux = self.loss_fn(
                reconstruction=reconstruction,
                input_tensor=x_batch,
                mu=mu,
                logvar=logvar,
                z=z,
                mask=mask,
                prior_mean=prior_mean,
                prior_var=prior_var,
                training=True,
                q0_Ah=norm_nominal_capacity*5.000,
                norm_cycle_numbers=norm_cycle_numbers,
                charging_rate=charging_rate,
            )

            # Store KL per dimension for this batch
            kl_per_dim_batches.append(kl_per_dim)

            # Early loss validation
            if torch.isnan(recon_loss) or torch.isnan(kl_loss):
                logger.error(f"Detected NaN in losses: Recon={recon_loss:.4f}, KL={kl_loss:.4f}")
                sys.exit(1)

            # SOH loss
            soh_loss = torch.nn.functional.mse_loss(soh_pred, soh_values.unsqueeze(1))
            latent_dependence_loss = aux["latent_dependence_loss"]
            latent_stats["cycle_corr"] += float(aux["cycle_corr"].item())
            latent_stats["crate_corr"] += float(aux["crate_corr"].item())
            latent_stats["cycle_swap_corr"] += float(aux["cycle_swap_corr"].item())
            latent_stats["crate_swap_corr"] += float(aux["crate_swap_corr"].item())


            # Use absolute reconstruction loss for sample weighting to preserve
            # early-cycle amplitude emphasis.
            per_sample = aux["recon_loss_per_sample"]

            # w = self._joint_rate_cycle_weights(
                # charging_rate=charging_rate,
                # norm_cycle_values=norm_cycle_numbers,
                # epoch=epoch,
            # )  # joint (rate, cycle-bin) weight
            
            # w_cycle = self._per_cycle_weights(
            #     norm_cycle_values=norm_cycle_numbers,
            #     epoch=epoch,
            # )  # additional continuous per-cycle weight
            # w = w_cycle
            # # w = torch.clamp(w, min=0.20, max=5.0)
            # w = w / w.mean().clamp_min(1e-8)                     # THIS is the weight normalization point
            # recon_loss = (per_sample * w.detach()).mean()

            total_loss = (
                recon_loss
                + latent_dependence_loss
                + kl_weight * kl_loss
                + gp_scale * gp_loss_val
                + soh_scale * soh_loss
            )

            # Update VAE
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # gradient cliping
            
            if self.use_gp:
                torch.nn.utils.clip_grad_norm_(self.gp_model.parameters(), max_norm=0.5)  # GP (usually smaller)
                torch.nn.utils.clip_grad_norm_(self.likelihood.parameters(), max_norm=0.5)  # GP likelihood

            self.optimizer.step()  # Update VAE parameters
            if self.use_gp:
                self.gp_optimizer.step()  # Update GP parameters

            batch_losses = {
                "total": total_loss.item(),
                "reconstruction": recon_loss.item(),
                "kl": kl_loss.item(),
                "gp_loss": gp_loss_val.item() if self.use_gp else 0.0,
                "soh_loss": soh_loss.item(),
                "latent_dependence": latent_dependence_loss.item(),
            }

            self._update_epoch_losses(epoch_losses, batch_losses)

            progress_bar.set_postfix(
                total_loss=f"{batch_losses['total']:.5f}",
                recon_loss=f"{batch_losses['reconstruction']:.5f}",
                kl_loss=f"{batch_losses['kl']:.5f}",
                gp_loss=f"{batch_losses['gp_loss']:.5f}",
                soh_loss=f"{batch_losses['soh_loss']:.5f}",
                latent_dep=f"{batch_losses['latent_dependence']:.5f}",
            )

            # print("z.requires_grad:", z.requires_grad)
            # for name, param in self.gp_model.named_parameters():
            #     if param.requires_grad:
            #         print(f"{name}: grad norm = {param.grad.norm() if param.grad is not None else 'None'}")


            # Save the reconstructed data after each epoch
            if self.hyper_params.save_every_reconstructions:
                save_reconstructed_data(
                    reconstructed_data=reconstruction,
                    cycle_numbers=cycle_numbers,
                    batch_idx=batch_idx,
                    save_dir=self.config.PATHS.predicted_data,
                    epoch=epoch,
                )
                logger.info(f"📋 Reconstructed data saved at epoch {epoch + 1}")


        # Calculate mean KL per dimension across all batches
        if kl_per_dim_batches:
            kl_per_dim_mean = torch.mean(torch.stack(kl_per_dim_batches), dim=0)
        else:
            kl_per_dim_mean = torch.zeros(1)  # Fallback if no valid batches

        # Calculate epoch averages
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        for key in latent_stats:
            latent_stats[key] /= num_batches

        self.metrics.append(
            {
                "epoch": epoch + 1,
                "train_loss": epoch_losses["total"],
                "reconstruction_loss": epoch_losses["reconstruction"],
                "train_reconstruction_loss": epoch_losses["reconstruction"],
                "kl_loss": epoch_losses["kl"],
                "train_kl_loss": epoch_losses["kl"],
                "gp_loss": epoch_losses["gp_loss"],
                "train_gp_loss": epoch_losses["gp_loss"],
                "soh_loss": epoch_losses["soh_loss"],
                "train_soh_loss": epoch_losses["soh_loss"],
                "latent_dependence_loss": epoch_losses["latent_dependence"],
                "latent_cycle_corr": latent_stats["cycle_corr"],
                "latent_crate_corr": latent_stats["crate_corr"],
                "latent_cycle_swap_corr": latent_stats["cycle_swap_corr"],
                "latent_crate_swap_corr": latent_stats["crate_swap_corr"],
                "kl_per_dimension": kl_per_dim_mean.tolist(),  # Convert tensor to list for JSON serialization
            }
        )

        self._log_epoch_loss_summary(
            stage="Train",
            epoch=epoch,
            losses={
                "total": epoch_losses["total"],
                "reconstruction": epoch_losses["reconstruction"],
                "kl_loss": epoch_losses["kl"],
                "gp_loss": epoch_losses["gp_loss"],
                "soh_loss": epoch_losses["soh_loss"],
                "latent_dependence": epoch_losses["latent_dependence"],
                **latent_stats,
            },
            kl_weight=kl_weight,
            gp_weight=gp_scale,
            soh_weight=soh_scale,
            latent_stats=latent_stats,
        )

        return epoch_losses["total"]

    def _compute_gp_loss(self, gp_input: torch.Tensor, z: torch.Tensor, epoch: int) -> torch.Tensor:
        
        # if epoch > 40:
        #     z_target = z.detach() # [B, D]
        # else:
        #     z_target = z
        z_target = z # [B, D]
        
        gp_input_detach = gp_input.detach()    # [B, 2]
    
        try:
            with gpytorch.settings.cholesky_jitter(1e-5), \
                 gpytorch.settings.max_cholesky_size(2000), \
                 gpytorch.settings.cg_tolerance(1e-4), \
                 gpytorch.settings.max_cg_iterations(2000):
    
                output = self.gp_model(gp_input_detach)   # batched MVN
                pm = output.mean                          # either [D, B] or [B, D]
                pv = output.variance
    
                if pm.ndim != 2:
                    raise RuntimeError(f"Expected 2D GP mean, got {tuple(pm.shape)}")
    
                B = int(gp_input_detach.size(0))
                D = int(z_target.size(1))
    
                # Make target match pm’s layout
                if pm.shape == (D, B):       # GP returned [D, B]
                    target = z_target.transpose(0, 1).contiguous()   # [D, B]
                elif pm.shape == (B, D):     # GP returned [B, D]
                    target = z_target.contiguous()                   # [B, D]
                else:
                    raise RuntimeError(
                        f"Unexpected GP mean shape {tuple(pm.shape)}; "
                        f"z is {tuple(z_target.shape)}, expected (D,B)=({D},{B}) or (B,D)=({B},{D})"
                    )
    
                elbo = self.mll(output, target)
                gp_loss = -elbo.mean()
                return gp_loss
    
        except Exception as e:
            logger.error(f"Error computing GP loss: {str(e)}", exc_info=True)
            sys.exit(1)

    def _validate_epoch(self, val_loader: DataLoader, epoch: int) -> None:
        """Run one validation epoch"""
        self.model.eval()
        if self.use_gp:
            self.gp_model.eval()
            self.likelihood.eval()

        gp_scale_factor = float(self.hyper_params.gp_schedule.scale_factor)
        soh_factor = float(self.hyper_params.soh_factor)
        kl_factor = float(self.hyper_params.kl_annealing.target_factor)

        val_losses = {
            "total": 0.0,
            "vae": 0.0,
            "gp_loss": 0.0,
            "kl_loss": 0.0,
            "reconstruction": 0.0,
            "soh_loss": 0.0,
            "latent_dependence": 0.0,
            # "flatness": 0.0,
        }
        val_latent_stats = {
            "cycle_corr": 0.0,
            "crate_corr": 0.0,
            "cycle_swap_corr": 0.0,
            "crate_swap_corr": 0.0,
        }
        num_batches = len(val_loader)

        with torch.no_grad():
            progress_bar = tqdm(
                enumerate(val_loader),
                total=num_batches,
                desc="⛳️ Validation",
                leave=True,
                colour="green",
                disable=not self.show_progress,
            )

            for batch_idx, (x_batch, mask_batch, token_mask_batch, label_batch) in progress_bar:
                x_batch = x_batch.to(self.device)
                label_batch = label_batch.to(self.device)
                mask = mask_batch.to(self.device)
                token_mask = token_mask_batch.to(self.device)

                x_batch_noised = x_batch  # + noise * mask
                
                norm_cycle_numbers    = label_batch[:, 2]
                soh_values            = label_batch[:, 3]
                charging_rate         = label_batch[:, 5]
                norm_nominal_capacity = label_batch[:, 6]

                cond_vec = self._build_cond_vec(
                    norm_cycle_numbers=norm_cycle_numbers,
                    charging_rate=charging_rate,
                    norm_nominal_capacity=norm_nominal_capacity,
                )


                gp_input = torch.cat([norm_cycle_numbers.unsqueeze(-1), charging_rate.unsqueeze(-1)], dim=1)
                assert gp_input.dim() == 2 and gp_input.size(-1) == 2, f"gp_input must be [B,2], got {tuple(gp_input.shape)}"

                # VAE forward pass
                reconstruction, mu, logvar, z, soh_pred = self.model(x_batch_noised, token_mask, cond_vec)


                prior_mean, prior_var = 0.0, 0.0
                gp_loss_val = 0.0
                if self.use_gp:

                    gp_posterior = self.gp_model(gp_input)  # batched MVN
    
                    # Infer B and D safely
                    B = int(gp_input.size(0))
                    D = int(z.size(1))  # safest ground truth for latent dim
    
                    pm = gp_posterior.mean
                    pv = gp_posterior.variance
    
                    # Normalize to [B, D]
                    if pm.ndim != 2:
                        raise RuntimeError(f"Expected 2D GP mean, got {tuple(pm.shape)}")
                    if pm.shape == (D, B):            # model returned [D, B]
                        prior_mean = pm.transpose(0, 1).contiguous()
                        prior_var  = pv.transpose(0, 1).contiguous()
                    elif pm.shape == (B, D):          # already [B, D]
                        prior_mean = pm.contiguous()
                        prior_var  = pv.contiguous()
                    else:
                        raise RuntimeError(
                            f"Unexpected GP mean shape {tuple(pm.shape)}; expected (D,B)=({D},{B}) or (B,D)=({B},{D})"
                    )
    
                    # Detach before feeding into KL term (we’re using the detached-mean VARIATIONAL prior route)
                    prior_mean = prior_mean.detach()
                    prior_var  = prior_var.detach()
                    gp_loss_val = self._evaluate_gp(gp_input, z)

                recon_loss, kl_loss, kl_per_dim, aux = self.loss_fn(
                    reconstruction=reconstruction,
                    input_tensor=x_batch,
                    mu=mu,
                    logvar=logvar,
                    z=z,
                    mask=mask,
                    prior_mean=prior_mean,
                    prior_var=prior_var,
                    training=False,
                    q0_Ah=norm_nominal_capacity*5.000,
                    norm_cycle_numbers=norm_cycle_numbers,
                    charging_rate=charging_rate,
                )


                # per_sample = aux.get("recon_loss_per_sample_relative", aux["recon_loss_per_sample"])
                # recon_loss = per_sample.mean() 

                # flatness_penalty, slope_value = self._cycle_flatness_loss(
                #     per_sample_loss=aux["recon_loss_per_sample"],
                #     norm_cycle_values=norm_cycle_numbers,
                #     epoch=epoch,
                # )
                # flatness_loss = self.cycle_flatness_weight * flatness_penalty

                per_sample = aux["recon_loss_per_sample"]

                # w = self._joint_rate_cycle_weights(
                    # charging_rate=charging_rate,
                    # norm_cycle_values=norm_cycle_numbers,
                    # epoch=epoch,
                # )  # joint (rate, cycle-bin) weight

                # w_cycle = self._per_cycle_weights(
                #     norm_cycle_values=norm_cycle_numbers,
                #     epoch=epoch,
                # )  # additional continuous per-cycle weight
                # w = w_cycle
                # # w = torch.clamp(w, min=0.20, max=5.0)
                # w = w / w.mean().clamp_min(1e-8)                     # THIS is the weight normalization point
                # recon_loss = (per_sample * w.detach()).mean()

                soh_loss = torch.nn.functional.mse_loss(soh_pred, soh_values.unsqueeze(1))
                latent_dependence_loss = aux["latent_dependence_loss"]
                val_latent_stats["cycle_corr"] += float(aux["cycle_corr"].item())
                val_latent_stats["crate_corr"] += float(aux["crate_corr"].item())
                val_latent_stats["cycle_swap_corr"] += float(aux["cycle_swap_corr"].item())
                val_latent_stats["crate_swap_corr"] += float(aux["crate_swap_corr"].item())


                total_loss = (
                    recon_loss
                    + latent_dependence_loss
                    + kl_factor * kl_loss
                    + gp_scale_factor * gp_loss_val
                    + soh_factor * soh_loss
                )

                val_losses["gp_loss"] += gp_loss_val.item() if self.use_gp else 0.0
                val_losses["kl_loss"] += kl_loss.item()
                val_losses["total"] += total_loss.item()
                val_losses["reconstruction"] += recon_loss.item()
                val_losses["soh_loss"] += soh_loss.item()
                val_losses["latent_dependence"] += latent_dependence_loss.item()

                # Update progress bar
                progress_bar.set_postfix(
                    total_loss=f"{total_loss.item():.5f}",
                    recon_loss=f"{recon_loss.item():.5f}",
                    kl_loss=f"{kl_loss.item():.5f}",
                    gp_loss=f"{gp_loss_val:.5f}",
                    soh_loss=f"{soh_loss.item():.5f}",
                    latent_dep=f"{latent_dependence_loss.item():.5f}",
                )

        # Calculate final validation losses
        for key in val_losses:
            val_losses[key] /= num_batches
        for key in val_latent_stats:
            val_latent_stats[key] /= num_batches

        # Update metrics
        self.metrics[-1].update(
            {
                "val_loss": val_losses["total"],
                "val_gp_loss": val_losses["gp_loss"],
                "val_kl_loss": val_losses["kl_loss"],
                "val_reconstruction_loss": val_losses["reconstruction"],
                "val_soh_loss": val_losses["soh_loss"],
                "val_latent_dependence_loss": val_losses["latent_dependence"],
                "val_latent_cycle_corr": val_latent_stats["cycle_corr"],
                "val_latent_crate_corr": val_latent_stats["crate_corr"],
                "val_latent_cycle_swap_corr": val_latent_stats["cycle_swap_corr"],
                "val_latent_crate_swap_corr": val_latent_stats["crate_swap_corr"],
            }
        )

        self._log_epoch_loss_summary(
            stage="Validation",
            epoch=epoch,
            losses=val_losses,
            kl_weight=kl_factor,
            gp_weight=gp_scale_factor,
            soh_weight=soh_factor,
            latent_stats=val_latent_stats,
        )

        return val_losses['total']

    def _evaluate_gp(self, gp_input: torch.Tensor, z: torch.Tensor) -> float:
        """Evaluate GP model without updating parameters"""

        gp_input_detach = gp_input.detach()
        z_detach = z.detach()

        correlation = torch.corrcoef(torch.cat([gp_input, z], dim=1))
        logger.debug(f"GP input-latent correlation: {correlation}")

        # Check for numerical issues
        if torch.isnan(gp_input_detach).any() or torch.isinf(gp_input_detach).any():
            logger.critical("GP validation input contains NaN or Inf values")
            return 0.0

        if torch.isnan(z_detach).any() or torch.isinf(z_detach).any():
            logger.critical("GP validation target contains NaN or Inf values")
            return 0.0

        try:
            with gpytorch.settings.fast_pred_var(), torch.no_grad():

                output = self.gp_model(gp_input_detach)
                pm = output.mean                          # either [D, B] or [B, D]
                pv = output.variance

                if pm.ndim != 2:
                    raise RuntimeError(f"Expected 2D GP mean, got {tuple(pm.shape)}")

                B = int(gp_input_detach.size(0))
                D = int(z_detach.size(1))

                # Make target match pm’s layout
                if pm.shape == (D, B):       # GP returned [D, B]
                    target = z_detach.transpose(0, 1).contiguous()   # [D, B]
                elif pm.shape == (B, D):     # GP returned [B, D]
                    target = z_detach.contiguous()                   # [B, D]
                else:
                    raise RuntimeError(
                        f"Unexpected GP mean shape {tuple(pm.shape)}; "
                        f"z is {tuple(z_target.shape)}, expected (D,B)=({D},{B}) or (B,D)=({B},{D})"
                    )

                elbo = self.mll(output, target)
                gp_loss = -elbo.mean()
                return gp_loss

        except Exception as e:
            logger.error(f"Error in GP evaluation: {str(e)}", exc_info=True)
            sys.exit(1)


    def _update_epoch_losses(
        self, epoch_losses: Dict[str, float], batch_losses: Dict[str, float]
    ) -> None:
        """Update running loss totals for the epoch"""
        for key in epoch_losses:
            epoch_losses[key] += batch_losses[key]

    def _log_epoch_loss_summary(
        self,
        stage: str,
        epoch: int,
        losses: Dict[str, float],
        kl_weight: Optional[float] = None,
        gp_weight: Optional[float] = None,
        soh_weight: Optional[float] = None,
        latent_stats: Optional[Dict[str, float]] = None,
    ) -> None:
        """Emit an explicit epoch-level loss summary to the logger."""
        summary = (
            f"[{stage}][Epoch {epoch + 1}] "
            f"total={losses['total']:.6f} "
            f"recon={losses['reconstruction']:.6f} "
            f"kl={losses['kl_loss']:.6f} "
            f"gp={losses['gp_loss']:.6f} "
            f"soh={losses['soh_loss']:.6f} "
            f"latent_dep={losses.get('latent_dependence', 0.0):.6f}"
        )
        if kl_weight is not None or gp_weight is not None or soh_weight is not None:
            weight_parts = []
            if kl_weight is not None:
                weight_parts.append(f"kl_weight={kl_weight:.6f}")
            if gp_weight is not None:
                weight_parts.append(f"gp_weight={gp_weight:.6f}")
            if soh_weight is not None:
                weight_parts.append(f"soh_weight={soh_weight:.6f}")
            summary = f"{summary} ({', '.join(weight_parts)})"
        if latent_stats is not None:
            summary = (
                f"{summary} "
                f"cycle_corr={latent_stats.get('cycle_corr', 0.0):.4f} "
                f"crate_corr={latent_stats.get('crate_corr', 0.0):.4f} "
                f"cycle_swap_corr={latent_stats.get('cycle_swap_corr', 0.0):.4f} "
                f"crate_swap_corr={latent_stats.get('crate_swap_corr', 0.0):.4f}"
            )
        logger.info(summary)

    def save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint with all necessary components"""
        path = f"{self.model_save_path}/{filename}"

        checkpoint = {
            "vae_state_dict": self.model.state_dict(),
            "vae_optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "train_norm_stats": self.train_norm_stats,
            "metrics": self.metrics,
            "epoch": len(self.metrics),
            "best_train_loss": self.best_train_loss,
            "early_stopping_counter": self.early_stopping_counter,
            "config": self.config.model_dump(),
            "metadata": {
                "experiment_name": self.config.GENERAL.experiment_name,
                "latent_dim": self.config.HYPER_PARAMETERS.latent_dim,
                "use_gp": self.use_gp,
                "checkpoint_filename": filename,
            },
        }

        # Add GP components if available
        if self.use_gp:
            checkpoint.update(
                {
                    "gp_state_dict": self.gp_model.state_dict(),
                    "likelihood_state_dict": self.likelihood.state_dict(),
                    "gp_optimizer": self.gp_optimizer.state_dict(),
                }
            )

        try:
            torch.save(checkpoint, path)
            logger.info(f"💾 Model checkpoint saved with filename {filename}")
        except Exception as e:
            logger.error(f"Error saving checkpoint: {str(e)}", exc_info=True)

    def load_checkpoint(self, filename: str = "best_model.pth") -> int:
        """Load model checkpoint and restore training state"""
        path = Path(self.model_save_path) / filename
        if not path.exists():
            logger.warning(f"Checkpoint {filename} not found at {self.model_save_path}")
            return 0

        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["vae_state_dict"])
        self.optimizer.load_state_dict(checkpoint["vae_optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.metrics = checkpoint.get("metrics", [])
        self.best_train_loss = checkpoint.get("best_train_loss", float("inf"))
        self.early_stopping_counter = checkpoint.get("early_stopping_counter", 0)

        # GP state (if used)
        if self.use_gp:
            self.gp_model.load_state_dict(checkpoint["gp_state_dict"])
            self.likelihood.load_state_dict(checkpoint["likelihood_state_dict"])
            self.gp_optimizer.load_state_dict(checkpoint["gp_optimizer"])

        start_epoch = checkpoint.get("epoch", 0)
        logger.info(f"🔄 Resumed from checkpoint at epoch {start_epoch}")
        return start_epoch
   

def compute_elbo_num_data(loader) -> int:
    # Prefer sampler length when it exists (handles replacement/Subset)
    if hasattr(loader, "sampler") and hasattr(loader.sampler, "__len__"):
        try:
            return len(loader.sampler)
        except Exception:
            pass
    return len(loader.dataset)
