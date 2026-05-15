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
from typing import List, Optional, Tuple, Dict, Any

logger = get_logger(__name__)


class Trainer:
    """
    Trainer class for VAE.
    """

    def __init__(
        self,
        model: nn.Module,
        config: FullConfig,
        device: torch.device,
        train_norm_stats: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the trainer with configuration"""
        self.config = config
        self.device = device
        self.model = model.to(self.device)
        self.hyper_params = self.config.HYPER_PARAMETERS
        self.show_progress = self.config.GENERAL.show_progress
        self.metrics: dict[str, float] = []

        norm_cfg = config.NORMALIZATION.model_dump()

        self.train_norm_stats = train_norm_stats or {}
        self.loss_fn = LossFactory(
            norm_cfg=norm_cfg,
            norm_stats=self.train_norm_stats,
            enable_physics_head=config.HYPER_PARAMETERS.physics_head.enabled,
        )

        # Initialize basic components
        self.best_train_loss = float("inf")
        self.model_save_path = config.PATHS.model_save
        self.metrics_save_path = config.PATHS.metrics


        self._initialize_optimizers()

        # Early stopping setup
        self.early_stopping_counter = 0
        self.best_val_loss = float("inf")
        self.early_stopping_patience = self.hyper_params.early_stopping.patience
        self.min_delta = self.hyper_params.early_stopping.min_delta


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

        num_batches = len(train_loader)
        kl_weight = self._get_kl_weight(epoch)


        soh_scale = float(self.hyper_params.soh_factor)
        soh_scale = soh_scale * min(1.0, (epoch+1) / 10.0)

        epoch_losses = {
            "total": 0.0,
            "reconstruction": 0.0,
            "kl": 0.0,
            "soh_loss": 0.0,
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


            # VAE forward pass and loss
            self.optimizer.zero_grad()

            reconstruction, mu, logvar, z, soh_pred = self.model(x_batch, token_mask)

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
                


            recon_loss, kl_loss, kl_per_dim, aux = self.loss_fn(
                reconstruction=reconstruction,
                input_tensor=x_batch,
                mu=mu,
                logvar=logvar,
                z=z,
                mask=mask,
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


            # Use absolute reconstruction loss for sample weighting to preserve
            # early-cycle amplitude emphasis.
            per_sample = aux["recon_loss_per_sample"]


            total_loss = (
                recon_loss
                + kl_weight * kl_loss
                + soh_scale * soh_loss
            )

            # Update VAE
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # gradient cliping
            

            self.optimizer.step()  # Update VAE parameters

            batch_losses = {
                "total": total_loss.item(),
                "reconstruction": recon_loss.item(),
                "kl": kl_loss.item(),
                "soh_loss": soh_loss.item(),

            }

            self._update_epoch_losses(epoch_losses, batch_losses)

            progress_bar.set_postfix(
                total_loss=f"{batch_losses['total']:.5f}",
                recon_loss=f"{batch_losses['reconstruction']:.5f}",
                kl_loss=f"{batch_losses['kl']:.5f}",
                soh_loss=f"{batch_losses['soh_loss']:.5f}",

            )




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

        self.metrics.append(
            {
                "epoch": epoch + 1,
                "train_loss": epoch_losses["total"],
                "reconstruction_loss": epoch_losses["reconstruction"],
                "train_reconstruction_loss": epoch_losses["reconstruction"],
                "kl_loss": epoch_losses["kl"],
                "train_kl_loss": epoch_losses["kl"],
                "soh_loss": epoch_losses["soh_loss"],
                "train_soh_loss": epoch_losses["soh_loss"],

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
                "soh_loss": epoch_losses["soh_loss"],
            },
            kl_weight=kl_weight,
            soh_weight=soh_scale,
        )

        return epoch_losses["total"]


    def _validate_epoch(self, val_loader: DataLoader, epoch: int) -> None:
        """Run one validation epoch"""
        self.model.eval()

        soh_factor = float(self.hyper_params.soh_factor)
        kl_factor = float(self.hyper_params.kl_annealing.target_factor)

        val_losses = {
            "total": 0.0,
            "vae": 0.0,
            "kl_loss": 0.0,
            "reconstruction": 0.0,
            "soh_loss": 0.0,
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

                
                norm_cycle_numbers    = label_batch[:, 2]
                soh_values            = label_batch[:, 3]
                charging_rate         = label_batch[:, 5]
                norm_nominal_capacity = label_batch[:, 6]


                # VAE forward pass
                reconstruction, mu, logvar, z, soh_pred = self.model(x_batch, token_mask)



                recon_loss, kl_loss, kl_per_dim, aux = self.loss_fn(
                    reconstruction=reconstruction,
                    input_tensor=x_batch,
                    mu=mu,
                    logvar=logvar,
                    z=z,
                    mask=mask,
                    training=False,
                    q0_Ah=norm_nominal_capacity*5.000,
                    norm_cycle_numbers=norm_cycle_numbers,
                    charging_rate=charging_rate,
                )


                per_sample = aux["recon_loss_per_sample"]


                soh_loss = torch.nn.functional.mse_loss(soh_pred, soh_values.unsqueeze(1))


                total_loss = (
                    recon_loss
                    + kl_factor * kl_loss
                    + soh_factor * soh_loss
                )


                val_losses["kl_loss"] += kl_loss.item()
                val_losses["total"] += total_loss.item()
                val_losses["reconstruction"] += recon_loss.item()
                val_losses["soh_loss"] += soh_loss.item()


                # Update progress bar
                progress_bar.set_postfix(
                    total_loss=f"{total_loss.item():.5f}",
                    recon_loss=f"{recon_loss.item():.5f}",
                    kl_loss=f"{kl_loss.item():.5f}",
                    soh_loss=f"{soh_loss.item():.5f}",
                )

        # Calculate final validation losses
        for key in val_losses:
            val_losses[key] /= num_batches

        # Update metrics
        self.metrics[-1].update(
            {
                "val_loss": val_losses["total"],
                "val_kl_loss": val_losses["kl_loss"],
                "val_reconstruction_loss": val_losses["reconstruction"],
                "val_soh_loss": val_losses["soh_loss"],
            }
        )

        self._log_epoch_loss_summary(
            stage="Validation",
            epoch=epoch,
            losses=val_losses,
            kl_weight=kl_factor,
            soh_weight=soh_factor,
        )

        return val_losses['total']



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
        soh_weight: Optional[float] = None,
    ) -> None:
        """Emit an explicit epoch-level loss summary to the logger."""
        summary = (
            f"[{stage}][Epoch {epoch + 1}] "
            f"total={losses['total']:.6f} "
            f"recon={losses['reconstruction']:.6f} "
            f"kl={losses['kl_loss']:.6f} "
            f"soh={losses['soh_loss']:.6f} "

        )
        if kl_weight is not None or soh_weight is not None:
            weight_parts = []
            if kl_weight is not None:
                weight_parts.append(f"kl_weight={kl_weight:.6f}")
            if soh_weight is not None:
                weight_parts.append(f"soh_weight={soh_weight:.6f}")
            summary = f"{summary} ({', '.join(weight_parts)})"

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
                "checkpoint_filename": filename,
            },
        }

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

        start_epoch = checkpoint.get("epoch", 0)
        logger.info(f"🔄 Resumed from checkpoint at epoch {start_epoch}")
        return start_epoch
   
