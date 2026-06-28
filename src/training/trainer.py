"""
Training loop for cephalometric landmark heatmap regression.

Handles:
- AdaptiveWingLoss between predicted and ground-truth heatmaps
- Adam optimiser with differential LR (encoder / decoder)
- CosineAnnealingWarmRestarts scheduler
- Best-model checkpointing
- Device auto-detection: MPS → CUDA → CPU
- Mixed-precision training (AMP)
- Per-channel validation loss logging for landmark collapse detection
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader

from src.models.unet import create_model, get_parameter_groups
from src.training.mlflow_utils import (
    log_epoch_extended,
    log_epoch_metrics,
    log_hyperparams,
    log_model_artifact,
    log_per_channel_val_loss,
)


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """Auto-detect the best available device.

    Priority: MPS (Apple Silicon) → CUDA → CPU.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """End-to-end trainer for heatmap-based landmark detection.

    Parameters
    ----------
    encoder_name : str
        Backbone encoder (default ``"resnet34"``).
    encoder_weights : str | None
        Pre-trained weights (default ``"imagenet"``).
    in_channels : int
        Image channels (1 for grayscale).
    num_classes : int
        Number of landmark channels.
    encoder_lr : float
        Learning rate for the encoder.
    decoder_lr : float
        Learning rate for the decoder / head.
    epochs : int
        Maximum training epochs.
    checkpoint_dir : Path | str
        Where to save checkpoints.
    device : torch.device | None
        Override device; auto-detected if ``None``.
    landmark_symbols : list[str] | None
        Landmark symbol names for per-channel loss logging.
    """

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: str | None = "imagenet",
        in_channels: int = 1,
        num_classes: int = 6,
        encoder_lr: float = 1e-4,
        decoder_lr: float = 1e-3,
        epochs: int = 100,
        checkpoint_dir: Path | str = "checkpoints",
        device: Optional[torch.device] = None,
        landmark_symbols: Optional[list[str]] = None,
    ) -> None:
        self.epochs = epochs
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.device = device or get_device()
        self.landmark_symbols = landmark_symbols

        # Mixed-precision setup
        # Enabled on CUDA (with GradScaler protection) but NOT on MPS
        # because AWL's pow/log operations can lose precision in FP16
        # and MPS doesn't support GradScaler to prevent gradient underflow.
        self._amp_dtype: torch.dtype | None = None
        if self.device.type == "cuda":
            self._amp_dtype = torch.float16
        self.scaler = torch.amp.GradScaler(enabled=self._amp_dtype is not None)

        # Model
        self.model = create_model(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            num_classes=num_classes,
        ).to(self.device)

        # Optimiser with differential LR
        param_groups = get_parameter_groups(
            self.model,
            encoder_lr=encoder_lr,
            decoder_lr=decoder_lr,
        )
        self.optimizer = Adam(param_groups)

        # Scheduler
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=20,
            T_mult=2,
            eta_min=1e-6,
        )

        # Loss
        from src.training.losses import AdaptiveWingLoss
        self.criterion = AdaptiveWingLoss(omega=14, theta=0.5, epsilon=1, alpha=2.1)

        # Tracking
        self.best_val_loss = float("inf")
        self.best_epoch = -1

    # ------------------------------------------------------------------

    def _train_one_epoch(self, loader: DataLoader) -> float:
        """Run one training epoch; return mean loss."""
        self.model.train()
        total_loss = 0.0
        for batch in loader:
            images = batch["image"].to(self.device)
            targets = batch["heatmaps"].to(self.device)

            with torch.autocast(
                device_type=self.device.type,
                dtype=self._amp_dtype,
                enabled=self._amp_dtype is not None,
            ):
                preds = self.model(images)
                loss = self.criterion(preds, targets)

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item() * images.size(0)

        return total_loss / len(loader.dataset)  # type: ignore[arg-type]

    @torch.no_grad()
    def _validate(
        self,
        loader: DataLoader,
        compute_per_channel: bool = False,
    ) -> tuple[float, dict[str, float] | None]:
        """Run validation; return mean loss and optionally per-channel losses.

        Parameters
        ----------
        loader : DataLoader
            Validation data loader.
        compute_per_channel : bool
            If True, compute and return per-channel (per-landmark) mean loss.

        Returns
        -------
        tuple[float, dict[str, float] | None]
            ``(mean_val_loss, per_channel_losses)`` where per_channel_losses
            is ``None`` if ``compute_per_channel`` is False.
        """
        self.model.eval()
        total_loss = 0.0
        num_channels = None
        channel_losses_sum = None
        total_samples = 0

        for batch in loader:
            images = batch["image"].to(self.device)
            targets = batch["heatmaps"].to(self.device)

            with torch.autocast(
                device_type=self.device.type,
                dtype=self._amp_dtype,
                enabled=self._amp_dtype is not None,
            ):
                preds = self.model(images)
                loss = self.criterion(preds, targets)
            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

            # Accumulate per-channel losses
            if compute_per_channel:
                B, C, H, W = preds.shape
                if num_channels is None:
                    num_channels = C
                    channel_losses_sum = torch.zeros(C, device=self.device)
                # Compute MSE per channel as a proxy metric (interpretable,
                # independent of AWL's foreground weighting)
                per_ch = ((preds - targets) ** 2).mean(dim=(2, 3))  # (B, C)
                channel_losses_sum += per_ch.sum(dim=0)  # (C,)

        mean_loss = total_loss / len(loader.dataset)  # type: ignore[arg-type]

        per_channel_dict = None
        if compute_per_channel and channel_losses_sum is not None:
            per_ch_mean = (channel_losses_sum / total_samples).cpu().tolist()
            symbols = self.landmark_symbols or [
                f"ch_{i}" for i in range(num_channels)  # type: ignore[arg-type]
            ]
            per_channel_dict = {
                sym: val for sym, val in zip(symbols, per_ch_mean)
            }

        return mean_loss, per_channel_dict

    # ------------------------------------------------------------------

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        log_to_mlflow: bool = True,
    ) -> Path:
        """Train the model and return the path to the best checkpoint.

        Parameters
        ----------
        train_loader, val_loader : DataLoader
            Training and validation data loaders.
        log_to_mlflow : bool
            Whether to log metrics via the mlflow_utils helpers.

        Returns
        -------
        Path
            Path to the best model checkpoint file.
        """
        best_ckpt_path = self.checkpoint_dir / "best_model.pth"

        print(
            f"Training on {self.device} | "
            f"{len(train_loader.dataset)} train, "  # type: ignore[arg-type]
            f"{len(val_loader.dataset)} val samples"  # type: ignore[arg-type]
        )
        print("-" * 60)

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()
            train_loss = self._train_one_epoch(train_loader)

            # Compute per-channel val loss every 10 epochs and on the last epoch
            compute_channels = (epoch % 10 == 0) or (epoch == self.epochs)
            val_loss, per_channel_losses = self._validate(
                val_loader, compute_per_channel=compute_channels,
            )
            elapsed = time.time() - t0

            self.scheduler.step()

            # Current learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Checkpoint best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_loss": val_loss,
                    },
                    best_ckpt_path,
                )
                marker = " ★ best"
            else:
                marker = ""

            print(
                f"Epoch {epoch:>3d}/{self.epochs} | "
                f"train {train_loss:.6f} | val {val_loss:.6f} | "
                f"lr {current_lr:.2e} | {elapsed:.1f}s{marker}"
            )

            if log_to_mlflow:
                log_epoch_metrics(epoch, train_loss, val_loss)
                log_epoch_extended(
                    epoch,
                    best_val_loss=self.best_val_loss,
                    learning_rate=current_lr,
                    epoch_time_seconds=elapsed,
                )
                if per_channel_losses is not None:
                    log_per_channel_val_loss(epoch, per_channel_losses)

        print("-" * 60)
        print(
            f"Best val loss {self.best_val_loss:.6f} at epoch {self.best_epoch}"
        )

        if log_to_mlflow:
            log_model_artifact(str(best_ckpt_path))

        return best_ckpt_path

    # ------------------------------------------------------------------

    def load_checkpoint(self, path: Path | str) -> None:
        """Load model weights from a checkpoint file.

        Parameters
        ----------
        path : Path | str
            Path to the ``.pth`` checkpoint.
        """
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded checkpoint from {path} (epoch {ckpt.get('epoch', '?')})")
