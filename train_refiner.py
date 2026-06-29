#!/usr/bin/env python3
"""CLI entrypoint for training the Stage 2 landmark refinement model.

Usage
-----
    python train_refiner.py --stage1-checkpoint checkpoints/.../best_model.pth --epochs 100

Run ``python train_refiner.py --help`` for the full list of options.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mlflow
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.data.crop_dataset import CropRefinementDataset, generate_stage1_predictions
from src.data.dataset import ALL_LANDMARK_SYMBOLS, NUM_LANDMARKS, load_pixel_spacings
from src.models.refiner import create_refiner
from src.models.unet import create_model
from src.training.mlflow_utils import setup_experiment


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train the Stage 2 landmark refinement model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--stage1-checkpoint",
        type=str,
        required=True,
        help="Path to a trained Stage 1 model checkpoint.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Training batch size.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate.",
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        default=96,
        help="Crop size around each coarse landmark prediction.",
    )
    parser.add_argument(
        "--jitter",
        type=int,
        default=20,
        help="Random offset (px) added to crop centre during training.",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=2.0,
        help="Gaussian sigma for Stage 2 heatmap targets.",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=512,
        help="Image size used by Stage 1 model.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Root data directory.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="",
        help="Checkpoint directory (auto-generated if empty).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader workers.",
    )
    return parser.parse_args(argv)


def _get_device() -> torch.device:
    """Detect the best available compute device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main(argv: list[str] | None = None) -> None:
    """Train the Stage 2 refiner model."""
    args = parse_args(argv)
    device = _get_device()

    # Checkpoint directory
    if not args.checkpoint_dir:
        args.checkpoint_dir = (
            f"checkpoints/refiner-ep{args.epochs}-crop{args.crop_size}"
        )
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Checkpoint dir: {ckpt_dir}")

    # ------------------------------------------------------------------
    # Load Stage 1 model
    # ------------------------------------------------------------------
    print(f"\nLoading Stage 1 model from: {args.stage1_checkpoint}")
    stage1_model = create_model(
        encoder_name="resnet34",
        encoder_weights=None,  # we're loading from checkpoint
        in_channels=1,
        num_classes=NUM_LANDMARKS,
    )
    state_dict = torch.load(args.stage1_checkpoint, map_location="cpu")
    stage1_model.load_state_dict(state_dict)
    stage1_model.to(device)
    stage1_model.eval()

    # ------------------------------------------------------------------
    # Generate Stage 1 predictions
    # ------------------------------------------------------------------
    data_dir = Path(args.data_dir)

    print("\nGenerating Stage 1 predictions on train set...")
    train_preds = generate_stage1_predictions(
        model=stage1_model,
        data_root=data_dir / "train",
        image_size=args.img_size,
        device=device,
    )
    print(f"  {len(train_preds)} images processed.")

    print("Generating Stage 1 predictions on valid set...")
    val_preds = generate_stage1_predictions(
        model=stage1_model,
        data_root=data_dir / "valid",
        image_size=args.img_size,
        device=device,
    )
    print(f"  {len(val_preds)} images processed.")

    # Free Stage 1 model from GPU
    del stage1_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Build datasets and dataloaders
    # ------------------------------------------------------------------
    train_dataset = CropRefinementDataset(
        data_root=data_dir / "train",
        stage1_predictions=train_preds,
        image_size=args.img_size,
        crop_size=args.crop_size,
        sigma=args.sigma,
        jitter=args.jitter,
    )
    val_dataset = CropRefinementDataset(
        data_root=data_dir / "valid",
        stage1_predictions=val_preds,
        image_size=args.img_size,
        crop_size=args.crop_size,
        sigma=args.sigma,
        jitter=0,  # no jitter for validation
    )

    print(f"\nTrain samples: {len(train_dataset)}")
    print(f"Val samples:   {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # ------------------------------------------------------------------
    # Create refiner model
    # ------------------------------------------------------------------
    model = create_refiner(in_channels=1, base_channels=32)
    model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nRefiner model: {num_params:,} parameters")

    # ------------------------------------------------------------------
    # Optimizer, scheduler, loss
    # ------------------------------------------------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = nn.MSELoss()

    # ------------------------------------------------------------------
    # MLflow setup
    # ------------------------------------------------------------------
    setup_experiment("cephalometric-refiner")

    run_name = f"refiner-ep{args.epochs}-crop{args.crop_size}"
    run_tags = {
        "stage": "2",
        "stage1_checkpoint": str(args.stage1_checkpoint),
    }

    best_val_loss = float("inf")
    best_epoch = 0

    with mlflow.start_run(run_name=run_name, tags=run_tags):
        mlflow.log_params({
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "crop_size": args.crop_size,
            "jitter": args.jitter,
            "sigma": args.sigma,
            "img_size": args.img_size,
            "num_params": num_params,
        })

        # ------------------------------------------------------------------
        # Training loop
        # ------------------------------------------------------------------
        for epoch in range(1, args.epochs + 1):
            epoch_start = time.time()

            # --- Train ---
            model.train()
            train_loss_sum = 0.0
            train_valid_count = 0

            for batch in train_loader:
                crops = batch["crop"].to(device)
                targets = batch["heatmap"].to(device)
                valid = batch["valid"]

                preds = model(crops)

                # Only compute loss on valid samples (GT inside crop)
                if valid.any():
                    valid_mask = valid.bool()
                    loss = criterion(preds[valid_mask], targets[valid_mask])

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    train_loss_sum += loss.item() * valid_mask.sum().item()
                    train_valid_count += valid_mask.sum().item()

            train_loss = (
                train_loss_sum / max(train_valid_count, 1)
            )

            # --- Validate ---
            model.eval()
            val_loss_sum = 0.0
            val_valid_count = 0

            with torch.no_grad():
                for batch in val_loader:
                    crops = batch["crop"].to(device)
                    targets = batch["heatmap"].to(device)
                    valid = batch["valid"]

                    preds = model(crops)

                    if valid.any():
                        valid_mask = valid.bool()
                        loss = criterion(preds[valid_mask], targets[valid_mask])

                        val_loss_sum += loss.item() * valid_mask.sum().item()
                        val_valid_count += valid_mask.sum().item()

            val_loss = val_loss_sum / max(val_valid_count, 1)

            scheduler.step()

            epoch_time = time.time() - epoch_start

            # --- Logging ---
            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "epoch_time_seconds": epoch_time,
                },
                step=epoch,
            )

            # --- Best model ---
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                torch.save(model.state_dict(), ckpt_dir / "best_refiner.pth")

            # --- Print progress ---
            if epoch % 5 == 0 or epoch == 1:
                print(
                    f"Epoch {epoch:3d}/{args.epochs} | "
                    f"Train: {train_loss:.6f} | "
                    f"Val: {val_loss:.6f} | "
                    f"Best: {best_val_loss:.6f} (ep {best_epoch}) | "
                    f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
                    f"{epoch_time:.1f}s"
                )

        # Save final model
        torch.save(model.state_dict(), ckpt_dir / "final_refiner.pth")
        mlflow.log_metric("best_val_loss", best_val_loss)
        mlflow.log_metric("best_epoch", best_epoch)

    print(f"\nTraining complete.")
    print(f"Best val loss: {best_val_loss:.6f} at epoch {best_epoch}")
    print(f"Checkpoints saved to: {ckpt_dir}")


if __name__ == "__main__":
    main()
