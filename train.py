#!/usr/bin/env python3
"""
CLI entrypoint for training the cephalometric landmark detection model.

Usage
-----
    python train.py --encoder resnet34 --epochs 100 --batch-size 4

Run ``python train.py --help`` for the full list of options.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mlflow
import torch
from torch.utils.data import DataLoader

from src.data.dataset import (
    CephalometricDataset,
    NUM_LANDMARKS,
    SELECTED_SYMBOLS,
    SELECTED_DISPLAY_NAMES,
    load_pixel_spacings,
)
from src.evaluation import evaluate_model
from src.training.mlflow_utils import (
    log_evaluation_metrics,
    log_hyperparams,
    setup_experiment,
)
from src.training.trainer import Trainer, get_device


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a cephalometric landmark detection model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="resnet34",
        help="Encoder backbone name (segmentation_models_pytorch).",
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
        default=4,
        help="Training batch size.",
    )
    parser.add_argument(
        "--encoder-lr",
        type=float,
        default=1e-4,
        help="Learning rate for the encoder (backbone).",
    )
    parser.add_argument(
        "--decoder-lr",
        type=float,
        default=1e-3,
        help="Learning rate for the decoder / segmentation head.",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=5.0,
        help="Gaussian sigma for heatmap generation.",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=512,
        help="Input image size (square).",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Root dataset directory (contains train/valid/test).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory to save model checkpoints.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker count (0 = main process).",
    )
    return parser.parse_args(argv)


def build_dataloaders(
    args: argparse.Namespace,
    pixel_spacings: dict[str, float],
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Construct train, val, and test DataLoaders.

    Expects the Aariz dataset directory structure::

        data/Aariz/
        ├── train/
        │   ├── Cephalograms/
        │   └── Annotations/
        │       └── Cephalometric Landmarks/
        │           ├── Junior Orthodontists/
        │           └── Senior Orthodontists/
        ├── valid/
        └── test/

    Returns
    -------
    tuple[DataLoader, DataLoader, DataLoader]
    """
    data_root = Path(args.data_dir)

    common_kwargs = dict(
        image_size=args.img_size,
        sigma=args.sigma,
        pixel_spacings=pixel_spacings,
    )

    train_ds = CephalometricDataset(
        data_root=data_root / "train",
        split="train",
        **common_kwargs,
    )
    val_ds = CephalometricDataset(
        data_root=data_root / "valid",
        split="val",
        **common_kwargs,
    )
    test_ds = CephalometricDataset(
        data_root=data_root / "test",
        split="test",
        **common_kwargs,
    )

    loader_kwargs = dict(
        num_workers=args.num_workers,
        pin_memory=False,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs,
    )

    return train_loader, val_loader, test_loader


def main(argv: list[str] | None = None) -> None:
    """Main training entrypoint."""
    args = parse_args(argv)
    device = get_device()

    print("=" * 60)
    print("Cephalometric Landmark Detection — Training")
    print("=" * 60)
    print(f"Device       : {device}")
    print(f"Encoder      : {args.encoder}")
    print(f"Epochs       : {args.epochs}")
    print(f"Batch size   : {args.batch_size}")
    print(f"Image size   : {args.img_size}")
    print(f"Sigma        : {args.sigma}")
    print(f"Encoder LR   : {args.encoder_lr}")
    print(f"Decoder LR   : {args.decoder_lr}")
    print(f"Data dir     : {args.data_dir}")
    print(f"Checkpoint   : {args.checkpoint_dir}")
    print(f"Landmarks    : {NUM_LANDMARKS} ({', '.join(SELECTED_SYMBOLS)})")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Load pixel spacings from CSV
    # ------------------------------------------------------------------
    csv_path = Path(args.data_dir) / "cephalogram_machine_mappings.csv"
    pixel_spacings: dict[str, float] = {}
    if csv_path.exists():
        pixel_spacings = load_pixel_spacings(csv_path)
        unique_spacings = sorted(set(pixel_spacings.values()))
        print(f"\n  Pixel spacings loaded: {len(pixel_spacings)} images")
        print(f"  Unique values (mm/px): {unique_spacings}")
    else:
        print(f"\n  WARNING: {csv_path} not found — using default 0.1 mm/px")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    print("\n[1/4] Loading datasets …")
    train_loader, val_loader, test_loader = build_dataloaders(args, pixel_spacings)
    print(
        f"  Train: {len(train_loader.dataset)} | "  # type: ignore[arg-type]
        f"Val: {len(val_loader.dataset)} | "  # type: ignore[arg-type]
        f"Test: {len(test_loader.dataset)}"  # type: ignore[arg-type]
    )

    # ------------------------------------------------------------------
    # MLflow
    # ------------------------------------------------------------------
    print("\n[2/4] Setting up MLflow …")
    experiment_id = setup_experiment("cephalometric-landmark-detection")
    print(f"  Experiment ID: {experiment_id}")

    hyperparams = {
        "encoder_name": args.encoder,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "encoder_lr": args.encoder_lr,
        "decoder_lr": args.decoder_lr,
        "sigma": args.sigma,
        "image_size": args.img_size,
        "num_landmarks": NUM_LANDMARKS,
        "dataset": "Aariz",
    }

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    print("\n[3/4] Training …")
    with mlflow.start_run(tags={"encoder": args.encoder}):
        log_hyperparams(hyperparams)

        trainer = Trainer(
            encoder_name=args.encoder,
            encoder_weights="imagenet",
            in_channels=1,
            num_classes=NUM_LANDMARKS,
            encoder_lr=args.encoder_lr,
            decoder_lr=args.decoder_lr,
            epochs=args.epochs,
            checkpoint_dir=args.checkpoint_dir,
            device=device,
        )

        best_ckpt = trainer.fit(
            train_loader, val_loader, log_to_mlflow=True,
        )

        # ------------------------------------------------------------------
        # Evaluation
        # ------------------------------------------------------------------
        print("\n[4/4] Evaluating on test set …")
        trainer.load_checkpoint(best_ckpt)

        results = evaluate_model(
            model=trainer.model,
            dataloader=test_loader,
            device=device,
        )

        # Log to MLflow
        log_evaluation_metrics(
            mre_per_landmark=results["mre_per_landmark"],
            mre_overall=results["mre_overall"],
            sdr_dict=results["sdr"],
        )

        # Print summary
        print("\n" + "=" * 60)
        print("Evaluation Results")
        print("=" * 60)
        print(f"  Overall MRE: {results['mre_overall']:.2f} mm")
        print("  Per-landmark MRE:")
        for name, mre in results["mre_per_landmark"].items():
            print(f"    {name:>20s}: {mre:.2f} mm")
        print("  SDR:")
        for key, val in results["sdr"].items():
            print(f"    {key:>12s}: {val:.1f}%")
        print("=" * 60)
        print(f"\nBest checkpoint saved to: {best_ckpt}")
        print("Done!")


if __name__ == "__main__":
    main()
