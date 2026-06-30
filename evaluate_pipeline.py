#!/usr/bin/env python3
"""Evaluate the full two-stage pipeline on the test set.

Usage
-----
    python evaluate_pipeline.py \
        --stage1-checkpoint checkpoints/stage1-.../best_model.pth \
        --stage2-checkpoint checkpoints/refiner-.../best_refiner.pth

Runs Stage 1 + Stage 2 end-to-end and reports MRE and SDR metrics,
with comparison to Stage 1 alone.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mlflow
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import (
    CephalometricDataset,
    NUM_LANDMARKS,
    SELECTED_DISPLAY_NAMES,
    SELECTED_SYMBOLS,
    load_pixel_spacings,
)
from src.evaluation import compute_mre, compute_sdr, evaluate_model
from src.inference.pipeline import TwoStagePipeline, evaluate_pipeline
from src.models.unet import create_model
from src.training.mlflow_utils import log_evaluation_metrics, setup_experiment


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the two-stage pipeline on the test set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--stage1-checkpoint",
        type=str,
        required=True,
        help="Path to trained Stage 1 model checkpoint.",
    )
    parser.add_argument(
        "--stage2-checkpoint",
        type=str,
        required=True,
        help="Path to trained Stage 2 refiner checkpoint.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Root data directory.",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=512,
        help="Image size for Stage 1.",
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        default=96,
        help="Crop size for Stage 2.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for evaluation (1 recommended for pipeline).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader workers.",
    )
    return parser.parse_args(argv)


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    device = _get_device()
    data_dir = Path(args.data_dir)

    print(f"Device: {device}")
    print(f"Stage 1: {args.stage1_checkpoint}")
    print(f"Stage 2: {args.stage2_checkpoint}")
    print()

    # ------------------------------------------------------------------
    # Build test dataloader
    # ------------------------------------------------------------------
    pixel_spacings = load_pixel_spacings(
        data_dir / "cephalogram_machine_mappings.csv"
    )
    test_dataset = CephalometricDataset(
        data_root=data_dir / "test",
        image_size=args.img_size,
        sigma=5.0,
        split="test",
        pixel_spacings=pixel_spacings,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print(f"Test samples: {len(test_dataset)}")

    # ------------------------------------------------------------------
    # Evaluate Stage 1 alone (baseline comparison)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  STAGE 1 ONLY (coarse)")
    print("=" * 60)

    stage1_model = create_model(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=1,
        num_classes=NUM_LANDMARKS,
    )
    checkpoint = torch.load(args.stage1_checkpoint, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        stage1_model.load_state_dict(checkpoint["model_state_dict"])
    else:
        stage1_model.load_state_dict(checkpoint)
    stage1_model.to(device)

    t0 = time.time()
    stage1_results = evaluate_model(
        stage1_model, test_loader, device,
        landmark_names=list(SELECTED_DISPLAY_NAMES),
    )
    stage1_time = time.time() - t0

    print(f"\n  Overall MRE: {stage1_results['mre_overall']:.2f} mm")
    print(f"  SDR@2mm: {stage1_results['sdr'].get('sdr_2.0mm', 0)*100:.1f}%")
    print(f"  SDR@2.5mm: {stage1_results['sdr'].get('sdr_2.5mm', 0)*100:.1f}%")
    print(f"  SDR@3mm: {stage1_results['sdr'].get('sdr_3.0mm', 0)*100:.1f}%")
    print(f"  SDR@4mm: {stage1_results['sdr'].get('sdr_4.0mm', 0)*100:.1f}%")
    print(f"  Time: {stage1_time:.1f}s")

    print("\n  Per-landmark MRE (mm):")
    sorted_landmarks = sorted(
        stage1_results["mre_per_landmark"].items(),
        key=lambda x: x[1],
        reverse=True,
    )
    for name, mre in sorted_landmarks:
        status = "✓" if mre < 3.0 else "⚠" if mre < 5.0 else "✗"
        print(f"    {status} {name:40s} {mre:6.2f}")

    del stage1_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Evaluate full two-stage pipeline
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  TWO-STAGE PIPELINE (coarse → refine)")
    print("=" * 60)

    pipeline = TwoStagePipeline(
        stage1_checkpoint=args.stage1_checkpoint,
        stage2_checkpoint=args.stage2_checkpoint,
        device=device,
        num_landmarks=NUM_LANDMARKS,
        crop_size=args.crop_size,
        image_size=args.img_size,
    )

    t0 = time.time()
    pipeline_results = evaluate_pipeline(
        pipeline, test_loader, device,
        landmark_names=list(SELECTED_DISPLAY_NAMES),
    )
    pipeline_time = time.time() - t0

    print(f"\n  Overall MRE: {pipeline_results['mre_overall']:.2f} mm")
    print(f"  SDR@2mm: {pipeline_results['sdr'].get('sdr_2.0mm', 0)*100:.1f}%")
    print(f"  SDR@2.5mm: {pipeline_results['sdr'].get('sdr_2.5mm', 0)*100:.1f}%")
    print(f"  SDR@3mm: {pipeline_results['sdr'].get('sdr_3.0mm', 0)*100:.1f}%")
    print(f"  SDR@4mm: {pipeline_results['sdr'].get('sdr_4.0mm', 0)*100:.1f}%")
    print(f"  Time: {pipeline_time:.1f}s")

    print("\n  Per-landmark MRE (mm):")
    sorted_landmarks = sorted(
        pipeline_results["mre_per_landmark"].items(),
        key=lambda x: x[1],
        reverse=True,
    )
    for name, mre in sorted_landmarks:
        status = "✓" if mre < 3.0 else "⚠" if mre < 5.0 else "✗"
        print(f"    {status} {name:40s} {mre:6.2f}")

    # ------------------------------------------------------------------
    # Side-by-side comparison
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  COMPARISON: Stage 1 Only vs Two-Stage Pipeline")
    print("=" * 60)

    s1_mre = stage1_results["mre_overall"]
    p_mre = pipeline_results["mre_overall"]
    improvement = ((s1_mre - p_mre) / s1_mre) * 100

    print(f"\n  {'Metric':<20s} {'Stage 1':>10s} {'Pipeline':>10s} {'Δ':>10s}")
    print(f"  {'-'*50}")
    print(f"  {'MRE (mm)':<20s} {s1_mre:>10.2f} {p_mre:>10.2f} {improvement:>+9.1f}%")

    for thresh in ["2.0", "2.5", "3.0", "4.0"]:
        key = f"sdr_{thresh}mm"
        s1_val = stage1_results["sdr"].get(key, 0) * 100
        p_val = pipeline_results["sdr"].get(key, 0) * 100
        delta = p_val - s1_val
        print(f"  {'SDR@'+thresh+'mm':<20s} {s1_val:>9.1f}% {p_val:>9.1f}% {delta:>+9.1f}%")

    print(f"\n  Per-landmark improvements:")
    for name in sorted(
        pipeline_results["mre_per_landmark"].keys(),
        key=lambda n: stage1_results["mre_per_landmark"].get(n, 0)
            - pipeline_results["mre_per_landmark"].get(n, 0),
        reverse=True,
    ):
        s1 = stage1_results["mre_per_landmark"].get(name, 0)
        p = pipeline_results["mre_per_landmark"].get(name, 0)
        delta = s1 - p
        arrow = "↓" if delta > 0 else "↑"
        print(f"    {name:40s} {s1:6.2f} → {p:6.2f}  ({arrow} {abs(delta):.2f}mm)")

    # ------------------------------------------------------------------
    # Log to MLflow
    # ------------------------------------------------------------------
    setup_experiment("cephalometric-pipeline-eval")
    with mlflow.start_run(run_name="two-stage-eval"):
        mlflow.log_params({
            "stage1_checkpoint": str(args.stage1_checkpoint),
            "stage2_checkpoint": str(args.stage2_checkpoint),
            "num_landmarks": NUM_LANDMARKS,
            "crop_size": args.crop_size,
        })
        # Log pipeline results
        log_evaluation_metrics(
            pipeline_results["mre_per_landmark"],
            pipeline_results["mre_overall"],
            pipeline_results["sdr"],
        )
        # Log stage1-only for comparison
        mlflow.log_metric("stage1_mre_overall_mm", s1_mre)
        mlflow.log_metric("improvement_pct", improvement)

    print("\nResults logged to MLflow experiment 'cephalometric-pipeline-eval'.")


if __name__ == "__main__":
    main()
