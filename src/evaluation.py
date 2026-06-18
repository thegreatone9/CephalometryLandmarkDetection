"""
Evaluation metrics for cephalometric landmark detection.

Implements Mean Radial Error (MRE) and Successful Detection Rate (SDR)
at configurable distance thresholds, following the standard evaluation
protocol used in the cephalometric literature.

Supports per-image pixel spacings (the Aariz dataset has images from
7 different X-ray machines with varying resolutions).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import SELECTED_DISPLAY_NAMES
from src.inference.predict import (
    heatmap_to_coordinates,
    refine_coordinates,
    rescale_coordinates,
)


def compute_mre(
    pred_coords: np.ndarray,
    gt_coords: np.ndarray,
    pixel_spacings: np.ndarray | float = 1.0,
) -> tuple[np.ndarray, float]:
    """Compute Mean Radial Error per landmark and overall.

    MRE is the Euclidean distance between predicted and ground-truth
    coordinates, converted to millimetres via ``pixel_spacings``.

    Parameters
    ----------
    pred_coords : np.ndarray
        Predicted coordinates ``(N, C, 2)`` where N = samples, C = landmarks.
    gt_coords : np.ndarray
        Ground-truth coordinates ``(N, C, 2)``.
    pixel_spacings : np.ndarray | float
        Per-sample conversion factor from pixels to mm.
        Either a scalar or array of shape ``(N,)`` or ``(N, 1)``.

    Returns
    -------
    tuple[np.ndarray, float]
        ``(per_landmark_mre, overall_mre)`` — per_landmark has shape ``(C,)``.
    """
    # Euclidean distance: (N, C)
    distances_px = np.sqrt(np.sum((pred_coords - gt_coords) ** 2, axis=-1))

    # Convert to mm — broadcast pixel_spacings to (N, 1) if needed
    if isinstance(pixel_spacings, (int, float)):
        distances_mm = distances_px * pixel_spacings
    else:
        spacings = np.asarray(pixel_spacings, dtype=np.float32)
        if spacings.ndim == 1:
            spacings = spacings[:, np.newaxis]  # (N, 1) for broadcasting
        distances_mm = distances_px * spacings

    per_landmark_mre = distances_mm.mean(axis=0)  # (C,)
    overall_mre = float(distances_mm.mean())

    return per_landmark_mre, overall_mre


def compute_sdr(
    pred_coords: np.ndarray,
    gt_coords: np.ndarray,
    pixel_spacings: np.ndarray | float = 1.0,
    thresholds: Optional[list[float]] = None,
) -> dict[str, float]:
    """Compute Successful Detection Rate at multiple distance thresholds.

    SDR is the percentage of landmarks detected within a given distance
    threshold (in mm).

    Parameters
    ----------
    pred_coords : np.ndarray
        Predicted coordinates ``(N, C, 2)``.
    gt_coords : np.ndarray
        Ground-truth coordinates ``(N, C, 2)``.
    pixel_spacings : np.ndarray | float
        Per-sample pixels → mm conversion factor.
    thresholds : list[float] | None
        Distance thresholds in mm (default ``[2.0, 2.5, 3.0, 4.0]``).

    Returns
    -------
    dict[str, float]
        Keys like ``"SDR_2.0mm"``, values are percentages in [0, 100].
    """
    if thresholds is None:
        thresholds = [2.0, 2.5, 3.0, 4.0]

    distances_px = np.sqrt(np.sum((pred_coords - gt_coords) ** 2, axis=-1))

    if isinstance(pixel_spacings, (int, float)):
        distances_mm = distances_px * pixel_spacings
    else:
        spacings = np.asarray(pixel_spacings, dtype=np.float32)
        if spacings.ndim == 1:
            spacings = spacings[:, np.newaxis]
        distances_mm = distances_px * spacings

    results: dict[str, float] = {}
    total_count = distances_mm.size  # N × C
    for t in thresholds:
        success_count = int((distances_mm < t).sum())
        results[f"SDR_{t}mm"] = (success_count / total_count) * 100.0

    return results


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    refine: bool = True,
    landmark_names: Optional[list[str]] = None,
) -> dict:
    """Run full evaluation on a dataset split.

    Per-image pixel spacings are read from the ``meta`` dict in each batch,
    so no single ``pixel_spacing`` argument is needed.

    Parameters
    ----------
    model : nn.Module
        Trained heatmap regression model.
    dataloader : DataLoader
        Evaluation data loader.
    device : torch.device
        Compute device.
    refine : bool
        Whether to apply weighted-centroid refinement.
    landmark_names : list[str] | None
        Names for per-landmark reporting; defaults to ``SELECTED_DISPLAY_NAMES``.

    Returns
    -------
    dict
        Contains ``"mre_per_landmark"``, ``"mre_overall"``, ``"sdr"``,
        ``"all_pred_coords"``, ``"all_gt_coords"``.
    """
    model.eval()
    landmark_names = landmark_names or list(SELECTED_DISPLAY_NAMES)

    all_preds: list[np.ndarray] = []
    all_gts: list[np.ndarray] = []
    all_spacings: list[float] = []

    for batch in dataloader:
        images = batch["image"].to(device)
        gt_landmarks = batch["landmarks"].numpy()  # (B, C, 2)
        meta = batch["meta"]

        pred_heatmaps = model(images).cpu()  # (B, C, H, W)

        for i in range(images.size(0)):
            hm = pred_heatmaps[i]  # (C, H, W)
            coords = heatmap_to_coordinates(hm).numpy()  # (C, 2)

            if refine:
                coords = refine_coordinates(hm.numpy(), coords)

            # Rescale to original resolution
            orig_size = (
                meta["original_size"][0][i].item(),
                meta["original_size"][1][i].item(),
            )
            resized_size = (
                meta["resized_size"][0][i].item(),
                meta["resized_size"][1][i].item(),
            )
            coords = rescale_coordinates(coords, orig_size, resized_size)

            # Ground truth was stored in resized coords → also rescale
            gt_i = gt_landmarks[i]  # (C, 2)
            gt_i_rescaled = rescale_coordinates(gt_i, orig_size, resized_size)

            all_preds.append(coords)
            all_gts.append(gt_i_rescaled)

            # Per-image pixel spacing
            spacing = meta["pixel_spacing"][i].item()
            all_spacings.append(spacing)

    pred_array = np.stack(all_preds)  # (N, C, 2)
    gt_array = np.stack(all_gts)      # (N, C, 2)
    spacings_array = np.array(all_spacings, dtype=np.float32)  # (N,)

    per_landmark_mre, overall_mre = compute_mre(
        pred_array, gt_array, spacings_array,
    )
    sdr = compute_sdr(pred_array, gt_array, spacings_array)

    mre_per_landmark = {
        name: float(per_landmark_mre[i])
        for i, name in enumerate(landmark_names)
    }

    return {
        "mre_per_landmark": mre_per_landmark,
        "mre_overall": overall_mre,
        "sdr": sdr,
        "all_pred_coords": pred_array,
        "all_gt_coords": gt_array,
    }
