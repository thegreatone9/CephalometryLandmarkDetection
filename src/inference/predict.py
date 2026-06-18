"""
Post-processing utilities for heatmap-based landmark prediction.

Converts model output heatmaps to (x, y) coordinates via argmax or
weighted centroid refinement, computes confidence scores, and rescales
coordinates back to the original image resolution.
"""

from __future__ import annotations

import torch
import numpy as np


def heatmap_to_coordinates(heatmaps: torch.Tensor) -> torch.Tensor:
    """Extract landmark coordinates from heatmaps via argmax.

    Parameters
    ----------
    heatmaps : Tensor
        Shape ``(C, H, W)`` or ``(B, C, H, W)``.

    Returns
    -------
    Tensor
        Shape ``(C, 2)`` or ``(B, C, 2)`` with (x, y) per landmark.
    """
    batched = heatmaps.dim() == 4
    if not batched:
        heatmaps = heatmaps.unsqueeze(0)  # (1, C, H, W)

    B, C, H, W = heatmaps.shape
    flat = heatmaps.view(B, C, -1)
    max_indices = flat.argmax(dim=-1)  # (B, C)

    y_coords = max_indices // W
    x_coords = max_indices % W

    coords = torch.stack([x_coords, y_coords], dim=-1).float()  # (B, C, 2)

    if not batched:
        coords = coords.squeeze(0)  # (C, 2)
    return coords


def weighted_centroid(
    heatmap: np.ndarray,
    peak_x: int,
    peak_y: int,
    window_size: int = 5,
) -> tuple[float, float]:
    """Refine a peak location using a weighted centroid around the argmax.

    Parameters
    ----------
    heatmap : np.ndarray
        Single-channel heatmap of shape ``(H, W)``.
    peak_x, peak_y : int
        Initial peak coordinates (from argmax).
    window_size : int
        Side length of the square window around the peak.

    Returns
    -------
    tuple[float, float]
        Refined ``(x, y)`` coordinates (sub-pixel).
    """
    H, W = heatmap.shape
    half = window_size // 2

    y_min = max(0, peak_y - half)
    y_max = min(H, peak_y + half + 1)
    x_min = max(0, peak_x - half)
    x_max = min(W, peak_x + half + 1)

    patch = heatmap[y_min:y_max, x_min:x_max]

    total = patch.sum()
    if total < 1e-8:
        return float(peak_x), float(peak_y)

    ys, xs = np.mgrid[y_min:y_max, x_min:x_max]
    refined_x = float((xs * patch).sum() / total)
    refined_y = float((ys * patch).sum() / total)

    return refined_x, refined_y


def refine_coordinates(
    heatmaps: np.ndarray,
    coords: np.ndarray,
    window_size: int = 5,
) -> np.ndarray:
    """Apply weighted-centroid refinement to all landmarks.

    Parameters
    ----------
    heatmaps : np.ndarray
        Shape ``(C, H, W)``.
    coords : np.ndarray
        Shape ``(C, 2)`` with integer (x, y) from argmax.
    window_size : int
        Refinement window.

    Returns
    -------
    np.ndarray
        Refined (x, y) coordinates of shape ``(C, 2)``.
    """
    C = heatmaps.shape[0]
    refined = np.zeros_like(coords, dtype=np.float32)
    for i in range(C):
        px, py = int(coords[i, 0]), int(coords[i, 1])
        rx, ry = weighted_centroid(heatmaps[i], px, py, window_size)
        refined[i] = [rx, ry]
    return refined


def compute_confidence(heatmaps: torch.Tensor) -> torch.Tensor:
    """Compute per-landmark confidence as normalised peak activation.

    The confidence is the peak value of each heatmap channel, scaled to
    a 0–100 % range (the Gaussian target peaks at 1.0).

    Parameters
    ----------
    heatmaps : Tensor
        Shape ``(C, H, W)`` or ``(B, C, H, W)``.

    Returns
    -------
    Tensor
        Shape ``(C,)`` or ``(B, C)`` with confidence in [0, 100].
    """
    batched = heatmaps.dim() == 4
    if not batched:
        heatmaps = heatmaps.unsqueeze(0)

    B, C, H, W = heatmaps.shape
    flat = heatmaps.view(B, C, -1)
    peak_vals = flat.max(dim=-1).values  # (B, C)
    confidence = peak_vals.clamp(0, 1) * 100.0

    if not batched:
        confidence = confidence.squeeze(0)
    return confidence


def rescale_coordinates(
    coords: np.ndarray,
    original_size: tuple[int, int],
    resized_size: tuple[int, int],
) -> np.ndarray:
    """Rescale predicted coordinates from resized space back to original.

    Parameters
    ----------
    coords : np.ndarray
        Shape ``(C, 2)`` with (x, y) in resized image space.
    original_size : tuple[int, int]
        ``(height, width)`` of the original image.
    resized_size : tuple[int, int]
        ``(height, width)`` of the resized image.

    Returns
    -------
    np.ndarray
        Coordinates in the original image space.
    """
    orig_h, orig_w = original_size
    resz_h, resz_w = resized_size

    scale_x = orig_w / resz_w
    scale_y = orig_h / resz_h

    rescaled = coords.copy().astype(np.float32)
    rescaled[:, 0] *= scale_x
    rescaled[:, 1] *= scale_y
    return rescaled
