"""
Visualization utilities for cephalometric landmark detection.

Provides functions to overlay predicted and ground-truth landmarks on
cephalogram images, draw angle construction lines, and create full
results figures with annotations.
"""

from __future__ import annotations

from typing import Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.figure import Figure


# Default colour palette for the 6 selected landmarks
DEFAULT_COLORS: list[str] = [
    "#FF4444",  # Sella — red
    "#44FF44",  # Nasion — green
    "#4444FF",  # A-point — blue
    "#FFAA00",  # B-point — orange
    "#FF44FF",  # Pogonion — magenta
    "#00DDDD",  # Menton — cyan
]


def draw_landmarks(
    image: np.ndarray,
    landmarks: np.ndarray,
    names: list[str],
    colors: Optional[list[str]] = None,
    gt_landmarks: Optional[np.ndarray] = None,
    ax: Optional[plt.Axes] = None,
    marker_size: int = 6,
    font_size: int = 8,
) -> plt.Axes:
    """Overlay predicted (and optionally ground truth) landmarks on an image.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image ``(H, W)`` or ``(H, W, 1)``.
    landmarks : np.ndarray
        Predicted landmarks ``(C, 2)`` as (x, y).
    names : list[str]
        Landmark names for labels.
    colors : list[str] | None
        Per-landmark colours; defaults to ``DEFAULT_COLORS``.
    gt_landmarks : np.ndarray | None
        Ground-truth landmarks for comparison.
    ax : plt.Axes | None
        Existing axes; created if ``None``.
    marker_size : int
        Marker size.
    font_size : int
        Label font size.

    Returns
    -------
    plt.Axes
    """
    colors = colors or DEFAULT_COLORS
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(8, 8))

    img_display = image.squeeze()
    ax.imshow(img_display, cmap="gray")

    for i, (name, color) in enumerate(zip(names, colors)):
        x, y = landmarks[i]
        ax.plot(x, y, "o", color=color, markersize=marker_size, label=name)
        ax.annotate(
            name,
            (x, y),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=font_size,
            color=color,
            fontweight="bold",
        )

        if gt_landmarks is not None:
            gx, gy = gt_landmarks[i]
            ax.plot(gx, gy, "x", color=color, markersize=marker_size + 2)
            ax.plot(
                [gx, x], [gy, y],
                "--", color=color, alpha=0.4, linewidth=1,
            )

    ax.axis("off")
    return ax


def draw_angle_lines(
    ax: plt.Axes,
    sella: np.ndarray,
    nasion: np.ndarray,
    a_point: np.ndarray,
    b_point: np.ndarray,
    line_width: float = 1.5,
    alpha: float = 0.7,
) -> None:
    """Draw S-N, N-A, and N-B construction lines for SNA/SNB angles.

    Parameters
    ----------
    ax : plt.Axes
        Axes to draw on.
    sella, nasion, a_point, b_point : np.ndarray
        Landmark coordinates, each ``(2,)`` as (x, y).
    line_width : float
        Line width.
    alpha : float
        Line transparency.
    """
    # S-N line
    ax.plot(
        [sella[0], nasion[0]], [sella[1], nasion[1]],
        "-", color="yellow", linewidth=line_width, alpha=alpha, label="S-N",
    )
    # N-A line
    ax.plot(
        [nasion[0], a_point[0]], [nasion[1], a_point[1]],
        "-", color="lime", linewidth=line_width, alpha=alpha, label="N-A",
    )
    # N-B line
    ax.plot(
        [nasion[0], b_point[0]], [nasion[1], b_point[1]],
        "-", color="orange", linewidth=line_width, alpha=alpha, label="N-B",
    )


def create_results_figure(
    image: np.ndarray,
    pred_landmarks: np.ndarray,
    gt_landmarks: Optional[np.ndarray],
    angles_dict: dict[str, float],
    landmark_names: list[str],
    save_path: Optional[str] = None,
) -> Figure:
    """Create a full results figure with landmarks, lines, and angle text.

    Parameters
    ----------
    image : np.ndarray
        Grayscale cephalogram ``(H, W)``.
    pred_landmarks : np.ndarray
        Predicted coordinates ``(C, 2)``.
    gt_landmarks : np.ndarray | None
        Ground-truth coordinates for overlay.
    angles_dict : dict[str, float]
        Dictionary of computed angles (e.g. ``{"SNA": 82.1, ...}``).
    landmark_names : list[str]
        Names of the selected landmarks.
    save_path : str | None
        If provided, save figure to this path.

    Returns
    -------
    Figure
    """
    fig, (ax_img, ax_txt) = plt.subplots(
        1, 2, figsize=(14, 7),
        gridspec_kw={"width_ratios": [3, 1]},
    )

    # Image with landmarks
    draw_landmarks(
        image, pred_landmarks, landmark_names,
        gt_landmarks=gt_landmarks, ax=ax_img,
    )

    # Draw angle construction lines (S, N, A, B are indices 0-3)
    if pred_landmarks.shape[0] >= 4:
        draw_angle_lines(
            ax_img,
            sella=pred_landmarks[0],
            nasion=pred_landmarks[1],
            a_point=pred_landmarks[2],
            b_point=pred_landmarks[3],
        )

    ax_img.set_title("Cephalometric Analysis", fontsize=14, fontweight="bold")

    # Legend patches
    patches = []
    for name, color in zip(landmark_names, DEFAULT_COLORS):
        patches.append(mpatches.Patch(color=color, label=name))
    if gt_landmarks is not None:
        patches.append(
            mpatches.Patch(
                facecolor="none", edgecolor="gray",
                label="× = Ground Truth",
            )
        )
    ax_img.legend(
        handles=patches, loc="lower left", fontsize=7, framealpha=0.8,
    )

    # Text panel for angles
    ax_txt.axis("off")
    text_lines = ["Cephalometric Angles", "=" * 25, ""]
    for name, value in angles_dict.items():
        text_lines.append(f"  {name}: {value:.1f}°")
    text_lines.append("")
    text_lines.append("-" * 25)
    text_lines.append("Normal ranges:")
    text_lines.append("  SNA: 82° ± 2°")
    text_lines.append("  SNB: 80° ± 2°")
    text_lines.append("  ANB: 2° ± 2°")

    ax_txt.text(
        0.1, 0.95, "\n".join(text_lines),
        transform=ax_txt.transAxes,
        fontsize=11,
        verticalalignment="top",
        fontfamily="monospace",
    )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
