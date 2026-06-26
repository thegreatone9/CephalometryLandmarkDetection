"""
Cephalometric Landmark Detection Dataset.

Loads lateral cephalogram images and their corresponding landmark annotations
from the Aariz dataset (Nature Scientific Data, 2025).

The Aariz dataset structure:
    data/Aariz/
    ├── train/
    │   ├── Cephalograms/           ← images (.png/.jpg)
    │   └── Annotations/
    │       ├── Cephalometric Landmarks/
    │       │   ├── Junior Orthodontists/   ← per-image .json
    │       │   └── Senior Orthodontists/   ← per-image .json
    │       └── CVM Stages/
    ├── valid/                      ← same structure
    ├── test/                       ← same structure
    └── cephalogram_machine_mappings.csv   ← per-image pixel spacing

Ground truth = mean of averaged Junior and averaged Senior landmark positions.

The dataset provides 29 landmarks per image.  We select 6 key skeletal
landmarks for our pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.preprocessing import get_train_transforms, get_val_transforms

# ---------------------------------------------------------------------------
# Landmark configuration
# ---------------------------------------------------------------------------

# The 29 Aariz landmark symbols, in the order they appear in the JSON files.
# (The JSON stores them as objects with "symbol" keys, so order doesn't matter
#  for loading — we look up by symbol.)
ALL_LANDMARK_SYMBOLS: list[str] = [
    "A",     # A-point
    "ANS",   # Anterior Nasal Spine
    "B",     # B-point
    "Me",    # Menton
    "N",     # Nasion
    "Or",    # Orbitale
    "Pog",   # Pogonion
    "PNS",   # Posterior Nasal Spine
    "Pn",    # Pronasale
    "R",     # Ramus
    "S",     # Sella
    "Ar",    # Articulare
    "Co",    # Condylion
    "Gn",    # Gnathion
    "Go",    # Gonion
    "Po",    # Porion
    "LPM",   # Lower 2nd PM Cusp Tip
    "LIT",   # Lower Incisor Tip
    "LMT",   # Lower Molar Cusp Tip
    "UPM",   # Upper 2nd PM Cusp Tip
    "UIA",   # Upper Incisor Apex
    "UIT",   # Upper Incisor Tip
    "UMT",   # Upper Molar Cusp Tip
    "LIA",   # Lower Incisor Apex
    "Li",    # Labrale inferius
    "Ls",    # Labrale superius
    "N`",    # Soft Tissue Nasion
    "Pog`",  # Soft Tissue Pogonion
    "Sn",    # Subnasale
]

# The landmarks we use for heatmap regression, by their Aariz symbol.
# Using all 29 Aariz landmarks for comprehensive benchmark evaluation.
SELECTED_SYMBOLS: list[str] = list(ALL_LANDMARK_SYMBOLS)

SELECTED_DISPLAY_NAMES: list[str] = [
    "A-point (A)",
    "Anterior Nasal Spine (ANS)",
    "B-point (B)",
    "Menton (Me)",
    "Nasion (N)",
    "Orbitale (Or)",
    "Pogonion (Pog)",
    "Posterior Nasal Spine (PNS)",
    "Pronasale (Pn)",
    "Ramus (R)",
    "Sella (S)",
    "Articulare (Ar)",
    "Condylion (Co)",
    "Gnathion (Gn)",
    "Gonion (Go)",
    "Porion (Po)",
    "Lower 2nd PM Cusp (LPM)",
    "Lower Incisor Tip (LIT)",
    "Lower Molar Cusp (LMT)",
    "Upper 2nd PM Cusp (UPM)",
    "Upper Incisor Apex (UIA)",
    "Upper Incisor Tip (UIT)",
    "Upper Molar Cusp (UMT)",
    "Lower Incisor Apex (LIA)",
    "Labrale Inferius (Li)",
    "Labrale Superius (Ls)",
    "Soft Tissue Nasion (N`)",
    "Soft Tissue Pogonion (Pog`)",
    "Subnasale (Sn)",
]

NUM_LANDMARKS: int = len(SELECTED_SYMBOLS)

# Per-landmark sigma values for heatmap generation.
# Dental landmarks get tighter Gaussians (less overlap in dense region),
# soft tissue gets wider Gaussians (higher annotation uncertainty).
SIGMA_MAP: dict[str, float] = {
    # Skeletal — clear bony edges, well-separated
    "A": 5.0, "ANS": 5.0, "B": 5.0, "Me": 5.0, "N": 5.0, "Or": 5.0,
    "Pog": 5.0, "PNS": 5.0, "R": 5.0, "S": 5.0, "Ar": 5.0, "Co": 5.0,
    "Gn": 5.0, "Go": 5.0, "Po": 5.0,
    # Dental — tightly clustered, need narrow Gaussians
    "LPM": 3.0, "LIT": 3.0, "LMT": 3.0, "UPM": 3.0,
    "UIA": 3.0, "UIT": 3.0, "UMT": 3.0, "LIA": 3.0,
    # Soft tissue — fuzzy boundaries, higher annotation uncertainty
    "Pn": 7.0, "Sn": 7.0, "Ls": 7.0, "Li": 7.0, "N`": 7.0, "Pog`": 7.0,
}


# ---------------------------------------------------------------------------
# Heatmap helpers
# ---------------------------------------------------------------------------

def _generate_gaussian_heatmap(
    height: int,
    width: int,
    center_x: float,
    center_y: float,
    sigma: float,
) -> np.ndarray:
    """Generate a single 2-D Gaussian heatmap centred at (center_x, center_y).

    Parameters
    ----------
    height, width : int
        Spatial dimensions of the output heatmap.
    center_x, center_y : float
        Landmark position in pixel coordinates (can be sub-pixel).
    sigma : float
        Standard deviation of the Gaussian kernel.

    Returns
    -------
    np.ndarray
        Array of shape (height, width) with values in [0, 1].
    """
    xs = np.arange(0, width, dtype=np.float32)
    ys = np.arange(0, height, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)

    heatmap = np.exp(
        -((xx - center_x) ** 2 + (yy - center_y) ** 2) / (2 * sigma ** 2)
    )
    return heatmap


def generate_heatmaps(
    landmarks: np.ndarray,
    height: int,
    width: int,
    sigma: float = 5.0,
    sigma_per_landmark: Optional[list[float]] = None,
) -> np.ndarray:
    """Generate multi-channel heatmap tensor from landmark coordinates.

    Parameters
    ----------
    landmarks : np.ndarray
        Shape (N, 2) with (x, y) per landmark.
    height, width : int
        Spatial dimensions.
    sigma : float
        Default Gaussian sigma (used when *sigma_per_landmark* is ``None``).
    sigma_per_landmark : list[float] | None
        Per-channel sigma overrides.  When provided, ``sigma_per_landmark[i]``
        is used for landmark *i*; otherwise *sigma* is used as the fallback.

    Returns
    -------
    np.ndarray
        Shape (N, height, width).
    """
    num = landmarks.shape[0]
    heatmaps = np.zeros((num, height, width), dtype=np.float32)
    for i in range(num):
        x, y = landmarks[i]
        # Skip landmarks outside the image (e.g. missing annotation coded as -1)
        if x < 0 or y < 0:
            continue
        s = sigma_per_landmark[i] if sigma_per_landmark is not None else sigma
        heatmaps[i] = _generate_gaussian_heatmap(height, width, x, y, s)
    return heatmaps


# ---------------------------------------------------------------------------
# Annotation I/O
# ---------------------------------------------------------------------------

def _load_landmarks_from_json(filepath: Path) -> dict[str, tuple[float, float]]:
    """Load landmark coordinates from an Aariz JSON annotation file.

    Returns a dict mapping landmark symbol → (x, y).
    """
    with open(filepath, "r") as fh:
        data = json.load(fh)

    landmarks: dict[str, tuple[float, float]] = {}
    for lm in data["landmarks"]:
        symbol = lm["symbol"]
        x = float(lm["value"]["x"])
        y = float(lm["value"]["y"])
        landmarks[symbol] = (x, y)

    return landmarks


def _average_annotations(
    junior_path: Path,
    senior_path: Path,
) -> dict[str, tuple[float, float]]:
    """Compute ground truth as the mean of junior and senior annotations.

    If only one source is available, returns that source's annotations.
    """
    junior_exists = junior_path.exists()
    senior_exists = senior_path.exists()

    if junior_exists and senior_exists:
        junior = _load_landmarks_from_json(junior_path)
        senior = _load_landmarks_from_json(senior_path)
        averaged: dict[str, tuple[float, float]] = {}
        # Use the union of both symbol sets
        all_symbols = set(junior.keys()) | set(senior.keys())
        for sym in all_symbols:
            if sym in junior and sym in senior:
                jx, jy = junior[sym]
                sx, sy = senior[sym]
                averaged[sym] = ((jx + sx) / 2.0, (jy + sy) / 2.0)
            elif sym in junior:
                averaged[sym] = junior[sym]
            else:
                averaged[sym] = senior[sym]
        return averaged
    elif junior_exists:
        return _load_landmarks_from_json(junior_path)
    elif senior_exists:
        return _load_landmarks_from_json(senior_path)
    else:
        raise FileNotFoundError(
            f"Neither junior nor senior annotation found:\n"
            f"  junior: {junior_path}\n"
            f"  senior: {senior_path}"
        )


def load_pixel_spacings(csv_path: Path) -> dict[str, float]:
    """Load per-cephalogram pixel spacing from the machine mappings CSV.

    Returns a dict mapping cephalogram_id → pixel_size (mm/pixel).
    """
    df = pd.read_csv(csv_path)
    return dict(zip(df["cephalogram_id"], df["pixel_size"].astype(float)))


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CephalometricDataset(Dataset):
    """PyTorch Dataset for cephalometric landmark detection (Aariz format).

    Each sample yields:
    - ``image``: float32 tensor [1, H, W] normalised to [0, 1].
    - ``heatmaps``: float32 tensor [C, H, W] with one Gaussian per landmark.
    - ``landmarks``: float32 tensor [C, 2] with (x, y) in resized coords.
    - ``meta``: dict with original size, file path, and pixel spacing.

    Parameters
    ----------
    data_root : Path
        Root of a split directory, e.g. ``data/Aariz/train``.
        Must contain ``Cephalograms/`` and ``Annotations/`` subdirs.
    image_size : int
        Target square size for resizing (default 512).
    sigma : float
        Gaussian sigma for heatmap generation.
    split : str
        One of ``"train"``, ``"val"``, ``"test"``.  Controls augmentation.
    selected_symbols : list[str] | None
        Landmark symbols to select.  Defaults to ``SELECTED_SYMBOLS``.
    pixel_spacings : dict[str, float] | None
        Per-image pixel spacing (mm/pixel).  Loaded from CSV externally.
    """

    def __init__(
        self,
        data_root: Path | str,
        image_size: int = 512,
        sigma: float = 5.0,
        split: str = "train",
        selected_symbols: Optional[list[str]] = None,
        pixel_spacings: Optional[dict[str, float]] = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.image_size = image_size
        self.sigma = sigma
        self.split = split
        self.selected_symbols = selected_symbols or SELECTED_SYMBOLS
        self.sigma_per_landmark = [SIGMA_MAP.get(sym, sigma) for sym in self.selected_symbols]
        self.pixel_spacings = pixel_spacings or {}

        # Resolve directory paths
        self.image_dir = self.data_root / "Cephalograms"
        self.junior_dir = (
            self.data_root / "Annotations" / "Cephalometric Landmarks"
            / "Junior Orthodontists"
        )
        self.senior_dir = (
            self.data_root / "Annotations" / "Cephalometric Landmarks"
            / "Senior Orthodontists"
        )

        # Validate directories exist
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not self.junior_dir.exists() and not self.senior_dir.exists():
            raise FileNotFoundError(
                f"No annotation directories found under {self.data_root / 'Annotations'}"
            )

        # Discover samples by listing image files
        self.samples: list[dict] = []
        image_extensions = {".png", ".jpg", ".jpeg", ".bmp"}
        for img_path in sorted(self.image_dir.iterdir()):
            if img_path.suffix.lower() not in image_extensions:
                continue
            stem = img_path.stem
            junior_json = self.junior_dir / f"{stem}.json"
            senior_json = self.senior_dir / f"{stem}.json"
            # Need at least one annotation source
            if junior_json.exists() or senior_json.exists():
                self.samples.append({
                    "image_path": img_path,
                    "junior_json": junior_json,
                    "senior_json": senior_json,
                    "ceph_id": stem,
                })

        if not self.samples:
            raise FileNotFoundError(
                f"No valid image+annotation pairs found in {self.data_root}"
            )

        # Augmentation transforms
        if split == "train":
            self.transforms = get_train_transforms(image_size)
        else:
            self.transforms = get_val_transforms(image_size)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        """Return a single sample dict.

        Keys
        ----
        image : Tensor [1, H, W]
        heatmaps : Tensor [C, H, W]
        landmarks : Tensor [C, 2]
        meta : dict
        """
        sample = self.samples[idx]
        img_path = sample["image_path"]

        # Load image as grayscale numpy array
        img = Image.open(img_path).convert("L")
        orig_w, orig_h = img.size
        img_np = np.array(img, dtype=np.float32)  # (H, W) in [0, 255]

        # Load ground truth (mean of junior + senior)
        all_landmarks = _average_annotations(
            sample["junior_json"], sample["senior_json"]
        )

        # Extract our selected landmarks in order
        landmarks = np.full((len(self.selected_symbols), 2), -1.0, dtype=np.float32)
        for i, sym in enumerate(self.selected_symbols):
            if sym in all_landmarks:
                x, y = all_landmarks[sym]
                landmarks[i] = [x, y]

        # Apply augmentation / resize (albumentations)
        keypoints = [(float(x), float(y)) for x, y in landmarks if x >= 0 and y >= 0]
        # Track which indices have valid keypoints
        valid_indices = [i for i, (x, y) in enumerate(landmarks) if x >= 0 and y >= 0]

        transformed = self.transforms(
            image=img_np,
            keypoints=keypoints,
        )
        img_np = transformed["image"]  # (H, W) after resize
        keypoints_out = transformed["keypoints"]

        # Reconstruct landmarks array in original order
        h_out, w_out = img_np.shape[:2]
        landmarks_out = np.full(
            (len(self.selected_symbols), 2), -1.0, dtype=np.float32
        )
        for kp_idx, orig_idx in enumerate(valid_indices):
            if kp_idx < len(keypoints_out):
                landmarks_out[orig_idx, 0] = keypoints_out[kp_idx][0]
                landmarks_out[orig_idx, 1] = keypoints_out[kp_idx][1]

        # Normalise image to [0, 1]
        img_np = img_np / 255.0

        # Generate heatmaps
        heatmaps = generate_heatmaps(
            landmarks_out, h_out, w_out, sigma=self.sigma,
            sigma_per_landmark=self.sigma_per_landmark,
        )

        # Convert to tensors
        image_tensor = torch.from_numpy(img_np).unsqueeze(0)  # (1, H, W)
        heatmap_tensor = torch.from_numpy(heatmaps)           # (C, H, W)
        landmark_tensor = torch.from_numpy(landmarks_out)     # (C, 2)

        # Pixel spacing for this image (for MRE computation)
        pixel_spacing = self.pixel_spacings.get(sample["ceph_id"], 0.1)

        meta = {
            "image_path": str(img_path),
            "ceph_id": sample["ceph_id"],
            "original_size": (orig_h, orig_w),
            "resized_size": (h_out, w_out),
            "pixel_spacing": pixel_spacing,
        }

        return {
            "image": image_tensor,
            "heatmaps": heatmap_tensor,
            "landmarks": landmark_tensor,
            "meta": meta,
        }
