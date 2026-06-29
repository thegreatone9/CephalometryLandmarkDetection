"""Crop-based dataset for Stage 2 landmark refinement training.

Generates small image crops centred on Stage 1 predictions, with
single-channel ground-truth heatmaps for precise localisation.
"""
from __future__ import annotations
import json
import random
from pathlib import Path
from typing import Optional
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from src.data.dataset import ALL_LANDMARK_SYMBOLS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_landmarks_from_json(filepath: Path) -> dict[str, tuple[float, float]]:
    """Load landmark coordinates from an Aariz JSON annotation file.

    Returns a dict mapping landmark symbol → (x, y) in *original* pixel
    coordinates.
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
    """Compute GT as the mean of junior and senior annotations.

    Falls back to whichever source is available when only one exists.
    """
    junior_exists = junior_path.exists()
    senior_exists = senior_path.exists()

    if junior_exists and senior_exists:
        junior = _load_landmarks_from_json(junior_path)
        senior = _load_landmarks_from_json(senior_path)
        averaged: dict[str, tuple[float, float]] = {}
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


def _generate_gaussian_heatmap(
    height: int,
    width: int,
    center_x: float,
    center_y: float,
    sigma: float,
) -> np.ndarray:
    """Generate a single 2-D Gaussian heatmap of shape (height, width)."""
    xs = np.arange(0, width, dtype=np.float32)
    ys = np.arange(0, height, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    heatmap = np.exp(
        -((xx - center_x) ** 2 + (yy - center_y) ** 2) / (2 * sigma ** 2)
    )
    return heatmap


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CropRefinementDataset(Dataset):
    """PyTorch Dataset for Stage 2 landmark refinement.

    For each (image, landmark) pair the dataset:
    1. Loads the full image and resizes to *image_size* × *image_size*.
    2. Centres a *crop_size* × *crop_size* crop on the Stage 1 coarse
       prediction (with optional jitter and zero-padding at boundaries).
    3. Generates a tight single-channel Gaussian heatmap at the ground-
       truth position relative to the crop.

    Parameters
    ----------
    data_root : Path | str
        Split directory (e.g. ``data/Aariz/train``).
    stage1_predictions : dict[str, np.ndarray]
        Mapping ``ceph_id`` → ``(num_landmarks, 2)`` array of coarse
        (x, y) coordinates in **resized** (``image_size`` × ``image_size``)
        space.
    image_size : int
        Square size that images were resized to for Stage 1.
    crop_size : int
        Side length of the crop extracted around each landmark.
    sigma : float
        Standard deviation of the Gaussian in the crop heatmap.
    jitter : int
        Maximum random offset (pixels) added to the crop centre during
        training.  Set to 0 to disable.
    selected_symbols : list[str] | None
        Landmark symbols to include.  Defaults to ``ALL_LANDMARK_SYMBOLS``.
    """

    def __init__(
        self,
        data_root: Path | str,
        stage1_predictions: dict[str, np.ndarray],
        image_size: int = 512,
        crop_size: int = 96,
        sigma: float = 2.0,
        jitter: int = 0,
        selected_symbols: Optional[list[str]] = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.stage1_predictions = stage1_predictions
        self.image_size = image_size
        self.crop_size = crop_size
        self.sigma = sigma
        self.jitter = jitter
        self.selected_symbols = selected_symbols or ALL_LANDMARK_SYMBOLS

        # Annotation directories
        self.image_dir = self.data_root / "Cephalograms"
        self.junior_dir = (
            self.data_root / "Annotations" / "Cephalometric Landmarks"
            / "Junior Orthodontists"
        )
        self.senior_dir = (
            self.data_root / "Annotations" / "Cephalometric Landmarks"
            / "Senior Orthodontists"
        )

        # Build per-(image, landmark) sample list
        # Each entry: (image_path, ceph_id, landmark_idx, gt_x_resized, gt_y_resized)
        self.samples: list[tuple[Path, str, int, float, float]] = []

        image_extensions = {".png", ".jpg", ".jpeg", ".bmp"}
        for img_path in sorted(self.image_dir.iterdir()):
            if img_path.suffix.lower() not in image_extensions:
                continue
            ceph_id = img_path.stem

            # Must have a Stage 1 prediction for this image
            if ceph_id not in self.stage1_predictions:
                continue

            # Need at least one annotation file
            junior_json = self.junior_dir / f"{ceph_id}.json"
            senior_json = self.senior_dir / f"{ceph_id}.json"
            if not junior_json.exists() and not senior_json.exists():
                continue

            # Load averaged GT annotations (original pixel coords)
            all_landmarks = _average_annotations(junior_json, senior_json)

            # Load original image size for coordinate scaling
            with Image.open(img_path) as img_tmp:
                orig_w, orig_h = img_tmp.size
            scale_x = self.image_size / orig_w
            scale_y = self.image_size / orig_h

            for lm_idx, sym in enumerate(self.selected_symbols):
                if sym in all_landmarks:
                    ox, oy = all_landmarks[sym]
                    gt_x = ox * scale_x
                    gt_y = oy * scale_y
                else:
                    # Sentinel: landmark missing from annotation
                    gt_x, gt_y = -1.0, -1.0

                self.samples.append(
                    (img_path, ceph_id, lm_idx, gt_x, gt_y)
                )

    # -----------------------------------------------------------------
    # Dataset interface
    # -----------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        """Return a single crop sample.

        Returns
        -------
        dict
            crop : Tensor [1, crop_size, crop_size]
                Greyscale crop normalised to [0, 1].
            heatmap : Tensor [1, crop_size, crop_size]
                Single-channel Gaussian target (zeros if invalid).
            landmark_idx : int
                Index of the landmark within ``selected_symbols``.
            offset : Tensor [2]
                ``(crop_x_start, crop_y_start)`` in resized image coords.
            valid : bool
                ``False`` when the GT is missing or falls outside the crop.
        """
        img_path, ceph_id, lm_idx, gt_x, gt_y = self.samples[idx]

        # 1. Load greyscale image and resize to (image_size, image_size)
        img = Image.open(img_path).convert("L")
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        img_np = np.array(img, dtype=np.float32)  # (H, W) in [0, 255]

        # 2. Stage 1 coarse prediction for this landmark
        coarse_coords = self.stage1_predictions[ceph_id]  # (num_landmarks, 2)
        coarse_x = float(coarse_coords[lm_idx, 0])
        coarse_y = float(coarse_coords[lm_idx, 1])

        # 3. Optional random jitter
        if self.jitter > 0:
            coarse_x += random.randint(-self.jitter, self.jitter)
            coarse_y += random.randint(-self.jitter, self.jitter)

        # 4. Compute crop bounds (may extend outside the image)
        half = self.crop_size // 2
        cx = int(round(coarse_x))
        cy = int(round(coarse_y))

        crop_x_start = cx - half
        crop_y_start = cy - half
        crop_x_end = crop_x_start + self.crop_size
        crop_y_end = crop_y_start + self.crop_size

        # 5. Extract crop with zero-padding at edges
        crop = np.zeros((self.crop_size, self.crop_size), dtype=np.float32)

        # Source region clamped to image bounds
        src_x0 = max(crop_x_start, 0)
        src_y0 = max(crop_y_start, 0)
        src_x1 = min(crop_x_end, self.image_size)
        src_y1 = min(crop_y_end, self.image_size)

        # Destination offsets inside the crop array
        dst_x0 = src_x0 - crop_x_start
        dst_y0 = src_y0 - crop_y_start
        dst_x1 = dst_x0 + (src_x1 - src_x0)
        dst_y1 = dst_y0 + (src_y1 - src_y0)

        if src_x1 > src_x0 and src_y1 > src_y0:
            crop[dst_y0:dst_y1, dst_x0:dst_x1] = img_np[src_y0:src_y1, src_x0:src_x1]

        # 6. GT coordinate in crop-local space
        local_x = gt_x - crop_x_start
        local_y = gt_y - crop_y_start

        # 7. Determine validity
        valid = True
        if gt_x < 0 or gt_y < 0:
            # Missing landmark (sentinel -1)
            valid = False
        elif (
            local_x < 0
            or local_y < 0
            or local_x >= self.crop_size
            or local_y >= self.crop_size
        ):
            # GT fell outside the crop
            valid = False

        # 8. Generate heatmap
        if valid:
            heatmap = _generate_gaussian_heatmap(
                self.crop_size, self.crop_size, local_x, local_y, self.sigma,
            )
        else:
            heatmap = np.zeros(
                (self.crop_size, self.crop_size), dtype=np.float32,
            )

        # 9. Normalise crop to [0, 1]
        crop = crop / 255.0

        # 10. Convert to tensors
        crop_tensor = torch.from_numpy(crop).unsqueeze(0)      # (1, H, W)
        heatmap_tensor = torch.from_numpy(heatmap).unsqueeze(0) # (1, H, W)
        offset_tensor = torch.tensor(
            [crop_x_start, crop_y_start], dtype=torch.float32,
        )

        return {
            "crop": crop_tensor,
            "heatmap": heatmap_tensor,
            "landmark_idx": lm_idx,
            "offset": offset_tensor,
            "valid": valid,
        }


# ---------------------------------------------------------------------------
# Stage 1 prediction generator
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_stage1_predictions(
    model: torch.nn.Module,
    data_root: Path | str,
    image_size: int,
    device: torch.device,
    selected_symbols: Optional[list[str]] = None,
) -> dict[str, np.ndarray]:
    """Run Stage 1 model on all images to get coarse predictions.

    Returns dict mapping ``ceph_id`` to ``(num_landmarks, 2)`` array of
    ``(x, y)`` coordinates in resized image space.

    Parameters
    ----------
    model : torch.nn.Module
        Trained Stage 1 heatmap regression model.
    data_root : Path | str
        Split directory (e.g. ``data/Aariz/train``).
    image_size : int
        Square resize dimension matching Stage 1 training.
    device : torch.device
        Device to run inference on.
    selected_symbols : list[str] | None
        Landmark symbols the model was trained on.  Defaults to
        ``ALL_LANDMARK_SYMBOLS``.
    """
    from torch.utils.data import DataLoader

    from src.data.dataset import CephalometricDataset
    from src.inference.predict import heatmap_to_coordinates

    selected_symbols = selected_symbols or ALL_LANDMARK_SYMBOLS

    # Build dataset with validation (deterministic) transforms
    dataset = CephalometricDataset(
        data_root=data_root,
        image_size=image_size,
        split="val",
        selected_symbols=selected_symbols,
    )

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    model.eval()
    predictions: dict[str, np.ndarray] = {}

    for batch in loader:
        images = batch["image"].to(device)          # (1, 1, H, W)
        ceph_id = batch["meta"]["ceph_id"][0]       # str (un-batched from list)

        output = model(images)                      # (1, C, H, W)
        coords = heatmap_to_coordinates(output)     # (1, C, 2)
        coords_np = coords[0].cpu().numpy()         # (C, 2)

        predictions[ceph_id] = coords_np

    return predictions
