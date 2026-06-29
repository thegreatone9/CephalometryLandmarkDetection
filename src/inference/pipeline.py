"""Two-stage coarse-to-fine landmark detection pipeline.

Combines a Stage 1 U-Net (coarse detection on full image) with a
Stage 2 refiner (precise localisation on crops) for end-to-end
cephalometric landmark prediction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import (
    ALL_LANDMARK_SYMBOLS,
    NUM_LANDMARKS,
    SELECTED_DISPLAY_NAMES,
)
from src.evaluation import compute_mre, compute_sdr
from src.inference.predict import (
    heatmap_to_coordinates,
    refine_coordinates,
    rescale_coordinates,
)
from src.models.refiner import create_refiner
from src.models.unet import create_model


def _extract_crop(
    image: torch.Tensor,
    center_x: int,
    center_y: int,
    crop_size: int,
) -> tuple[torch.Tensor, int, int]:
    """Extract a crop from the image with zero-padding at edges.

    Parameters
    ----------
    image : Tensor
        Shape ``(1, H, W)`` — single grayscale image.
    center_x, center_y : int
        Centre of the crop in pixel coordinates.
    crop_size : int
        Side length of the square crop.

    Returns
    -------
    tuple
        ``(crop, x_start, y_start)`` where crop has shape
        ``(1, crop_size, crop_size)`` and ``(x_start, y_start)``
        is the top-left corner of the crop in the original image.
    """
    _, H, W = image.shape
    half = crop_size // 2

    # Compute crop bounds in the original image
    x_start = center_x - half
    y_start = center_y - half
    x_end = x_start + crop_size
    y_end = y_start + crop_size

    # Compute valid region (clipped to image bounds)
    src_x_start = max(0, x_start)
    src_y_start = max(0, y_start)
    src_x_end = min(W, x_end)
    src_y_end = min(H, y_end)

    # Compute where the valid region maps to in the crop
    dst_x_start = src_x_start - x_start
    dst_y_start = src_y_start - y_start
    dst_x_end = dst_x_start + (src_x_end - src_x_start)
    dst_y_end = dst_y_start + (src_y_end - src_y_start)

    crop = torch.zeros(1, crop_size, crop_size, device=image.device)
    crop[
        :,
        dst_y_start:dst_y_end,
        dst_x_start:dst_x_end,
    ] = image[:, src_y_start:src_y_end, src_x_start:src_x_end]

    return crop, x_start, y_start


class TwoStagePipeline:
    """End-to-end two-stage landmark detection pipeline.

    Parameters
    ----------
    stage1_checkpoint : Path | str
        Path to the trained Stage 1 (U-Net) model checkpoint.
    stage2_checkpoint : Path | str
        Path to the trained Stage 2 (refiner) model checkpoint.
    device : torch.device
        Compute device.
    num_landmarks : int
        Number of landmarks (default 29 for all Aariz landmarks).
    crop_size : int
        Crop size for Stage 2 refinement.
    image_size : int
        Image size used by Stage 1.
    """

    def __init__(
        self,
        stage1_checkpoint: Path | str,
        stage2_checkpoint: Path | str,
        device: torch.device,
        num_landmarks: int = NUM_LANDMARKS,
        crop_size: int = 96,
        image_size: int = 512,
    ) -> None:
        self.device = device
        self.crop_size = crop_size
        self.image_size = image_size
        self.num_landmarks = num_landmarks

        # Load Stage 1 model
        self.stage1 = create_model(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=1,
            num_classes=num_landmarks,
        )
        s1_checkpoint = torch.load(stage1_checkpoint, map_location="cpu")
        if isinstance(s1_checkpoint, dict) and "model_state_dict" in s1_checkpoint:
            self.stage1.load_state_dict(s1_checkpoint["model_state_dict"])
        else:
            self.stage1.load_state_dict(s1_checkpoint)
        self.stage1.to(device)
        self.stage1.eval()

        # Load Stage 2 model
        self.stage2 = create_refiner(in_channels=1, base_channels=32)
        s2_state = torch.load(stage2_checkpoint, map_location="cpu")
        self.stage2.load_state_dict(s2_state)
        self.stage2.to(device)
        self.stage2.eval()

    @torch.no_grad()
    def predict(self, image: torch.Tensor) -> np.ndarray:
        """Run two-stage prediction on a single image.

        Parameters
        ----------
        image : Tensor
            Shape ``(1, 1, H, W)`` — preprocessed grayscale image.

        Returns
        -------
        np.ndarray
            Shape ``(num_landmarks, 2)`` with ``(x, y)`` coordinates
            in the resized image space.
        """
        # Stage 1: coarse detection
        heatmaps = self.stage1(image.to(self.device))  # (1, C, H, W)
        coarse_coords = heatmap_to_coordinates(heatmaps[0])  # (C, 2)
        coarse_np = coarse_coords.cpu().numpy()

        img_single = image[0]  # (1, H, W)
        refined_coords = np.zeros((self.num_landmarks, 2), dtype=np.float32)

        # Stage 2: refine each landmark
        for i in range(self.num_landmarks):
            cx = int(coarse_np[i, 0])
            cy = int(coarse_np[i, 1])

            crop, x_start, y_start = _extract_crop(
                img_single.to(self.device), cx, cy, self.crop_size,
            )

            # Run refiner
            crop_batch = crop.unsqueeze(0)  # (1, 1, crop_size, crop_size)
            refined_hm = self.stage2(crop_batch)  # (1, 1, crop_size, crop_size)

            # Extract peak from refined heatmap
            refined_hm_np = refined_hm[0, 0].cpu().numpy()
            peak_idx = refined_hm_np.argmax()
            local_y, local_x = divmod(peak_idx, self.crop_size)

            # Map back to full image space
            refined_coords[i, 0] = x_start + local_x
            refined_coords[i, 1] = y_start + local_y

        return refined_coords


@torch.no_grad()
def evaluate_pipeline(
    pipeline: TwoStagePipeline,
    dataloader: DataLoader,
    device: torch.device,
    landmark_names: Optional[list[str]] = None,
) -> dict:
    """Evaluate the two-stage pipeline on a dataset.

    Parameters
    ----------
    pipeline : TwoStagePipeline
        Initialised two-stage pipeline.
    dataloader : DataLoader
        Evaluation data loader (batch_size=1 recommended).
    device : torch.device
        Compute device.
    landmark_names : list[str] | None
        Names for per-landmark reporting.

    Returns
    -------
    dict
        Contains ``"mre_per_landmark"``, ``"mre_overall"``, ``"sdr"``,
        ``"all_pred_coords"``, ``"all_gt_coords"``.
    """
    landmark_names = landmark_names or list(SELECTED_DISPLAY_NAMES)

    all_preds: list[np.ndarray] = []
    all_gts: list[np.ndarray] = []
    all_spacings: list[float] = []

    for batch in dataloader:
        images = batch["image"]  # (B, 1, H, W)
        gt_landmarks = batch["landmarks"].numpy()  # (B, C, 2)
        meta = batch["meta"]

        for i in range(images.size(0)):
            img = images[i : i + 1]  # (1, 1, H, W)

            # Two-stage prediction
            pred_coords = pipeline.predict(img)  # (C, 2)

            # Rescale to original resolution
            orig_size = (
                meta["original_size"][0][i].item(),
                meta["original_size"][1][i].item(),
            )
            resized_size = (
                meta["resized_size"][0][i].item(),
                meta["resized_size"][1][i].item(),
            )
            pred_rescaled = rescale_coordinates(
                pred_coords, orig_size, resized_size,
            )

            gt_i = gt_landmarks[i]
            gt_rescaled = rescale_coordinates(gt_i, orig_size, resized_size)

            all_preds.append(pred_rescaled)
            all_gts.append(gt_rescaled)

            spacing = meta["pixel_spacing"][i].item()
            all_spacings.append(spacing)

    pred_array = np.stack(all_preds)
    gt_array = np.stack(all_gts)
    spacings_array = np.array(all_spacings, dtype=np.float32)

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
