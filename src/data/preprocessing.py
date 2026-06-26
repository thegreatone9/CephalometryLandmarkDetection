"""
Augmentation pipelines for cephalometric landmark detection.

Uses *albumentations* with keypoint-aware transforms so that image
augmentations and landmark coordinate updates stay synchronised.

**No horizontal flip** is applied because cephalometric landmarks are
anatomically asymmetric (left-right has clinical meaning).
"""

from __future__ import annotations

import albumentations as A
from albumentations.core.composition import Compose


def get_train_transforms(image_size: int = 512) -> Compose:
    """Return the training augmentation pipeline.

    Augmentations applied:
    - Resize to ``(image_size, image_size)``
    - Random rotation ±15°
    - Random shift (±5 %) and scale (±10 %)
    - Elastic deformation
    - Random brightness / contrast jitter
    - CLAHE adaptive histogram equalisation
    - Gaussian noise injection
    - Gaussian blur
    - No horizontal flip (anatomy is asymmetric)

    Keypoints are tracked via the ``"xy"`` format.

    Parameters
    ----------
    image_size : int
        Target square image size.

    Returns
    -------
    Compose
        Albumentations pipeline.
    """
    return A.Compose(
        [
            A.Resize(height=image_size, width=image_size),
            A.Rotate(limit=15, border_mode=0, p=0.5),
            A.Affine(
                translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                scale=(0.9, 1.1),
                rotate=0,
                border_mode=0,
                p=0.5,
            ),
            A.ElasticTransform(alpha=30, sigma=5, p=0.2),
            A.RandomBrightnessContrast(
                brightness_limit=0.15,
                contrast_limit=0.15,
                p=0.5,
            ),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),
            A.GaussNoise(
                std_range=(0.01, 0.05),
                p=0.3,
            ),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        ],
        keypoint_params=A.KeypointParams(
            format="xy",
            remove_invisible=False,
        ),
    )


def get_val_transforms(image_size: int = 512) -> Compose:
    """Return the validation / test transform pipeline (resize only).

    Parameters
    ----------
    image_size : int
        Target square image size.

    Returns
    -------
    Compose
        Albumentations pipeline.
    """
    return A.Compose(
        [
            A.Resize(height=image_size, width=image_size),
        ],
        keypoint_params=A.KeypointParams(
            format="xy",
            remove_invisible=False,
        ),
    )
