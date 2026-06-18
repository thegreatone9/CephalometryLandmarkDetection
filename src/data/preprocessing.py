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
    - Random brightness / contrast jitter
    - Gaussian noise injection
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
            A.RandomBrightnessContrast(
                brightness_limit=0.15,
                contrast_limit=0.15,
                p=0.5,
            ),
            A.GaussNoise(
                std_range=(0.01, 0.05),
                p=0.3,
            ),
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
