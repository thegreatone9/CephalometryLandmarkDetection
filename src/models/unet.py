"""
U-Net model factory using *segmentation_models_pytorch*.

Provides helpers to instantiate a U-Net with a configurable encoder
and to create parameter groups for differential learning rates
(slower encoder, faster decoder).
"""

from __future__ import annotations

from typing import Any

import segmentation_models_pytorch as smp
import torch.nn as nn


def create_model(
    encoder_name: str = "resnet34",
    encoder_weights: str | None = "imagenet",
    in_channels: int = 1,
    num_classes: int = 6,
) -> nn.Module:
    """Instantiate a U-Net model via *segmentation_models_pytorch*.

    Parameters
    ----------
    encoder_name : str
        Name of the encoder backbone (e.g. ``"resnet34"``, ``"efficientnet-b0"``).
    encoder_weights : str | None
        Pre-trained weight identifier.  ``"imagenet"`` or ``None``.
    in_channels : int
        Number of input image channels (1 for grayscale).
    num_classes : int
        Number of output channels (one per landmark).

    Returns
    -------
    nn.Module
        The U-Net model.
    """
    model: nn.Module = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes,
        decoder_attention_type="scse",
    )
    return model


def get_parameter_groups(
    model: nn.Module,
    encoder_lr: float = 1e-4,
    decoder_lr: float = 1e-3,
) -> list[dict[str, Any]]:
    """Create parameter groups with differential learning rates.

    The encoder (pre-trained backbone) uses a lower learning rate to
    preserve transferred features, while the decoder head trains faster.

    Parameters
    ----------
    model : nn.Module
        A *segmentation_models_pytorch* U-Net (must have ``encoder`` and
        ``decoder`` / ``segmentation_head`` attributes).
    encoder_lr : float
        Learning rate for encoder parameters.
    decoder_lr : float
        Learning rate for decoder + segmentation head parameters.

    Returns
    -------
    list[dict]
        Parameter groups suitable for ``torch.optim.Adam``.
    """
    encoder_params = list(model.encoder.parameters())
    decoder_params = (
        list(model.decoder.parameters())
        + list(model.segmentation_head.parameters())
    )

    return [
        {"params": encoder_params, "lr": encoder_lr},
        {"params": decoder_params, "lr": decoder_lr},
    ]
