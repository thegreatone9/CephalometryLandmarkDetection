"""Lightweight CNN for Stage-2 landmark refinement.

Takes a small grayscale crop (96×96) centred on a coarse landmark prediction
and produces a single-channel heatmap whose peak indicates the refined
landmark position.

Architecture (~200 K params)
----------------------------
Initial conv  → ResBlock(32→32) → ResBlock(32→64, stride 2)
→ ResBlock(64→64) → bilinear upsample → conv 64→32 → conv 32→1 → sigmoid
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Basic residual block with two 3×3 convolutions and an optional skip
    projection for channel / spatial dimension mismatches.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int, optional
        Stride for the first convolution.  When ``stride > 1`` the spatial
        dimensions are down-sampled and a 1×1 projection is used on the
        skip connection.  Default: ``1``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
    ) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        # Skip projection when channels or spatial size change.
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(B, C_in, H, W)``.

        Returns
        -------
        torch.Tensor
            Output tensor of shape ``(B, C_out, H', W')``.
        """
        identity = self.skip(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + identity)
        return out


class LandmarkRefiner(nn.Module):
    """Lightweight residual CNN that refines a single landmark location.

    The network consumes a small grayscale crop around a coarse landmark
    prediction and outputs a heatmap of the same spatial size whose peak
    corresponds to the refined landmark position.

    Parameters
    ----------
    in_channels : int, optional
        Number of input channels (``1`` for grayscale).  Default: ``1``.
    base_channels : int, optional
        Number of channels after the initial convolution.  Default: ``32``.
    """

    def __init__(self, in_channels: int = 1, base_channels: int = 32) -> None:
        super().__init__()

        mid_channels = base_channels * 2  # 64

        # --- Encoder -----------------------------------------------------------
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )

        self.block1 = ResidualBlock(base_channels, base_channels, stride=1)
        self.block2 = ResidualBlock(base_channels, mid_channels, stride=2)   # 96→48
        self.block3 = ResidualBlock(mid_channels, mid_channels, stride=1)

        # --- Decoder -----------------------------------------------------------
        self.decode = nn.Sequential(
            nn.Conv2d(mid_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )

        self.head = nn.Sequential(
            nn.Conv2d(base_channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Produce a refined heatmap from a coarse landmark crop.

        Parameters
        ----------
        x : torch.Tensor
            Grayscale crop of shape ``(B, 1, 96, 96)``.

        Returns
        -------
        torch.Tensor
            Heatmap of shape ``(B, 1, 96, 96)`` with values in ``[0, 1]``.
        """
        h, w = x.shape[2], x.shape[3]

        out = self.initial(x)       # (B, 32, 96, 96)
        out = self.block1(out)      # (B, 32, 96, 96)
        out = self.block2(out)      # (B, 64, 48, 48)
        out = self.block3(out)      # (B, 64, 48, 48)

        # Upsample back to original spatial size.
        out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=False)

        out = self.decode(out)      # (B, 32, 96, 96)
        out = self.head(out)        # (B,  1, 96, 96)
        return out


def create_refiner(in_channels: int = 1, base_channels: int = 32) -> nn.Module:
    """Factory function for :class:`LandmarkRefiner`.

    Parameters
    ----------
    in_channels : int, optional
        Number of input channels.  Default: ``1``.
    base_channels : int, optional
        Base channel width.  Default: ``32``.

    Returns
    -------
    nn.Module
        An initialised :class:`LandmarkRefiner` instance.
    """
    return LandmarkRefiner(in_channels=in_channels, base_channels=base_channels)
