"""
Adaptive Wing Loss for heatmap-based landmark regression.

Implements the loss from:
    Wang et al., "Adaptive Wing Loss for Robust Face Alignment via
    Heatmap Regression", ICCV 2019.

Standard MSE treats every pixel equally. Adaptive Wing Loss uses a
logarithmic branch near zero (for fine-grained gradient signal) and a
linear branch further away (to avoid gradient explosion), with the
transition shaped by the ground-truth heatmap value at each pixel.

A foreground weighting map is applied so that landmark pixels (where
the Gaussian target > 0) receive substantially more gradient than the
background.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Adaptive Wing Loss
# ---------------------------------------------------------------------------

class AdaptiveWingLoss(nn.Module):
    """Adaptive Wing Loss for heatmap regression.

    Parameters
    ----------
    omega : float
        Multiplicative scale for the logarithmic branch.  Controls the
        overall magnitude of the loss curve (default ``14``).
    theta : float
        Threshold that separates the non-linear (log) region from the
        linear region.  Errors below *theta* use the log branch
        (default ``0.5``).
    epsilon : float
        Curvature control inside the log branch.  Prevents the gradient
        from becoming too large near zero error (default ``1``).
    alpha : float
        Exponent offset.  The effective exponent at each pixel is
        ``alpha - y_target``, making the loss adaptive to the local
        heatmap value (default ``2.1``).
    """

    def __init__(
        self,
        omega: float = 14,
        theta: float = 0.5,
        epsilon: float = 1,
        alpha: float = 2.1,
    ) -> None:
        super().__init__()
        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha

    # ------------------------------------------------------------------

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the weighted Adaptive Wing Loss.

        Parameters
        ----------
        pred : torch.Tensor
            Predicted heatmaps, shape ``(B, C, H, W)``.
        target : torch.Tensor
            Ground-truth heatmaps, same shape as *pred*.

        Returns
        -------
        torch.Tensor
            Scalar mean loss.
        """
        # Per-pixel error
        delta = (target - pred).abs()

        # Adaptive exponent: alpha - y_target  (clamped for stability)
        exp = self.alpha - target
        # Clamp to avoid negative or zero exponents in pow / log
        exp = exp.clamp(min=0.1)

        # ----------------------------------------------------------
        # Pre-compute A and C (depend on theta, epsilon, target)
        # ----------------------------------------------------------
        theta_over_eps = self.theta / self.epsilon  # scalar

        # (theta / epsilon) ^ (alpha - y_target)
        # Adding a tiny constant inside the base keeps pow stable for
        # very small theta / epsilon ratios.
        base = torch.tensor(
            theta_over_eps, dtype=pred.dtype, device=pred.device,
        ).clamp(min=1e-8)
        pow_term = base.pow(exp)                         # same shape as exp
        pow_term_m1 = base.pow(exp - 1.0)                # exponent - 1

        # A = omega / (1 + pow_term) * (alpha - y) * pow_term_m1 / epsilon
        denom = 1.0 + pow_term
        A = self.omega * (1.0 / denom) * exp * pow_term_m1 * (1.0 / self.epsilon)

        # C = theta * A - omega * ln(1 + pow_term)
        C = self.theta * A - self.omega * torch.log1p(pow_term)

        # ----------------------------------------------------------
        # Two-branch loss
        # ----------------------------------------------------------
        # Log branch:  omega * ln(1 + |delta / epsilon| ^ (alpha - y))
        inside = 1.0 + (delta / self.epsilon).clamp(min=1e-8).pow(exp)
        loss_log = self.omega * torch.log(inside.clamp(min=1e-8))

        # Linear branch: A * |delta| - C
        loss_linear = A * delta - C

        awl = torch.where(delta < self.theta, loss_log, loss_linear)

        # ----------------------------------------------------------
        # Foreground weighting
        # ----------------------------------------------------------
        w_foreground = 10.0
        weight_map = torch.ones_like(target)
        weight_map = weight_map + w_foreground * (target > 0).float()

        loss = awl * weight_map

        return loss.mean(dim=(0, 2, 3)).mean()
