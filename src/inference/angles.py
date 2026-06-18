"""
Cephalometric angle computations.

Implements SNA, SNB, and ANB angle calculations from predicted
landmark coordinates, along with a plain-English interpretation
of the ANB angle for clinical context.
"""

from __future__ import annotations

import math

import numpy as np


def compute_angle(
    point_a: np.ndarray,
    vertex: np.ndarray,
    point_b: np.ndarray,
) -> float:
    """Compute the angle at *vertex* formed by rays to *point_a* and *point_b*.

    Uses ``atan2`` so the result is always in [0, 180] degrees.

    Parameters
    ----------
    point_a, vertex, point_b : np.ndarray
        Each is a 1-D array of shape ``(2,)`` with (x, y).

    Returns
    -------
    float
        Angle in degrees.
    """
    va = point_a - vertex
    vb = point_b - vertex

    angle_a = math.atan2(va[1], va[0])
    angle_b = math.atan2(vb[1], vb[0])

    angle = abs(angle_a - angle_b)
    if angle > math.pi:
        angle = 2 * math.pi - angle

    return math.degrees(angle)


def compute_sna(
    sella: np.ndarray,
    nasion: np.ndarray,
    a_point: np.ndarray,
) -> float:
    """Compute the SNA angle (angle at Nasion between Sella and A-point).

    Normal range: ~82° ± 2°.

    Parameters
    ----------
    sella, nasion, a_point : np.ndarray
        Landmark coordinates, each shape ``(2,)``.

    Returns
    -------
    float
        SNA angle in degrees.
    """
    return compute_angle(sella, nasion, a_point)


def compute_snb(
    sella: np.ndarray,
    nasion: np.ndarray,
    b_point: np.ndarray,
) -> float:
    """Compute the SNB angle (angle at Nasion between Sella and B-point).

    Normal range: ~80° ± 2°.

    Parameters
    ----------
    sella, nasion, b_point : np.ndarray
        Landmark coordinates, each shape ``(2,)``.

    Returns
    -------
    float
        SNB angle in degrees.
    """
    return compute_angle(sella, nasion, b_point)


def compute_anb(sna: float, snb: float) -> float:
    """Compute the ANB angle as SNA − SNB.

    Normal range: ~2°.

    Parameters
    ----------
    sna, snb : float
        SNA and SNB angles in degrees.

    Returns
    -------
    float
        ANB angle in degrees.
    """
    return sna - snb


def interpret_anb(anb_value: float) -> str:
    """Return a plain-English clinical interpretation of an ANB angle.

    Parameters
    ----------
    anb_value : float
        ANB angle in degrees.

    Returns
    -------
    str
        Human-readable interpretation string.
    """
    if anb_value < 0:
        return (
            f"ANB = {anb_value:.1f}°: Class III skeletal relationship. "
            "The mandible is positioned anteriorly relative to the maxilla "
            "(skeletal underbite / mandibular prognathism)."
        )
    elif anb_value <= 1:
        return (
            f"ANB = {anb_value:.1f}°: Borderline Class I / Class III. "
            "Near-normal relationship with slight mandibular tendency."
        )
    elif anb_value <= 4:
        return (
            f"ANB = {anb_value:.1f}°: Class I skeletal relationship (normal). "
            "The maxilla and mandible are in harmonious anteroposterior relation."
        )
    elif anb_value <= 7:
        return (
            f"ANB = {anb_value:.1f}°: Mild Class II skeletal relationship. "
            "The maxilla is moderately anterior to the mandible."
        )
    else:
        return (
            f"ANB = {anb_value:.1f}°: Class II skeletal relationship. "
            "Significant maxillary protrusion or mandibular retrusion "
            "(skeletal overbite)."
        )
