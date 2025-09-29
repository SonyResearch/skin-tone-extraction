"""Formulas to calculate derivative metrics e.g. ITA / hue angle."""

import numpy as np


def get_ita(lum: np.ndarray, lab_b: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Calculate the Individual Typology Angle (ITA) in degrees.

    ITA = arctan((L* - 50) / b*) * (180 / pi)

    Args:
        lum: Array of L* (luminance) values in CIELAB.
        lab_b: Array of b* values in CIELAB.
        eps: Small value to avoid division by zero.

    Returns:
        Array of ITA values in degrees.
    """
    return np.degrees(np.arctan((lum - 50) / (lab_b + eps)))


def get_corrected_ita(
    lum: np.ndarray, lab_b: np.ndarray, eps: float = 1e-8
) -> np.ndarray:
    """Calculate the corrected Individual Typology Angle (cITA) in degrees.

    ITA = arctan((L* - 50) / |b*|) * (180 / pi)

    Args:
        lum: Array of L* (luminance) values in CIELAB.
        lab_b: Array of b* values in CIELAB.
        eps: Small value to avoid division by zero.

    Returns:
        Array of cITA values in degrees.
    """
    return np.degrees(np.arctan((lum - 50) / (np.abs(lab_b) + eps)))


def get_hue(lab_a: np.ndarray, lab_b: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Compute hue angle in CIELAB color space.

    Args:
        lab_a: Array of a* values in CIELAB.
        lab_b: Array of b* values in CIELAB.
        eps: Small value to avoid division by zero.

    Returns:
        Array of hue angle values in degrees.
    """
    return np.degrees(np.arctan(lab_b / (lab_a + eps)))
