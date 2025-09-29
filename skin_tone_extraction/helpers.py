"""Various smaller helper functions."""

import numpy as np
from skimage.color import rgb2lab

from skin_tone_extraction.image import MaskedImage
from skin_tone_extraction.metrics import get_corrected_ita, get_hue, get_ita
from skin_tone_extraction.visualizer import Visualizer


def mode_hist(x: np.ndarray, bins: int | str = "sturges") -> float:
    """Compute a histogram and return the mode (most frequent value).

    Args:
        x: Input array of values.
        bins: Method for determining histogram bins or number of bins
            (see numpy.histogram).
        plot: If True, plot the histogram.

    Returns:
        The mode value from the histogram.
    """
    hist, hist_bins = np.histogram(x, bins=bins)
    mode = hist_bins[hist.argmax()]

    return mode, hist_bins


def prepare_skin_tone_columns(skin):
    """Prepare comprehensive skin tone measurements from RGB skin pixel data.

    Args:
        skin: Array of RGB skin pixel values with shape (n_pixels, 3).

    Returns:
        Dictionary containing various skin tone measurements:
        - lum: L* values from CIELAB
        - lab_a: a* values from CIELAB
        - lab_b: b* values from CIELAB
        - hue: Hue angle values
        - ita: Individual Typology Angle values
        - cita: Corrected Individual Typology Angle values
        - red: Red channel values
        - green: Green channel values
        - blue: Blue channel values
    """
    if skin.size == 0:
        import warnings

        warnings.warn("Input 'skin' is empty. Returning empty arrays.")
        empty = np.empty(0)
        return {
            "lum": empty,
            "lab_a": empty,
            "lab_b": empty,
            "hue": empty,
            "ita": empty,
            "cita": empty,
            "red": empty,
            "green": empty,
            "blue": empty,
        }
    skin_lab = rgb2lab(skin)
    ita = get_ita(skin_lab[:, 0], skin_lab[:, 2])
    cita = get_corrected_ita(skin_lab[:, 0], skin_lab[:, 2])
    hue = get_hue(skin_lab[:, 1], skin_lab[:, 2])
    columns = {
        "lum": skin_lab[:, 0],
        "lab_a": skin_lab[:, 1],
        "lab_b": skin_lab[:, 2],
        "hue": hue,
        "ita": ita,
        "cita": cita,
        "red": skin[:, 0],
        "green": skin[:, 1],
        "blue": skin[:, 2],
    }
    return columns


def calculate_average(columns, std=True):
    """Calculate average values and optionally standard deviations for raw values.

    Args:
        columns: Dictionary of skin tone measurement arrays.
        std: Whether to also calculate standard deviations.

    Returns:
        Dictionary containing mean values and optionally standard deviations
        for each measurement type.
    """
    result = {}
    for key, values in columns.items():
        if len(values) == 0:
            result[key] = np.nan
            if std:
                result[f"{key}_std"] = np.nan
        else:
            result[key] = np.mean(values)
            if std:
                result[f"{key}_std"] = np.std(values)
    return result


def calculate_mode(columns, visualizer=None):
    """Calculate mode values for skin tone measurements.

    Args:
        columns: Dictionary of skin tone measurement arrays.
        visualizer: Optional Visualizer instance for displaying histograms.

    Returns:
        Dictionary containing mode values for each measurement type.
    """
    result = {}
    for key, values in columns.items():
        if len(values) == 0:
            result[key] = np.nan
        else:
            result[key], bins = mode_hist(values)

            if visualizer is not None:
                visualizer.visualize_mode(
                    metric_name=key, values=values, bins=bins, mode=result[key]
                )
    return result


def visualize_mask(
    img_path: str, mask_path: str, mask_value: int, bg_val: int = 128
) -> None:
    """Visualize only the masked pixels from the image for a given mask value."""
    img, binary_mask = MaskedImage(img_path, mask_path, mask_value).load()

    visualizer = Visualizer()
    visualizer.visualize_loaded_mask(
        img=img,
        binary_mask=binary_mask,
        label=f"Image Mask (mask_value: {mask_value})",
        bg_val=bg_val,
    )
