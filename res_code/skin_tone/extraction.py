"""Base classes for extraction methods."""

import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np

from res_code.skin_tone.helpers import (
    calculate_average,
    calculate_mode,
    prepare_skin_tone_columns,
)
from res_code.skin_tone.image import MaskedImage
from res_code.skin_tone.visualizer import Visualizer


class ExtractionResult:
    """Container for skin tone extraction results."""

    def __init__(
        self, measurements: Dict[str, Any], method: str = "", image_id: str = ""
    ):
        """Initialize extraction result.

        Args:
            measurements: Dictionary containing skin tone measurements
            method: Name of the extraction method used
            image_id: Identifier of the processed image
        """
        self.measurements = measurements
        self.method = method
        self.image_id = image_id

    def get(self, key: str, default: Any = None) -> Any:
        """Get a measurement value by key."""
        return self.measurements.get(key, default)

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access to measurements."""
        return self.measurements[key]

    def __contains__(self, key: str) -> bool:
        """Check if a measurement key exists."""
        return key in self.measurements

    def keys(self):
        """Return measurement keys."""
        return self.measurements.keys()

    def values(self):
        """Return measurement values."""
        return self.measurements.values()

    def items(self):
        """Return measurement items."""
        return self.measurements.items()


class ExtractionMethod(ABC):
    """Base class for skin tone extraction methods."""

    method_id: str

    def __init__(self, debug: bool = False, debug_image_dir: str | None = None):
        """Initialize extraction method.

        Args:
            debug: If True, output debug visualizations and information.
            debug_image_dir: Directory to save visualizations images (if debug is True).
        """
        self.debug = debug
        self.visualizer = Visualizer()
        self.visualizer.prefix = self.method_id + "_"
        if debug_image_dir is not None:
            self.visualizer.output_dir = debug_image_dir
            if not debug:
                warnings.warn(
                    "debug_image_dir is set but debug is False. "
                    "No debug images will be saved.",
                    UserWarning,
                )

    def extract(self, image: MaskedImage) -> ExtractionResult:
        """Extract skin tone measurements from a MaskedImage."""
        self.visualizer.image_id = image.id

        self.image = image

        return self._extract()

    @abstractmethod
    def _extract(self) -> ExtractionResult:
        """Actual extraction logic to be implemented by subclasses."""
        pass

    def calculate_mode_skin_tone(
        self, skin: np.ndarray, debug_mask: np.ndarray | None = None
    ):
        """Calculate mode skin tone measurements from RGB skin pixel data.

        Args:
            skin: Array of RGB skin pixel values with shape (n_pixels, 3).
            debug_mask: Binary mask to apply during visualization.

        Returns:
            Dictionary containing mode values for various skin tone measurements.
        """
        columns = prepare_skin_tone_columns(skin)

        if self.debug and debug_mask is not None:
            self.visualizer.visualize_metrics(
                columns,
                debug_mask,
            )

        return calculate_mode(columns, visualizer=self.visualizer)

    def calculate_average_skin_tone(
        self,
        skin: np.ndarray,
        average_first: bool = False,
        debug_mask: np.ndarray = None,
    ):
        """Calculate average skin tone measurements from RGB skin pixel data.

        Args:
            skin: Array of RGB skin pixel values with shape (n_pixels, 3).
            average_first: If True, average the RGB values first before calculating
                measurements. If False, calculate measurements for each pixel then
                average.
            debug_mask: Binary mask to apply during visualization.

        Returns:
            Dictionary containing average values for various skin tone measurements.
        """
        if average_first:
            skin = np.mean(skin, axis=0, keepdims=True)
        columns = prepare_skin_tone_columns(skin)
        if not average_first:
            # Only visualize metrics if they aren't averaged before conversion
            if self.debug and debug_mask is not None:
                self.visualizer.visualize_metrics(
                    columns,
                    debug_mask,
                )

            # Calculate average across all columns
            columns = calculate_average(columns)
        else:
            # Convert to scalars
            for key in columns:
                assert isinstance(columns[key], np.ndarray) and columns[key].size == 1
                columns[key] = columns[key][0]
        return columns
