"""Skin tone extraction approach building on Krishnapriya et al.

Reference:
Krishnapriya, K. S., Pangelinan, G., King, M. C., & Bowyer, K. W. (2022). Analysis of
manual and automated skin tone assignments. In Proceedings of the IEEE/CVF Winter
Conference on Applications of Computer Vision (pp. 429-438).
"""

import numpy as np
from skimage.color import rgb2ycbcr

from skin_tone_extraction.extraction import ExtractionMethod, ExtractionResult


class KrishnapriyaMethod(ExtractionMethod):
    """Extract skin tone using an approach inspired by Krishnapriya et al. (2022)."""

    method_id = "krishnapriya"

    def __init__(
        self,
        cr_thresh: tuple[float, float] = (136.0, 173.0),
        cb_thresh: tuple[float, float] = (77.0, 127.0),
        average_before_conversion: bool = True,
        debug: bool = False,
        debug_image_dir: str = None,
    ):
        """Initialize the extraction method.

        Args:
            cr_thresh: Min and max thresholds for Cr (red-difference) component in
                YCbCr space.
            cb_thresh: Min and max thresholds for Cb (blue-difference) component in
                YCbCr space.
            average_before_conversion: If True, averages RGB values before converting to
                 final color space. If False, converts each pixel individually then
                 averages.
            debug: If True, output debug visualizations and information.
            debug_image_dir: Directory to save visualizations images (if debug is True).
        """
        super().__init__(debug=debug, debug_image_dir=debug_image_dir)
        self.cr_thresh = cr_thresh
        self.cb_thresh = cb_thresh
        self.average_before_conversion = average_before_conversion

    def _extract(self) -> ExtractionResult:
        """Extract skin tone using the Krishnapriya method.

        Returns:
            ExtractionResult: Results containing skin tone measurements.
        """
        img = self.image.img
        mask = self.image.mask

        if self.debug:
            self.visualizer.visualize_image(img, "Original")
            self.visualizer.visualize_loaded_mask(
                img=img,
                binary_mask=mask,
                label="Initial Mask",
                plot_name="initial_mask",
            )

        diagnostics = {}

        # Apply initial mask
        skin = img[mask] / 255.0

        # Convert to YCbCr color space
        skin_ycbcr = rgb2ycbcr(skin)
        cb = skin_ycbcr[:, 1]
        cr = skin_ycbcr[:, 2]

        # Filter skin pixels based on Cb and Cr thresholds
        filter_mask = (
            (cr >= self.cr_thresh[0])
            & (cr <= self.cr_thresh[1])
            & (cb >= self.cb_thresh[0])
            & (cb <= self.cb_thresh[1])
        )
        skin_filtered = skin[filter_mask]
        diagnostics["n_pixels"] = np.sum(filter_mask)
        diagnostics["filter_frac"] = diagnostics["n_pixels"] / len(filter_mask)

        if self.debug:
            # Create a mask with the same shape as the original mask
            filter_mask_viz = np.zeros_like(mask, dtype=bool)
            # Set True for the filtered skin pixels
            filter_mask_viz[mask] = filter_mask
            self.visualizer.visualize_loaded_mask(
                img=img,
                binary_mask=filter_mask_viz,
                label="Filtered Mask (Cb, Cr)",
                plot_name="filtered_mask",
            )

        res = self.calculate_average_skin_tone(
            skin_filtered,
            average_first=self.average_before_conversion,
            debug_mask=filter_mask_viz if self.debug else None,
        )

        if self.debug:
            self.visualizer.visualize_colors(
                [(res["red"], res["green"], res["blue"])],
                names=["Average (filtered skin)"],
            )

        res["diagnostics"] = diagnostics

        return ExtractionResult(
            measurements=res, method="krishnapriya", image_id=self.image.id
        )
