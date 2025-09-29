"""Skin tone extraction approach building on Merler et al.

Reference:
Merler, M., Ratha, N., Feris, R. S., & Smith, J. R. (2019). Diversity in faces. arXiv
preprint arXiv:1901.10436.
"""

from typing import Literal

import numpy as np
from skimage.filters import gaussian

from res_code.skin_tone.extraction import ExtractionMethod, ExtractionResult
from res_code.skin_tone.helpers import (
    calculate_average,
    calculate_mode,
    prepare_skin_tone_columns,
)


class MerlerMethod(ExtractionMethod):
    """Extract skin tone using an approach inspired by Merler (2019)."""

    method_id = "merler"

    def __init__(
        self,
        upper_pct: float = 0.60,
        smooth: bool = True,
        aggregate_regions: Literal["average", "mode"] = "mode",
        absolute_vertical: bool = True,
        debug: bool = False,
        debug_image_dir: str = None,
    ):
        """Initialize the extraction method.

        Args:
            upper_pct: Fraction of the skin height to consider for the upper regions.
            smooth: If True, applies Gaussian smoothing to color dimensions.
            aggregate_regions: Method to aggregate colors within each region.
            absolute_vertical: If True, uses absolute pixel coordinates for vertical
                cut.
            debug: If True, output debug visualizations and information.
            debug_image_dir: Directory to save visualizations images (if debug is True).
        """
        super().__init__(debug=debug, debug_image_dir=debug_image_dir)
        self.upper_pct = upper_pct
        self.smooth = smooth
        self.aggregate_regions = aggregate_regions
        self.absolute_vertical = absolute_vertical

    def _extract(self) -> ExtractionResult:
        """Extract skin tone using the Merler method.

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
                label="Mask",
            )

        diagnostics = {}

        # Get coordinates of skin pixels
        skin_indices = np.argwhere(mask)
        if skin_indices.shape[0] == 0:
            return ExtractionResult(
                measurements={
                    "red": np.nan,
                    "green": np.nan,
                    "blue": np.nan,
                    "diagnostics": {"n_pixels": 0},
                },
                method="merler",
                image_id=self.image.id,
            )

        # Get min and max coordinates of skin pixels
        min_y = skin_indices[:, 0].min()
        max_y = skin_indices[:, 0].max()
        min_x = skin_indices[:, 1].min()
        max_x = skin_indices[:, 1].max()
        skin_height = max_y - min_y + 1

        upper = int(skin_height * self.upper_pct)
        region_y_start = min_y
        region_y_end = region_y_start + upper
        region_y_end = min(region_y_end, max_y + 1)

        # For 2x2 grid, cut at vertical and horizontal middle
        y_mid = (region_y_start + region_y_end) // 2
        if self.absolute_vertical:
            x_mid = img.shape[1] // 2
        else:
            x_mid = (min_x + max_x) // 2

        # Define 4 regions: top-left, top-right, bottom-left, bottom-right
        regions = [
            ((region_y_start, y_mid), (min_x, x_mid)),  # top-left
            ((region_y_start, y_mid), (x_mid, max_x + 1)),  # top-right
            ((y_mid, region_y_end), (min_x, x_mid)),  # bottom-left
            ((y_mid, region_y_end), (x_mid, max_x + 1)),  # bottom-right
        ]
        diagnostics["region_bounds"] = regions

        # Visualize regions if debugging
        if self.debug:
            # Assign region labels to each skin pixel for visualization (2x2 grid)
            region_labels = np.full(skin_indices.shape[0], -1, dtype=int)
            for i, ((y0, y1), (x0, x1)) in enumerate(regions):
                region_mask = (
                    (skin_indices[:, 0] >= y0)
                    & (skin_indices[:, 0] < y1)
                    & (skin_indices[:, 1] >= x0)
                    & (skin_indices[:, 1] < x1)
                )
                region_labels[region_mask] = i
            # Create a mask for all skin pixels
            skin_mask = np.zeros_like(mask, dtype=bool)
            skin_mask[skin_indices[:, 0], skin_indices[:, 1]] = True

            self.visualizer.visualize_clusters_on_image(
                labels=region_labels,
                img=img,
                mask=skin_mask,
                alpha=0.6,
            )

        skin = img[mask]

        # Process each region
        region_results = []
        region_sizes = []
        all_region_dimensions = []
        all_region_masks = []

        for (y0, y1), (x0, x1) in regions:
            region_mask = (
                (skin_indices[:, 0] >= y0)
                & (skin_indices[:, 0] < y1)
                & (skin_indices[:, 1] >= x0)
                & (skin_indices[:, 1] < x1)
            )
            all_region_masks.append(region_mask)

            region_skin = skin[region_mask]
            skin_dimensions = prepare_skin_tone_columns(region_skin)

            # Smooth over each dimension
            if self.smooth:
                for dimension in skin_dimensions.keys():
                    skin_dimensions[dimension] = gaussian(
                        skin_dimensions[dimension], sigma=1, truncate=4
                    )

            # Store dimensions for combined visualization
            all_region_dimensions.append(skin_dimensions)

            if self.aggregate_regions == "average":
                result = calculate_average(skin_dimensions, std=False)
            elif self.aggregate_regions == "mode":
                first_region = len(region_results) == 0
                result = calculate_mode(
                    skin_dimensions,
                    visualizer=self.visualizer if first_region and self.debug else None,
                )
            else:
                raise ValueError(
                    f"Invalid aggregate_regions value: {self.aggregate_regions}. "
                    "Expected 'average' or 'mode'."
                )
            region_results.append(result)

            region_sizes.append(np.sum(region_mask))

        diagnostics["region_sizes"] = region_sizes
        total_pixels = np.sum(region_sizes)
        diagnostics["total_pixels"] = total_pixels

        if self.debug:
            region_masks_2d = []

            for i, region_mask in enumerate(all_region_masks):
                # Create 2D mask for this specific region
                region_coords = skin_indices[region_mask]
                individual_mask = np.zeros_like(mask, dtype=bool)
                individual_mask[region_coords[:, 0], region_coords[:, 1]] = True
                region_masks_2d.append(individual_mask)

            # Use new multi-region visualization functionality
            self.visualizer.visualize_metrics(
                all_region_dimensions,
                region_masks_2d,
            )

        # Calculate weighted averages over regions
        res = {}
        for key in region_results[0].keys():
            # Filter out NaNs for aggregation
            values = np.array([result[key] for result in region_results])
            weights = np.array(region_sizes)
            valid = ~np.isnan(values)
            if np.any(valid):
                res[key] = np.average(values[valid], weights=weights[valid])
                res[key + "_std"] = np.sqrt(
                    np.average(
                        (values[valid] - res[key]) ** 2,
                        weights=weights[valid],
                    )
                )
            else:
                res[key] = np.nan
                res[key + "_std"] = np.nan

        if self.debug:
            # Show individual region colors
            colors = [(reg["red"], reg["green"], reg["blue"]) for reg in region_results]
            region_labels = ["Top-Left", "Top-Right", "Bot.-Left", "Bot.-Right"]
            labels = [
                f"{label}\n({size / total_pixels:.0%})"
                for label, size in zip(region_labels, region_sizes)
            ]

            # Add final average
            colors = colors + [(res["red"], res["green"], res["blue"])]
            labels = labels + ["Average\n(weighted)"]

            self.visualizer.visualize_colors(
                colors,
                names=labels,
            )

        res["diagnostics"] = diagnostics

        return ExtractionResult(
            measurements=res, method="merler", image_id=self.image.id
        )
