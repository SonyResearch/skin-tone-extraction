"""Simple skin tone extraction approach utilizing the whole skin mask."""

from skin_tone_extraction.extraction import ExtractionMethod, ExtractionResult


class OverallAverageMethod(ExtractionMethod):
    """Extract skin tone using the average color of all skin pixels."""

    method_id = "average"

    def _extract(self) -> ExtractionResult:
        """Extract skin tone using the average color of all skin pixels.

        Returns:
            ExtractionResult: Results containing skin tone measurements.
        """
        img = self.image.img
        mask = self.image.mask

        skin = img[mask] / 255.0

        if self.debug:
            self.visualizer.visualize_image(img, "Original")
            self.visualizer.visualize_loaded_mask(
                img=img,
                binary_mask=mask,
                label="Mask",
            )

        res = self.calculate_average_skin_tone(skin, debug_mask=mask)

        if self.debug:
            self.visualizer.visualize_colors(
                [(res["red"], res["green"], res["blue"])],
                names=["Average"],
            )

        return ExtractionResult(
            measurements=res, method="overall_average", image_id=self.image.id
        )


class OverallModeMethod(ExtractionMethod):
    """Extract skin tone using the mode (most frequent) color of all skin pixels."""

    method_id = "mode"

    def _extract(self) -> ExtractionResult:
        """Extract skin tone using the mode (most frequent) color of all skin pixels.

        Returns:
            ExtractionResult: Results containing skin tone measurements.
        """
        img = self.image.img
        mask = self.image.mask

        skin = img[mask] / 255.0

        if self.debug:
            self.visualizer.visualize_image(img, "Original")
            self.visualizer.visualize_loaded_mask(
                img=img,
                binary_mask=mask,
                label="Mask",
            )

        res = self.calculate_mode_skin_tone(skin, debug_mask=mask)

        if self.debug:
            self.visualizer.visualize_colors(
                [(res["red"], res["green"], res["blue"])],
                names=["Mode"],
            )

        return ExtractionResult(
            measurements=res, method="overall_mode", image_id=self.image.id
        )
