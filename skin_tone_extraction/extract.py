"""Main entrypoint to extract skin tone from images."""

from skin_tone_extraction.image import MaskedImage
from skin_tone_extraction.methods.krishnapriya import KrishnapriyaMethod
from skin_tone_extraction.methods.merler import MerlerMethod
from skin_tone_extraction.methods.overall import OverallAverageMethod, OverallModeMethod
from skin_tone_extraction.methods.thong import ThongMethod

METHODS = {
    "thong": ThongMethod,
    "average": OverallAverageMethod,
    "mode": OverallModeMethod,
    "krishnapriya": KrishnapriyaMethod,
    "merler": MerlerMethod,
}
DEFAULT_METHOD = "average"


def extract_skin_tone(
    masked_image: MaskedImage,
    method: str = DEFAULT_METHOD,
    **kwargs,
) -> dict[str, float]:
    """Extract skin tone from a MaskedImage using the specified method.

    Args:
        masked_image (MaskedImage): The MaskedImage instance containing image and mask.
        method (str, optional): Extraction method to use. Options are
            'thong', 'average', 'mode', 'krishnapriya', 'merler'. Defaults to 'average'.
        **kwargs: Additional keyword arguments passed to the extraction method.

    Returns:
        The result of the skin tone extraction method.

    Raises:
        ValueError: If an unknown method is specified.
    """
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}")

    # Instantiate the extraction method with any provided kwargs
    extraction_method = METHODS[method](**kwargs)

    # Extract skin tone and return measurements
    result = extraction_method.extract(masked_image)
    return result.measurements


def extract_skin_tone_from_paths(
    img_path: str,
    mask_path: str,
    mask_value: int,
    method: str = DEFAULT_METHOD,
    **kwargs,
) -> dict[str, float]:
    """Extract skin tone from an image using the specified method.

    Args:
        img_path (str): Path to the image file.
        mask_path (str): Path to the mask file.
        mask_value (int): Value in the mask to select the region of interest.
        method (str, optional): Extraction method to use. Options are
            'thong', 'average', 'mode', 'krishnapriya', 'merler'. Defaults to 'average'.
        **kwargs: Additional keyword arguments passed to the extraction method.

    Returns:
        The result of the skin tone extraction method.

    Raises:
        ValueError: If an unknown method is specified.
    """
    # Create MaskedImage instance
    masked_image = MaskedImage(img_path, mask_path, mask_value)

    return extract_skin_tone(masked_image, method=method, **kwargs)
