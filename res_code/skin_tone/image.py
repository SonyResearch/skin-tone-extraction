"""Module for handling images with associated masks and parsing paths."""

import os

import numpy as np
from skimage.io import imread
from skimage.transform import resize


class GrayscaleError(Exception):
    """Custom exception for handling grayscale image errors."""

    pass


class MaskedImage:
    """Class representing an image with an associated mask."""

    def __init__(
        self,
        img_path: str,
        mask_path: str,
        mask_value: int,
        id: str = None,
        collection=None,
    ):
        """Initialize a MaskedImage with paths and mask value.

        Args:
            img_path: Path to the image file.
            mask_path: Path to the mask file.
            mask_value: The value in the mask to be considered as the region of
                interest.
            id: Optional ID for the image. If None, uses filename without extension.
            collection: Optional reference to the MaskedImageCollection for
                configuration.
        """
        self.img_path = img_path
        self.mask_path = mask_path
        self.mask_value = mask_value
        self._collection = collection

        # Use provided ID or filename without extension
        if id is None:
            id = os.path.splitext(os.path.basename(img_path))[0]
        self.id = id

        # Initialize cached image/mask data
        self._img = None
        self._mask = None

    @classmethod
    def in_collection(cls, collection, img_path: str, mask_path: str, id: str = None):
        """Create a MaskedImage instance connected to a MaskedImageCollection.

        This method creates a MaskedImage that retrieves its mask_value and other
        configuration settings from the provided collection.

        Args:
            collection: The MaskedImageCollection instance
            img_path: Path to the image file
            mask_path: Path to the mask file
            id: Optional ID for the image. If None, uses filename without extension.

        Returns:
            MaskedImage instance connected to the collection
        """
        return cls(
            img_path=img_path,
            mask_path=mask_path,
            mask_value=collection.mask_value,
            id=id,
            collection=collection,
        )

    def _get_config_value(self, attr_name: str, default_value):
        """Get a configuration value from the collection with fallback to default.

        Args:
            attr_name: Name of the attribute to retrieve from collection
            default_value: Default value to use if collection is None or attribute
                is None

        Returns:
            Configuration value from collection or default value
        """
        if self._collection is not None:
            collection_value = getattr(self._collection, attr_name, None)
            if collection_value is not None:
                return collection_value
        return default_value

    def load(self, allow_grayscale: bool = None) -> tuple[np.ndarray, np.ndarray]:
        """Load and return the image and its corresponding mask.

        Ensures both have matching dimensions and the mask is binary for the specified
        mask_value.

        Args:
            allow_grayscale: If False, raises GrayscaleError for grayscale images.
                If None, uses collection setting if available, otherwise defaults
                to False.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing:
                - img (np.ndarray): The loaded image as an array. If the image is
                    grayscale, it is converted to RGB.
                - mask (np.ndarray): A boolean mask array where True corresponds to
                    pixels with the specified mask_value.

        Raises:
            ValueError: If the image and mask dimensions do not match after resizing.
            GrayscaleError: If the image is grayscale and allow_grayscale=False.
        """
        # Determine allow_grayscale setting
        if allow_grayscale is None:
            allow_grayscale = self._get_config_value("allow_grayscale", False)

        img = imread(self.img_path)

        # Handle (potentially) grayscale images
        is_grayscale = False
        if img.ndim == 2:
            # Grayscale image with no RGB channel info
            is_grayscale = True
            if allow_grayscale:
                # Option 1: Convert to RGB
                print(
                    f"Warning: Image {self.img_path} is grayscale, converting to RGB."
                )
                img = np.stack([img] * 3, axis=-1)
        else:
            # Check whether r=g=b to detect grayscale
            is_grayscale = np.all(img[:, :, 0] == img[:, :, 1]) and np.all(
                img[:, :, 0] == img[:, :, 2]
            )

        if is_grayscale and not allow_grayscale:
            raise GrayscaleError(
                f"Image {self.img_path} is grayscale, set allow_grayscale=True "
                "if you want it to be used."
            )

        raw_mask = imread(self.mask_path)
        mask = raw_mask == self.mask_value

        # Check for equal dimensions, else resize the mask
        if img.shape[:2] != mask.shape[:2]:
            # Resize mask to match image dimensions
            mask = resize(
                mask,
                img.shape[:2],
                order=0,  # Nearest neighbor to preserve binary values
                preserve_range=True,
                anti_aliasing=False,
            ).astype(bool)

            if img.shape[:2] != mask.shape[:2]:
                raise ValueError(
                    f"Image shape {img.shape[:2]} and mask shape {mask.shape[:2]} "
                    "do not match after resizing."
                )

        # Check how much of the image the mask covers, give a warning based on
        # collection settings or defaults
        mask_coverage = np.sum(mask) / np.prod(mask.shape)
        if mask_coverage < self._get_config_value(
            "min_mask_coverage_warning", 0.01
        ) or mask_coverage > self._get_config_value("max_mask_coverage_warning", 0.90):
            print(
                f"Warning: Mask for image {self.img_path} covers "
                f"{mask_coverage:.2%} of the image, this is an unusual amount."
            )

        # Cache the loaded data
        self._img = img
        self._mask = mask

        return img, mask

    @property
    def img(self) -> np.ndarray:
        """Get the loaded image array. Loads if not already cached."""
        if self._img is None:
            self.load()
        return self._img

    @property
    def mask(self) -> np.ndarray:
        """Get the loaded mask array. Loads if not already cached."""
        if self._mask is None:
            self.load()
        return self._mask
