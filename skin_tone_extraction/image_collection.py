"""Collection class for managing and defining multiple MaskedImage(s)."""

import glob
import os
import re
from typing import Iterator, List, Optional, Tuple, Union

from skin_tone_extraction.dataset import DatasetConfig

from .image import MaskedImage


class MaskedImageCollection:
    """A collection of MaskedImages."""

    allow_grayscale: Optional[bool] = None
    min_mask_coverage_warning: Optional[float] = None
    max_mask_coverage_warning: Optional[float] = None

    def __init__(
        self,
        images: Union[str, List[str], Tuple[str, ...]],
        masks: Union[str, List[str], Tuple[str, ...]],
        mask_value: int,
        image_ids: Optional[List[str]] = None,
    ):
        """Initialize the collection with image and mask sources.

        Args:
            images: Image source - directory path, pattern with {image_id},
                   list of file paths, or tuple of patterns/directories
            masks: Mask source - directory path, pattern with {image_id},
                  list of file paths, or tuple of patterns/directories
            mask_value: Integer value used to select skin region in mask
            image_ids: Optional list of image IDs to validate against

        Raises:
            ValueError: If validation fails or no matching files found
        """
        self.mask_value = mask_value

        # Parse and validate inputs
        self._image_paths, self._mask_paths, self._image_ids = self._parse_inputs(
            images, masks, image_ids
        )
        self._validate_inputs()

        # Cache for loaded MaskedImage instances
        self._loaded_images = {}

    @classmethod
    def from_dataset_config(
        cls, dataset_config: DatasetConfig
    ) -> "MaskedImageCollection":
        """Create a MaskedImageCollection from a dataset configuration.

        Args:
            dataset_config: Dataset configuration object with images, masks,
                mask_value, and optionally image_ids attributes

        Returns:
            MaskedImageCollection instance

        Raises:
            ValueError: If required fields are missing from dataset config
        """
        # Note: DatasetConfig might not always be correctly typed due to parallel
        # processing. We can only really rely on it as a dataclass.
        images = getattr(dataset_config, "images", None)
        masks = getattr(dataset_config, "masks", None)
        mask_value = getattr(dataset_config, "mask_value", None)
        image_ids = getattr(dataset_config, "image_ids", None)

        if images is None or masks is None or mask_value is None:
            raise ValueError(
                "Dataset config is missing required fields: "
                "images, masks, or mask_value"
            )

        # Create instance
        collection = cls(
            images=images,
            masks=masks,
            mask_value=mask_value,
            image_ids=image_ids,
        )

        # Apply optional configuration parameters
        allow_grayscale = getattr(dataset_config, "allow_grayscale", None)
        if allow_grayscale is not None:
            collection.allow_grayscale = allow_grayscale
        min_mask_coverage_warning = getattr(
            dataset_config, "min_mask_coverage_warning", None
        )
        if min_mask_coverage_warning is not None:
            collection.min_mask_coverage_warning = min_mask_coverage_warning
        max_mask_coverage_warning = getattr(
            dataset_config, "max_mask_coverage_warning", None
        )
        if max_mask_coverage_warning is not None:
            collection.max_mask_coverage_warning = max_mask_coverage_warning

        return collection

    def _parse_inputs(
        self,
        images: Union[str, List[str], Tuple[str, ...]],
        masks: Union[str, List[str], Tuple[str, ...]],
        image_ids: Optional[List[str]] = None,
    ) -> Tuple[List[str], List[str], Optional[List[str]]]:
        """Parse images and masks inputs into lists of file paths.

        Supports multiple input formats:
        - Strings: Scans directories for matching files, if they contain {image_id} this
            will be treated as a wildcard and extracted. Files will be searched based on
            images and image_ids will then be placed into the mask paths.
        - Tuples: Multiple strings to be parsed with their outputs concatenated.
        - Lists: Direct file path lists (must have matching lengths), will NOT be parsed

        Args:
            images: Image source - directory path, pattern with {image_id},
                   list of file paths, or tuple of patterns/directories
            masks: Mask source - directory path, pattern with {image_id},
                  list of file paths, or tuple of patterns/directories
            image_ids: Optional list of image IDs to validate against

        Returns:
            Tuple containing:
            - List of image file paths
            - List of mask file paths
            - List of image IDs (or None if not available)

        Raises:
            ValueError: If validation fails or no matching files found
        """

        def is_pattern(path):
            return isinstance(path, str) and "{image_id}" in path

        # If lists are provided, just return them
        if isinstance(images, list) and isinstance(masks, list):
            if len(images) != len(masks):
                raise ValueError("Length of images and masks lists must match.")
            # If image_ids is provided, check length
            if image_ids is not None and len(image_ids) != len(images):
                raise ValueError("Length of image_ids must match images/masks lists.")
            return images, masks, image_ids

        # If either images or masks is a tuple, make sure both are
        if isinstance(images, tuple) and not isinstance(masks, tuple):
            masks = (masks,)
        if not isinstance(images, tuple) and isinstance(masks, tuple):
            images = (images,)

        if isinstance(images, tuple) and isinstance(masks, tuple):
            # Tuples detected -> Recursively iterate over patterns and concat the output
            if len(images) != len(masks):
                raise ValueError("Length of images and masks tuples must match.")
            if image_ids is not None:
                raise ValueError("image_ids not supported with tuple inputs.")

            img_paths = []
            mask_paths = []
            image_ids = []

            for i, m in zip(images, masks):
                # Avoid infinite recursion
                if not isinstance(i, str) or not isinstance(m, str):
                    raise ValueError("Tuple elements must be strings.")
                res_images, res_masks, res_ids = self._parse_inputs(i, m)

                img_paths.extend(res_images)
                mask_paths.extend(res_masks)
                if res_ids:
                    image_ids.extend(res_ids)

            return img_paths, mask_paths, image_ids if image_ids else None
        else:
            if is_pattern(images) and is_pattern(masks):
                # Pattern mode
                img_pattern = images.replace("{image_id}", "*")
                img_files = sorted(glob.glob(img_pattern))
                if not img_files:
                    raise ValueError(f"No images found for pattern: {img_pattern}")

                # Extract image_id using regex
                pattern_regex = re.escape(images).replace("\\{image_id\\}", "(.+)")
                img_id_re = re.compile(f"^{pattern_regex}$")
                image_ids = []
                for f in img_files:
                    m = img_id_re.match(f)
                    if m:
                        image_ids.append(m.group(1))
                    else:
                        # Try matching just the basename
                        m = img_id_re.match(os.path.basename(f))
                        if m:
                            image_ids.append(m.group(1))
                        else:
                            image_ids.append(os.path.splitext(os.path.basename(f))[0])

                mask_files = [
                    masks.replace("{image_id}", image_id) for image_id in image_ids
                ]
                img_paths = img_files
                mask_paths = mask_files
            elif not is_pattern(images) and not is_pattern(masks):
                # Directory mode (read all files in the directories)
                img_files = sorted(
                    [f for f in os.listdir(images) if not f.startswith(".")]
                )
                mask_files = sorted(
                    [f for f in os.listdir(masks) if not f.startswith(".")]
                )
                # Match files by name (assume same filenames in both dirs)
                common_files = sorted(set(img_files) & set(mask_files))
                if not common_files:
                    raise ValueError("No matching files found in both directories.")
                img_paths = [os.path.join(images, f) for f in common_files]
                mask_paths = [os.path.join(masks, f) for f in common_files]
                image_ids = None
            else:
                raise ValueError(
                    "Both images and masks must be either directories or patterns "
                    "with {image_id}."
                )

        return img_paths, mask_paths, image_ids

    def _validate_inputs(self) -> None:
        """Validate that image and mask paths exist and match.

        Raises:
            ValueError: If validation fails
        """
        if len(self._image_paths) != len(self._mask_paths):
            # Check for matching lengths
            raise ValueError(
                "Number of images and masks must be the same. "
                f"Got {len(self._image_paths)} images and "
                f"{len(self._mask_paths)} masks."
            )

        if self._image_ids:
            # Check for matching lengths
            if len(self._image_ids) != len(self._image_paths):
                raise ValueError(
                    "Number of image IDs must match number of images. "
                    f"Got {len(self._image_ids)} IDs and "
                    f"{len(self._image_paths)} images."
                )

            # Check for duplicates
            num_duplicates = len(self._image_ids) - len(set(self._image_ids))
            if num_duplicates > 0:
                fraction_duplicates = num_duplicates / len(self._image_ids)
                raise ValueError(
                    "Duplicate image IDs found."
                    f"Share of duplicates: {fraction_duplicates:.3f}"
                )
        else:
            # If there are no image IDs, ensure all image paths are unique
            num_duplicates = len(self._image_paths) - len(set(self._image_paths))
            if num_duplicates > 0:
                fraction_duplicates = num_duplicates / len(self._image_paths)
                raise ValueError(
                    "Duplicate image paths found."
                    f"Share of duplicates: {fraction_duplicates:.3f}"
                )

    def __len__(self) -> int:
        """Return the number of images in the collection."""
        return len(self._image_paths)

    def __getitem__(self, index: int) -> MaskedImage:
        """Get a MaskedImage instance by index, creating it if necessary.

        Args:
            index: Index of the image to retrieve

        Returns:
            MaskedImage instance

        Raises:
            IndexError: If index is out of range
        """
        if index < 0 or index >= len(self._image_paths):
            raise IndexError("Index out of range")

        # Check if already loaded
        if index not in self._loaded_images:
            # Create MaskedImage instance
            image_id = self._image_ids[index] if self._image_ids else None
            self._loaded_images[index] = MaskedImage(
                img_path=self._image_paths[index],
                mask_path=self._mask_paths[index],
                mask_value=self.mask_value,
                id=image_id,
                collection=self,
            )

        return self._loaded_images[index]

    def __iter__(self) -> Iterator[MaskedImage]:
        """Iterate over all MaskedImage instances in the collection."""
        for i in range(len(self)):
            yield self[i]

    @property
    def image_paths(self) -> List[str]:
        """Get list of image file paths."""
        return self._image_paths.copy()

    @property
    def mask_paths(self) -> List[str]:
        """Get list of mask file paths."""
        return self._mask_paths.copy()

    @property
    def image_ids(self) -> Optional[List[str]]:
        """Get list of image IDs if available."""
        return self._image_ids.copy() if self._image_ids else None

    def get_id(self, index: int) -> Optional[str]:
        """Get image ID by index if available.

        Args:
            index: Index of the ID to retrieve

        Returns:
            Image ID string or None if not available

        Raises:
            IndexError: If index is out of range
        """
        if index < 0 or index >= len(self._image_paths):
            raise IndexError("Index out of range")

        return self._image_ids[index] if self._image_ids else None

    def clear_cache(self) -> None:
        """Clear all loaded MaskedImage instances to free memory."""
        self._loaded_images.clear()

    def slice(
        self, start: int = None, stop: int = None, step: int = None
    ) -> "MaskedImageCollection":
        """Create a new collection with a slice of the current images.

        Args:
            start: Start index for slicing
            stop: Stop index for slicing
            step: Step size for slicing

        Returns:
            New MaskedImageCollection with sliced data
        """
        sliced_image_paths = self._image_paths[start:stop:step]
        sliced_mask_paths = self._mask_paths[start:stop:step]
        sliced_image_ids = None
        if self._image_ids:
            sliced_image_ids = self._image_ids[start:stop:step]

        return MaskedImageCollection(
            images=sliced_image_paths,
            masks=sliced_mask_paths,
            mask_value=self.mask_value,
            image_ids=sliced_image_ids,
        )

    def __repr__(self) -> str:
        """String representation of the collection."""
        return (
            f"MaskedImageCollection(length={len(self)}, "
            f"mask_value={self.mask_value}, "
            f"cached={len(self._loaded_images)})"
        )
