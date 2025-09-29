"""Pre-specified dataset configurations for use with the CLI.

You can pre-specify datasets by creating a `datasets.py` file in the working directory.
These datasets can then be referenced by name via the CLI using the `--dataset`
argument, allowing one to specify dataset paths / patterns only once. This is especially
helpful for more complex datasets, which require e.g. parsing of a file to retrieve
paths. This can be handled in the constructor of a DatasetConfig subclass.
"""

import os
import runpy
import warnings


class DatasetConfig:
    """Configuration for a prespecified dataset to be used via the CLI.

    Do not instantiate this class directly, but create subclasses
    instead. The CLI will then instantiate the subclass as required.
    """

    images: str | list[str]
    "Path/pattern to image files or list of image file paths."

    masks: str | list[str]
    "Path/pattern to mask files or list of mask file paths."

    image_ids: list[str] | None = None
    "Optional list of image ids."

    mask_value: int
    "Integer value used to select region in mask."

    allow_grayscale: bool | None = None
    "Whether to allow grayscale images. If None, uses collection default."

    min_mask_coverage_warning: float | None = None
    "Minimum mask coverage below which to warn. If None, uses collection default."

    max_mask_coverage_warning: float | None = None
    "Maximum mask coverage above which to warn. If None, uses collection default."


def get_dataset_config(
    dataset: str, local_datasets_path: str = "./datasets.py"
) -> DatasetConfig:
    """Loads and validates the dataset configuration from datasets.py."""
    if not os.path.isfile(local_datasets_path):
        raise FileNotFoundError(
            "Local datasets.py file not found. "
            "Please ensure it exists in the current directory."
        )

    # Run the datasets.py file to get the dataset configurations
    datasets_results = runpy.run_path(local_datasets_path)

    # Extract the configuration from the results
    if dataset not in datasets_results:
        available_options = ", ".join(datasets_results.keys())
        raise ValueError(
            f"Dataset '{dataset}' not found in dataset configurations. "
            f"Available options: {available_options}"
        )

    dataset_config = datasets_results[dataset]

    # Check whether dataset_config is a class or an instance
    if isinstance(dataset_config, type):
        # If it's a class, instantiate it
        dataset_config = dataset_config()

    # NOTE: We cannot do `isinstance(dataset_config, DatasetConfig)` here
    # because it will fail as the classes are not _exactly_ the same, therefore
    # the following workaround relying on class names
    is_config_instance = False
    if hasattr(dataset_config, "__class__"):
        is_config_instance = DatasetConfig.__name__ in [
            base.__name__ for base in dataset_config.__class__.__bases__
        ]
    if not is_config_instance:
        warnings.warn(
            "Dataset configuration should be an instance of DatasetConfig "
            "(and doesn't seem to be).",
            UserWarning,
        )

    return dataset_config
