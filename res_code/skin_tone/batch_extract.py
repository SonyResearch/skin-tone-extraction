"""Helper functions for parallel batch extraction."""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm

from .extract import extract_skin_tone
from .image import GrayscaleError
from .image_collection import MaskedImageCollection

RESULT_EMPTY = {}


def batch_extract(
    collection: MaskedImageCollection,
    max_workers: int = None,
    **kwargs,
) -> list[dict[str, float]]:
    """Batch extract skin values from multiple images in parallel with progress bar."""

    def _extract(masked_image):
        return extract_skin_tone(masked_image, **kwargs)

    results = [None] * len(collection)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_extract, collection[idx]): idx
            for idx in range(len(collection))
        }
        with tqdm(total=len(collection), desc="Extracting", smoothing=0) as pbar:
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except GrayscaleError as e:
                    masked_image = collection[idx]
                    print(
                        f"Warning: Skipping grayscale image "
                        f"'{masked_image.img_path}': {e}"
                    )
                    results[idx] = RESULT_EMPTY
                except Exception as e:
                    masked_image = collection[idx]
                    results[idx] = RESULT_EMPTY
                    print(
                        f"Exception for img_path='{masked_image.img_path}', "
                        f"mask_path='{masked_image.mask_path}'"
                    )
                    raise e
                pbar.update(1)
    return results


def batch_extract_df(
    collection: MaskedImageCollection,
    max_workers: int = None,
    expand_diagnostics: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """Batch extract skin values and return as a DataFrame with metadata."""
    results = batch_extract(
        collection=collection,
        max_workers=max_workers,
        **kwargs,
    )
    results_df = pd.DataFrame(results)
    meta_df = pd.DataFrame(
        {
            "img_filename": [os.path.basename(p) for p in collection.image_paths],
            "mask_filename": [os.path.basename(p) for p in collection.mask_paths],
            "img_path": collection.image_paths,
            "mask_path": collection.mask_paths,
        }
    )
    df = pd.concat([meta_df, results_df], axis=1)

    if expand_diagnostics and "diagnostics" in df.columns:
        diag_df = df["diagnostics"].apply(pd.Series)
        if not diag_df.empty:
            diag_df = diag_df.add_prefix("diag_")
            df = pd.concat([df.drop(columns=["diagnostics"]), diag_df], axis=1)

    return df
