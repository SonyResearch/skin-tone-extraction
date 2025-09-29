"""Tests for extraction functions."""

import glob
import os
import tempfile

import numpy as np
import pytest

from skin_tone_extraction.batch_extract import batch_extract_df
from skin_tone_extraction.extract import extract_skin_tone_from_paths
from skin_tone_extraction.helpers import (
    mode_hist,
    prepare_skin_tone_columns,
    visualize_mask,
)
from skin_tone_extraction.image_collection import MaskedImageCollection
from skin_tone_extraction.methods.overall import OverallAverageMethod
from skin_tone_extraction.methods.thong import ThongMethod
from skin_tone_extraction.metrics import get_hue
from skin_tone_extraction.visualizer import Visualizer

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "./assets")
IMG_PATH = os.path.abspath(os.path.join(ASSETS_DIR, "00000.png"))
MASK_PATH = os.path.abspath(os.path.join(ASSETS_DIR, "00000_mask.png"))


def test_extract_image_basic():
    result = extract_skin_tone_from_paths(
        IMG_PATH, MASK_PATH, mask_value=1, method="thong"
    )

    # Hard-coded values based on example in https://github.com/SonyResearch/apparent_skincolor
    expected_values = {
        "lum": 60.32474192162227,
        "hue": 27.9874036146538,
        "lum_std": 7.537910024242527,
        "hue_std": 68.11535999823427,
        "red": 0.5670517343182768,
        "green": 0.5587551551471699,
        "blue": 0.4942237944973208,
        "red_std": 0.09923820062668536,
        "green_std": 0.059575927609366405,
        "blue_std": 0.055943684718035125,
    }

    # Check if all expected keys are present
    for key in expected_values:
        assert key in result, f"Missing key: {key}"
        assert isinstance(result[key], float), f"{key} is not a float"

    # Compare values with expected values
    for key, expected_value in expected_values.items():
        assert abs(result[key] - expected_value) < 1e-5, (
            f"Value for {key} does not match: {result[key]} != {expected_value}"
        )


def test_extract_snapshot_thong(snapshot):
    result = extract_skin_tone_from_paths(
        IMG_PATH, MASK_PATH, mask_value=1, method="thong"
    )
    assert result == snapshot


def test_extract_snapshot_average(snapshot):
    result = extract_skin_tone_from_paths(
        IMG_PATH, MASK_PATH, mask_value=1, method="average"
    )
    assert result == snapshot


def test_extract_snapshot_mode(snapshot):
    result = extract_skin_tone_from_paths(
        IMG_PATH, MASK_PATH, mask_value=1, method="mode"
    )
    assert result == snapshot


def test_extract_snapshot_merler(snapshot):
    result = extract_skin_tone_from_paths(
        IMG_PATH, MASK_PATH, mask_value=1, method="merler"
    )
    assert result == snapshot


def test_extract_snapshot_krishnapriya(snapshot):
    result = extract_skin_tone_from_paths(
        IMG_PATH, MASK_PATH, mask_value=1, method="krishnapriya"
    )
    assert result == snapshot


def test_extract_snapshot_merler_nonsmooth_avg(snapshot):
    result = extract_skin_tone_from_paths(
        IMG_PATH,
        MASK_PATH,
        mask_value=1,
        method="merler",
        smooth=False,
        aggregate_regions="average",
    )
    assert result == snapshot


def test_get_hue():
    a = np.array([1.0, 0.0, -1.0])
    b = np.array([0.0, 1.0, 0.0])
    hue = get_hue(a, b)
    assert hue.shape == a.shape
    assert np.all(np.isfinite(hue))


def test_mode_hist():
    x = np.array([1, 2, 2, 3, 3, 3, 4])
    mode, _ = mode_hist(x, bins=4)
    assert isinstance(mode, float)


def test_clustering():
    x = np.random.rand(10, 2)
    method = ThongMethod()
    labels, model = method.clustering(x, n_clusters=2)
    assert labels.shape[0] == 10
    assert hasattr(model, "predict")


def test_get_scalar_values():
    # Use synthetic LAB data and labels
    lab = np.random.rand(10, 3) * 100
    rgb = np.random.rand(10, 3) * 100
    labels = np.array([0, 1] * 5)
    method = ThongMethod()
    res = method.get_scalar_values(lab, labels, rgb, topk=1, skip_topk=0)
    assert "lum" in res and "hue" in res


def test_visualize_mask():
    # Should not raise
    visualize_mask(IMG_PATH, MASK_PATH, mask_value=1)


def test_visualize_clusters_on_image():
    img = np.ones((5, 5, 3), dtype=np.float32)
    mask = np.ones((5, 5), dtype=bool)
    labels = np.zeros(mask.sum(), dtype=int)
    viz = Visualizer()
    viz.visualize_clusters_on_image(labels, img, mask)


def test_show():
    img = np.ones((5, 5, 3), dtype=np.float32)
    viz = Visualizer()
    viz.visualize_image(img, title="Test Show")


def test_calculate_average_skin_tone_empty():
    empty = np.array([])
    method = OverallAverageMethod()
    with pytest.warns(Warning):
        result = method.calculate_average_skin_tone(empty)
    # Should return means and stds as nan or handle gracefully
    assert all(np.isnan(v) for v in result.values())


def test_prepare_skin_tone_columns_empty():
    empty = np.array([])
    with pytest.warns(Warning):
        columns = prepare_skin_tone_columns(empty)
    # Should return dict with empty arrays for each color channel
    assert isinstance(columns, dict)
    for v in columns.values():
        assert isinstance(v, np.ndarray)
        assert v.size == 0


def test_batch_extract_df_basic():
    # Create MaskedImageCollection
    collection = MaskedImageCollection(
        images=[IMG_PATH], masks=[MASK_PATH], mask_value=1
    )
    df = batch_extract_df(
        collection=collection, method="merler", expand_diagnostics=True
    )
    # Check DataFrame shape
    assert df.shape[0] == 1
    # Check metadata columns
    for col in ["img_filename", "mask_filename", "img_path", "mask_path"]:
        assert col in df.columns
    # Check skin tone columns
    for col in [
        "lum",
        "hue",
        "lum_std",
        "hue_std",
        "red",
        "green",
        "blue",
        "red_std",
        "green_std",
        "blue_std",
    ]:
        assert col in df.columns
    # Check values are floats
    for col in [
        "lum",
        "hue",
        "lum_std",
        "hue_std",
        "red",
        "green",
        "blue",
        "red_std",
        "green_std",
        "blue_std",
    ]:
        assert isinstance(df.iloc[0][col], float)
    # Check whether columns exist with the diag_ prefix
    assert "diagnostics" not in df.columns, "Diagnostics column should not be present"
    diag_cols = [col for col in df.columns if col.startswith("diag_")]
    assert len(diag_cols) > 0, "No expanded diagnostic columns found"


def test_debug_image_saving():
    """Test that debug images are saved when debug=True and debug_image_dir is set."""
    # Create a temporary directory for debug images
    with tempfile.TemporaryDirectory() as temp_dir:
        # Extract skin tone with debug enabled
        result = extract_skin_tone_from_paths(
            IMG_PATH,
            MASK_PATH,
            mask_value=1,
            method="thong",
            debug=True,
            debug_image_dir=temp_dir,
        )

        # Check that result contains expected measurements
        assert isinstance(result, dict)
        assert "lum" in result
        assert "hue" in result

        # Check that debug images were saved
        saved_files = glob.glob(os.path.join(temp_dir, "*.png"))
        assert len(saved_files) > 0, "No debug images were saved"

        # Check for specific expected debug images based on ThongMethod
        # visualization calls. The ThongMethod creates these visualizations
        # when debug=True:
        # - "Original" (via visualize_image)
        # - "Smoothed" (via visualize_image)
        # - "Smoothed (Masked)" (via visualize_loaded_mask)
        # - "Lum" (via visualize_1dim_mask)
        # - "Hue" (via visualize_1dim_mask)
        # - clusters visualization (via visualize_clusters_on_image)

        # Check for at least some expected debug image files
        expected_patterns = [
            "*original*.png",
            "*smoothed*.png",
            "*lum*.png",
            "*hue*.png",
        ]

        found_patterns = []
        for pattern in expected_patterns:
            matches = glob.glob(os.path.join(temp_dir, pattern))
            if matches:
                found_patterns.append(pattern)

        # Ensure at least some debug images were created
        saved_file_names = [os.path.basename(f) for f in saved_files]
        assert len(found_patterns) > 0, (
            f"Expected debug image patterns not found. Saved files: {saved_file_names}"
        )
