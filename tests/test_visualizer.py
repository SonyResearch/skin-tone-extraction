"""Tests for visualization functions."""

import numpy as np

from skin_tone_extraction.visualizer import Visualizer


def test_visualize_loaded_mask_basic():
    img = np.ones((5, 5, 3), dtype=np.uint8) * 100
    mask = np.ones((5, 5), dtype=bool)
    visualizer = Visualizer()
    # Should not raise an exception
    visualizer.visualize_loaded_mask(img, mask, "Test")


def test_visualize_loaded_mask_float():
    img = np.ones((5, 5, 3), dtype=np.float32) * 0.5
    mask = np.ones((5, 5), dtype=bool)
    visualizer = Visualizer()
    # Should not raise an exception
    visualizer.visualize_loaded_mask(img, mask)


def test_visualize_1dim_mask_basic():
    mask = np.zeros((5, 5), dtype=bool)
    mask[1:4, 1:4] = True  # 9 pixels
    metric = np.random.rand(9)
    visualizer = Visualizer()
    # Should not raise an exception
    visualizer.visualize_1dim_mask(metric, mask, "Test Metric")


def test_visualize_1dim_mask_multiple_regions():
    mask1 = np.zeros((5, 5), dtype=bool)
    mask1[1:3, 1:3] = True  # 4 pixels
    mask2 = np.zeros((5, 5), dtype=bool)
    mask2[3:5, 3:5] = True  # 4 pixels

    metric1 = np.random.rand(4)
    metric2 = np.random.rand(4)

    visualizer = Visualizer()
    # Should not raise an exception
    visualizer.visualize_1dim_mask([metric1, metric2], [mask1, mask2], "Multi Region")


def test_visualize_metrics_basic():
    mask = np.zeros((5, 5), dtype=bool)
    mask[1:4, 1:4] = True  # 9 pixels

    metrics = {
        "lum": np.random.rand(9),
        "hue": np.random.rand(9) * 360,
        "ita": np.random.rand(9) * 100 - 50,
    }

    visualizer = Visualizer()
    # Should not raise an exception
    visualizer.visualize_metrics(metrics, mask)


def test_visualize_metrics_with_postfix():
    mask = np.zeros((5, 5), dtype=bool)
    mask[1:3, 1:3] = True  # 4 pixels

    metrics = {"lum": np.random.rand(4)}

    visualizer = Visualizer()
    # Should not raise an exception
    visualizer.visualize_metrics(metrics, mask, postfix=" - Test")


def test_visualize_clusters_on_image():
    img = np.ones((5, 5, 3), dtype=np.uint8) * 128
    mask = np.ones((5, 5), dtype=bool)
    labels = np.zeros(25, dtype=int)  # All same cluster

    visualizer = Visualizer()
    # Should not raise an exception
    visualizer.visualize_clusters_on_image(labels, img, mask)


def test_visualize_clusters_multiple_clusters():
    img = np.ones((5, 5, 3), dtype=np.float32) * 0.5
    mask = np.ones((5, 5), dtype=bool)
    labels = np.random.randint(0, 3, size=25)  # 3 clusters

    visualizer = Visualizer()
    # Should not raise an exception
    visualizer.visualize_clusters_on_image(labels, img, mask, alpha=0.7)


def test_show_uint8():
    img = np.ones((5, 5, 3), dtype=np.uint8) * 128
    visualizer = Visualizer()
    # Should not raise an exception
    visualizer.visualize_image(img, "Test Image")


def test_show_float():
    img = np.ones((5, 5, 3), dtype=np.float32) * 0.5
    visualizer = Visualizer()
    # Should not raise an exception
    visualizer.visualize_image(img)


def test_show_colors_basic():
    colors = ["red", "green", "blue"]
    visualizer = Visualizer()
    # Should not raise an exception
    visualizer.visualize_colors(colors)


def test_show_colors_with_names():
    colors = ["#FF0000", "#00FF00", "#0000FF"]
    names = ["Red", "Green", "Blue"]
    visualizer = Visualizer()
    # Should not raise an exception
    visualizer.visualize_colors(colors, names=names, title="RGB Colors")


def test_show_colors_rgb_tuples():
    colors = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
    visualizer = Visualizer()
    # Should not raise an exception
    visualizer.visualize_colors(colors, circle_size=400)


def test_mask_metric_compatibility():
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:8, 3:7] = True  # Create a rectangular region
    n_masked_pixels = np.sum(mask)

    # Create metric with correct number of values
    metric = np.random.rand(n_masked_pixels)
    assert len(metric) == n_masked_pixels
    assert metric.ndim == 1


def test_image_format_handling():
    # uint8 format
    img_uint8 = np.random.randint(0, 255, (5, 5, 3), dtype=np.uint8)
    assert img_uint8.dtype == np.uint8

    # float format
    img_float = np.random.rand(5, 5, 3)
    assert img_float.dtype == np.float64
    assert img_float.max() <= 1.0
