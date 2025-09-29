"""Visualization methods for skin tone analysis."""

import os

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


class Visualizer:
    """Class containing visualization methods for skin tone analysis."""

    def __init__(
        self,
        image_id: str | None = None,
        output_dir: str | None = None,
        prefix: str = "",
        debug_metrics: list | None = [
            "lum",
            "hue",
            "ita",
        ],
    ):
        """Initialize visualizer.

        Args:
            image_id: Identifier for the image being analyzed. Used in output filenames
                when saving plots. If None, filenames won't include an image identifier.
            output_dir: Directory path where plots should be saved. If None, plots are
                only displayed and not saved to disk.
            prefix: String prefix to add to output filenames, inserted between image_id
                and the plot name.
            debug_metrics: List of metric names to visualize in e.g. visualize_metrics.
                Defaults to ["lum", "hue", "ita"]. Set to None or empty list to disable
                default metric visualization.
        """
        self.image_id = image_id
        self.output_dir = output_dir
        self.prefix = prefix
        self.debug_metrics = debug_metrics

    def _show_plt(self, name: str):
        # Save plot to file
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            filename = name
            if self.image_id:
                filename = f"{self.image_id}_{self.prefix}{filename}"
            filepath = os.path.join(self.output_dir, f"{filename}.png")
            plt.savefig(filepath, bbox_inches="tight", dpi=300)

        # Show plot in interactive contexts
        plt.show()

    def visualize_loaded_mask(
        self,
        img: np.ndarray,
        binary_mask: np.ndarray,
        label: str = None,
        bg_val: int = 255,
        plot_name: str = None,
    ) -> None:
        """Visualize only the masked pixels from the image for a given mask value."""
        # Scale bg_val to match image scale
        if img.dtype != np.uint8 and (
            img.max() <= 1.0 or not np.issubdtype(img.dtype, np.integer)
        ):
            scaled_bg_val = bg_val / 255.0
        else:
            scaled_bg_val = bg_val

        # Create an output image where only masked pixels are shown, others are bg_val
        masked_img = np.full_like(img, scaled_bg_val)
        for c in range(img.shape[2]):
            masked_img[..., c][binary_mask] = img[..., c][binary_mask]
        plt.imshow(masked_img)
        if label:
            plt.title(label)
        plt.axis("off")
        name = plot_name if plot_name else "loaded_mask"
        self._show_plt(name)

    def visualize_1dim_mask(
        self,
        metric,
        binary_mask,
        label: str = None,
        colormap: str = "viridis",
        bg_color: str = "white",
        plot_name: str = None,
    ) -> None:
        """Visualize a one-dimensional metric using a colormap.

        Args:
            metric: 1D array of metric values for each masked pixel, or list of 1D
                arrays for multiple regions.
            binary_mask: 2D boolean array indicating which pixels to visualize, or list
                of 2D boolean arrays for multiple regions.
            label: Optional title for the plot
            colormap: Matplotlib colormap name (default: 'viridis').
            bg_color: Background color for non-masked pixels (default: 'white').
            plot_name: Optional name for saving the plot.
        """
        # Handle single region case
        if not isinstance(metric, list):
            metric = [metric]
            binary_mask = [binary_mask]

        # Ensure all inputs are lists of same length
        if not isinstance(binary_mask, list):
            binary_mask = [binary_mask] * len(metric)

        # Visualize all regions in a single plot
        # Create combined visualization array
        vis_img = np.full(binary_mask[0].shape, np.nan, dtype=np.float32)

        # Overlay all regions (they have non-overlapping masks)
        for m, mask in zip(metric, binary_mask):
            vis_img[mask] = m

        # Create masked array so background pixels are excluded from colormap
        masked_img = np.ma.masked_invalid(vis_img)

        # Get the colormap and set background color
        cmap = plt.cm.get_cmap(colormap)
        cmap.set_bad(bg_color)

        plt.figure()
        plt.imshow(masked_img, cmap=cmap)
        plt.colorbar()

        # Create title from all labels
        if label:
            plt.title(label)

        plt.axis("off")
        name = (
            plot_name
            if plot_name
            else (label.lower().replace(" ", "_") if label else "1dim_mask")
        )
        self._show_plt(name)

    def visualize_metrics(
        self,
        metrics_dict,
        binary_mask,
        debug_metrics: list = None,
        postfix: str = "",
        plot_name: str = None,
        **kwargs,
    ) -> None:
        """Visualize selected metrics from a metrics dictionary.

        Args:
            metrics_dict: Dictionary containing metric arrays or list of dictionaries
                for multiple regions (e.g., from prepare_skin_tone_columns).
            binary_mask: 2D boolean array indicating which pixels to visualize,
                or list of 2D boolean arrays for multiple regions.
            debug_metrics: List of metric keys to visualize. Defaults to
                self.debug_metrics.
            postfix: String to append to metric labels.
            plot_name: Optional name for saving the plot.
            **kwargs: Additional keyword arguments to pass to visualize_1dim_mask.
        """
        if debug_metrics is None:
            debug_metrics = self.debug_metrics

        # Handle single region case
        if not isinstance(metrics_dict, list):
            metrics_dict = [metrics_dict]
            binary_mask = [binary_mask]
            if postfix and not isinstance(postfix, list):
                postfix = [postfix]

        # Ensure all inputs are lists of same length
        if not isinstance(binary_mask, list):
            binary_mask = [binary_mask] * len(metrics_dict)

        for metric_key in debug_metrics:
            # Collect metric values and masks for all regions that have this metric
            metric_values_list = []
            binary_mask_list = []

            for i, (m_dict, mask) in enumerate(zip(metrics_dict, binary_mask)):
                if metric_key in m_dict:
                    metric_values_list.append(m_dict[metric_key])
                    binary_mask_list.append(mask)
                else:
                    print(
                        f"Warning: Metric '{metric_key}' not found in region {i} "
                        "metrics dictionary"
                    )

            # Visualize all regions for this metric if any were found
            if metric_values_list:
                metric_plot_name = plot_name if plot_name else f"metric_{metric_key}"
                self.visualize_1dim_mask(
                    metric=metric_values_list,
                    binary_mask=binary_mask_list,
                    label=f"{metric_key.capitalize()}{postfix}",
                    plot_name=metric_plot_name,
                    **kwargs,
                )

    def visualize_clusters_on_image(
        self,
        labels: np.ndarray,
        img: np.ndarray,
        mask: np.ndarray,
        alpha: float = 0.5,
        plot_name: str = None,
    ) -> None:
        """Visualize clusters by coloring the masked pixels in the image.

        Args:
            labels: 1D array of cluster labels for each masked pixel.
            img: Original image as a numpy array (H, W, 3).
            mask: 2D boolean array, True for masked pixels.
            alpha: Transparency for overlay.
            plot_name: Optional name for saving the plot.
        """
        unique_labels = np.unique(labels)
        # Ignore -1 label if present (indicating unassigned pixels)
        if -1 in unique_labels:
            unique_labels = unique_labels[unique_labels != -1]
        n_clusters = len(unique_labels)
        colors = plt.cm.get_cmap(lut=n_clusters)

        # Prepare overlay
        overlay = np.zeros_like(img, dtype=np.float32)
        mask_indices = np.argwhere(mask)
        legend_patches = []
        for idx, label in enumerate(unique_labels):
            cluster_mask = np.zeros(mask.shape, dtype=bool)
            cluster_mask[tuple(mask_indices[labels == label].T)] = True
            color = np.array(mcolors.to_rgb(colors(idx))) * 255
            for c in range(3):
                overlay[..., c][cluster_mask] = color[c]
            # Add legend patch
            legend_patches.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    label=f"Cluster {label}",
                    markerfacecolor=np.array(mcolors.to_rgb(colors(idx))),
                    markersize=10,
                )
            )

        # Handle both float (0-1) and int (0-255) images
        if img.dtype == np.uint8 or (
            np.issubdtype(img.dtype, np.integer) and img.max() > 1
        ):
            img_float = img.astype(np.float32) / 255.0
        else:
            img_float = img.astype(np.float32)
        overlay_float = overlay / 255.0

        blended = img_float * (1 - alpha) + overlay_float * alpha
        blended = np.clip(blended, 0, 1)

        plt.figure(figsize=(8, 8))
        plt.imshow(blended)
        plt.title("Clusters")
        plt.axis("off")
        plt.legend(handles=legend_patches, loc="upper right", bbox_to_anchor=(1.15, 1))
        name = plot_name if plot_name else "clusters"
        self._show_plt(name)

    def visualize_image(
        self, img: np.ndarray, title: str = None, plot_name: str = None
    ) -> None:
        """Display an image, handling both integer (0-255) and float (0-1) formats.

        Args:
            img: Image as a numpy array (H, W, 3).
            title: Optional title for the plot.
            plot_name: Optional name for saving the plot.
        """
        if img.dtype == np.uint8 or (
            np.issubdtype(img.dtype, np.integer) and img.max() > 1
        ):
            img_disp = img.astype(np.float32) / 255.0
        else:
            img_disp = img.astype(np.float32)
        plt.figure()
        plt.imshow(np.clip(img_disp, 0, 1))
        if title:
            plt.title(title)
        plt.axis("off")
        name = (
            plot_name
            if plot_name
            else (title.lower().replace(" ", "_") if title else "image")
        )
        self._show_plt(name)

    def visualize_colors(
        self, colors, names=None, circle_size=300, title=None, plot_name: str = None
    ):
        """Plot a list of colors as individual circles.

        Args:
            colors: List of colors (as RGB tuples, hex strings, or matplotlib color
                format).
            names: Optional list of names for each color.
            circle_size: Size of the circles.
            title: Optional title for the plot.
            plot_name: Optional name for saving the plot.
        """
        n = len(colors)
        fig, ax = plt.subplots(figsize=(max(2, n), 2))
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(-0.5, 0.5)
        ax.axis("off")

        for i, color in enumerate(colors):
            ax.scatter(i, 0, s=circle_size, color=color, edgecolors="k")
            if names is not None:
                ax.text(i, -0.25, str(names[i]), ha="center", va="top", fontsize=10)

        if title:
            plt.title(title)
        name = (
            plot_name
            if plot_name
            else (title.lower().replace(" ", "_") if title else "colors")
        )
        self._show_plt(name)

    def visualize_mode(
        self,
        metric_name: str,
        values: np.ndarray,
        bins: int,
        mode: float,
        plot_name: str = None,
    ):
        """Visualize a histogram of a metric's values with the mode indicated.

        Args:
            metric_name: Name of the metric being visualized.
            values: 1D array of metric values.
            bins: Number of bins or array of bin edges for the histogram.
            mode: The mode value to indicate on the histogram.
            plot_name: Optional name for saving the plot.
        """
        if metric_name not in self.debug_metrics:
            return

        plt.hist(values, bins=bins, edgecolor="black")
        plt.axvline(mode, color="red", linestyle="dashed", linewidth=1)
        if metric_name:
            plt.xlabel(metric_name)
        name = plot_name if plot_name else f"mode_{metric_name}"
        self._show_plt(name)
