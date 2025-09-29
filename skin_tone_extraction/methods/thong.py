"""Skin tone extraction approach building on Thong et al.

Reference:
Thong, W., Joniak, P., & Xiang, A. (2023). Beyond skin tone: A multidimensional measure
of apparent skin color. In Proceedings of the IEEE/CVF International Conference on
Computer Vision (pp. 4903-4913)

See Also:
    https://github.com/SonyResearch/apparent_skincolor
"""

import numpy as np
from skimage.color import rgb2lab
from skimage.filters import gaussian
from sklearn import cluster

from skin_tone_extraction.extraction import ExtractionMethod, ExtractionResult
from skin_tone_extraction.helpers import calculate_mode, prepare_skin_tone_columns
from skin_tone_extraction.metrics import get_hue


class ThongMethod(ExtractionMethod):
    """Extract skin tone using an approach inspired by Thong et al. (2023)."""

    method_id = "thong"

    def __init__(
        self,
        n_clusters: int = 5,
        topk: int = 3,
        skip_topk: int = 0,
        bins: str = "sturges",
        debug: bool = False,
        debug_image_dir: str = None,
    ):
        """Initialize the extraction method.

        Args:
            n_clusters: Number of clusters for K-means clustering.
            topk: Number of top clusters (by luminance) to use for final averaging.
            skip_topk: Number of brightest clusters to skip before selecting topk
                clusters.
            bins: Binning strategy for histogram mode calculation.
            debug: If True, output debug visualizations and information.
            debug_image_dir: Directory to save visualizations images (if debug is True).
        """
        super().__init__(debug=debug, debug_image_dir=debug_image_dir)
        self.n_clusters = n_clusters
        self.topk = topk
        self.skip_topk = skip_topk
        self.bins = bins

    def _extract(self) -> ExtractionResult:
        """Extract skin tone using the Thong method.

        Returns:
            ExtractionResult: Results containing skin tone measurements.
        """
        img = self.image.img
        mask = self.image.mask

        if self.debug:
            self.visualizer.visualize_image(img, "Original")

        # smoothing
        img_smoothed = gaussian(img, sigma=(1, 1), truncate=4, channel_axis=-1)
        if self.debug:
            self.visualizer.visualize_image(img_smoothed, "Smoothed")

            self.visualizer.visualize_loaded_mask(
                img=img_smoothed,
                binary_mask=mask,
                label="Smoothed (Masked)",
            )

        # get skin pixels (shape will be Mx3) and go to Lab
        skin_smoothed = img_smoothed[mask]
        skin_smoothed_lab = rgb2lab(skin_smoothed)

        # Cluster on L and hue
        hue_angle = get_hue(skin_smoothed_lab[:, 1], skin_smoothed_lab[:, 2])
        data_to_cluster = np.vstack([skin_smoothed_lab[:, 0], hue_angle]).T
        labels, model = self.clustering(data_to_cluster, n_clusters=self.n_clusters)

        if self.debug:
            self.visualizer.visualize_1dim_mask(
                skin_smoothed_lab[:, 0], mask, label="Lum"
            )
            self.visualizer.visualize_1dim_mask(hue_angle, mask, label="Hue")

            self.visualizer.visualize_clusters_on_image(labels, img_smoothed, mask)

        # Get scalar values from clusters
        res = self.get_scalar_values(
            skin_smoothed_lab,
            labels,
            skin_smoothed,
            topk=self.topk,
            skip_topk=self.skip_topk,
            bins=self.bins,
        )

        return ExtractionResult(
            measurements=res, method="thong", image_id=self.image.id
        )

    def get_scalar_values(
        self,
        skin_smoothed_lab: np.ndarray,
        labels: np.ndarray,
        skin_smoothed: np.ndarray,
        topk: int,
        skip_topk: int,
        bins: str = "sturges",
    ) -> dict[str, float]:
        """Extract scalar color values from clustered skin pixels.

        Args:
            skin_smoothed_lab: Skin pixels in L*a*b* color space, shape(N, 3).
            labels: Cluster labels for each skin pixel, shape (N,).
            skin_smoothed: Skin pixels in RGB color space, shape (N, 3).
            topk: Number of top clusters (by luminance) to use for final averaging.
            skip_topk: Number of brightest clusters to skip before selecting
                topk clusters.
            bins: Binning strategy for histogram mode calculation.
                Defaults to "sturges".

        Returns:
            dict: Results dictionary.
        """
        # Extract skin pixels for each cluster
        n_clusters = len(np.unique(labels))
        n_pixels = np.asarray([np.sum(labels == i) for i in range(n_clusters)])

        # Process each cluster to get mode values
        cluster_results = []
        for i in range(n_clusters):
            cluster_mask = labels == i
            cluster_skin = skin_smoothed[cluster_mask]

            # Use prepare_skin_tone_columns to get comprehensive measurements
            skin_dimensions = prepare_skin_tone_columns(cluster_skin)

            # Calculate mode for this cluster
            cluster_mode = calculate_mode(
                skin_dimensions,
                visualizer=self.visualizer if i == 0 and self.debug else None,
            )
            cluster_results.append(cluster_mode)

        # Extract luminance values for sorting
        lum_values = np.array([result["lum"] for result in cluster_results])
        idx_sorted = np.argsort(lum_values)[::-1]  # sort by luminance (brightest first)
        idx = idx_sorted[skip_topk : skip_topk + topk]  # skip and take topk

        if self.debug:
            print(
                f"Skipping {skip_topk} brightest clusters, "
                f"using top {topk} clusters (by luminance): {idx}"
            )

        # Calculate weighted averages over selected clusters
        keys = list(cluster_results[0].keys())
        res_topk = {}

        for key in keys:
            values = np.array([cluster_results[i][key] for i in idx])
            weights = n_pixels[idx]

            # Handle potential NaN values
            valid = ~np.isnan(values)
            if np.any(valid):
                res_topk[key] = np.average(values[valid], weights=weights[valid])
                res_topk[key + "_std"] = np.sqrt(
                    np.average(
                        (values[valid] - res_topk[key]) ** 2, weights=weights[valid]
                    )
                )
            else:
                res_topk[key] = np.nan
                res_topk[key + "_std"] = np.nan

        if self.debug:
            total_pixels = np.sum(n_pixels)
            cluster_colors = [
                (
                    cluster_results[i]["red"],
                    cluster_results[i]["green"],
                    cluster_results[i]["blue"],
                )
                for i in idx
            ]
            final_color = [(res_topk["red"], res_topk["green"], res_topk["blue"])]

            self.visualizer.visualize_colors(
                cluster_colors + final_color,
                names=[f"Cluster {i}\n({n_pixels[i] / total_pixels:.0%})" for i in idx]
                + ["Average\n(weighted)"],
            )

        return res_topk

    def clustering(
        self, x: np.ndarray, n_clusters: int = 5, random_state: int = 2021
    ) -> tuple[np.ndarray, cluster.KMeans]:
        """Perform K-means clustering on input data.

        Args:
            x: Input data for clustering.
            n_clusters: Number of clusters to form.
            random_state: Random state for reproducibility.

        Returns:
            Tuple containing cluster labels and the fitted KMeans model.
        """
        model = cluster.KMeans(n_clusters, random_state=random_state)
        model.fit(x)
        return model.labels_, model
