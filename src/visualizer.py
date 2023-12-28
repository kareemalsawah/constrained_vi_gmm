"""
Visualizer class for debugging visualizations
"""
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Ellipse
from matplotlib.cm import ScalarMappable
import matplotlib.transforms as transforms


class Visualizer:
    """
    Visualizer class for visualizing the algorithm with gaussians as ellipses
    Works only for 2D data

    Parameters
    ----------
    dataset: np.array
        shape=(n_samples, n_dims)
    enabled: bool, by default False
        Whether to enable plotting or not
    min_weight_visible: float, by default 0.0
        Any gaussian with weight less than this value will not be plotted
    pause_time: float, by default 0.2
        Time to pause the plot after each update (in secs)
    cmap: plt.cm, by default plt.cm.jet
        Color map to use for the ellipses
        Can be any of the color maps from matplotlib.cm
    """

    def __init__(
        self,
        dataset: np.array,
        enabled: bool = False,
        min_weight_visible: float = 0.0,
        pause_time: float = 0.2,
        cmap: plt.cm = plt.cm.jet,
    ):
        self.dataset = dataset
        self.enabled = enabled
        self.min_weight_visible = min_weight_visible
        self.pause_time = pause_time
        self.cmap = cmap
        if self.enabled:
            if self.dataset.shape[1] != 2:
                raise ValueError("Dataset must have 2 dimensions to enable plotting")
            self.fig, self.ax = plt.subplots()
            self.added_first_plot = False

    def plot_gaussians(self, means: np.array, covariances: np.array, weights: np.array):
        """
        Plot the gaussians as ellipses with the given weights to represent the
        color of the ellipse.
        Created with the help of chatGPT

        Parameters
        ----------
        means: np.array
            shape=(n_clusters, n_dims), must have n_dims=2
        covariances: np.array
            shape=(n_clusters, n_dims, n_dims), must have n_dims=2
        weights: np.array
            shape=(n_clusters,)
        """
        if not self.enabled:
            return

        n_clusters, _ = means.shape
        weights = weights.reshape(-1, 1)

        # Check if inputs are valid
        if means.shape != (n_clusters, 2):
            raise ValueError(f"means.shape must be (n_clusters, 2), got {means.shape}")
        if covariances.shape != (n_clusters, 2, 2):
            raise ValueError(
                f"covariances.shape must be (n_clusters, 2, 2), got {covariances.shape}"
            )
        if weights.shape != (n_clusters, 1):
            raise ValueError(
                f"weights.shape must be (n_clusters, 1), got {weights.shape}"
            )

        self.ax.clear()
        self.ax.scatter(self.dataset[:, 0], self.dataset[:, 1], c="blue")
        n_std = 2.0
        for mean, cov, weight in zip(means, covariances, weights):
            if weight < self.min_weight_visible:  # Don't plot if weight is too small
                continue

            pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
            ell_radius_x = np.sqrt(1 + pearson)
            ell_radius_y = np.sqrt(1 - pearson)
            ellipse = Ellipse(
                (0, 0),
                width=ell_radius_x * 2,
                height=ell_radius_y * 2,
                alpha=0.5,
                color=self.cmap(weight),
            )
            scale_x = np.sqrt(cov[0, 0]) * n_std
            scale_y = np.sqrt(cov[1, 1]) * n_std

            transf = (
                transforms.Affine2D()
                .rotate_deg(45)
                .scale(scale_x, scale_y)
                .translate(mean[0], mean[1])
            )

            ellipse.set_transform(transf + self.ax.transData)

            self.ax.add_patch(ellipse)

        # Set plot limits and labels, only for the first plot
        if not self.added_first_plot:
            plt.xlim(means[:, 0].min() - 3, means[:, 0].max() + 3)
            plt.ylim(means[:, 1].min() - 3, means[:, 1].max() + 3)
            plt.xlabel("X-axis")
            plt.ylabel("Y-axis")
            plt.title("Gaussian Distributions")

            # Colorbar for the weights
            sm = ScalarMappable(cmap=self.cmap)
            sm.set_array([])
            plt.colorbar(sm, ax=self.ax, label="Weights")
            self.added_first_plot = True

        plt.pause(self.pause_time)
