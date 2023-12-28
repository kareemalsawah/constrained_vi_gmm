"""
Visualizer class for debugging visualizations
"""
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Ellipse
from matplotlib.cm import ScalarMappable


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

        for mean, cov, weight in zip(means, covariances, weights):
            if weight < self.min_weight_visible:  # Don't plot if weight is too small
                continue

            # Eigenvalues and eigenvectors for the covariance matrix
            vals, vecs = np.linalg.eigh(cov)
            order = vals.argsort()[::-1]
            vals, vecs = vals[order], vecs[:, order]

            # Width and height of the ellipse, using a scaling factor for visualization
            width, height = 2 * np.sqrt(vals)
            angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

            # Create an ellipse
            ellipse = Ellipse(
                xy=mean,
                width=width,
                height=height,
                angle=angle,
                alpha=0.5,
                color=self.cmap(weight),
            )
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
