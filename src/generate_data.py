"""
Functions to generate synthetic data for testing
"""
import numpy as np
import matplotlib.pyplot as plt


class GaussianClusters:
    """
    Generate a mixture of gaussians with given means, covariances and weights

    Parameters
    ----------
    means: np.array
        shape=(n_clusters, n_dims)
    covariances: np.array
        shape=(n_clusters, n_dims, n_dims)
    weights: np.array
        shape=(n_clusters,)
    """

    def __init__(self, means: np.array, covariances: np.array, weights: np.array):
        self.means = means
        self.covariances = covariances
        self.weights = weights
        self.n_clusters = means.shape[0]
        self.n_dims = means.shape[1]

        # normalize weights
        self.weights = self.weights / np.sum(self.weights)

        # validate shapes
        assert self.covariances.shape[0] == self.n_clusters
        assert self.covariances.shape[1] == self.covariances.shape[2] == self.n_dims
        assert self.weights.shape[0] == self.n_clusters

        # validate covariances are symmetric and positive definite
        for i in range(self.n_clusters):
            assert np.all(np.linalg.eigvals(self.covariances[i]) > 0)
            assert np.all(self.covariances[i] == self.covariances[i].T)

        # validate weights are positive
        assert np.all(self.weights > 0)

    def sample(self, n_samples: int, plot: bool = False) -> np.array:
        """
        Sample from the gaussian clusters

        Parameters
        ----------
        n_samples: int
            Number of samples to generate
        plot: bool, by default False
            If true, plots the generate data

        Returns
        -------
        np.array
            shape=(n_samples, n_dims)
        """
        samples = []
        for i in range(self.n_clusters):
            to_add = np.random.multivariate_normal(
                self.means[i], self.covariances[i], int(n_samples * self.weights[i])
            )
            samples.extend(to_add.tolist())
        samples = np.array(samples)

        if plot:  # plot samples, each cluster with a different color
            plt.title("Generated Data")
            cur_pos = 0
            for i in range(self.n_clusters):
                to_plot = samples[cur_pos : cur_pos + int(n_samples * self.weights[i])]
                cur_pos += int(n_samples * self.weights[i])
                plt.scatter(to_plot[:, 0], to_plot[:, 1])
            plt.show()

        # shuffle samples
        samples = samples[np.random.permutation(samples.shape[0])]
        return samples
