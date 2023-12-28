import unittest
import numpy as np

from src.generate_data import GaussianClusters
from src.gaussian_mixture_model import GaussianMixtureModel


class TestGMM(unittest.TestCase):
    def test_gmm(self):
        # generate clusters
        np.random.seed(0)
        cluster_means = np.array([[-1, -2], [3, 3]])
        cluster_covariances = np.array([[[1, 0], [0, 1]], [[2, 0], [0, 0.1]]])
        cluster_weights = np.array([0.2, 0.8])
        n_samples = 1000

        clusters = GaussianClusters(cluster_means, cluster_covariances, cluster_weights)
        dataset = clusters.sample(n_samples)

        gmm = GaussianMixtureModel(2, 200, 1e-4)
        updated_priors, expected_z = gmm.fit(dataset, plot=True)
        cluster_ratios = np.sum(expected_z, axis=0) / n_samples
        pred_cov = updated_priors["w_inv_k"] / updated_priors["beta_k"].reshape(2, 1, 1)

        # match the clusters
        if cluster_ratios[0] < cluster_ratios[1]:
            cluster_means = np.array([[-1, -2], [3, 3]])
            expected_cov = np.array([[[1, 0], [0, 1]], [[2, 0], [0, 0.1]]])
        else:
            cluster_means = np.array([[3, 3], [-1, -2]])
            expected_cov = np.array([[[2, 0], [0, 0.1]], [[1, 0], [0, 1]]])

        self.assertTrue(np.abs(np.min(cluster_ratios) - 0.2) < 0.05)
        self.assertTrue(np.abs(np.max(cluster_ratios) - 0.8) < 0.05)
        self.assertTrue(np.allclose(cluster_means, updated_priors["m_k"], atol=0.1))
        self.assertTrue(np.allclose(expected_cov, pred_cov, atol=0.2))


if __name__ == "__main__":
    unittest.main()
