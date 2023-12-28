import unittest
import numpy as np

from src.generate_data import GaussianClusters
from src.constrained_gmm import ConstrainedGMM


class TestConstrainedGMM(unittest.TestCase):
    def test_normal_gmm(self):
        np.random.seed(42)
        cluster_means = np.array([[-1, -2], [3, 3]])
        cluster_covariances = np.array([[[1, 0], [0, 1]], [[2, 0], [0, 0.1]]])
        cluster_weights = np.array([0.2, 0.8])
        n_samples = 1000

        clusters = GaussianClusters(cluster_means, cluster_covariances, cluster_weights)
        dataset = clusters.sample(n_samples)

        gmm = ConstrainedGMM(cluster_weights, 0.1, 2, 15, 1e-4)
        updated_priors, expected_z = gmm.fit(dataset, plot=True)
        cluster_ratios = np.sum(expected_z, axis=0) / n_samples
        pred_cov = updated_priors["w_inv_k"] / updated_priors["beta_k"].reshape(2, 1, 1)

        self.assertTrue(np.abs(np.min(cluster_ratios) - 0.2) < 0.05)
        self.assertTrue(np.abs(np.max(cluster_ratios) - 0.8) < 0.05)
        self.assertTrue(np.allclose(cluster_means, updated_priors["m_k"], atol=0.2))
        self.assertTrue(np.allclose(cluster_covariances, pred_cov, atol=0.3))

    def test_specific_case(self):
        np.random.seed(42)
        points_cluster_1 = np.array([[1, 1], [1.2, 0.8], [1.4, 0.6], [1.6, 0.4]])
        points_cluster_2 = np.array([[-1.2, -1.5], [-0.8, -1.8]])

        point_between_them = (
            np.mean(points_cluster_1, axis=0) + np.mean(points_cluster_2, axis=0)
        ) / 2

        all_points = np.concatenate(
            [points_cluster_1, points_cluster_2, point_between_them.reshape(1, 2)],
            axis=0,
        )

        # ratios wanting to assign it to smaller cluster at idx 0
        counts = np.array([4, 3]) / 7
        gmm = ConstrainedGMM(counts, 0.01, 2, 100, 1e-3)
        _, expected_z = gmm.fit(all_points, plot=True)
        cluster_ratios = np.sum(expected_z, axis=0) / 7
        self.assertTrue(np.allclose(cluster_ratios, counts, atol=0.1))

        # ratios wanting to assign it to smaller cluster at idx 1
        counts = np.array([3, 4]) / 7
        gmm = ConstrainedGMM(counts, 0.01, 2, 100, 1e-3)
        _, expected_z = gmm.fit(all_points, plot=True)
        cluster_ratios = np.sum(expected_z, axis=0) / 7
        self.assertTrue(np.allclose(cluster_ratios, counts, atol=0.1))

        # ratios wanting to assign it to larger cluster at idx 0
        counts = np.array([5, 2]) / 7
        gmm = ConstrainedGMM(counts, 0.01, 2, 100, 1e-3)
        _, expected_z = gmm.fit(all_points, plot=True)
        cluster_ratios = np.sum(expected_z, axis=0) / 7
        self.assertTrue(np.allclose(cluster_ratios, counts, atol=0.1))

        # ratios wanting to assign it to larger cluster at idx 1
        counts = np.array([2, 5]) / 7
        gmm = ConstrainedGMM(counts, 0.01, 2, 100, 1e-3)
        _, expected_z = gmm.fit(all_points, plot=True)
        cluster_ratios = np.sum(expected_z, axis=0) / 7
        self.assertTrue(np.allclose(cluster_ratios, counts, atol=0.1))


if __name__ == "__main__":
    unittest.main()
