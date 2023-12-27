"""
Synthetic data unittests
"""
import unittest

import numpy as np
from src.generate_data import GaussianClusters


class TestGenerateData(unittest.TestCase):
    """
    Test GaussianClusters class
    """

    def test_generate_data(self):
        """
        Simple shape and mean test
        """
        np.random.seed(42)
        generator = GaussianClusters(
            means=np.array([[1, 1], [-1, -2]]),
            covariances=np.array([np.eye(2) * 0.3, np.eye(2) * 0.3]),
            weights=np.array([0.8, 0.2]),
        )

        samples = generator.sample(1000)

        self.assertEqual(samples.shape, (1000, 2))

        # test that the means are close to the expected values
        self.assertTrue(np.abs(np.mean(samples[:, 0]) - (1 * 0.8 + -1 * 0.2)) < 0.1)
        self.assertTrue(np.abs(np.mean(samples[:, 1]) - (1 * 0.8 + -2 * 0.2)) < 0.1)


if __name__ == "__main__":
    unittest.main()
