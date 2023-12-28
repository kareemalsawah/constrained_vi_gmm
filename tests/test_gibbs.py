import unittest
import numpy as np

from src.gibbs import PermInvGibbsSampler


class TestGibbs(unittest.TestCase):
    def test_prob_c_given_z(self):
        np.random.seed(42)
        observed_ratios = np.array([0.2, 0.8])
        observation_variance = 0.1
        gibbs_sampler = PermInvGibbsSampler(
            observed_ratios,
            observation_variance,
            n_chains=100,
            burn_in=50,
            chain_length=200,
        )
        z = np.array([[0, 1], [1, 0]])

        pc_z0 = gibbs_sampler.prob_c_given_z(z, 0)
        pc_z1 = gibbs_sampler.prob_c_given_z(z, 1)

        # manually computed values
        expected_pc_z0 = np.array([2.7e-6, 0.027])
        expected_pc_z1 = np.array([0.027, 0.449**2])

        self.assertTrue(np.all(np.abs(pc_z0 - expected_pc_z0) < 0.01))
        self.assertTrue(np.all(np.abs(pc_z1 - expected_pc_z1) < 0.01))

    def test_matching(self):
        np.random.seed(42)
        observed_ratios = np.array([0.2, 0.8])
        observation_variance = 0.1
        gibbs_sampler = PermInvGibbsSampler(
            observed_ratios,
            observation_variance,
            n_chains=100,
            burn_in=50,
            chain_length=200,
        )

        norm_probs = np.array([[0.7, 0.3], [0.6, 0.4]])
        gibbs_sampler.match_observation_ratios(norm_probs)
        self.assertTrue(np.all(gibbs_sampler.cur_cols == np.array([1, 0])))

        gibbs_sampler.match_observation_ratios(norm_probs[:, ::-1])
        self.assertTrue(np.all(gibbs_sampler.cur_cols == np.array([0, 1])))

    def test_expected_z(self):
        np.random.seed(42)
        observed_ratios = np.array([0.8, 0.2])
        observation_variance = 0.1
        gibbs_sampler = PermInvGibbsSampler(
            observed_ratios,
            observation_variance,
            n_chains=100,
            burn_in=50,
            chain_length=200,
        )
        norm_probs = np.array([[0.7, 0.3], [0.6, 0.4]])

        expected_z = gibbs_sampler.expected_z(norm_probs)

        # expected z manually computed (over 4 possible values for z)
        manual_z = np.array([[0.95, 0.05], [0.93, 0.07]])

        diff = np.abs(expected_z - manual_z)
        self.assertTrue(np.all(diff < 0.05))


if __name__ == "__main__":
    unittest.main()
