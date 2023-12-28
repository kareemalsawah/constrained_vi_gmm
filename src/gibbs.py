"""
Gibbs Sampler for estimating the expecation of z under q(z)
"""
import numpy as np
from scipy.optimize import linear_sum_assignment


class PermInvGibbsSampler:
    """
    Permutation Invariant Gibbs Sampler for estimating the expectation of z under q(z)
    Uses Hungarian Algorithm to find the best permutation

    Parameters
    ----------
    observed_ratios : np.array
        Observed ratios of each cluster
    observation_variance : float
        Variance of the observation noise
    n_chains : int, by default 50
        Number of chains to run
    burn_in : int, by default 20
        Number of iterations to run for burn-in
    chain_length : int, by default 100
        Number of iterations to run for each chain after burn-in
    """

    def __init__(
        self,
        observed_ratios: np.array,
        observation_variance: float,
        n_chains: int = 50,
        burn_in: int = 20,
        chain_length: int = 100,
    ):
        self.n_chains = n_chains
        self.burn_in = burn_in
        self.chain_length = chain_length

        self.observed_ratios = observed_ratios
        self.observation_variance = observation_variance
        self.cur_cols = np.arange(len(observed_ratios))

    def expected_z(self, norm_probs: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        norm_probs : np.ndarray
            Normalized probabilities of z

        Returns
        -------
        np.ndarray
            Expected value of z, shape=(n_samples, n_clusters)
        """
        n_samples, n_clusters = norm_probs.shape
        z_samples = []
        self.match_observation_ratios(norm_probs)

        for _ in range(self.n_chains):  # TODO: parallelize this
            # Sample initial z from norm_probs
            cumsum = np.cumsum(norm_probs, axis=1)
            samps = (np.random.uniform(0, 1, (n_samples, 1)) < cumsum).astype(int)
            z_sample = np.argmax(samps, axis=1)
            curr_z = np.eye(n_clusters)[z_sample]

            for j in range(self.burn_in + self.chain_length):
                cur_idx = np.random.choice(n_samples)  # index to update

                row_probs = norm_probs[cur_idx]  # shape=(n_clusters,)
                c_probs = self.prob_c_given_z(curr_z, cur_idx)  # shape=(n_clusters,)

                # Sample new row
                sample_probs = row_probs * c_probs
                if np.sum(sample_probs) == 0:
                    sample_probs = row_probs
                sample_probs /= np.sum(sample_probs)
                curr_z[cur_idx] = np.random.multinomial(1, sample_probs)

                if j > self.burn_in:
                    z_samples.append(np.copy(curr_z))

        z_samples = np.array(z_samples)
        expected_z = np.mean(z_samples, axis=0)
        expected_z /= np.sum(expected_z, axis=1, keepdims=True)
        return expected_z

    def prob_c_given_z(self, z: np.ndarray, idx: int) -> np.ndarray:
        """
        Computes P(C | z) enumerating all possible values of z[idx]
        Note: z[idx] is one-hot, so all possible values have count of n_clusters
        C: observed ratios

        Parameters
        ----------
        z : np.ndarray
            shape=(n_samples, n_clusters)
        idx : int
            Index of the row to fix, between (0, n_samples)

        Returns
        -------
        np.ndarray
            shape=(n_clusters,)
        """

        def gaussian_loss(x, mu, var):
            return np.exp(-0.5 * (x - mu) ** 2 / var)

        n_samples, _ = z.shape

        # shape=(n_clusters,) for each i: equal to observed count if z[idx][i] = 1
        ck_sums = np.sum(z, axis=0) - z[idx] + 1

        c_diff = ck_sums - self.observed_ratios[self.cur_cols] * n_samples

        # P(C[i]|Z) for each i: where z[idx][i] = 1
        c_z_i_1 = gaussian_loss(c_diff, 0, self.observation_variance)

        # P(C[i]|Z) for each i: where z[idx][i] = 0
        c_z_i_0 = gaussian_loss(c_diff - 1, 0, self.observation_variance)

        # P(C|Z) for each i: where z[idx][i] = 1
        c_probs = c_z_i_1 * np.prod(c_z_i_0) / (c_z_i_0 + 1e-6)

        return c_probs

    def match_observation_ratios(self, norm_probs: np.array):
        """
        Match the observation ratios using Hungarian Algorithm

        Parameters
        ----------
        norm_probs : np.array
            Normalized probabilities of z, shape=(n_samples, n_clusters)
        """
        n_samples, n_clusters = norm_probs.shape
        curr_ratios = np.sum(norm_probs, axis=0) / n_samples

        # Compute cost matrix
        cost_matrix = (curr_ratios.reshape(n_clusters, 1) - self.observed_ratios) ** 2

        # Hungarian Algorithm Matching
        _, col_ind = linear_sum_assignment(cost_matrix)
        self.cur_cols = col_ind
