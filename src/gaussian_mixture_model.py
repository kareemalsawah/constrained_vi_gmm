"""
Variational inference implementation of GMMs
"""
from typing import Tuple, Dict
from dataclasses import dataclass

import numpy as np
from scipy.special import digamma

from .visualizer import Visualizer

# pylint: disable=too-many-function-args


@dataclass
class PriorParameters:
    """
    Prior parameters for all variables

    Parameters
    ----------
    mu : np.array
        Mean of the normal prior of the mean of the features
        Part of the Normal inverse Wishart prior
        shape=(n_dims)
    beta : float
        Precision of the normal prior of the mean of the features
        Part of the Normal inverse Wishart prior
    w_inv : np.array
        Inverse covariance matrix of the wishart prior of the covariance of the features
        Part of the Normal inverse Wishart prior
        shape=(n_dims, n_dims)
    v : float
        Degrees of freedom of the wishart prior of the covariance of the features
        Part of the Normal inverse Wishart prior
    alpha : np.array
        Dirichlet prior of the mixture weights
        shape=(n_clusters)
    """

    mu: np.array = None
    beta: float = None
    w_inv: np.array = None
    v: float = None

    alpha: np.array = None

    def set_default(self, n_dims: int, n_clusters: int):
        """
        Set default values for the prior parameters
        Set to a weakly informative prior

        Parameters
        ----------
        n_dims : int
            Dimensionality of the dataset
        n_clusters : int
            Number of clusters
        """
        self.mu = np.zeros(n_dims)
        self.beta = 0.01
        self.w_inv = np.eye(n_dims) / 10
        self.v = 1

        self.alpha = np.ones(n_clusters) * 10


class GaussianMixtureModel:
    """
    Variational inference implementation of GMMs

    Parameters
    ----------
    n_clusters : int
        Number of clusters
    max_iters : int
        Maximum number of iterations
    convergence_eps : float
        Convergence threshold for the variational inference algorithm
    """

    def __init__(
        self, n_clusters: int, max_iters: int = 100, convergence_eps: float = 1e-2
    ):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.convergence_eps = convergence_eps

    def compute_cluster_statistics(
        self, dataset: np.array, expected_z: np.array
    ) -> Tuple[np.array, np.array, np.array]:
        """
        Compute the cluster statistics:
        - x_bar: cluster means
        - S_k: cluster covariance
        - N_k: cluster ratios

        Parameters
        ----------
        dataset : np.array
            shape=(n_samples, n_dims)
        expected_z : np.array
            shape=(n_samples, n_clusters)

        Returns
        -------
        Tuple[np.array, np.array, np.array]
            x_bar: shape=(n_clusters, n_dims)
            S_k: shape=(n_clusters, n_dims, n_dims)
            N_k: shape=(n_clusters)
        """
        n_samples, n_dims = dataset.shape

        cluster_ratios = np.sum(
            expected_z, axis=0
        )  # shape=(n_clusters), N_k in derivation

        # mean of each cluster, shape = (n_clusters, n_dims)
        x_bar = np.sum(
            expected_z.reshape(n_samples, self.n_clusters, 1)
            * dataset.reshape(n_samples, 1, n_dims),
            axis=0,
        ) / (cluster_ratios.reshape(self.n_clusters, 1) + 1e-5)

        # cluster covariance, shape = (n_clusters, n_dims, n_dims)
        diff = dataset.reshape(n_samples, 1, n_dims, 1) - x_bar.reshape(
            1, self.n_clusters, n_dims, 1
        )
        cov_s = np.einsum(
            "nkij,nkjb->nkib", diff, diff.transpose(0, 1, 3, 2), optimize=True
        )  # shape=(n_samples, n_clusters, n_dims, n_dims)
        cov_s_weighted = expected_z.reshape(n_samples, self.n_clusters, 1, 1) * cov_s
        s_k = np.sum(cov_s_weighted, axis=0) / (
            cluster_ratios.reshape(self.n_clusters, 1, 1) + 1e-5
        )  # shape=(n_clusters, n_dims, n_dims)

        return cluster_ratios, x_bar, s_k

    def update_prior_parameters(
        self, dataset: np.array, expected_z: np.array, prior_params: PriorParameters
    ) -> Dict[str, np.array]:
        """
        Update the prior parameters

        Parameters
        ----------
        dataset : np.array
            shape=(n_samples, n_dims)
        expected_z : np.array
            Current estimate of the cluster assignments
            shape=(n_samples, n_clusters)
        prior_params : PriorParameters
            Prior parameters

        Returns
        -------
        Dict[str, np.array]
            Updated prior parameters: dictionary with the following
            - m_k: shape=(n_clusters, n_dims)
            - beta_k: shape=(n_clusters)
            - v_k: shape=(n_clusters)
            - w_inv_k: shape=(n_clusters, n_dims, n_dims)
            - alpha_k: shape=(n_clusters)
        """
        cluster_ratios, x_bar, s_k = self.compute_cluster_statistics(
            dataset, expected_z
        )
        _, n_clusters = expected_z.shape
        n_dims = dataset.shape[1]

        beta_k = prior_params.beta + cluster_ratios  # shape = (n_clusters)
        m_k = (
            prior_params.beta * prior_params.mu.reshape(1, n_dims)
            + cluster_ratios.reshape(n_dims, 1) * x_bar
        ) / (
            beta_k.reshape(n_clusters, 1) + 1e-5
        )  # shape=(n_clusters, n_dims)
        v_k = prior_params.v + cluster_ratios  # shape=(n_clusters)
        w_inv_k = (
            prior_params.w_inv.reshape(1, n_dims, n_dims)
            + cluster_ratios.reshape(n_clusters, 1, 1) * s_k
        )  # shape=(n_clusters, n_dims, n_dims)
        factor = (
            prior_params.beta
            * cluster_ratios.reshape(n_clusters, 1, 1)
            / (prior_params.beta + cluster_ratios.reshape(n_clusters, 1, 1))
        )  # shape=(n_clusters, 1, 1)
        diff = x_bar.reshape(n_clusters, n_dims, 1) - prior_params.mu.reshape(
            1, n_dims, 1
        )
        w_cov_mat = np.einsum(
            "ijk,ikb->ijb", diff, diff.transpose(0, 2, 1), optimize=True
        )  # shape=(n_clusters, n_dims, n_dims)
        w_inv_k += factor * w_cov_mat

        alpha_k = prior_params.alpha + cluster_ratios

        return {
            "m_k": m_k,
            "beta_k": beta_k,
            "v_k": v_k,
            "w_inv_k": w_inv_k,
            "alpha_k": alpha_k,
        }

    def compute_log_det_sig_inv(self, updated_priors: Dict[str, np.array]) -> np.array:
        """
        Compute the log determinant of the covariance matrix

        Parameters
        ----------
        updated_priors : Dict[str, np.array]
            Updated priors parameters, see self.update_prior_parameters

        Returns
        -------
        np.array
            log_det_sig_inv: shape=(n_clusters)
        """
        n_clusters, n_dims = updated_priors["m_k"].shape
        inp_digamma = (1 - np.arange(1, n_dims + 1)) / 2  # shape=(n_dims)
        inp_digamma = inp_digamma.reshape(1, n_dims) + updated_priors["v_k"].reshape(
            n_clusters, 1
        )  # shape=(n_clusters,n_dims)
        log_determ = np.linalg.slogdet(
            np.linalg.inv(updated_priors["w_inv_k"])
        ).logabsdet  # shape=(n_clusters), TODO: speedup by removing inv
        log_det_sig_inv = (
            np.sum(digamma(inp_digamma), axis=1) + log_determ + n_dims * np.log(2)
        )  # shape=(n_clusters)
        return log_det_sig_inv

    def update_expected_z(
        self,
        updated_priors: Dict[str, np.array],
        dataset: np.array,
        log_pi: np.array,
        log_det_sig_inv: np.array,
    ) -> np.array:
        """
        Update the expected cluster assignments
        See derivation for more details

        Parameters
        ----------
        updated_priors : Dict[str, np.array]
            Updated priors parameters, see self.update_prior_parameters
        dataset : np.array
            shape=(n_samples, n_dims)
        log_pi : np.array
            Updated log mixture weights, shape=(n_clusters)
        log_det_sig_inv : np.array
            Updated log determinant of the covariance matrix, shape=(n_clusters)

        Returns
        -------
        np.array
            expected_z: shape=(n_samples, n_clusters)
        """
        n_samples, n_dims = dataset.shape
        n_clusters = self.n_clusters

        # compute gaussian loss
        diff = dataset.reshape(n_samples, 1, 1, n_dims) - updated_priors["m_k"].reshape(
            1, n_clusters, 1, n_dims
        )
        term1 = np.einsum(
            "nkij,nkjb->nkib",
            diff,
            np.linalg.inv(updated_priors["w_inv_k"]).reshape(
                1, n_clusters, n_dims, n_dims
            ),
            optimize=True,
        )
        term2 = np.einsum(
            "nkij,nkjb->nkib",
            term1,
            diff.transpose(0, 1, 3, 2),
            optimize=True,
        ).reshape(n_samples, n_clusters)

        # compute expected_z
        term3 = -1 * updated_priors["v_k"].reshape(1, n_clusters) * term2 / 2
        term4 = term3 - n_dims / (2 * updated_priors["beta_k"].reshape(1, n_clusters))

        unnormalized_log_probs = (
            log_pi.reshape(1, n_clusters)
            + 0.5 * log_det_sig_inv.reshape(1, n_clusters)
            + term4
        )

        unnormalized_probs = np.exp(unnormalized_log_probs)
        expected_z = unnormalized_probs / (
            np.sum(unnormalized_probs, axis=1, keepdims=True) + 1e-5
        )
        return expected_z

    def fit(self, dataset: np.array, plot: bool = False):
        """
        Variational inference implementation of GMMs

        Parameters
        ----------
        dataset : np.array
            Input dataset, shape = (N,D)
            N: number of samples
            D: dimensionality of the dataset
        """
        visualizer = Visualizer(dataset, plot)
        n_samples, n_dims = dataset.shape
        n_clusters = self.n_clusters

        # prior parameters
        prior_params = PriorParameters()
        prior_params.set_default(n_dims, n_clusters)

        # randomly initialize expected_z
        expected_z = np.random.uniform(
            0, 1, (n_samples, n_clusters)
        )  # rnk in derivation
        expected_z = expected_z / np.sum(expected_z, axis=1, keepdims=True)  # normalize
        prev_means = None

        for _ in range(self.max_iters):
            updated_priors = self.update_prior_parameters(
                dataset, expected_z, prior_params
            )

            # update pi, mu, sigma
            log_pi = digamma(updated_priors["alpha_k"]) - digamma(
                np.sum(updated_priors["alpha_k"])
            )
            log_det_sig_inv = self.compute_log_det_sig_inv(updated_priors)

            # update expected_z
            expected_z = self.update_expected_z(
                updated_priors, dataset, log_pi, log_det_sig_inv
            )
            cluster_ratios = np.sum(expected_z, axis=0) / n_samples

            # plot if enabled
            cluster_cov = updated_priors["w_inv_k"] / updated_priors["beta_k"].reshape(
                n_clusters, 1, 1
            )
            visualizer.plot_gaussians(
                updated_priors["m_k"],
                cluster_cov,
                cluster_ratios,
            )

            # check for convergence
            if prev_means is not None:
                if (
                    np.linalg.norm(prev_means - updated_priors["m_k"], axis=1).mean()
                    < self.convergence_eps
                ):
                    break

            prev_means = np.copy(updated_priors["m_k"])

        return updated_priors, expected_z
