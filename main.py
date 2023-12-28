"""
Example on using the ConstrainedGMM class to cluster data with constraints
"""
import numpy as np
from src import ConstrainedGMM, GaussianClusters


def main():
    """
    Main
    """
    cluster_means = np.array([[-1, -2], [3, 3]])
    cluster_covariances = np.array([[[1, 0], [0, 1]], [[2, 0], [0, 0.1]]])
    cluster_weights = np.array([0.2, 0.8])
    n_samples = 10
    gmm_generator = GaussianClusters(
        cluster_means, cluster_covariances, cluster_weights
    )

    dataset = gmm_generator.sample(n_samples)

    n_clusters = 2
    max_iters = 100
    tol = 1e-3
    ratio_var = 0.1
    gmm = ConstrainedGMM(cluster_weights, ratio_var, n_clusters, max_iters, tol)

    updated_priors, expected_z = gmm.fit(dataset, plot=True)
    pred_cov = updated_priors["w_inv_k"] / updated_priors["beta_k"].reshape(2, 1, 1)

    print("Cluster ratios:", np.sum(expected_z, axis=0) / n_samples)
    print("Cluster means:", updated_priors["m_k"])
    print("Cluster covariances:", pred_cov)


if __name__ == "__main__":
    main()
