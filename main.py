"""
Example on using the ConstrainedGMM class to cluster data with constraints
"""
import numpy as np
from src import ConstrainedGMM, GaussianClusters, GaussianMixtureModel


def example_1():
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


def example_2_unconstrained():
    """
    Example added as GIF in the README
    """
    np.random.seed(42)
    points_cluster_1 = np.array([[1, 1], [1.2, 0.8], [1.4, 0.6], [1.6, 0.4]])
    points_cluster_2 = np.array([[-1.2, -1.5], [-0.8, -1.8]])

    point_between_them = (
        np.mean(points_cluster_1, axis=0) + np.mean(points_cluster_2, axis=0)
    ) / 2

    # make it closer to the smaller cluster
    point_between_them -= 0.4

    all_points = np.concatenate(
        [points_cluster_1, points_cluster_2, point_between_them.reshape(1, 2)],
        axis=0,
    )

    gmm = GaussianMixtureModel(2, 100, 1e-3)
    _, expected_z = gmm.fit(all_points, plot=True)


def example_2_constrained():
    """
    Example added as GIF in the README
    """
    np.random.seed(42)
    points_cluster_1 = np.array([[1, 1], [1.2, 0.8], [1.4, 0.6], [1.6, 0.4]])
    points_cluster_2 = np.array([[-1.2, -1.5], [-0.8, -1.8]])

    point_between_them = (
        np.mean(points_cluster_1, axis=0) + np.mean(points_cluster_2, axis=0)
    ) / 2

    # make it closer to the smaller cluster
    point_between_them -= 0.4

    all_points = np.concatenate(
        [points_cluster_1, points_cluster_2, point_between_them.reshape(1, 2)],
        axis=0,
    )

    # ratios wanting to assign it to larger cluster at idx 0
    counts = np.array([5, 2]) / 7
    gmm = ConstrainedGMM(counts, 0.01, 2, 100, 1e-3)
    _, expected_z = gmm.fit(all_points, plot=True)


def example_2_unconstrained():
    """
    Example added as GIF in the README
    """
    np.random.seed(42)
    points_cluster_1 = np.array([[1, 1], [1.2, 0.8], [1.4, 0.6], [1.6, 0.4]])
    points_cluster_2 = np.array([[-1.2, -1.5], [-0.8, -1.8]])

    point_between_them = (
        np.mean(points_cluster_1, axis=0) + np.mean(points_cluster_2, axis=0)
    ) / 2

    # make it closer to the smaller cluster
    point_between_them -= 0.4

    all_points = np.concatenate(
        [points_cluster_1, points_cluster_2, point_between_them.reshape(1, 2)],
        axis=0,
    )

    gmm = GaussianMixtureModel(2, 100, 1e-3)
    _, expected_z = gmm.fit(all_points, plot=True)


def example_3():
    """
    Another example
    """
    np.random.seed(42)
    max_iters = 30
    convergence_tol = 1e-3
    precision_of_counts = 0.01  # lower means more confident

    all_points = np.array([[-1, 0], [1, 0], [0.2, 0]])
    counts = np.array([0.5, 0.5])

    # all_points = np.array([[-1, 0], [1, 0], [-0.2, 0]])
    # counts = np.array([0.5, 0.5])

    # all_points = np.array([[-1, 0.2], [-1, -0.2], [1, 0], [0.2, 0]])
    # counts = np.array([0.5, 0.5])

    # all_points = np.array([[-1, 0.2], [-1, -0.2], [1, 0], [0.2, 0]])
    # counts = np.array([0.75, 0.25])

    # all_points = np.array([[-1, 0], [1, 0]])
    # counts = np.array([0.45, 0.45, 0.1])

    # all_points = np.array([[-1, 0.2], [-1, -0.2], [1, 0]])
    # counts = np.array([0.6, 0.3, 0.1])

    gmm = ConstrainedGMM(
        counts, precision_of_counts, counts.shape[0], max_iters, convergence_tol
    )
    _, expected_z = gmm.fit(all_points, plot=True)

    print(expected_z)


if __name__ == "__main__":
    # example_2_unconstrained()
    # example_2_constrained()
    example_3()
