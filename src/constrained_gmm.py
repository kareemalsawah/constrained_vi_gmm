"""
Constrained Gaussian Mixture Model
"""
from typing import Dict

import numpy as np

from .gibbs import PermInvGibbsSampler
from .gaussian_mixture_model import GaussianMixtureModel


class ConstrainedGMM(GaussianMixtureModel):
    def __init__(
        self, cluster_ratios: np.array, cluster_ratio_var: float, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.gibbs_sampler = PermInvGibbsSampler(cluster_ratios, cluster_ratio_var)

    def update_expected_z(
        self,
        updated_priors: Dict[str, np.array],
        dataset: np.array,
        log_pi: np.array,
        log_det_sig_inv: np.array,
    ) -> np.array:
        """
        Update the expected z matrix
        Uses the gibbs sampler to update the expected z matrix with the P(C|Z) weights
        C: observed cluster ratios
        """
        expected_z = super().update_expected_z(
            updated_priors, dataset, log_pi, log_det_sig_inv
        )

        expected_z = self.gibbs_sampler.expected_z(expected_z)
        return expected_z

    def fit(self, *args, **kwargs):
        """
        Fit the model
        """
        updated_priors, expected_z = super().fit(*args, **kwargs)

        # reorder according to cluster ratios in gibbs sampler
        for key, val in updated_priors.items():
            updated_priors[key] = val[self.gibbs_sampler.cur_cols]
        expected_z = expected_z[:, self.gibbs_sampler.cur_cols]

        return updated_priors, expected_z
