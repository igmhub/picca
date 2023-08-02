"""This module defines a set of functions to compute the deltas.

This module provides three functions:
    - compute_mean_cont
    - compute_var_stats
    - stack
See the respective documentation for details
"""
import numpy as np

from .data import Forest

def stack(data, stack_from_deltas=False):
    """Computes a stack of the delta field as a function of wavelength

    Args:
        data: dict
            A dictionary with the forests in each healpix. If stack_from_deltas
            is passed, then the dictionary must contain the deltas in each
            healpix
        stack_from_deltas: bool - default: False
            Flag to determine whether to stack from deltas or compute them

    Returns:
        The following variables:
            stack_log_lambda: Logarithm of the wavelengths (in Angs).
            stack_delta: The stacked delta field.
            stack_weight: Total weight on the delta_stack
    """
    num_bins = int((Forest.log_lambda_max - Forest.log_lambda_min) /
                   Forest.delta_log_lambda) + 1
    stack_log_lambda = (Forest.log_lambda_min +
                        np.arange(num_bins) * Forest.delta_log_lambda)
    stack_delta = np.zeros(num_bins)
    stack_weight = np.zeros(num_bins)
    for healpix in sorted(list(data.keys())):
        for forest in data[healpix]:
            if stack_from_deltas:
                delta = forest.delta
                weights = forest.weights
            else:
                delta = forest.flux / forest.cont
                var_lss = Forest.get_var_lss(forest.log_lambda)
                eta = Forest.get_eta(forest.log_lambda)
                fudge = Forest.get_fudge(forest.log_lambda)
                var = 1. / forest.ivar / forest.cont**2
                variance = eta * var + var_lss + fudge / var
                weights = 1. / variance

            bins = ((forest.log_lambda - Forest.log_lambda_min) /
                    Forest.delta_log_lambda + 0.5).astype(int)
            rebin = np.bincount(bins, weights=delta * weights)
            stack_delta[:len(rebin)] += rebin
            rebin = np.bincount(bins, weights=weights)
            stack_weight[:len(rebin)] += rebin

    w = stack_weight > 0
    stack_delta[w] /= stack_weight[w]

    return stack_log_lambda, stack_delta, stack_weight
