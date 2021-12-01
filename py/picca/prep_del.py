"""This module defines a set of functions to compute the deltas.

This module provides three functions:
    - compute_mean_cont
    - compute_var_stats
    - stack
See the respective documentation for details
"""
import numpy as np
import iminuit

from .data import Forest
from .utils import userprint


def compute_mean_cont(data):
    """Computes the mean quasar continuum over the whole sample.

    Args:
        data: dict
            A dictionary with the read forests in each healpix

    Returns:
        The following variables:
            log_lambda_rest_frame: Logarithm of the wavelengths at rest-frame
                (in Angs).
            mean_cont: Mean quasar continuum over the whole sample
            mean_cont_weight: Total weight on the mean quasar continuum
    """
    num_bins = (int(
        (Forest.log_lambda_max_rest_frame - Forest.log_lambda_min_rest_frame) /
        Forest.delta_log_lambda) + 1)
    mean_cont = np.zeros(num_bins)
    mean_cont_weight = np.zeros(num_bins)
    log_lambda_rest_frame = (
        Forest.log_lambda_min_rest_frame + (np.arange(num_bins) + .5) *
        (Forest.log_lambda_max_rest_frame - Forest.log_lambda_min_rest_frame) /
        num_bins)
    for healpix in sorted(list(data.keys())):
        for forest in data[healpix]:
            bins = ((forest.log_lambda - Forest.log_lambda_min_rest_frame -
                     np.log10(1 + forest.z_qso)) /
                    (Forest.log_lambda_max_rest_frame -
                     Forest.log_lambda_min_rest_frame) * num_bins).astype(int)
            var_lss = Forest.get_var_lss(forest.log_lambda)
            eta = Forest.get_eta(forest.log_lambda)
            fudge = Forest.get_fudge(forest.log_lambda)
            var_pipe = 1. / forest.ivar / forest.cont**2
            variance = eta * var_pipe + var_lss + fudge / var_pipe
            weights = 1 / variance
            cont = np.bincount(bins,
                               weights=forest.flux / forest.cont * weights)
            mean_cont[:len(cont)] += cont
            cont = np.bincount(bins, weights=weights)
            mean_cont_weight[:len(cont)] += cont

    w = mean_cont_weight > 0
    mean_cont[w] /= mean_cont_weight[w]
    mean_cont /= mean_cont.mean()
    return log_lambda_rest_frame, mean_cont, mean_cont_weight


def compute_var_stats(data, limit_eta=(0.5, 1.5), limit_var_lss=(0., 0.3)):
    """Computes variance functions and statistics

    This function computes the statistics required to fit the mapping functions
    eta, var_lss, and fudge. It also computes the functions themselves. See
    equation 4 of du Mas des Bourboux et al. 2020 for details.

    Args:
        data: dict
            A dictionary with the read forests in each healpix
        limit_eta: tuple of floats
            Limits on the correction factor to the contribution of the pipeline
            estimate of the instrumental noise to the variance.
        limit_var_lss: tuple of floats
            Limits on the pixel variance due to Large Scale Structure
    Returns:
        The following variables:
            log_lambda: Logarithm of the wavelengths (in Angs).
            eta: Correction factor to the contribution of the pipeline
                estimate of the instrumental noise to the variance.
            var_lss: Pixel variance due to the Large Scale Strucure.
            fudge: Fudge contribution to the pixel variance.
            num_pixels: Number of pixels contributing to the array.
            var_pipe_values: Value of the pipeline variance in pipeline
                variance bins
            var_delta: Variance of the delta field. Binned according to var.
            var2_delta: Square of the variance of the delta field. Binned
                according to var.
            count: Number of pixels in each pipeline variance bin.
            num_qso: Number of quasars in each pipeline variance bin.
            chi2_in_bin: chi2 value obtained when fitting the functions eta,
                var_lss, and fudge for each of the wavelengths
            error_eta: Error on the correction factor to the contribution of the
                pipeline  estimate of the instrumental noise to the variance.
            error_var_lss: Error on the pixel variance due to the Large Scale
                Strucure.
            error_fudge: Error on the fudge contribution to the pixel variance.
    """
    # initialize arrays
    num_bins = 20
    eta = np.zeros(num_bins)
    var_lss = np.zeros(num_bins)
    fudge = np.zeros(num_bins)
    error_eta = np.zeros(num_bins)
    error_var_lss = np.zeros(num_bins)
    error_fudge = np.zeros(num_bins)
    num_pixels = np.zeros(num_bins)
    log_lambda = (Forest.log_lambda_min + (np.arange(num_bins) + .5) *
                  (Forest.log_lambda_max - Forest.log_lambda_min) / num_bins)

    # define an array to contain the possible values of pipeline variances
    # the measured pipeline variance of the deltas will be averaged using the
    # same binning, and the two arrays will be compared to fit the functions
    # eta, var_lss, and fudge
    num_var_bins = 100
    var_pipe_min = np.log10(1e-5)
    var_pipe_max = np.log10(2.)
    var_pipe_values = 10**(var_pipe_min +
                           ((np.arange(num_var_bins) + .5) *
                            (var_pipe_max - var_pipe_min) / num_var_bins))

    # initialize arrays to compute the statistics of deltas
    var_delta = np.zeros(num_bins * num_var_bins)
    mean_delta = np.zeros(num_bins * num_var_bins)
    var2_delta = np.zeros(num_bins * num_var_bins)
    count = np.zeros(num_bins * num_var_bins)
    num_qso = np.zeros(num_bins * num_var_bins)

    # compute delta statistics, binning the variance according to 'var'
    for healpix in sorted(list(data.keys())):
        for forest in data[healpix]:

            var_pipe = 1 / forest.ivar / forest.cont**2
            w = ((np.log10(var_pipe) > var_pipe_min) &
                 (np.log10(var_pipe) < var_pipe_max))

            # select the wavelength and the pipeline variance bins
            log_lambda_bins = ((forest.log_lambda - Forest.log_lambda_min) /
                               (Forest.log_lambda_max - Forest.log_lambda_min) *
                               num_bins).astype(int)
            var_pipe_bins = np.floor(
                (np.log10(var_pipe) - var_pipe_min) /
                (var_pipe_max - var_pipe_min) * num_var_bins).astype(int)

            # filter the values with a pipeline variance out of range
            log_lambda_bins = log_lambda_bins[w]
            var_pipe_bins = var_pipe_bins[w]

            # compute overall bin
            bins = var_pipe_bins + num_var_bins * log_lambda_bins

            # compute deltas
            delta = (forest.flux / forest.cont - 1)
            delta = delta[w]

            # add contributions to delta statistics
            rebin = np.bincount(bins, weights=delta)
            mean_delta[:len(rebin)] += rebin

            rebin = np.bincount(bins, weights=delta**2)
            var_delta[:len(rebin)] += rebin

            rebin = np.bincount(bins, weights=delta**4)
            var2_delta[:len(rebin)] += rebin

            rebin = np.bincount(bins)
            count[:len(rebin)] += rebin
            num_qso[np.unique(bins)] += 1

    # normalise and finish the computation of delta statistics
    w = count > 0
    var_delta[w] /= count[w]
    mean_delta[w] /= count[w]
    var_delta -= mean_delta**2
    var2_delta[w] /= count[w]
    var2_delta -= var_delta**2
    var2_delta[w] /= count[w]

    # fit the functions eta, var_lss, and fudge
    chi2_in_bin = np.zeros(num_bins)
    fudge_ref = 1e-7

    userprint(" Mean quantities in observer-frame")
    userprint(" loglam    eta      var_lss  fudge    chi2     num_pix ")
    for index in range(num_bins):
        # pylint: disable-msg=cell-var-from-loop
        # this function is defined differntly at each step of the loop
        def chi2(eta, var_lss, fudge):
            """Computes the chi2 of the fit of eta, var_lss, and fudge for a
            wavelength bin

            Args:
                eta: float
                    Correction factor to the contribution of the pipeline
                    estimate of the instrumental noise to the variance.
                var_lss: float
                    Pixel variance due to the Large Scale Strucure
                fudge: float
                    Fudge contribution to the pixel variance

            Global args (defined only in the scope of function
            compute_var_stats):
                var_delta: array of floats
                    Variance of the delta field
                var2_delta: array of floats
                    Square of the variance of the delta field
                index: int
                    Index with the selected wavelength bin
                num_bins: int
                    Number of wavelength bins
                num_var_bins: int
                    Number of bins in which the pipeline variance values are
                    split
                var_pipe_values: array of floats
                    Value of the pipeline variance in pipeline variance bins
                num_qso: array of ints
                    Number of quasars in each pipeline variance bin

            Returns:
                The obtained chi2
            """
            variance = eta * var_pipe_values + var_lss + fudge*fudge_ref / var_pipe_values
            chi2_contribution = (
                var_delta[index * num_var_bins:(index + 1) * num_var_bins] - variance)
            weights = var2_delta[index * num_var_bins:(index + 1) *
                                 num_var_bins]
            w = num_qso[index * num_var_bins:(index + 1) * num_var_bins] > 100
            return np.sum(chi2_contribution[w]**2 / weights[w])

        minimizer = iminuit.Minuit(chi2,
                                   name=("eta", "var_lss", "fudge"),
                                   eta=1.,
                                   var_lss=0.1,
                                   fudge=1.)
        minimizer.errors["eta"] = 0.05
        minimizer.errors["var_lss"] = 0.05
        minimizer.errors["fudge"] = 0.05
        minimizer.errordef = 1.
        minimizer.print_level = 0
        minimizer.limits["eta"] = limit_eta
        minimizer.limits["var_lss"] = limit_var_lss
        minimizer.limits["fudge"] = (0, None)
        minimizer.migrad()

        if minimizer.valid:
            minimizer.hesse()
            eta[index] = minimizer.values["eta"]
            var_lss[index] = minimizer.values["var_lss"]
            fudge[index] = minimizer.values["fudge"] * fudge_ref
            error_eta[index] = minimizer.errors["eta"]
            error_var_lss[index] = minimizer.errors["var_lss"]
            error_fudge[index] = minimizer.errors["fudge"] * fudge_ref
        else:
            eta[index] = 1.
            var_lss[index] = 0.1
            fudge[index] = 1. * fudge_ref
            error_eta[index] = 0.
            error_var_lss[index] = 0.
            error_fudge[index] = 0.
        num_pixels[index] = count[index * num_var_bins:(index + 1) *
                                  num_var_bins].sum()
        chi2_in_bin[index] = minimizer.fval

        #note that this has been changed for debugging purposes
        userprint(f" {log_lambda[index]:.3e} "
                  f"{eta[index]:.2e} {var_lss[index]:.2e} {fudge[index]:.2e} "+
                  f"{chi2_in_bin[index]:.2e} {int(num_pixels[index]):d} ") 
                  #f"{error_eta[index]:.2e} {error_var_lss[index]:.2e} {error_fudge[index]:.2e}")

    return (log_lambda, eta, var_lss, fudge, num_pixels, var_pipe_values,
            var_delta.reshape(num_bins, -1), var2_delta.reshape(num_bins, -1),
            count.reshape(num_bins, -1), num_qso.reshape(num_bins, -1),
            chi2_in_bin, error_eta, error_var_lss, error_fudge)


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
