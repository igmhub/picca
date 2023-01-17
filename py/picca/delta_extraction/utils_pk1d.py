"""This module defines a set of functions to manage specifics of Pk1D analysis
when computing the deltas.

This module provides three functions:
    - exp_diff
    - exp_diff_desi
    - spectral_resolution
    - spectral_resolution_desi
See the respective documentation for details
"""
import logging
import warnings
import numpy as np
from numba import njit

from picca.delta_extraction.utils import SPEED_LIGHT

# create logger
module_logger = logging.getLogger(__name__)


def exp_diff(hdul, log_lambda):
    """Compute the difference between exposures.

    More precisely compute de semidifference between two customized coadded
    spectra obtained from weighted averages of the even-number exposures, for
    the first spectrum, and of the odd-number exposures, for the second one
    (see section 3.2 of Chabanier et al. 2019).

    Arguments
    ---------
    hdul: fitsio.fitslib.FITS
    Header Data Unit List opened by fitsio

    log_lambda: array of floats
    Array containing the logarithm of the wavelengths (in Angs)

    Return
    ------
    exposures_diff: array of float
    The difference between exposures
    """
    num_exp_per_col = hdul[0].read_header()['NEXP'] // 2
    flux_total_odd = np.zeros(log_lambda.size)
    ivar_total_odd = np.zeros(log_lambda.size)
    flux_total_even = np.zeros(log_lambda.size)
    ivar_total_even = np.zeros(log_lambda.size)

    if num_exp_per_col < 2:
        module_logger.debug("Not enough exposures for diff")

    for index_exp in range(num_exp_per_col):
        for index_col in range(2):
            log_lambda_exp = hdul[(4 + index_exp +
                                   index_col * num_exp_per_col)]["loglam"][:]
            flux_exp = hdul[(4 + index_exp +
                             index_col * num_exp_per_col)]["flux"][:]
            ivar_exp = hdul[(4 + index_exp +
                             index_col * num_exp_per_col)]["ivar"][:]
            mask = hdul[4 + index_exp + index_col * num_exp_per_col]["mask"][:]
            log_lambda_bins = np.searchsorted(log_lambda, log_lambda_exp)

            # exclude masks 25 (COMBINEREJ), 23 (BRIGHTSKY)?
            rebin_ivar_exp = np.bincount(log_lambda_bins,
                                         weights=ivar_exp * (mask & 2**25 == 0))
            rebin_flux_exp = np.bincount(log_lambda_bins,
                                         weights=(ivar_exp * flux_exp *
                                                  (mask & 2**25 == 0)))

            if index_exp % 2 == 1:
                flux_total_odd[:len(rebin_ivar_exp) - 1] += rebin_flux_exp[:-1]
                ivar_total_odd[:len(rebin_ivar_exp) - 1] += rebin_ivar_exp[:-1]
            else:
                flux_total_even[:len(rebin_ivar_exp) - 1] += rebin_flux_exp[:-1]
                ivar_total_even[:len(rebin_ivar_exp) - 1] += rebin_ivar_exp[:-1]

    w = ivar_total_odd > 0
    flux_total_odd[w] /= ivar_total_odd[w]
    w = ivar_total_even > 0
    flux_total_even[w] /= ivar_total_even[w]

    alpha = 1
    if num_exp_per_col % 2 == 1:
        num_even_exp = (num_exp_per_col - 1) // 2
        alpha = np.sqrt(4. * num_even_exp *
                        (num_even_exp + 1)) / num_exp_per_col
    # TODO: CHECK THE * alpha (Nathalie)
    exposures_diff = 0.5 * (flux_total_even - flux_total_odd) * alpha

    return exposures_diff


def exp_diff_desi(spec_dict, mask_targetid):
    """Computes the difference between exposures.

    More precisely computes de semidifference between two customized coadded
    spectra obtained from weighted averages of the even-number exposures, for
    the first spectrum, and of the odd-number exposures, for the second one.

    Args:
        spec_dict: dict
            spec dictionary from desi_healpix/tile
        mask targetid: array of ints
            Targetids to select for calculating the exp differences

    Returns:
        The difference between exposures
    """
    ivar_unsorted = np.atleast_2d(spec_dict["IVAR"][mask_targetid])
    num_exp = ivar_unsorted.shape[0]

    # Putting the lowest ivar exposure at the end if the number of exposures is odd
    argsort = np.arange(num_exp)
    if num_exp % 2 == 1:
        argmin_ivar = np.argmin(np.mean(ivar_unsorted, axis=1))
        argsort[-1], argsort[argmin_ivar] = argsort[argmin_ivar], argsort[-1]

    flux = np.atleast_2d(spec_dict["FLUX"][mask_targetid])[argsort, :]
    ivar = ivar_unsorted[argsort, :]
    if num_exp < 2:
        module_logger.debug("Not enough exposures for diff, Spectra rejected")
        return None
    if num_exp > 100:
        module_logger.debug("More than 100 exposures, potentially wrong file "
                            "type and using wavelength axis here, skipping?")
        return None

    # Computing ivar and flux for odd and even exposures
    ivar_total = np.zeros(flux.shape[1])
    flux_total_odd = np.zeros(flux.shape[1])
    ivar_total_odd = np.zeros(flux.shape[1])
    flux_total_even = np.zeros(flux.shape[1])
    ivar_total_even = np.zeros(flux.shape[1])
    for index_exp in range(2 * (num_exp // 2)):
        flexp = flux[index_exp]
        ivexp = ivar[index_exp]
        if index_exp % 2 == 1:
            flux_total_odd += flexp * ivexp
            ivar_total_odd += ivexp
        else:
            flux_total_even += flexp * ivexp
            ivar_total_even += ivexp
    ivar_total = ivar.sum(axis=0)

    # Masking and dividing flux by ivar
    w_odd = ivar_total_odd > 0
    flux_total_odd[w_odd] /= ivar_total_odd[w_odd]
    w_even = ivar_total_even > 0
    flux_total_even[w_even] /= ivar_total_even[w_even]

    # Computing alpha correction
    w = w_odd & w_even & (ivar_total > 0)
    alpha_array = np.ones(flux.shape[1])
    alpha_array[w] = (1 / np.sqrt(ivar_total[w])) / (0.5 * np.sqrt(
        (1 / ivar_total_even[w]) + (1 / ivar_total_odd[w])))
    diff = 0.5 * (flux_total_even - flux_total_odd) * alpha_array
    return diff


def spectral_resolution(wdisp,
                        with_correction=False,
                        fiberid=None,
                        log_lambda=None):
    # TODO: fix docstring
    """Compute the spectral resolution

    Arguments
    ---------
    wdisp: array of floats
    ?

    with_correction: bool - default: False
    If True, applies the correction to the pipeline noise described
    in section 2.4.3 of Palanque-Delabrouille et al. 2013

    fiberid: int or None - default: None
    Fiberid of the observations

    log_lambda: array or None - default: None
    Logarithm of the wavelength (in Angstroms)

    Return
    ------
    reso: array of floats
    The spectral resolution
    """
    reso = wdisp * SPEED_LIGHT * 1.0e-4 * np.log(10.)

    if with_correction:
        lambda_ = np.power(10., log_lambda)
        # compute the wavelength correction
        correction = 1.267 - 0.000142716 * lambda_ + 1.9068e-08 * lambda_ * lambda_
        correction[lambda_ > 6000.0] = 1.097

        # add the fiberid correction
        # fiberids greater than 500 corresponds to the second spectrograph
        fiberid = fiberid % 500
        if fiberid < 100:
            correction = (1. + (correction - 1) * .25 + (correction - 1) * .75 *
                          (fiberid) / 100.)
        elif fiberid > 400:
            correction = (1. + (correction - 1) * .25 + (correction - 1) * .75 *
                          (500 - fiberid) / 100.)

        # apply the correction
        reso *= correction

    return reso


@njit  #("f8[:](float32[:, :], i8)")
def _find_nonzero_abs_min_per_row(reso_matrix, num_rows):
    """Find the non-zero absolute minimum of the resolution matrix
    for each row

    Arguments
    ---------
    reso_matrix: 2D array of shape (num_diags, num_rows)
    Resolution matrix

    num_rows: int
    Number of rows

    Return
    ------
    nonzero_abs_min_per_row: 1D array
    An array of size num_rows with non-zero abs. min
    """
    nonzero_abs_min_per_row = np.zeros(num_rows)
    for irow in range(num_rows):
        row = reso_matrix[:, irow]
        nonzero_idx = row.nonzero()[0]
        if nonzero_idx.size == 0:
            continue

        nonzero_abs_min_per_row[irow] = np.abs(row[nonzero_idx]).min()

    return nonzero_abs_min_per_row


def spectral_resolution_desi(reso_matrix, lambda_):
    """Compute the spectral resolution for DESI spectra

    Arguments
    ---------
    reso_matrix: 2D float32 array of shape (num_diags, num_rows)
    Resolution matrix

    lambda_: 1D array
    Wavelength (in Angstroms)

    Return
    ------
    reso_in_km_per_s: array
    The spectral resolution
    """
    if lambda_ is None:
        return None

    delta_log_lambda = np.empty_like(lambda_)
    delta_log_lambda[:-1] = np.diff(np.log10(lambda_))
    #note that this would be the same result as before (except for the missing bug) in
    #case of log-uniform binning, but for linear binning pixel size chenges wrt lambda
    delta_log_lambda[-1] = delta_log_lambda[-2] + (delta_log_lambda[-2] -
                                                   delta_log_lambda[-3])

    num_diags, num_rows = reso_matrix.shape
    num_offdiags = num_diags // 2

    # shift every row to a small positive value (not done anymore)
    # new resolution model is then Gaussian + constant
    # using an arbitrary epsilon is not stable
    # use nonzero absolute minimum of the row
    nonzero_abs_min_per_row = _find_nonzero_abs_min_per_row(
        reso_matrix, num_rows)

    # mask out rows that are all zeros
    w = nonzero_abs_min_per_row > 0
    if not np.any(w):
        return np.zeros_like(lambda_), np.zeros_like(lambda_)

    reso = reso_matrix[:, w]  #- shift[w]

    #assume reso = A*exp(-(x-central_pixel_pos)**2 / 2 / sigma**2)
    #=> sigma = sqrt((x-central_pixel_pos)/2)**2 / log(A/reso)
    #   A = reso(central_pixel_pos)
    # the following averages over estimates for four symmetric values of x
    indices = np.array([-2, -1, 1, 2], dtype=int)
    with warnings.catch_warnings():
        # this ignores all warnings in this part of the code, could in principle
        # specify the exact warnings to be supressed
        warnings.filterwarnings(
            'ignore'
        )
        ratios = reso[num_offdiags, :] / reso[num_offdiags + indices, :]
        ratios = np.log(ratios)
        w2 = ratios > 0
        norm = np.sum(w2, axis=0)
        new_ratios = np.zeros_like(ratios)
        new_ratios[w2] = 1. / np.sqrt(ratios[w2])
        # ratios = 1./np.sqrt(np.log(ratios))

        rms_in_pixel = np.empty_like(lambda_)
        rms_in_pixel[w] = np.abs(indices).dot(new_ratios) / np.sqrt(2.) / norm
        rms_in_pixel[~w] = rms_in_pixel[w].mean()

        reso_in_km_per_s = (rms_in_pixel * SPEED_LIGHT * delta_log_lambda *
                            np.log(10.0))  #this is FWHM

    return rms_in_pixel, reso_in_km_per_s
