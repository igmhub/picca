"""This module defines a set of functions to manage specifics of Pk1D analysis
when computing the deltas.

This module provides three functions:
    - exp_diff
    - spectral_resolution
    - spectral_resolution_desi
See the respective documentation for details
"""
import numpy as np

from picca.constants import SPEED_LIGHT
from picca.utils import userprint


def exp_diff(hdul, log_lambda):
    """Computes the difference between exposures.

    More precisely compute de semidifference between two customized coadded
    spectra obtained from weighted averages of the even-number exposures, for
    the first spectrum, and of the odd-number exposures, for the second one
    (see section 3.2 of Chabanier et al. 2019).

    Arguments
    ---------
    hdul: fitsio.fitslib.FITS
    Header Data Unit List opened by fitsio

    log_lambda: array of float
    Array containing the logarithm of the wavelengths (in Angs)

    Return
    ------
    exposure_diff: array of float
    The difference between exposures
    """
    num_exp_per_col = hdul[0].read_header()['NEXP'] // 2
    flux_total_odd = np.zeros(log_lambda.size)
    ivar_total_odd = np.zeros(log_lambda.size)
    flux_total_even = np.zeros(log_lambda.size)
    ivar_total_even = np.zeros(log_lambda.size)

    if num_exp_per_col < 2:
        userprint("DBG : not enough exposures for diff")

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


def exp_diff_desi(hdul, mask_targetid):
    """Compute the difference between exposures.

    More precisely compute de semidifference between two customized coadded
    spectra obtained from weighted averages of the even-number exposures, for
    the first spectrum, and of the odd-number exposures, for the second one
    (see section 3.2 of Chabanier et al. 2019).

    Arguments
    ---------
    hdul: fitsio.fitslib.FITS
    Header Data Unit List opened by fitsio

    mask targetid: array of int
    Targetids to select for calculating the exp differences

    Return
    ------
    diff: array of float
    The difference between exposures
    """
    argsort = np.flip(np.argsort(hdul["TEFF_LYA"][mask_targetid][:]))
    flux = hdul["FL"][mask_targetid][argsort, :]
    ivar = hdul["IV"][mask_targetid][argsort, :]
    teff_lya = hdul["TEFF_LYA"][mask_targetid][argsort]

    num_exp = len(flux)
    if num_exp < 2:
        userprint("DBG : not enough exposures for diff, spectra rejected")
        return None
    flux_total_odd = np.zeros(flux.shape[1])
    ivar_total_odd = np.zeros(flux.shape[1])
    flux_total_even = np.zeros(flux.shape[1])
    ivar_total_even = np.zeros(flux.shape[1])
    teff_even = 0
    teff_odd = 0
    teff_total = np.sum(teff_lya)
    teff_last = teff_lya[-1]
    for index_exp in range(2 * (num_exp // 2)):
        flexp = flux[index_exp]
        ivexp = ivar[index_exp]
        teff_lya_exp = teff_lya[index_exp]
        if index_exp % 2 == 1:
            flux_total_odd += flexp * ivexp
            ivar_total_odd += ivexp
            teff_odd += teff_lya_exp
        else:
            flux_total_even += flexp * ivexp
            ivar_total_even += ivexp
            teff_even += teff_lya_exp

    w = ivar_total_odd > 0
    flux_total_odd[w] /= ivar_total_odd[w]
    w = ivar_total_even > 0
    flux_total_even[w] /= ivar_total_even[w]

    alpha = 1
    if num_exp % 2 == 1:
        # Ignasi: these variables were unsused so I commented them out
        # n_even = (num_exp - 1) // 2
        # alpha_n_old = np.sqrt(4. * n_even * (num_exp - n_even)) / num_exp
        # alpha_N = np.sqrt(4.*t_even*(t_exp-t_even))/t_exp
        # alpha_c_new = np.sqrt((teff_total - teff_last) / teff_total)
        alpha_n_new = np.sqrt(
            (teff_total - teff_last) * (teff_total + teff_last)) / teff_total
        alpha = alpha_n_new
    diff = 0.5 * (flux_total_even - flux_total_odd) * alpha

    return diff


def spectral_resolution(wdisp,
                        with_correction=False,
                        fiberid=None,
                        log_lambda=None):
    # TODO: fix docstring
    """Compute the spectral resolution

    Arguments
    ---------
    wdisp: array of float
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
    reso: array of float
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


def spectral_resolution_desi(reso_matrix, log_lambda):
    """Compute the spectral resolution for DESI spectra
    Note that this is only giving rough estimates, it relies on a Gaussian resolution matrix

    Arguments
    ---------
    reso_matrix: array
    Resolution matrix

    log_lambda: array or None - default: None
    Logarithm of the wavelength (in Angstroms)

    Return
    ------
    rms_in_pixel: float
    The spectral resolution

    avg_reso_in_km_per_s: float
    The average resolution in km/s
    """
    delta_log_lambda = ((log_lambda[-1] - log_lambda[0]) /
                        float(len(log_lambda) - 1))
    reso = np.clip(reso_matrix, 1.0e-6, 1.0e6)

    #
    rms_in_pixel = (
        np.sqrt(1.0 / 2.0 /
                np.log(reso[len(reso) // 2][:] / reso[len(reso) // 2 - 1][:])) +
        np.sqrt(4.0 / 2.0 /
                np.log(reso[len(reso) // 2][:] / reso[len(reso) // 2 - 2][:])) +
        np.sqrt(1.0 / 2.0 /
                np.log(reso[len(reso) // 2][:] / reso[len(reso) // 2 + 1][:])) +
        np.sqrt(4.0 / 2.0 / np.log(
            reso[len(reso) // 2][:] / reso[len(reso) // 2 + 2][:]))) / 4.0

    avg_reso_in_km_per_s = (rms_in_pixel * SPEED_LIGHT * delta_log_lambda *
                            np.log(10.0))

    return rms_in_pixel, avg_reso_in_km_per_s
