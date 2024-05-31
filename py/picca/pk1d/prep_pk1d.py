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
    num_exp_per_col = hdul[0].read_header()["NEXP"] // 2
    flux_total_odd = np.zeros(log_lambda.size)
    ivar_total_odd = np.zeros(log_lambda.size)
    flux_total_even = np.zeros(log_lambda.size)
    ivar_total_even = np.zeros(log_lambda.size)

    if num_exp_per_col < 2:
        userprint("DBG : not enough exposures for diff")

    for index_exp in range(num_exp_per_col):
        for index_col in range(2):
            log_lambda_exp = hdul[(4 + index_exp + index_col * num_exp_per_col)][
                "loglam"
            ][:]
            flux_exp = hdul[(4 + index_exp + index_col * num_exp_per_col)]["flux"][:]
            ivar_exp = hdul[(4 + index_exp + index_col * num_exp_per_col)]["ivar"][:]
            mask = hdul[4 + index_exp + index_col * num_exp_per_col]["mask"][:]
            log_lambda_bins = np.searchsorted(log_lambda, log_lambda_exp)

            # exclude masks 25 (COMBINEREJ), 23 (BRIGHTSKY)?
            rebin_ivar_exp = np.bincount(
                log_lambda_bins, weights=ivar_exp * (mask & 2**25 == 0)
            )
            rebin_flux_exp = np.bincount(
                log_lambda_bins, weights=(ivar_exp * flux_exp * (mask & 2**25 == 0))
            )

            if index_exp % 2 == 1:
                flux_total_odd[: len(rebin_ivar_exp) - 1] += rebin_flux_exp[:-1]
                ivar_total_odd[: len(rebin_ivar_exp) - 1] += rebin_ivar_exp[:-1]
            else:
                flux_total_even[: len(rebin_ivar_exp) - 1] += rebin_flux_exp[:-1]
                ivar_total_even[: len(rebin_ivar_exp) - 1] += rebin_ivar_exp[:-1]

    w = ivar_total_odd > 0
    flux_total_odd[w] /= ivar_total_odd[w]
    w = ivar_total_even > 0
    flux_total_even[w] /= ivar_total_even[w]

    alpha = 1
    if num_exp_per_col % 2 == 1:
        num_exp_even = (num_exp_per_col - 1) // 2
        alpha = np.sqrt(4.0 * num_exp_even * (num_exp_even + 1)) / num_exp_per_col
    # TODO: CHECK THE * alpha (Nathalie)
    exposures_diff = 0.5 * (flux_total_even - flux_total_odd) * alpha

    return exposures_diff


def exp_diff_desi(
    file,
    mask_targetid,
    method_alpha="desi_array",
    use_only_even=False,
):
    """Compute the difference between exposures.

    More precisely compute de semidifference between two customized coadded
    spectra obtained from weighted averages of the even-number exposures, for
    the first spectrum, and of the odd-number exposures, for the second one
    (see section 3.2 of Chabanier et al. 2019).

    Arguments
    ---------
    file: fitsio.fitslib.FITS
    Header Data Unit List opened by fitsio.

    mask_targetid: array of int
    Targetids to select for calculating the exposure differences.

    method_alpha: str
    Method to compute alpha. Defaults to "desi_array".

    use_only_even: bool
    Flag to indicate whether to use only even-number exposures. Defaults to False.

    Return
    ------
    diff: array of float
    The difference between exposures
    """
    argsort = np.flip(np.argsort(np.mean(file["IV"][mask_targetid], axis=1)))

    teff_lya = file["TEFF_LYA"][mask_targetid][argsort]
    flux = file["FL"][mask_targetid][argsort, :]
    ivar = file["IV"][mask_targetid][argsort, :]

    num_exp = len(flux)
    if num_exp < 2:
        print("Not enough exposures for diff, spectra rejected")
        return None
    if use_only_even:
        if num_exp % 2 == 1:
            print("Odd number of exposures discarded")
            return None

    time_even = 0
    time_odd = 0
    time_exp = np.sum(teff_lya)

    even_inds = slice(0, 2 * (num_exp // 2), 2)
    odd_inds = slice(1, 2 * (num_exp // 2), 2)
    flux_tot_odd = (flux[odd_inds] * ivar[odd_inds]).sum(axis=0)
    ivar_tot_odd = (ivar[odd_inds]).sum(axis=0)
    flux_tot_even = (flux[even_inds] * ivar[even_inds]).sum(axis=0)
    ivar_tot_even = (ivar[even_inds]).sum(axis=0)
    ivar_tot = ivar[:num_exp].sum(axis=0)

    mask_odd = ivar_tot_odd > 0
    flux_tot_odd[mask_odd] /= ivar_tot_odd[mask_odd]
    mask_even = ivar_tot_even > 0
    flux_tot_even[mask_even] /= ivar_tot_even[mask_even]

    alpha = 1
    if method_alpha == "eboss":
        if num_exp % 2 == 1:
            n_even = num_exp // 2
            alpha = np.sqrt(4.0 * n_even * (n_even + 1)) / num_exp

    elif method_alpha == "eboss_corr":
        if num_exp % 2 == 1:
            n_even = num_exp // 2
            alpha = np.sqrt((2 * n_even) / (num_exp))

    elif method_alpha == "desi_array":
        mask = mask_odd & mask_even & (ivar_tot > 0)
        alpha_array = np.ones(flux.shape[1])
        alpha_array[mask] = (1 / np.sqrt(ivar_tot[mask])) / (
            0.5 * np.sqrt((1 / ivar_tot_even[mask]) + (1 / ivar_tot_odd[mask]))
        )
        alpha = alpha_array

    elif method_alpha == "desi_mean_array":
        alpha_array = np.ones(flux.shape[1])
        alpha_array[mask] = (1 / np.sqrt(ivar_tot[mask])) / (
            0.5 * np.sqrt((1 / ivar_tot_even[mask]) + (1 / ivar_tot_odd[mask]))
        )
        alpha = np.nanmean((1 / np.sqrt(ivar_tot[mask]))) / np.nanmean(
            (0.5 * np.sqrt((1 / ivar_tot_even[mask]) + (1 / ivar_tot_odd[mask])))
        )

    elif method_alpha == "desi_time":
        alpha = 2 * np.sqrt(
            (time_odd * time_even) / (time_exp * (time_odd + time_even))
        )

    else:
        raise ValueError(f"Unknown method_alpha: {method_alpha}")

    diff = 0.5 * (flux_tot_even - flux_tot_odd) * alpha
    return diff


def spectral_resolution(
    wdisp,
    with_correction=False,
    fiberid=None,
    log_lambda=None,
):
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
    reso = wdisp * SPEED_LIGHT * 1.0e-4 * np.log(10.0)

    if with_correction:
        lambda_ = np.power(10.0, log_lambda)
        # compute the wavelength correction
        correction = 1.267 - 0.000142716 * lambda_ + 1.9068e-08 * lambda_ * lambda_
        correction[lambda_ > 6000.0] = 1.097

        # add the fiberid correction
        # fiberids greater than 500 corresponds to the second spectrograph
        fiberid = fiberid % 500
        if fiberid < 100:
            correction = (
                1.0
                + (correction - 1) * 0.25
                + (correction - 1) * 0.75 * (fiberid) / 100.0
            )
        elif fiberid > 400:
            correction = (
                1.0
                + (correction - 1) * 0.25
                + (correction - 1) * 0.75 * (500 - fiberid) / 100.0
            )

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
    delta_log_lambda = (log_lambda[-1] - log_lambda[0]) / float(len(log_lambda) - 1)
    reso = np.clip(reso_matrix, 1.0e-6, 1.0e6)

    #
    rms_in_pixel = (
        np.sqrt(
            1.0 / 2.0 / np.log(reso[len(reso) // 2][:] / reso[len(reso) // 2 - 1][:])
        )
        + np.sqrt(
            4.0 / 2.0 / np.log(reso[len(reso) // 2][:] / reso[len(reso) // 2 - 2][:])
        )
        + np.sqrt(
            1.0 / 2.0 / np.log(reso[len(reso) // 2][:] / reso[len(reso) // 2 + 1][:])
        )
        + np.sqrt(
            4.0 / 2.0 / np.log(reso[len(reso) // 2][:] / reso[len(reso) // 2 + 2][:])
        )
    ) / 4.0

    avg_reso_in_km_per_s = rms_in_pixel * SPEED_LIGHT * delta_log_lambda * np.log(10.0)

    return rms_in_pixel, avg_reso_in_km_per_s
