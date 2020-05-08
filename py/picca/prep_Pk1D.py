"""This module defines a set of functions to manage specifics of Pk1D analysis
when computing the deltas.

This module provides three functions:
    - exp_diff
    - spectral_resolution
    - spectral_resolution_desi
See the respective documentation for details
"""
import numpy as np
import scipy as sp

from picca import constants
from picca.utils import userprint


def exp_diff(hdul, log_lambda):
    """Computes the difference between exposures.

    More precisely computes de semidifference between two customized coadded
    spectra obtained from weighted averages of the even-number exposures, for
    the first spectrum, and of the odd-number exposures, for the second one
    (see section 3.2 of Chabanier et al. 2019).

    Args:
        hdul: fitsio.fitslib.FITS
            Header Data Unit List opened by fitsio
        log_lambda: array of floats
            Array containing the logarithm of the wavelengths (in Angs)

    Returns:
        The difference between exposures
    """
    num_exp_per_col = hdul[0].read_header()['NEXP']//2
    flux_total_odd = np.zeros(log_lambda.size)
    ivar_total_odd = np.zeros(log_lambda.size)
    flux_total_even = np.zeros(log_lambda.size)
    ivar_total_even = np.zeros(log_lambda.size)

    if num_exp_per_col < 2:
        userprint("DBG : not enough exposures for diff")

    for index_exp in range(num_exp_per_col):
        for index_col in range(2):
            log_lambda_exp = hdul[(4 + index_exp +
                                   index_col*num_exp_per_col)]["loglam"][:]
            flux_exp = hdul[(4 + index_exp +
                             index_col*num_exp_per_col)]["flux"][:]
            ivar_exp = hdul[(4 + index_exp +
                             index_col*num_exp_per_col)]["ivar"][:]
            mask = hdul[4 + index_exp + index_col*num_exp_per_col]["mask"][:]
            bins = np.searchsorted(log_lambda, log_lambda_exp)

            # exclude masks 25 (COMBINEREJ), 23 (BRIGHTSKY)?
            rebin_ivar_exp = np.bincount(bins,
                                         weights=ivar_exp*(mask & 2**25 == 0))
            rebin_flux_exp = np.bincount(bins,
                                         weights=(ivar_exp*flux_exp*
                                                  (mask & 2**25 == 0)))

            if index_exp%2 == 1:
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
    if num_exp_per_col%2 == 1:
        num_even_exp = (num_exp_per_col - 1)//2
        alpha = np.sqrt(4.*num_even_exp*(num_even_exp + 1))/num_exp_per_col
    # TODO: CHECK THE * alpha (Nathalie)
    exposures_diff = 0.5 * (flux_total_even - flux_total_odd) * alpha

    return exposures_diff


def spectral_resolution(wdisp,with_correction=None,fiber=None,log_lambda=None) :

    reso = wdisp*constants.speed_light/1000.*1.0e-4*sp.log(10.)

    if (with_correction):
        wave = sp.power(10.,log_lambda)
        corrPlateau = 1.267 - 0.000142716*wave + 1.9068e-08*wave*wave;
        corrPlateau[wave>6000.0] = 1.097

        fibnum = fiber%500
        if(fibnum<100):
            corr = 1. + (corrPlateau-1)*.25 + (corrPlateau-1)*.75*(fibnum)/100.
        elif (fibnum>400):
            corr = 1. + (corrPlateau-1)*.25 + (corrPlateau-1)*.75*(500-fibnum)/100.
        else:
            corr = corrPlateau
        reso *= corr
    return reso

def spectral_resolution_desi(reso_matrix, log_lambda) :

    delta_log_lambda = (log_lambda[-1]-log_lambda[0])/float(len(log_lambda)-1)
    reso= sp.clip(reso_matrix,1.0e-6,1.0e6)
    rms_in_pixel = (sp.sqrt(1.0/2.0/sp.log(reso[len(reso)//2][:]/reso[len(reso)//2-1][:]))
                    + sp.sqrt(4.0/2.0/sp.log(reso[len(reso)//2][:]/reso[len(reso)//2-2][:])))/2.0

    reso_in_km_per_s = rms_in_pixel*constants.speed_light/1000.*delta_log_lambda*sp.log(10.0)

    return reso_in_km_per_s
