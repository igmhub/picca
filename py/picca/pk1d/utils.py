"""This module define several functions and variables used throughout the
pk1d package"""

import numpy as np
from picca import constants
from picca.utils import userprint

DEFAULT_K_MIN_VEL = 0.000813
DEFAULT_K_BIN_VEL = 0.000542
# Default k_min in AA: assumed average redshift of 3.4,
#  and a forest defined between 1050 and 1200 Angstrom:
DEFAULT_K_MIN_LIN = 2 * np.pi / ((1200 - 1050) * (1 + 3.4))
# Default k bin is tuned to be close to k_min*DEFAULT_K_BINNING_FACTOR:
DEFAULT_K_BINNING_FACTOR = 4

# Fit dispersion vs SNR,
# used in mean Pk computation when weight_method=='fit_snr':
MEANPK_FITRANGE_SNR = [1, 10]

# Wavelength grid to estimate skyline masks:
# default values from eg. desispec/scripts/proc.py
DEFAULT_MINWAVE_SKYMASK = 3600.0
DEFAULT_MAXWAVE_SKYMASK = 9824.0
DEFAULT_DWAVE_SKYMASK = 0.8


def fitfunc_variance_pk1d(snr, amp, zero_point):
    """Compute variance

    Arguments
    ---------
    snr (float): 
    The signal-to-noise ratio of the signal. Must be greater than 1.

    amp (float): 
    The amplitude of the signal.

    zero_point (float): 
    The zero point offset of the signal.

    Return
    ------
    float: The variance of the signal.
    """
    return (amp / (snr - 1)**2) + zero_point


def skyline_mask_matrices_desi(z_parts, skyline_mask_file,
                               minwave=DEFAULT_DWAVE_SKYMASK,
                               maxwave=DEFAULT_MAXWAVE_SKYMASK,
                               dwave=DEFAULT_DWAVE_SKYMASK):
    """ Compute matrices to correct for the masking effect on FFTs,
    when chunks are defined with the `--parts-in-redshift` option in picca_Pk1D.
    Only implemented for DESI data, ie delta lambda = 0.8 and no rebinning in the deltas.

    Arguments
    ---------
    z_parts: list
    Edges of redshift chunks

    skyline_mask_file: str
    Name of file containing the list of skyline masks

    minwave, maxwave, dwave: float
    Parameter defining the wavelength grid used for the skymask. Should be consistent with
    arguments lambda min, lambda max, delta lambda used in picca_delta_extraction.py.

    Return
    ------
    A list with N items [meanz, matrix] where N = len(z_parts)-1 is the number of redshift chunks.
    Each matrix is the inverse of, symbolically, M_ij ~ |M_{i-j}|^2 where M_k is the
    Fourier transform of the mask function associated to skylines. Within the current
    implementation, the matrix sizes are dictated by the DESI wavelength binning.
    """
    skyline_list = np.genfromtxt(skyline_mask_file,
                                 names=('type', 'wave_min', 'wave_max', 'frame'))
    ref_wavegrid = np.arange(minwave, maxwave, dwave)
    num_parts = len(z_parts)-1
    out = []

    for iz in range(num_parts):
        lmin = constants.ABSORBER_IGM['LYA'] * (1+z_parts[iz])
        lmax = constants.ABSORBER_IGM['LYA'] * (1+z_parts[iz+1])
        # - the following selection is identical to compute_pk1d.split_forest_in_z_parts:
        wave = ref_wavegrid[ (ref_wavegrid>=lmin) & (ref_wavegrid<lmax) ]
        npts = len(wave)
        skymask = np.ones(npts)
        selection = ( (skyline_list['wave_min']<=lmax) & (skyline_list['wave_min']>=lmin)
                    ) | ( (skyline_list['wave_max']<=lmax) & (skyline_list['wave_max']>=lmin) )
        for skyline in skyline_list[selection]:
            skymask[(wave>=skyline['wave_min']) & (wave<=skyline['wave_max'])] = 0
        skymask_tilde = np.fft.fft(skymask)/npts
        mask_matrix = np.zeros((npts, npts))
        for j in range(npts):
            for l in range(npts):
                index_mask = j-l if j>=l else j-l+npts
                mask_matrix[j, l] = (
                    skymask_tilde[index_mask].real ** 2
                    + skymask_tilde[index_mask].imag ** 2
                )
        try:
            inv_matrix = np.linalg.inv(mask_matrix)
        except np.linalg.LinAlgError:
            userprint(
                """Warning: cannot invert sky mask matrix """
                f"""for z bin {z_parts[iz]} - {z_parts[iz+1]}"""
            )
            userprint("No correction will be applied for this bin")
            inv_matrix = np.eye(npts)
        meanz = (z_parts[iz]+z_parts[iz+1])/2
        out.append([meanz, inv_matrix])

    return out
