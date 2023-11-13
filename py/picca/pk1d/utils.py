"""This module define several functions and variables used throughout the
pk1d package"""

import numpy as np
from picca.delta_extraction.utils import SPEED_LIGHT  # pylint: disable=unused-import

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
