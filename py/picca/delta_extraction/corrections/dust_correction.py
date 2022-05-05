"""This module defines the class DustCorrection. It also defines
the function unred (https://github.com/sczesla/PyAstronomy in /src/pyasl/asl/unred)
to compute the reddening correction."""
import fitsio
import numpy as np
from scipy import interpolate

from picca.delta_extraction.correction import Correction
from picca.delta_extraction.errors import CorrectionError

accepted_options = ["extinction_conversion_r", "filename"]

defaults = {
    "extinction_conversion_r": 3.793,
}

class DustCorrection(Correction):
    """Class to correct for dust absortpion

    Methods
    -------
    __init__
    apply_correction

    Attributes
    ----------
    extinction_bv_map: dict
    B-V extinction due to dust. Keys are THING_ID
    """
    def __init__(self, config):
        """Initialize class instance.

        Arguments
        ---------
        config: configparser.SectionProxy
        Parsed options to initialize class

        Raise
        -----
        CorrectionError if input file does not have extension CATALOG
        CorrectionError if input file does not have fields THING_ID and/or
        EXTINCTION in extension CATALOG
        """
        extinction_conversion_r = config.getfloat("extinction_conversion_r")
        if extinction_conversion_r is None:
            raise CorrectionError("Missing argument 'extinction_conversion_r' "
                                  "required by DustCorrection")

        filename = config.get("filename")
        if filename is None:
            raise CorrectionError("Missing argument 'filename' "
                                  "required by DustCorrection")
        try:
            hdu = fitsio.read(filename, ext="CATALOG")
            thingid = hdu['THING_ID']
            ext = hdu['EXTINCTION'][:, 1] / extinction_conversion_r
        except OSError as error:
            raise CorrectionError(
                "Error loading DustCorrection. "
                f"File {filename} does not have extension "
                "'CATALOG'"
            ) from error
        except ValueError as error:
            raise CorrectionError(
                "Error loading DustCorrection. "
                f"File {filename} does not have fields "
                "'THING_ID' and/or 'EXTINCTION' in HDU "
                "'CATALOG'"
            ) from error
        self.extinction_bv_map = dict(zip(thingid, ext))

    def apply_correction(self, forest):
        """Apply the correction. Correction is computed using the unread
        function and applied by dividing the data flux by the loaded correction,
        and the subsequent correction of the inverse variance. If the forest
        instance has the attribute exposures_diff and it is not None, it is
        divided by the correction

        Arguments
        ---------
        forest: Forest
        A Forest instance to which the correction is applied

        Raise
        -----
        CorrectionError if forest instance does not have the attribute
        'log_lambda'
        """
        thingid = forest.los_id
        extinction = self.extinction_bv_map.get(thingid)
        if extinction is None:
            return

        correction = unred(10**forest.log_lambda, extinction)
        forest.flux /= correction
        forest.ivar *= correction**2
        if hasattr(forest, "exposures_diff"):
            forest.exposures_diff /= correction

# pylint: disable=invalid-name,locally-disabled
# we keep variable names since this function is adopted from elsewhere
def unred(wave, ebv, R_V=3.1, LMC2=False, AVGLMC=False):
    """
    https://github.com/sczesla/PyAstronomy
    in /src/pyasl/asl/unred
    """

    # Convert to inverse microns
    x = 10000. / wave
    curve = x * 0.

    # Set some standard values:
    x0 = 4.596
    gamma = 0.99
    c3 = 3.23
    c4 = 0.41
    c2 = -0.824 + 4.717 / R_V
    c1 = 2.030 - 3.007 * c2

    if LMC2:
        x0 = 4.626
        gamma = 1.05
        c4 = 0.42
        c3 = 1.92
        c2 = 1.31
        c1 = -2.16
    elif AVGLMC:
        x0 = 4.596
        gamma = 0.91
        c4 = 0.64
        c3 = 2.73
        c2 = 1.11
        c1 = -1.28

    # Compute UV portion of A(lambda)/E(B-V) curve using FM fitting function and
    # R-dependent coefficients
    xcutuv = np.array([10000.0 / 2700.0])
    xspluv = 10000.0 / np.array([2700.0, 2600.0])

    iuv = np.where(x >= xcutuv)[0]
    N_UV = iuv.size
    iopir = np.where(x < xcutuv)[0]
    Nopir = iopir.size
    if N_UV > 0:
        xuv = np.concatenate((xspluv, x[iuv]))
    else:
        xuv = xspluv

    yuv = c1 + c2 * xuv
    yuv = yuv + c3 * xuv**2 / ((xuv**2 - x0**2)**2 + (xuv * gamma)**2)
    yuv = yuv + c4 * (0.5392 * (np.maximum(xuv, 5.9) - 5.9)**2 + 0.05644 *
                      (np.maximum(xuv, 5.9) - 5.9)**3)
    yuv = yuv + R_V
    yspluv = yuv[0:2]  # save spline points

    if N_UV > 0:
        curve[iuv] = yuv[2::]  # remove spline points

    # Compute optical portion of A(lambda)/E(B-V) curve
    # using cubic spline anchored in UV, optical, and IR
    xsplopir = np.concatenate(
        ([0], 10000.0 /
         np.array([26500.0, 12200.0, 6000.0, 5470.0, 4670.0, 4110.0])))
    ysplir = np.array([0.0, 0.26469, 0.82925]) * R_V / 3.1
    ysplop = np.array(
        (np.polyval([-4.22809e-01, 1.00270, 2.13572e-04][::-1], R_V),
         np.polyval([-5.13540e-02, 1.00216, -7.35778e-05][::-1], R_V),
         np.polyval([7.00127e-01, 1.00184, -3.32598e-05][::-1], R_V),
         np.polyval([1.19456, 1.01707, -5.46959e-03, 7.97809e-04,
                     -4.45636e-05][::-1], R_V)))
    ysplopir = np.concatenate((ysplir, ysplop))

    if Nopir > 0:
        tck = interpolate.splrep(np.concatenate((xsplopir, xspluv)),
                                 np.concatenate((ysplopir, yspluv)),
                                 s=0)
        curve[iopir] = interpolate.splev(x[iopir], tck)

    #Now apply extinction correction to input flux vector
    curve *= ebv
    corr = 1. / (10.**(0.4 * curve))

    return corr
