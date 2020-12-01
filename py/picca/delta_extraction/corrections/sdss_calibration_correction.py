"""This module defines the abstract class SdssCalibrationCorrection"""
from scipy.interpolate import interp1d
import fitsio

from picca.delta_extraction.correction import Correction
from picca.delta_extraction.errors import CorrectionError

class SdssCalibrationCorrection(Correction):
    """Class to correct calibration errors in SDSS spectra

    Methods
    -------
    __init__
    apply_correction

    Attributes
    ----------
    correct_flux: scipy.interpolate.interp1d
    Interpolation function to adapt the correction to slightly different
    grids of wavelength
    """
    def __init__(self, config):
        """Initializes class instance.

        Arguments
        ---------
        config: configparser.SectionProxy
        Parsed options to initialize class
        """
        filename = config.get("filename")
        hdu = fitsio.read(filename, ext=1)
        stack_log_lambda = hdu['loglam']
        stack_delta = hdu['stack']
        w = (stack_delta != 0.)
        self.correct_flux = interp1d(stack_log_lambda[w],
                                     stack_delta[w],
                                     fill_value="extrapolate",
                                     kind="nearest")

    def apply_correction(self, forest):
        """Applies the correction. Correction is applied by dividing the
        data flux by the loaded correction, and the subsequent correction
        of the inverse variance

        Arguments
        ---------
        forest: Forest
        A Forest instance to which the correction is applied

        Raises
        ------
        CorrectionError if forest instance does not have the attribute
        'log_lambda'
        """
        if not hasattr(forest, "log_lambda"):
            raise CorrectionError("Correction from SdssCalibrationCorrection "
                                  "should only be applied to data with the "
                                  "attribute 'log_lambda'")
        correction = self.correct_flux(forest.log_lambda)
        forest.flux /= correction
        forest.ivar *= correction**2
