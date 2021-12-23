"""This module defines the class SdssCalibrationCorrection"""
import fitsio
from scipy.interpolate import interp1d

from picca.delta_extraction.correction import Correction
from picca.delta_extraction.errors import CorrectionError

accepted_options = ["filename"]

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
        """Initialize class instance.

        Arguments
        ---------
        config: configparser.SectionProxy
        Parsed options to initialize class

        Raise
        -----
        CorrectionError if input file does not have extension STACK_DELTAS
        CorrectionError if input file does not have fields loglam and/or stack
        in extension STACK_DELTAS
        """
        filename = config.get("filename")
        if filename is None:
            raise CorrectionError("Missing argument 'filename' required by SdssCalibrationCorrection")
        try:
            hdu = fitsio.read(filename, ext="STACK_DELTAS")
            stack_log_lambda = hdu['loglam']
            stack_delta = hdu['stack']
        except OSError:
            raise CorrectionError("Error loading SdssCalibrationCorrection. "
                                  f"File {filename} does not have extension "
                                  "'STACK_DELTAS'")
        except ValueError:
            raise CorrectionError("Error loading SdssCalibrationCorrection. "
                                  f"File {filename} does not have fields "
                                  "'loglam' and/or 'stack' in HDU 'STACK_DELTAS'")
        w = (stack_delta != 0.)
        self.correct_flux = interp1d(stack_log_lambda[w],
                                     stack_delta[w],
                                     fill_value="extrapolate",
                                     kind="nearest")

    def apply_correction(self, forest):
        """Apply the correction. Correction is applied by dividing the
        data flux by the loaded correction, and the subsequent correction
        of the inverse variance

        Arguments
        ---------
        forest: Forest
        A Forest instance to which the correction is applied

        Raise
        -----
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
