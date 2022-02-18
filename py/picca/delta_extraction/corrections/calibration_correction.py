"""This module defines the class CalibrationCorrection"""
import fitsio
from scipy.interpolate import interp1d

from picca.delta_extraction.correction import Correction
from picca.delta_extraction.errors import CorrectionError

accepted_options = ["filename"]

class CalibrationCorrection(Correction):
    """Class to correct calibration errors measured from other spectral regions.

    Methods
    -------
    __init__
    apply_correction

    Attributes
    ----------
    correct_flux: scipy.interpolate.interp1d
    Interpolation function to adapt the correction to slightly different
    grids of wavelength

    wave_solution: str
    Determines whether the wavelength solution has linear spacing ("lin") or
    logarithmic spacing ("log").
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
            if "loglam" in hdu.dtype.names:
                stack_log_lambda = hdu['loglam']
                self.wave_solution = "log"
            elif "lambda" in hdu.dtype.names:
                stack_lambda = hdu['lambda']
                self.wave_solution = "lin"
            else:
                raise CorrectionError("Error loading CalibrationCorrection. In "
                                      "extension 'STACK_DELTAS' in file "
                                      f"{filename} one of the fields 'loglam' "
                                      "or 'lambda' should be present. I did not"
                                      "find them.")
            stack_delta = hdu['stack']
        except OSError:
            raise CorrectionError("Error loading CalibrationCorrection. "
                                  f"File {filename} does not have extension "
                                  "'STACK_DELTAS'")
        except ValueError:
            raise CorrectionError("Error loading CalibrationCorrection. "
                                  f"File {filename} does not have fields "
                                  "'loglam' and/or 'stack' in HDU 'STACK_DELTAS'")
        w = (stack_delta != 0.)
        if self.wave_solution == "log":
            self.correct_flux = interp1d(stack_log_lambda[w],
                                         stack_delta[w],
                                         fill_value="extrapolate",
                                         kind="nearest")
        elif self.wave_solution == "lin":
            self.correct_flux = interp1d(stack_lambda[w],
                                         stack_delta[w],
                                         fill_value="extrapolate",
                                         kind="nearest")
        else:
            raise CorrectionError("In CalibrationCorrection wave_solution must "
                                  "be either 'log' or 'lin'")

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
        if self.wave_solution == "log":
            if not hasattr(forest, "log_lambda"):
                raise CorrectionError("Forest instance is missing "
                                      "attribute 'log_lambda' required by "
                                      "CalibrationCorrection")
            correction = self.correct_flux(forest.log_lambda)
        elif self.wave_solution == "lin":
            if not hasattr(forest, "lambda_"):
                raise CorrectionError("Forest instance is missing "
                                      "attribute 'lambda_' required by "
                                      "CalibrationCorrection")
            correction = self.correct_flux(forest.lambda_)
        else:
            raise CorrectionError("In CalibrationCorrection wave_solution must "
                                  "be either 'log' or 'lin'")
        forest.flux /= correction
        forest.ivar *= correction**2
