"""This module defines the abstract class SdssCalibrationCorrection"""
import fitsio
from scipy.interpolate import interp1d

from picca.delta_extraction.correction import Correction
from picca.delta_extraction.errors import CorrectionError

class SdssIvarCorrection(Correction):
    """Class to correct inverse variance errors in SDSS spectra

    Methods
    -------
    __init__
    apply_correction

    Attributes
    ----------
    correct_ivar: scipy.interpolate.interp1d
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
        CorrectionError if input file does not have extension VAR_FUNC
        CorrectionError if input file does not have fields loglam and/or eta
        in extension VAR_FUNC
        """
        filename = config.get("filename")
        try:
            hdu = fitsio.read(filename, ext="VAR_FUNC")
            log_lambda = hdu['loglam']
            eta = hdu['eta']
        except OSError:
            raise CorrectionError("Error loading SdssIvarCorrection. "
                                  f"File {filename} does not have extension "
                                  "'VAR_FUNC'")
        except ValueError:
            raise CorrectionError("Error loading SdssIvarCorrection. "
                                  f"File {filename} does not have fields "
                                  "'loglam' and/or 'eta' in HDU 'VAR_FUNC'")
        self.correct_ivar = interp1d(log_lambda,
                                     eta,
                                     fill_value="extrapolate",
                                     kind="nearest")

    def apply_correction(self, forest):
        """Apply the correction. Correction is applied by dividing the
        data inverse variance by the loaded correction.

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
            raise CorrectionError("Correction from SdssIvarCorrection "
                                  "should only be applied to data with the "
                                  "attribute 'log_lambda'")
        correction = self.correct_ivar(forest.log_lambda)
        forest.ivar /= correction
