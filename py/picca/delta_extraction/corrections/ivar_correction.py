"""This module defines the abstract class IvarCorrection"""
import logging

import fitsio
import numpy as np
from scipy.interpolate import interp1d

from picca.delta_extraction.correction import Correction
from picca.delta_extraction.errors import CorrectionError

accepted_options = ["filename"]

class IvarCorrection(Correction):
    """Class to correct inverse variance errors measured from other spectral
    regions.

    Methods
    -------
    __init__
    apply_correction

    Attributes
    ----------
    correct_ivar: scipy.interpolate.interp1d
    Interpolation function to adapt the correction to slightly different
    grids of wavelength

    logger: logging.Logger
    Logger object
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
        self.logger = logging.getLogger(__name__)

        filename = config.get("filename")
        if filename is None:
            raise CorrectionError("Missing argument 'filename' required by SdssIvarCorrection")
        try:
            hdu = fitsio.read(filename, ext="VAR_FUNC")
            if "loglam" in hdu.dtype.names:
                log_lambda = hdu['loglam']
            elif "lambda" in hdu.dtype.names:
                self.logger.warning("DeprecationWarning: Reading correction using 'lambda'. "
                                    "Newer versions of picca always save 'log_lambda' and "
                                    "so this option will be removed in the future.")
                log_lambda = np.log10(hdu['lambda'])
            else:
                raise CorrectionError("Error loading IvarCorrection. In "
                                      "extension 'VAR_FUNC' in file "
                                      f"{filename} one of the fields 'loglam' "
                                      "or 'lambda' should be present. I did not"
                                      "find them.")

            eta = hdu['eta']
        except OSError as error:
            raise CorrectionError(
                "Error loading CalibrationCorrection. "
                f"Failed to find or open file {filename}"
            ) from error
        except ValueError as error:
            raise CorrectionError(
                "Error loading IvarCorrection. "
                f"File {filename} does not have fields "
                "'loglam' and/or 'eta' in HDU 'VAR_FUNC'"
            ) from error

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
        correction = self.correct_ivar(forest.log_lambda)
        forest.ivar /= correction
