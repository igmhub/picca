"""This module defines the class AbsorberMask in the
masking of absorbers"""
import logging

import fitsio
import numpy as np

from picca.delta_extraction.astronomical_objects.forest import Forest
from picca.delta_extraction.errors import MaskError
from picca.delta_extraction.mask import Mask

defaults = {
    "absorber mask width": 2.5,
}

accepted_options = ["absorber mask width", "filename"]

class AbsorberMask(Mask):
    """Class to mask Absorbers

    Methods
    -------
    __init__
    apply_mask

    Attributes
    ----------
    los_ids: dict (from Mask)
    A dictionary with the absorbers contained in each line of sight. Keys are the
    identifier for the line of sight and values are lists of z_abs

    absorber_mask_width: float
    Mask width on each side of the absorber central observed wavelength in
    units of 1e4*dlog10(lambda)

    logger: logging.Logger
    Logger object
    """
    def __init__(self, config):
        """Initialize class instance.

        Arguments
        ---------
        config: configparser.SectionProxy
        Parsed options to initialize class
        """
        self.logger = logging.getLogger(__name__)

        super().__init__()

        # first load the absorbers catalogue
        filename = config.get("filename")
        if filename is None:
            raise MaskError("Missing argument 'filename' required by "
                            "AbsorbersMask")

        self.logger.progress(f"Reading absorbers from: {filename}")

        los_id_name = config.get("los_id name")
        if los_id_name is None:
            raise MaskError("Missing argument 'los_id name' required by AbsorberMask")

        columns_list = [los_id_name, "LAMBDA_ABS"]
        try:
            hdul = fitsio.FITS(filename)
            cat = {col: hdul["ABSORBERCAT"][col][:] for col in columns_list}
        except OSError:
            raise MaskError("Error loading AbsorberMask. File "
                            f"{filename} does not have extension "
                            "'ABSORBERCAT'")
        except ValueError:
            aux = "', '".join(columns_list)
            raise MaskError("Error loading AbsorberMask. File "
                            f"{filename} does not have fields '{aux}' "
                            "in HDU 'ABSORBERCAT'")
        finally:
            hdul.close()

        # group absorbers on the same line of sight together
        self.los_ids = {}
        for los_id in np.unique(cat[los_id_name]):
            w = (los_id == cat[los_id_name])
            self.los_ids[los_id] = list(cat[los_id_name][w])
        num_absorbers = np.sum([len(los_id) for los_id in self.los_ids.values()])

        self.logger.progress(" In catalog: {} absorbers".format(num_absorbers))
        self.logger.progress(" In catalog: {} forests have absorbers\n".format(len(self.los_ids)))

        # setup transmission limit
        # transmissions below this number are masked
        self.absorber_mask_width = config.getfloat("absorber mask width")
        if self.absorber_mask_width is None:
            raise MaskError("Missing argument 'absorber mask width' required by "
                            "AbsorbersMask")

    def apply_mask(self, forest):
        """Applies the mask. The mask is done by removing the affected
        pixels from the arrays in data.mask_fields

        Arguments
        ---------
        forest: Forest
        A Forest instance to which the correction is applied

        Raise
        -----
        CorrectionError if Forest.wave_solution is not 'log'
        """
        if Forest.wave_solution == "log":
            log_lambda = forest.log_lambda
        elif Forest.wave_solution == "lin":
            log_labmda = np.log10(forest.lambda_)
        else:
            raise MaskError("Forest.wave_solution must be either 'log' or 'lin'")

        # load DLAs
        if self.los_ids.get(forest.los_id) is not None:
            # find out which pixels to mask
            w = np.ones(log_lambda.size, dtype=bool)
            for lambda_absorber in self.los_ids.get(forest.los_id):
                w &= (np.fabs(1.e4 * (log_lambda - np.log10(lambda_absorber))) >
                      self.absorber_mask_width)

            # do the actual masking
            for param in Forest.mask_fields:
                setattr(forest, param, getattr(forest, param)[w])
