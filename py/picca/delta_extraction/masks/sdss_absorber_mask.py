"""This module defines the abstract class AbsorberMask in the
masking of absorbers"""
import numpy as np
import fitsio

from picca.delta_extraction.errors import MaskError
from picca.delta_extraction.mask import Mask
from picca.delta_extraction.userprint import userprint

defaults = {
    "absorber mask width": 2.5,
}

class SdssAbsorberMask(Mask):
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
    """
    def __init__(self, config):
        """Initializes class instance.

        Arguments
        ---------
        config: configparser.SectionProxy
        Parsed options to initialize class
        """
        # first load the absorbers catalogue
        absorbers_catalogue = config.get("absorbers catalogue")
        if absorbers_catalogue is None:
            raise MaskError("Missing argument 'absorbers catalogue' required by "
                            "AbsorbersMask")

        userprint('Reading absorbers from:', absorbers_catalogue)

        columns_list = ["THING_ID", "LAMBDA_ABS"]
        try:
            hdul = fitsio.FITS(absorbers_catalogue)
            cat = {col: hdul["ABSORBERCAT"][col][:] for col in columns_list}
        except OSError:
            raise MaskError("Error loading SdssAbsorberMask. File "
                            f"{absorbers_catalogue} does not have extension "
                            "'ABSORBERCAT'")
        except ValueError:
            aux = "', '".join(columns_list)
            raise MaskError("Error loading SdssAbsorberMask. File "
                            f"{absorbers_catalogue} does not have fields '{aux}' "
                            "in HDU 'ABSORBERCAT'")
        finally:
            hdul.close()

        # group absorbers on the same line of sight together
        self.los_ids = {}
        for thingid in np.unique(cat["THING_ID"]):
            w = (thingid == cat["THING_ID"])
            self.los_ids[thingid] = list(cat["LAMBDA_ABS"][w])
        num_absorbers = np.sum([len(thingid) for thingid in self.los_ids.values()])

        userprint(" In catalog: {} absorbers".format(num_absorbers))
        userprint(" In catalog: {} forests have absorbers".format(len(self.los_ids)))
        userprint("")

        # setup transmission limit
        # transmissions below this number are masked
        self.absorber_mask_width = config.getfloat("absorber mask width")
        if self.absorber_mask_width is None:
            self.absorber_mask_width = defaults.get("absorber mask width")

    def apply_mask(self, forest):
        """Applies the mask. The mask is done by removing the affected
        pixels from the arrays in data.mask_fields

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
            raise MaskError("Mask from SdssAbsorberMask should only be applied "
                            "to data with the attribute 'log_lambda'")
        # load DLAs
        if self.los_ids.get(forest.los_id) is not None:
            # find out which pixels to mask
            w = np.ones(forest.log_lambda.size, dtype=bool)
            for lambda_absorber in self.los_ids.get(forest.los_id):
                w &= (np.fabs(1.e4 * (forest.log_lambda - np.log10(lambda_absorber))) >
                      self.absorber_mask_width)

            # do the actual masking
            for param in forest.mask_fields:
                setattr(forest, param, getattr(forest, param)[w])
