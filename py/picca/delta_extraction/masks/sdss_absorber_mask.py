"""This module defines the abstract class AbsorberMask in the
masking of absorbers"""
import numpy as np

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
    A dictionary with the DLAs contained in each line of sight. Keys are the
    identifier for the line of sight and values are lists of (z_abs, nhi)

    absorber_mask_width: float
    Mask width on each side of the absorber central observed wavelength in
    units of 1e4*dlog10(lambda)
    """
    def __init__(self, config):
        """Initializes class instance.
        Arguments are required to be keyword arguments by the lack of
        order in Config

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
        file = open(absorbers_catalogue)
        self.los_ids = {}
        num_absorbers = 0
        col_names = None
        for line in file.readlines():
            cols = line.split()
            if len(cols) == 0:
                continue
            if cols[0][0] == "#":
                continue
            if cols[0] == "ThingID":
                col_names = cols
                continue
            if cols[0][0] == "-":
                continue
            thingid = int(cols[col_names.index("ThingID")])
            if thingid not in self.los_ids:
                self.los_ids[thingid] = []
            lambda_absorber = float(cols[col_names.index("lambda")])
            self.los_ids[thingid].append(lambda_absorber)
            num_absorbers += 1
        file.close()

        userprint(" In catalog: {} absorbers".format(num_absorbers))
        userprint(" In catalog: {} forests have absorbers".format(len(self.los_ids)))
        userprint("")

        # setup transmission limit
        # transmissions below this number are masked
        self.absorber_mask_width = config.get("absorber mask width")
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
            for lambda_absorber in self.los_ids.get(forest.los_ids):
                w &= (np.fabs(1.e4 * (forest.log_lambda - np.log10(lambda_absorber))) >
                      self.absorber_mask_width)

            # do the actual masking
            for param in forest.mask_fields:
                setattr(forest, param, getattr(forest, param)[w])
