"""This module defines the classes SdssDlaMask and Dla used in the
masking of DLAs"""
import logging

from astropy.table import Table
import fitsio
import numpy as np

from picca.delta_extraction.astronomical_objects.forest import Forest
from picca.delta_extraction.errors import MaskError
from picca.delta_extraction.mask import Mask
from picca.delta_extraction.utils import SPEED_LIGHT

defaults = {
    "bal index type": "ai",
}

# Wavelengths in Angstroms
lines = {
    "lCIV": 1549,
    "lNV": 1240.81,
    "lLya": 1216.1,
    "lCIII": 1175,
    "lPV1": 1117,
    "lPV2": 1128,
    "lSIV1": 1062,
    "lSIV2": 1074,
    "lLyb": 1020,
    "lOIV": 1031,
    "lOVI": 1037,
    "lOI": 1039
}

class SdssBalMask(Mask):
    """Class to mask BALs

    Methods
    -------
    __init__
    apply_mask

    Attributes
    ----------
    los_ids: dict (from Mask)
    A dictionary with the DLAs contained in each line of sight. Keys are the
    identifier for the line of sight and values are lists of (z_abs, nhi)

    bal_index_type: str
    BAL index type, choose either 'ai' or 'bi'. This will set which velocity
    the  BAL mask uses.

    cat: dict
    Dictionary with the BAL catalogue

    logger: logging.Logger
    Logger object

    mask: astropy.Table
    Table containing specific intervals of wavelength to be masked for DLAs
    """
    def __init__(self, config):
        """Initializes class instance.

        Arguments
        ---------
        config: configparser.SectionProxy
        Parsed options to initialize class

        Raise
        -----
        MaskError if there are missing variables
        MaskError if input file does not have extension DLACAT
        MaskError if input file does not have fields THING_ID, Z, NHI in extension
        DLACAT
        MaskError upon OsError when reading the mask file
        """
        self.logger = logging.getLogger(__name__)

        super().__init__()

        filename = config.get("filename")
        if filename is None:
            raise MaskError("Missing argument 'filename' required by SdssBalMask")

        self.logger.progress(f"Reading BAL catalog from: {filename}")
        columns_list = [
            'THING_ID', 'VMIN_CIV_450', 'VMAX_CIV_450', 'VMIN_CIV_2000',
            'VMAX_CIV_2000'
        ]
        try:
            hdul = fitsio.FITS(filename)
            self.cat = {col: hdul['BALCAT'][col][:] for col in columns_list}
        except OSError:
            raise MaskError(f"Error loading SdssBalMask. File {filename} does "
                            "not have extension 'BALCAT'")
        except ValueError:
            aux = "', '".join(columns_list)
            raise MaskError(f"Error loading SdssBalMask. File {filename} does "
                            f"not have fields '{aux}' in HDU 'BALCAT'")
        finally:
            hdul.close()

        # setup bal index limit
        self.bal_index_type = config.getfloat("bal index type")
        if self.bal_index_type is None:
            self.bal_index_type = MaskError("Missing argument 'dla catalogue' "
                                            "required by SdssBalMask")
        if self.bal_index_type not in ["ai", "bi"]:
            self.bal_index_type = MaskError("In SdssBalMask, unrecognized value "
                                            "for 'bal_index_type'. Expected one "
                                            "of 'ai' or 'bi'. Found "
                                            f"{self.bal_index_type}")

        # compute info for each line of sight
        self.los_ids = {}
        for thingid in np.unique(self.cat["THING_ID"]):
            self.los_ids[thingid] = self.add_bal_rest_frame(thingid)

        num_bals = np.sum([len(thingid) for thingid in self.los_ids.values()])
        self.logger.progress('In catalog: {} BAL quasars'.format(num_bals))

    def add_bal_rest_frame(self, thingid):
        """Creates a list of wavelengths to be masked out by forest.mask

        Args:
            thingid: str
                thingid of quasar (eBOSS)
        """
        if self.bal_index_type == 'bi':
            velocity_list = ['VMIN_CIV_2000', 'VMAX_CIV_2000']
        else:  # AI, the default
            velocity_list = ['VMIN_CIV_450', 'VMAX_CIV_450']

        mask_rest_frame_bal = Table(names=['log_wave_min', 'log_wave_max', 'frame'],
                                    dtype=['f4', 'f4', 'S2'])
        min_velocities = []  # list of minimum velocities
        max_velocities = []  # list of maximum velocities

        # Match thing_id of object to BAL catalog index
        match_index = np.where(self.cat['THING_ID'] == thingid)[0][0]

        # Store the min/max velocity pairs from the BAL catalog
        for col in velocity_list:
            if col.find('VMIN') == 0:
                velocity_list = self.cat[col]
                for vel in velocity_list[match_index]:
                    if vel > 0:
                        min_velocities.append(vel)
            else:
                velocity_list = self.cat[col]
                for vel in velocity_list[match_index]:
                    if vel > 0:
                        max_velocities.append(vel)

        # Calculate mask width for each velocity pair, for each emission line
        for min_vel, max_vel in zip(min_velocities, max_velocities):
            for line in lines.values():
                min_wavelength = np.log10(line * (1 - min_vel / SPEED_LIGHT))
                max_wavelength = np.log10(line * (1 - max_vel / SPEED_LIGHT))
                mask_rest_frame_bal.add_row([min_wavelength, max_wavelength, 'RF'])

        return mask_rest_frame_bal

    def apply_mask(self, forest):
        """Apply the mask. The mask is done by removing the affected
        pixels from the arrays in Forest.mask_fields

        Arguments
        ---------
        forest: Forest
        A Forest instance to which the correction is applied

        Raise
        -----
        MaskError if Forest.wave_solution is not 'log'
        """
        if Forest.wave_solution != "log":
            raise MaskError("SdssBalMask should only be applied when "
                            "Forest.wave_solution is 'log'. Found: "
                            f"{Forest.wave_solution}")

        mask_table = self.los_ids.get(forest.los_id)
        if (mask_table is not None) and len(mask_table) > 0:

            # find out which pixels to mask
            w = np.ones(forest.log_lambda.size, dtype=bool)
            for mask_range in mask_table:
                rest_frame_log_lambda = forest.log_lambda - np.log10(1. + forest.z)
                w &= ((rest_frame_log_lambda < mask_range['log_wave_min']) |
                      (rest_frame_log_lambda > mask_range['log_wave_max']))

            # do the actual masking
            for param in Forest.mask_fields:
                setattr(forest, param, getattr(forest, param)[w])
