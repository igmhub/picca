"""This module defines the classes BalMask and Dla used in the
masking of DLAs"""
import logging

import fitsio
import numpy as np
from numba import njit

from picca.delta_extraction.astronomical_objects.forest import Forest
from picca.delta_extraction.errors import MaskError
from picca.delta_extraction.mask import Mask, accepted_options, defaults
from picca.delta_extraction.utils import (
    SPEED_LIGHT, update_accepted_options, update_default_options)

accepted_options = update_accepted_options(accepted_options, [
    "bal index type", "filename", "los_id name", "keep pixels"
])

defaults = update_default_options(
    defaults, {
        "bal index type": "ai",
        "los_id name": "THING_ID",
    })

# Wavelengths in Angstroms
LINES = np.array([
    ("lCIV", 1549),
    ("lSiIV2", 1403),
    ("lSiIV1", 1394),
    ("lNV", 1240.81),
    ("lLya", 1216.1),
    ("lCIII", 1175),
    ("lPV2", 1128),
    ("lPV1", 1117),
    ("lSIV2", 1074),
    ("lSIV1", 1062),
    ("lOIV", 1031),
    ("lOVI", 1037),
    ("lOI", 1039),
    ("lLyb", 1025.7),
    ("lLy3", 972.5),
    ("lCIII", 977.0),
    ("lNIII", 989.9),
    ("lLy4", 949.7),
    ], dtype=[("name", "U10"), ("value", float)])

@njit()
def add_bal_rest_frame(los_id, los_id_array, min_velocities_array, max_velocities_array):
    """Creates a list of wavelengths to be masked out by forest.mask

    Arguments
    ---------
    los_id: str
    Line-of-sight id

    los_id_name: str
    Name of the line-of-sight id

    Return
    ------
    log_lambda_min: array of float
    Array with the minimum log(wavelength) values in the masks intervals

    log_lambda_max: array of float
    Array with the maximum log(wavelength) values in the masks intervals
    """
    # Match thing_id of object to BAL catalog index
    # Will there be duplicates or only one index?
    match_index = np.where(los_id_array == los_id)[0]

    # Store the min/max velocity pairs from the BAL catalog
    min_velocities = min_velocities_array[match_index].ravel()
    max_velocities = max_velocities_array[match_index].ravel()

    # Remove non-positive velocity rows
    w = (min_velocities > 0) & (max_velocities > 0)
    min_velocities = min_velocities[w]
    max_velocities = max_velocities[w]

    num_velocities = min_velocities.size
    if num_velocities == 0:
        return np.zeros(0), np.zeros(0)

    # Calculate mask width for each velocity pair, for each emission line
    # This might be  bit confusing, since BAL absorption is
    # blueshifted from the emission lines. The “minimum velocity”
    # corresponds to the red side of the BAL absorption (the larger
    # wavelength value), and the “maximum velocity” corresponds to
    # the blue side (the smaller wavelength value).
    lambda_max = np.outer(LINES['value'], 1 - min_velocities / SPEED_LIGHT).ravel()
    lambda_min = np.outer(LINES['value'], 1 - max_velocities / SPEED_LIGHT).ravel()
    log_lambda_min = np.log10(lambda_min)
    log_lambda_max = np.log10(lambda_max)

    return log_lambda_min, log_lambda_max

class BalMask(Mask):
    """Class to mask BALs

    Methods
    -------
    __init__
    add_bal_rest_frame
    apply_mask

    Attributes
    ----------
    (see Mask in py/picca/delta_extraction/mask.py)

    bal_index_type: str
    BAL index type, choose either 'ai' or 'bi'. This will set which velocity
    the  BAL mask uses.

    velocity_list: list of str
    List of column names for minimum and maximum velocities respectively.

    cat: dict
    Dictionary with the BAL catalogue

    logger: logging.Logger
    Logger object
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
        MaskError if input file does not have the expected extension
        MaskError if input file does not have the expected fields
        MaskError upon OsError when reading the mask file
        """
        self.logger = logging.getLogger(__name__)

        super().__init__(config)

        filename = config.get("filename")
        if filename is None:
            raise MaskError("Missing argument 'filename' required by BalMask")

        los_id_name = config.get("los_id name")
        if los_id_name is None:
            raise MaskError(
                "Missing argument 'los_id name' required by BalMask")
        if los_id_name == "THING_ID":
            ext_name = 'BALCAT'
        elif los_id_name == "TARGETID":
            extnames = [ext.get_extname() for ext in fitsio.FITS(filename)]
            if "QSO_CAT" in extnames:
                ext_name = "QSO_CAT"
            elif "ZCATALOG" in extnames:
                ext_name = "ZCATALOG"
            else:
                raise MaskError(
                    "Could not find valid quasar catalog extension in fits "
                    f"file: {filename}")
        else:
            raise MaskError(
                "Unrecognized los_id name. Expected one of 'THING_ID' "
                f" or 'TARGETID'. Found {los_id_name}")

        # setup bal index limit
        self.bal_index_type = config.get("bal index type")
        if self.bal_index_type is None:
            raise MaskError(
                "Missing argument 'bal index type' "
                "required by BalMask")
        columns_list = None
        if self.bal_index_type == "ai":
            columns_list = [los_id_name, 'VMIN_CIV_450', 'VMAX_CIV_450']
        elif self.bal_index_type == "bi":
            columns_list = [los_id_name, 'VMIN_CIV_2000', 'VMAX_CIV_2000']
        else:
            self.bal_index_type = MaskError(
                "In BalMask, unrecognized value "
                "for 'bal_index_type'. Expected one "
                "of 'ai' or 'bi'. Found "
                f"{self.bal_index_type}")

        self.velocity_list = columns_list[1:]

        self.logger.progress(f"Reading BAL catalog from: {filename}")

        try:
            hdul = fitsio.FITS(filename)
            self.cat = {
                col: (hdul[ext_name][col][:].astype(np.int64)
                    if col == los_id_name else
                    hdul[ext_name][col][:].astype(np.float64))
                for col in columns_list}
        except OSError as error:
            raise MaskError(
                f"Error loading BalMask. File {filename} does "
                f"not have extension '{ext_name}'"
            ) from error
        except ValueError as error:
            aux = "', '".join(columns_list)
            raise MaskError(
                f"Error loading BalMask. File {filename} does "
                f"not have fields '{aux}' in HDU '{ext_name}'"
            ) from error
        finally:
            hdul.close()

        # compute info for each line of sight
        self.los_ids = {
            los_id: (
                los_id,
                self.cat[los_id_name],
                self.cat[self.velocity_list[0]],
                self.cat[self.velocity_list[1]])
            for los_id in np.unique(self.cat[los_id_name])
        }

        num_bals = len(self.los_ids)
        self.logger.progress(f'In catalog: {num_bals} BAL quasars')

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
        if self.los_ids.get(forest.los_id) is None:
            return

        log_lambda_min, log_lambda_max = add_bal_rest_frame(
            *self.los_ids.get(forest.los_id))

        if len(log_lambda_min) == 0:
            return

        # find out which pixels to mask
        w = np.ones(forest.log_lambda.size, dtype=bool)
        rest_frame_log_lambda = forest.log_lambda - np.log10(1. + forest.z)
        mask_idx_ranges = np.searchsorted(
            rest_frame_log_lambda,
            [log_lambda_min, log_lambda_max]).T
        # Make sure first index comes before the second
        mask_idx_ranges.sort(axis=1)

        for idx1, idx2 in mask_idx_ranges:
            w[idx1:idx2] = 0

        # do the actual masking
        for param in Forest.mask_fields:
            self._masker(forest, param, w)
