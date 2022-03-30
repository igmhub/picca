"""This module defines the classes BalMask and Dla used in the
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
    "los_id name": "THING_ID",
}

accepted_options = ["bal index type", "filename", "los_id name"]

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


class BalMask(Mask):
    """Class to mask BALs

    Methods
    -------
    __init__
    apply_mask

    Attributes
    ----------
    los_ids: dict (from Mask)
    A dictionary with the BALs contained in each line of sight. Keys are the
    identifier for the line of sight and values are lists of (z_abs, nhi)

    bal_index_type: str
    BAL index type, choose either 'ai' or 'bi'. This will set which velocity
    the  BAL mask uses.

    cat: dict
    Dictionary with the BAL catalogue

    logger: logging.Logger
    Logger object

    mask: astropy.Table
    Table containing specific intervals of wavelength to be masked for BALs
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

        super().__init__()

        filename = config.get("filename")
        if filename is None:
            raise MaskError("Missing argument 'filename' required by BalMask")

        los_id_name = config.get("los_id name")
        if los_id_name is None:
            raise MaskError(
                "Missing argument 'los_id name' required by BalMask")
        elif los_id_name == "THING_ID":
            ext_name = 'BALCAT'
        elif los_id_name == "TARGETID":
            ext_name = 'ZCATALOG'
        else:
            raise MaskError(
                "Unrecognized los_id name. Expected one of 'THING_ID' "
                f" or 'TARGETID'. Found {los_id_name}")

        # setup bal index limit
        self.bal_index_type = config.getfloat("bal index type")
        if self.bal_index_type is None:
            self.bal_index_type = MaskError(
                "Missing argument 'bal index type' "
                "required by BalMask")
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

        self.logger.progress(f"Reading BAL catalog from: {filename}")

        try:
            hdul = fitsio.FITS(filename)
            self.cat = {col: hdul[ext_name][col][:] for col in columns_list}
        except OSError:
            raise MaskError(f"Error loading BalMask. File {filename} does "
                            f"not have extension '{ext_name}'")
        except ValueError:
            aux = "', '".join(columns_list)
            raise MaskError(f"Error loading BalMask. File {filename} does "
                            f"not have fields '{aux}' in HDU '{ext_name}'")
        finally:
            hdul.close()

        # compute info for each line of sight
        self.los_ids = {}
        for los_id in np.unique(self.cat[los_id_name]):
            self.los_ids[los_id] = self.add_bal_rest_frame(los_id)

        num_bals = np.sum([len(los_id) for los_id in self.los_ids.values()])
        self.logger.progress('In catalog: {} BAL quasars'.format(num_bals))

    def add_bal_rest_frame(self, los_id, los_id_name):
        """Creates a list of wavelengths to be masked out by forest.mask

        Arguments
        ---------
        los_id: str
        Line-of-sight id

        los_id_name: str
        Name of the line-of-sight id
        """
        if self.bal_index_type == 'bi':
            velocity_list = ['VMIN_CIV_2000', 'VMAX_CIV_2000']
        else:  # AI, the default
            velocity_list = ['VMIN_CIV_450', 'VMAX_CIV_450']

        mask_rest_frame_bal = Table(names=[
            'log_lambda_min', 'log_lambda_max', 'lambda_min', 'lambda_max'
        ],
                                    dtype=['f4', 'f4', 'f4', 'f4'])
        min_velocities = []  # list of minimum velocities
        max_velocities = []  # list of maximum velocities

        # Match thing_id of object to BAL catalog index
        match_index = np.where(self.cat[los_id_name] == los_id)[0][0]

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
                log_lambda_min = np.log10(line * (1 - min_vel / SPEED_LIGHT))
                log_lambda_max = np.log10(line * (1 - max_vel / SPEED_LIGHT))
                mask_rest_frame_bal.add_row([
                    log_lambda_min,
                    log_lambda_max,
                    10**log_lambda_min,
                    10**log_lambda_max,
                ])

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
        mask_table = self.los_ids.get(forest.los_id)
        if (mask_table is not None) and len(mask_table) > 0:

            # find out which pixels to mask
            if Forest.wave_solution == "log":
                w = np.ones(forest.log_lambda.size, dtype=bool)
                for mask_range in mask_table:
                    rest_frame_log_lambda = forest.log_lambda - np.log10(
                        1. + forest.z)
                    w &= (
                        (rest_frame_log_lambda < mask_range['log_lambda_min'])
                        |
                        (rest_frame_log_lambda > mask_range['log_lambda_max']))
            elif Forest.wave_solution == "lin":
                w = np.ones(forest.lambda_.size, dtype=bool)
                for mask_range in mask_table:
                    rest_frame_lambda = forest.lambda_ / (1. + forest.z)
                    w &= ((rest_frame_lambda < mask_range['lambda_min']) |
                          (rest_frame_lambda > mask_range['lambda_max']))
            else:
                raise MaskError(
                    "Forest.wave_solution must be either 'log' or 'lin'")

            # do the actual masking
            for param in Forest.mask_fields:
                if param in ['resolution_matrix']:
                    setattr(forest, param, getattr(forest, param)[:, w])
                else:
                    setattr(forest, param, getattr(forest, param)[w])