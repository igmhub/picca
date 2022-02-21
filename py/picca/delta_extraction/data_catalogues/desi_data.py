"""This module defines the class DesiData to load DESI data
"""
import os
import logging
import glob

import fitsio
import healpy
import numpy as np

from picca.delta_extraction.astronomical_objects.desi_forest import DesiForest
from picca.delta_extraction.astronomical_objects.desi_pk1d_forest import DesiPk1dForest
from picca.delta_extraction.astronomical_objects.forest import Forest
from picca.delta_extraction.data import Data, defaults, accepted_options
from picca.delta_extraction.errors import DataError
from picca.delta_extraction.quasar_catalogues.desi_quasar_catalogue import DesiQuasarCatalogue
from picca.delta_extraction.quasar_catalogues.desi_quasar_catalogue import accepted_options as accepted_options_quasar_catalogue
from picca.delta_extraction.utils import ACCEPTED_BLINDING_STRATEGIES
from picca.delta_extraction.utils_pk1d import spectral_resolution_desi

accepted_options = sorted(list(set(accepted_options + accepted_options_quasar_catalogue + [
    "blinding", "delta lambda", "input directory", "lambda max",
    "lambda max rest frame", "lambda min", "lambda min rest frame",
    "rebin", "wave solution"])))

defaults.update({
    "delta lambda": 0.8,
    "lambda max": 5500.0,
    "lambda max rest frame": 1200.0,
    "lambda min": 3600.0,
    "lambda min rest frame": 1040.0,
    "blinding": "corr_yshift",
    # TODO: update this to "lin" when we are sure that the linear binning work
    "wave solution": "log",
    "rebin": 3,
})

class DesiData(Data):
    """Abstract class to read DESI data and format it as a list of
    Forest instances.

    Methods
    -------
    filter_forests (from Data)
    __init__
    _parse_config
    read_data
    set_blinding

    Attributes
    ----------
    analysis_type: str (from Data)
    Selected analysis type. Current options are "BAO 3D" or "PK 1D"

    forests: list of Forest (from Data)
    A list of Forest from which to compute the deltas.

    min_num_pix: int (from Data)
    Minimum number of pixels in a forest. Forests with less pixels will be dropped.

    blinding: str
    A string specifying the chosen blinding strategies. Must be one of the
    accepted values in ACCEPTED_BLINDING_STRATEGIES

    catalogue: astropy.table.Table
    The quasar catalogue

    input_directory: str
    Directory to spectra files.

    logger: logging.Logger
    Logger object
    """
    def __init__(self, config):
        """Initialize class instance

        Arguments
        ---------
        config: configparser.SectionProxy
        Parsed options to initialize class
        """
        self.logger = logging.getLogger(__name__)

        super().__init__(config)

        # load variables from config
        self.input_directory = None
        self.blinding = None
        self._parse_config(config)

        # load z_truth catalogue
        self.catalogue = DesiQuasarCatalogue(config).catalogue

        # read data
        is_mock, is_sv = self.read_data()

        # set blinding
        self.set_blinding(is_mock, is_sv)

    def _parse_config(self, config):
        """Parse the configuration options

        Arguments
        ---------
        config: configparser.SectionProxy
        Parsed options to initialize class

        Raise
        -----
        DataError upon missing required variables
        """
        # setup Forest class variables
        wave_solution = config.get("wave solution")
        if wave_solution is None:
            raise DataError("Missing argument 'wave solution' required by DesiData")
        if wave_solution not in ["lin", "log"]:
            raise DataError("Unrecognised value for 'wave solution'. Expected either "
                            f"'lin' or 'lof'. Found {wave_solution}")
        Forest.wave_solution = wave_solution

        if Forest.wave_solution == "log":
            rebin = config.getint("rebin")
            if rebin is None:
                raise DataError("Missing argument 'rebin' required by DesiData when "
                                "'wave solution' is set to 'log'")
            Forest.delta_log_lambda = rebin * 1e-4
        elif Forest.wave_solution == "lin":
            delta_lambda = config.getfloat("delta lambda")
            if delta_lambda is None:
                raise DataError("Missing argument 'delta lambda' required by DesiData")
            Forest.delta_log_lambda = np.log10(delta_lambda)
        else:
            raise DataError("Forest.wave_solution must be either "
                            "'log' or 'lin'")

        lambda_max = config.getfloat("lambda max")
        if lambda_max is None:
            raise DataError("Missing argument 'lambda max' required by DesiData")
        Forest.log_lambda_max = np.log10(lambda_max)
        lambda_max_rest_frame = config.getfloat("lambda max rest frame")
        if lambda_max_rest_frame is None:
            raise DataError("Missing argument 'lambda max rest frame' required by DesiData")
        Forest.log_lambda_max_rest_frame = np.log10(lambda_max_rest_frame)
        lambda_min = config.getfloat("lambda min")
        if lambda_min is None:
            raise DataError("Missing argument 'lambda min' required by DesiData")
        Forest.log_lambda_min = np.log10(lambda_min)
        lambda_min_rest_frame = config.getfloat("lambda min rest frame")
        if lambda_min_rest_frame is None:
            raise DataError("Missing argument 'lambda min rest frame' required by DesiData")
        Forest.log_lambda_min_rest_frame = np.log10(lambda_min_rest_frame)

        # instance variables
        self.blinding = config.get("blinding")
        if self.blinding is None:
            raise DataError("Missing argument 'blinding' required by DesiData")
        if self.blinding not in ACCEPTED_BLINDING_STRATEGIES:
            raise DataError("Unrecognized blinding strategy. Accepted strategies "
                            f"are {ACCEPTED_BLINDING_STRATEGIES}. Found {self.blinding}")

        self.input_directory = config.get("input directory")
        if self.input_directory is None:
            raise DataError(
                "Missing argument 'input directory' required by DesiData")

    # pylint: disable=no-self-use
    # this method should use self in child classes
    def read_data(self):
        """Read the spectra and formats its data as Forest instances.

        Method to be implemented by child classes.

        Return
        ------
        is_mock: bool
        True if mocks are read, False otherwise

        is_sv: bool
        True if all the read data belong to SV. False otherwise

        Raise
        -----
        DataError if no quasars were found
        """
        raise DataError("Function 'read_data' was not overloaded by child class")

    def set_blinding(self, is_mock, is_sv):
        """Set the blinding in Forest.

        Update the stored value if necessary.

        Attributes
        ----------
        is_mock: boolean
        True if reading mocks, False otherwise

        is_sv: boolean
        True if reading SV data only, False otherwise
        """
        # blinding checks
        if is_mock:
            if self.blinding != "none":
                self.logger.warning(f"Selected blinding, {self.blinding} is "
                                    "being ignored as mocks should not be "
                                    "blinded. 'none' blinding engaged")
                self.blinding = "none"
        elif is_sv:
            if self.blinding != "none":
                self.logger.warning(f"Selected blinding, {self.blinding} is "
                                    "being ignored as SV data should not be "
                                    "blinded. 'none' blinding engaged")
                self.blinding = "none"
        # TODO: remove this when we are ready to unblind
        else:
            if self.blinding != "corr_yshift":
                self.logger.warning(f"Selected blinding, {self.blinding} is "
                                    "being ignored as data should be blinded. "
                                    "'corr_yshift' blinding engaged")
                self.blinding = "corr_yshift"

        # set blinding strategy
        Forest.blinding = self.blinding
