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
from picca.delta_extraction.quasar_catalogues.desi_quasar_catalogue import defaults as defaults_quasar_catalogue
from picca.delta_extraction.utils import ACCEPTED_BLINDING_STRATEGIES
from picca.delta_extraction.utils_pk1d import spectral_resolution_desi

accepted_options = sorted(list(set(accepted_options + accepted_options_quasar_catalogue + [
    "blinding", "wave solution"])))

defaults.update({
    "delta lambda": 0.8,
    "delta log lambda": 3e-4,
    "blinding": "corr_yshift",
    "wave solution": "lin",
})
defaults.update(defaults_quasar_catalogue)

class DesiData(Data):
    """Abstract class to read DESI data and format it as a list of
    Forest instances.

    Methods
    -------
    filter_forests (from Data)
    __init__
    __parse_config
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
        self.blinding = None
        self.__parse_config(config)

        # load z_truth catalogue
        self.catalogue = DesiQuasarCatalogue(config).catalogue

        # read data
        is_mock, is_sv = self.read_data()

        # set blinding
        self.set_blinding(is_mock, is_sv)

    def __parse_config(self, config):
        """Parse the configuration options

        Arguments
        ---------
        config: configparser.SectionProxy
        Parsed options to initialize class

        Raise
        -----
        DataError upon missing required variables
        """
        # instance variables
        self.blinding = config.get("blinding")
        if self.blinding is None:
            raise DataError("Missing argument 'blinding' required by DesiData")
        if self.blinding not in ACCEPTED_BLINDING_STRATEGIES:
            raise DataError("Unrecognized blinding strategy. Accepted strategies "
                            f"are {ACCEPTED_BLINDING_STRATEGIES}. "
                            f"Found '{self.blinding}'")

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
