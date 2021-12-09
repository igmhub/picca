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
from picca.delta_extraction.desi_data import DesiHealpix
from picca.delta_extraction.errors import DataError
from picca.delta_extraction.quasar_catalogues.ztruth_catalogue import ZtruthCatalogue
from picca.delta_extraction.utils import ACCEPTED_BLINDING_STRATEGIES
from picca.delta_extraction.utils_pk1d import spectral_resolution_desi

class DesisimMocks(DesiHealpix):
    """Reads the spectra from DESI using healpix mode and formats its data as a
    list of Forest instances.

    Should work for both data and mocks. This is specified using the 'mode'
    keyword. It is required to set the in_nside member.

    Methods
    -------
    filter_forests (from Data)
    set_blinding (from Data)
    read_file (from DesiHealpix)
    __init__
    read_data

    Attributes
    ----------
    analysis_type: str (from Data)
    Selected analysis type. Current options are "BAO 3D" or "PK 1D"

    forests: list of Forest (from Data)
    A list of Forest from which to compute the deltas.

    min_num_pix: int (from Data)
    Minimum number of pixels in a forest. Forests with less pixels will be dropped.

    blinding: str (from DesiData)
    A string specifying the chosen blinding strategies. Must be one of the
    accepted values in ACCEPTED_BLINDING_STRATEGIES

    input_directory: str (from DesiData)
    Directory to spectra files.

    in_nside: 64 or 16
    Nside used in the folder structure (64 for data and 16 for mocks)

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

    def read_data(self):
        """Read the spectra and formats its data as Forest instances.

        Method used to read healpix-based survey data.

        Return
        ------
        is_mock: bool
        True as we are loading mocks

        is_sv: bool
        False as mocks data is not part of DESI SV data

        Raise
        -----
        DataError if the analysis type is PK 1D and resolution data is not present
        """
        in_nside = 16

        healpix = [
            healpy.ang2pix(in_nside, np.pi / 2 - row["DEC"], row["RA"], nest=True)
            for row in self.catalogue
        ]
        self.catalogue["HEALPIX"] = healpix
        self.catalogue.sort("HEALPIX")
        grouped_catalogue = self.catalogue.group_by(["HEALPIX", "SURVEY"])

        forests_by_targetid = {}
        for (index,
             (healpix, survey)), group in zip(enumerate(grouped_catalogue.groups.keys),
                                    grouped_catalogue.groups):

            filename = (
                f"{self.input_directory}/{healpix//100}/{healpix}/spectra-"
                f"{in_nside}-{healpix}.fits")

            self.logger.progress(
                f"Read {index} of {len(grouped_catalogue.groups.keys)}. "
                f"num_data: {len(forests_by_targetid)}")

            self.read_file(filename)

        self.forests = list(forests_by_targetid.values())

        return True, False
