"""This module defines the class DesiData to load DESI data
"""
import logging

from picca.delta_extraction.data_catalogues.desi_healpix_fast import DesiHealpixFast
from picca.delta_extraction.data_catalogues.desi_healpix_fast import (# pylint: disable=unused-import
    defaults, accepted_options)

class DesisimMocksFast(DesiHealpixFast):
    """Reads the spectra from DESI using healpix mode and formats its data as a
    list of Forest instances.

    Should work for both data and mocks. This is specified using the 'mode'
    keyword. It is required to set the in_nside member.

    Methods
    -------
    (see DesiHealpix in py/picca/delta_extraction/data_catalogues/desi_healpix.py)
    __init__
    get_filename

    Attributes
    ----------
    (see DesiHealpix in py/picca/delta_extraction/data_catalogues/desi_healpix.py)

    in_nside: 16
    Parameter in_nside to compute the healpix indexes

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

        # overwrite value for mocks
        self.in_nside = 16
        config["in_nside"] = str(self.in_nside)

        super().__init__(config)

    def get_filename(self, survey, healpix, coadd_name):
        """Get the name of the file to read

        Arguments
        ---------
        survey: str
        Name of the survey (sv, sv1, sv2, sv3, main, special)

        healpix: int
        Healpix of observations

        Return
        ------
        filename: str
        The name of the file to read

        is_mock: bool
        True, as we are reading mocks
        """
        filename = (
            f"{self.input_directory}/{healpix//100}/{healpix}/spectra-"
            f"{self.in_nside}-{healpix}.fits")
        return filename

    def read_data(self, is_mock=True):
        """Read the data.

        Method used to read healpix-based survey data.

        Return
        ------
        is_mock: bool
        False for DESI data and True for mocks

        Raise
        -----
        DataError if no quasars were found
        """
        return super().read_data(is_mock=is_mock)
