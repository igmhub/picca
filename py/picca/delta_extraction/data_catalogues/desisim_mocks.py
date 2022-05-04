"""This module defines the class DesiData to load DESI data
"""
import logging

from picca.delta_extraction.data_catalogues.desi_healpix import DesiHealpix
from picca.delta_extraction.data_catalogues.desi_healpix import (# pylint: disable=unused-import
    defaults, accepted_options)

class DesisimMocks(DesiHealpix):
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
        if self.use_non_coadded_spectra:
            self.logger.warning(
                'the "use_non_coadded_spectra" option was set, '
                'but has no effect on Mocks, will proceed as normal')

    def get_filename(self, survey, healpix):
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
        return filename, True
