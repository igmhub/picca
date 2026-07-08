"""This module defines the class DesiData to load DESI data
"""
import logging
import multiprocessing
import time
import itertools

import fitsio
import numpy as np

from picca.delta_extraction.data_catalogues.desi_data import (  # pylint: disable=unused-import
    DesiData,
    accepted_options,
    defaults,
)
from picca.delta_extraction.data_catalogues.desi_healpix_fast import DesiHealpixFileHandler, combine_results
from picca.delta_extraction.errors import DataError


class DesiUniqPix(DesiData):
    """Reads the spectra from DESI using healpix mode and formats its data as a
    list of Forest instances.

    Methods
    -------
    (see DesiData in py/picca/delta_extraction/data_catalogues/desi_data.py)
    __init__
    __parse_config
    get_filename
    read_data
    read_file

    Attributes
    ----------
    (see DesiData in py/picca/delta_extraction/data_catalogues/desi_data.py)

    logger: logging.Logger
    Logger object

    num_processors: int
    Number of processors to be used for parallel reading
    """

    def __init__(self, config):
        """Initialize class instance

        Arguments
        ---------
        config: configparser.SectionProxy
        Parsed options to initialize class
        """
        self.logger = logging.getLogger(__name__)

        # overwrite in-nside value
        # this is hard-coded to 64 since it is the value used to compute the uniqpix values in the DESI data
        config["in_nside"] = str(64)

        # set add uniqpix flag
        config["add uniqpix"] = "True"

        super().__init__(config)

        #TODO: remove exception when this is implemented
        if self.analysis_type == "PK 1D":
            raise NotImplementedError("fast healpix reading is not implemented for PK 1D analyses")

    def get_filename(self, survey, uniqpix, coadd_name):
        """Get the name of the file to read

        Arguments
        ---------
        survey: str
        Name of the survey (sv, sv1, sv2, sv3, main, special)

        uniqpix: int
        Uniqpix of observations

        Return
        ------
        filename: str
        The name of the file to read

        is_mock: bool
        False, as we are reading DESI data
        """
        filename = (
            f"{self.input_directory}/{survey}/dark/{uniqpix//100}/{uniqpix}/{coadd_name}-{survey}-"
            f"dark-{uniqpix}.fits")
        # TODO: not sure if we want the dark survey to be hard coded
        # in here, probably won't run on anything else, but still
        return filename

    def read_data(self, is_mock=False):
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
        coadd_name = "spectra" if self.use_non_coadded_spectra else "coadd"

        grouped_catalogue = self.catalogue.group_by(["UNIQPIX", "SURVEY"])

        arguments = [
            (self.get_filename(group["SURVEY"][0], group["UNIQPIX"][0], coadd_name), group)
            for group in grouped_catalogue.groups
        ]

        self.logger.info(f"reading data from {len(arguments)} files")
        if self.num_processors > 1:
            context = multiprocessing.get_context('fork')
            with context.Pool(processes=self.num_processors) as pool:
                imap_it = pool.imap(
                    DesiHealpixFileHandler(self.logger), arguments)
                t0 = time.time()
                self.logger.progress("Merging threads")

                self.forests = combine_results(imap_it)
                t1 = time.time()
                self.logger.progress(f"Time spent meerging threads: {t1-t0}")
        else:
            raise NotImplementedError('uniqux reading is not implemented'
                                      'for analyses with "num processors=1"')

        if len(self.forests) == 0:
            raise DataError("No quasars found, stopping here")

        return is_mock

