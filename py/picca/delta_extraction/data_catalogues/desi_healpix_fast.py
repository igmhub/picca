"""This module defines the class DesiData to load DESI data
"""
import logging
import multiprocessing
import time
import itertools

import fitsio
import numpy as np

from picca.delta_extraction.astronomical_objects.desi_forest import DesiForest
from picca.delta_extraction.data_catalogues.desi_data import (  # pylint: disable=unused-import
    DesiData,
    DesiDataFileHandler,
    accepted_options,
    defaults,
    merge_new_forest,
    verify_exposures_shape,
)
from picca.delta_extraction.errors import DataError


class DesiHealpixFast(DesiData):
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

        super().__init__(config)

        #TODO: remove exception when this is implemented
        if self.analysis_type == "PK 1D":
            raise NotImplementedError("fast healpix reading is not implemented for PK 1D analyses")

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
        False, as we are reading DESI data
        """
        filename = (
            f"{self.input_directory}/{survey}/dark/{healpix//100}/{healpix}/{coadd_name}-{survey}-"
            f"dark-{healpix}.fits")
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

        grouped_catalogue = self.catalogue.group_by(["HEALPIX", "SURVEY"])

        arguments = [
            (self.get_filename(group["SURVEY"][0], group["HEALPIX"][0], coadd_name), group)
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

        if len(self.forests) == 0:
            raise DataError("No quasars found, stopping here")

        return is_mock


# Class to read in parallel
# Seems lightweight to copy all these 3 arguments
class DesiHealpixFileHandler():
    """File handler for class DesiHealpix

    Methods
    -------
    (see DesiDataFileHandler in py/picca/delta_extraction/data_catalogues/desi_data.py)
    read_file

    Attributes
    ----------
    (see DesiDataFileHandler in py/picca/delta_extraction/data_catalogues/desi_data.py)
    """
    def __init__(self, logger):
        """Initialize file handler

        Arguments
        ---------
        logger: logging.Logger
        Logger object from parent class. Trying to initialize it here
        without copying failed data_tests.py
        """
        # The next line gives failed tests
        # self.logger = logging.getLogger(__name__)
        self.logger = logger

    def __call__(self, args):
        """Call method read_file. Note imap can be called with
        only one argument, hence tuple as argument.

        Arguments
        ---------
        args: tuple
        Arguments to be passed to read_file. Should contain a string with the
        filename and a astropy.table.Table with the quasar catalogue
        """
        return self.read_file(*args)

    def read_file(self, filename, catalogue):
        """Read the spectra and formats its data as Forest instances.

        Arguments
        ---------
        filename: str
        Name of the file to read

        catalogue: astropy.table.Table
        The quasar catalogue fragment associated with this file

        Returns:
        ---------
        forests_by_targetid: dict
        Dictionary were forests are stored.

        num_data: int
        The number of instances loaded

        Raise
        -----
        DataError if the analysis type is PK 1D and resolution data is not present
        """
        try:
            hdul = fitsio.FITS(filename)
        except IOError:
            self.logger.warning(f"Error reading '{filename}'. Ignoring file")
            return {}, 0
        # Read targetid from fibermap to match to catalogue later
        fibermap = hdul['FIBERMAP'].read()

        colors = ["B", "R"]
        if "Z_FLUX" in hdul:
            colors.append("Z")

        # read wavelength
        wave = np.hstack([hdul[f"{color}_WAVELENGTH"].read() for color in colors])
        log_lambda = np.log10(wave)
        # read flux and ivar
        flux_colors = {
            color: hdul[f"{color}_FLUX"].read() for color in colors
        }
        ivar_colors = {
            color: (hdul[f"{color}_IVAR"].read() * (hdul[f"{color}_MASK"].read() == 0))
            for color in colors
        }

        # Loop over quasars in catalogue fragment
        forests = []
        for row in catalogue:
            # Find which row in tile contains this quasar
            # It should be there by construction
            targetid = row["TARGETID"]
            w_t = np.where(fibermap["TARGETID"] == targetid)[0]
            if len(w_t) == 0:
                self.logger.warning(
                    f"Error reading {targetid}. Ignoring object")
                continue

            w_t = w_t[0]
            # Construct DesiForest instance
            # Fluxes from the different spectrographs will be coadded
            flux = np.hstack([item[w_t] for item in flux_colors.values()])
            ivar = np.hstack([item[w_t] for item in ivar_colors.values()])

            args = {
                "flux": flux,
                "ivar": ivar,
                "targetid": targetid,
                "ra": row['RA'],
                "dec": row['DEC'],
                "z": row['Z'],
                "log_lambda": log_lambda,
            }
            forest = DesiForest(**args)
            forest.rebin()
            forests.append(forest)

        return forests


def combine_results(lists):
    """Combine the content of all the lists into a single one

    Arguments
    ---------
    lists: list of lists
    The content to be merged

    Return
    ------
    combined_list: list
    The combined list
    """
    # Combine a list of lists into a single list
    return list(itertools.chain(*lists))
