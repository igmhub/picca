"""This module defines the class DesiData to load DESI data
"""
import logging
import os
import multiprocessing

import fitsio
import numpy as np

from picca.delta_extraction.data_catalogues.desi_data import DesiData
from picca.delta_extraction.data_catalogues.desi_data import (# pylint: disable=unused-import
    defaults, accepted_options)
from picca.delta_extraction.errors import DataError

accepted_options = sorted(
    list(set(accepted_options + ["num processors"])))


class DesiHealpix(DesiData):
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

        self.num_processors = None
        self.__parse_config(config)

        # init of DesiData needs to come last, as it contains the actual data
        # reading and thus needs all config
        super().__init__(config)

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
        self.num_processors = config.getint("num processors")
        if self.num_processors is None:
            raise DataError(
                "Missing argument 'num processors' required by DesiHealpix")
        if self.num_processors == 0:
            self.num_processors = (multiprocessing.cpu_count() // 2)

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
        False, as we are reading DESI data
        """
        input_directory = f'{self.input_directory}/{survey}/dark'
        coadd_name = "spectra" if self.use_non_coadded_spectra else "coadd"
        filename = (
            f"{input_directory}/{healpix//100}/{healpix}/{coadd_name}-{survey}-"
            f"dark-{healpix}.fits")
        # TODO: not sure if we want the dark survey to be hard coded
        # in here, probably won't run on anything else, but still
        return filename, False

    def read_data(self):
        """Read the data.

        Method used to read healpix-based survey data.

        Return
        ------
        is_mock: bool
        False for DESI data and True for mocks

        is_sv: bool
        True if all the read data belong to SV. False otherwise

        Raise
        -----
        DataError if no quasars were found
        """
        grouped_catalogue = self.catalogue.group_by(["HEALPIX", "SURVEY"])

        is_sv = True
        is_mock = False
        if self.num_processors > 1:
            context = multiprocessing.get_context('fork')
            manager = multiprocessing.Manager()
            forests_by_targetid = manager.dict()
            arguments = []
            for (index, (healpix, survey)), group in zip(
                    enumerate(grouped_catalogue.groups.keys),
                    grouped_catalogue.groups):

                if survey not in ["sv", "sv1", "sv2", "sv3"]:
                    is_sv = False

                filename, is_mock_aux = self.get_filename(survey, healpix)
                if is_mock_aux:
                    is_mock = True
                #input_directory = f'{self.input_directory}/{survey}/dark'
                #coadd_name = "spectra" if self.use_non_coadded_spectra else "coadd"
                #filename = (
                #    f"{input_directory}/{healpix//100}/{healpix}/{coadd_name}-{survey}-"
                #    f"dark-{healpix}.fits")

                arguments.append((filename, group, forests_by_targetid))

                self.logger.info(f"reading data from {len(arguments)} files")
            with context.Pool(processes=self.num_processors) as pool:

                pool.starmap(self.read_file, arguments)
            for forest in forests_by_targetid.values():
                # TODO: the following just does the consistency checking again,
                # to avoid mask_fields not being populated. In the long run an
                # alternative way of running the multiprocessing is envisioned
                # which would be more stable, see discussion in PRs 879 and 883
                forest.consistency_check()
        else:
            forests_by_targetid = {}
            for (index, (healpix, survey)), group in zip(
                    enumerate(grouped_catalogue.groups.keys),
                    grouped_catalogue.groups):

                if survey not in ["sv", "sv1", "sv2", "sv3"]:
                    is_sv = False

                filename = self.get_filename(survey, healpix)
                filename, is_mock_aux = self.get_filename(survey, healpix)
                if is_mock_aux:
                    is_mock = True
                #input_directory = f'{self.input_directory}/{survey}/dark'
                #coadd_name = "spectra" if self.use_non_coadded_spectra else "coadd"
                #filename = (
                #    f"{input_directory}/{healpix//100}/{healpix}/{coadd_name}-{survey}-"
                #    f"dark-{healpix}.fits")
                # TODO: not sure if we want the dark survey to be hard coded
                # in here, probably won't run on anything else, but still
                self.read_file(filename, group, forests_by_targetid)
                self.logger.progress(
                    f"Read {index} of {len(grouped_catalogue.groups.keys)}. "
                    f"num_data: {len(forests_by_targetid)}")

        if len(forests_by_targetid) == 0:
            raise DataError("No quasars found, stopping here")
        self.forests = list(forests_by_targetid.values())

        return is_mock, is_sv

    def read_file(self, filename, catalogue, forests_by_targetid):
        """Read the spectra and formats its data as Forest instances.

        Arguments
        ---------
        filename: str
        Name of the file to read

        catalogue: astropy.table.Table
        The quasar catalogue fragment associated with this file

        forests_by_targetid: dict
        Dictionary were forests are stored. Its content is modified by this
        function with the new forests.

        Raise
        -----
        DataError if the analysis type is PK 1D and resolution data is not present
        """
        try:
            hdul = fitsio.FITS(filename)
        except IOError:
            self.logger.warning(f"Error reading '{filename}'. Ignoring file")
            return
        # Read targetid from fibermap to match to catalogue later
        fibermap = hdul['FIBERMAP'].read()
        # First read all wavelength, flux, ivar, mask, and resolution
        # from this file
        spectrographs_data = {}
        colors = ["B", "R"]
        if "Z_FLUX" in hdul:
            colors.append("Z")
        else:
            self.logger.warning(
                f"Missing Z band from {filename}. Ignoring color.")

        reso_from_truth = False
        for color in colors:
            spec = {}
            try:
                spec["WAVELENGTH"] = hdul[f"{color}_WAVELENGTH"].read()
                spec["FLUX"] = hdul[f"{color}_FLUX"].read()
                spec["IVAR"] = (hdul[f"{color}_IVAR"].read() *
                                (hdul[f"{color}_MASK"].read() == 0))
                w = np.isnan(spec["FLUX"]) | np.isnan(spec["IVAR"])
                for key in ["FLUX", "IVAR"]:
                    spec[key][w] = 0.
                if self.analysis_type == "PK 1D":
                    if f"{color}_RESOLUTION" in hdul:
                        spec["RESO"] = hdul[f"{color}_RESOLUTION"].read()
                    else:
                        if not reso_from_truth:
                            self.logger.debug(
                                "no resolution in files, reading from truth files"
                            )
                        reso_from_truth = True
                        basename_truth = os.path.basename(filename).replace(
                            'spectra-', 'truth-')
                        pathname_truth = os.path.dirname(filename)
                        filename_truth = f"{pathname_truth}/{basename_truth}"
                        if os.path.exists(filename_truth):
                            with fitsio.FITS(filename_truth) as hdul_truth:
                                spec["RESO"] = hdul_truth[
                                    f"{color}_RESOLUTION"].read()
                        else:
                            raise DataError(
                                f"Error while reading {color} band from "
                                f"{filename}. Analysis type is 'PK 1D', "
                                "but file does not contain HDU "
                                f"'{color}_RESOLUTION'")
                spectrographs_data[color] = spec
            except OSError:
                self.logger.warning(
                    f"Error while reading {color} band from {filename}. "
                    "Ignoring color.")
        hdul.close()

        self.format_data(catalogue, spectrographs_data, fibermap["TARGETID"],
                         forests_by_targetid, reso_from_truth=reso_from_truth)
