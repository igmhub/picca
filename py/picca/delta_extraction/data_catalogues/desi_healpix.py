"""This module defines the class DesiData to load DESI data
"""
import logging
import os
import multiprocessing

import fitsio
import numpy as np

from picca.delta_extraction.astronomical_objects.desi_pk1d_forest import DesiPk1dForest
from picca.delta_extraction.data_catalogues.desi_data import (
    DesiData, DesiDataFileHandler, merge_new_forest)
from picca.delta_extraction.data_catalogues.desi_data import (  # pylint: disable=unused-import
    defaults, accepted_options)
from picca.delta_extraction.errors import DataError


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

        super().__init__(config)

        if self.analysis_type == "PK 1D":
            DesiPk1dForest.update_class_variables()

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

        Raise
        -----
        DataError if no quasars were found
        """
        grouped_catalogue = self.catalogue.group_by(["HEALPIX", "SURVEY"])

        is_mock = False
        forests_by_targetid = {}

        arguments = []
        for group in grouped_catalogue.groups:
            healpix, survey = group["HEALPIX", "SURVEY"][0]

            filename, is_mock_aux = self.get_filename(survey, healpix)
            if is_mock_aux:
                is_mock = True

            arguments.append((filename, group))

        self.logger.info(f"reading data from {len(arguments)} files")

        if self.num_processors > 1:
            context = multiprocessing.get_context('fork')
            with context.Pool(processes=self.num_processors) as pool:
                imap_it = pool.imap(
                    DesiHealpixFileHandler(self.analysis_type,
                                           self.use_non_coadded_spectra,
                                           self.logger), arguments)
                for forests_by_targetid_aux, _ in imap_it:
                    # Merge each dict to master forests_by_targetid
                    merge_new_forest(forests_by_targetid,
                                     forests_by_targetid_aux)

        else:
            reader = DesiHealpixFileHandler(self.analysis_type,
                                            self.use_non_coadded_spectra,
                                            self.logger)
            num_data = 0
            for index, this_arg in enumerate(arguments):
                forests_by_targetid_aux, num_data_aux = reader(this_arg)
                merge_new_forest(forests_by_targetid, forests_by_targetid_aux)
                num_data += num_data_aux
                self.logger.progress(f"Read {index} of {len(arguments)}. "
                                     f"num_data: {num_data}")

        if len(forests_by_targetid) == 0:
            raise DataError("No quasars found, stopping here")
        self.forests = list(forests_by_targetid.values())

        return is_mock


# Class to read in parallel
# Seems lightweight to copy all these 3 arguments
class DesiHealpixFileHandler(DesiDataFileHandler):
    """File handler for class DesiHealpix

    Methods
    -------
    (see DesiDataFileHandler in py/picca/delta_extraction/data_catalogues/desi_data.py)
    read_file

    Attributes
    ----------
    (see DesiDataFileHandler in py/picca/delta_extraction/data_catalogues/desi_data.py)
    """

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
        # First read all wavelength, flux, ivar, mask, and resolution
        # from this file
        spectrographs_data = {}
        colors = ["B", "R"]
        if "Z_FLUX" in hdul:
            colors.append("Z")
        else:
            self.logger.warning(
                f"Missing Z band from {filename}. Ignoring color.")

        hdul_truth = None
        reso_from_truth = False
        if self.analysis_type == "PK 1D" and any(f"{c}_RESOLUTION" not in hdul for c in colors):
            self.logger.debug(
                    "no resolution in files, reading from truth files"
                )
            basename_truth = os.path.basename(filename).replace(
                            'spectra-', 'truth-')
            pathname_truth = os.path.dirname(filename)
            filename_truth = f"{pathname_truth}/{basename_truth}"
            if os.path.exists(filename_truth):
                hdul_truth = fitsio.FITS(filename_truth)
                reso_from_truth = True

        def _read_resolution(color):
            if f"{color}_RESOLUTION" in hdul:
                return hdul[f"{color}_RESOLUTION"].read()
            if hdul_truth is not None:
                return hdul_truth[f"{color}_RESOLUTION"].read()

            raise DataError(
                    f"Error while reading {color} band from "
                    f"{filename}. Analysis type is 'PK 1D', "
                    "but file does not contain HDU "
                    f"'{color}_RESOLUTION'")

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
                    spec["RESO"] = _read_resolution(color)

                spectrographs_data[color] = spec
            except OSError:
                self.logger.warning(
                    f"Error while reading {color} band from {filename}. "
                    "Ignoring color.")
        hdul.close()
        if hdul_truth is not None:
            hdul_truth.close()

        forests_by_targetid, num_data = self.format_data(
            catalogue,
            spectrographs_data,
            fibermap["TARGETID"],
            reso_from_truth=reso_from_truth)

        return forests_by_targetid, num_data
