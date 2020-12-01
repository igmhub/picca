"""This module defines the abstract class Data from which all
classes loading data must inherit
"""
import os
import warnings
import time
import numpy as np
import fitsio

from picca.delta_extraction.data import Data
from picca.delta_extraction.errors import DataError, DataWarning
from picca.delta_extraction.userprint import userprint

from picca.delta_extraction.data_catalogues.drq_catalogue import DrqCatalogue

from picca.delta_extraction.astronomical_objects.sdss_forest import SdssForest


defaults = {
    "lambda max": 5500.0,
    "lambda max rest frame": 1200.0,
    "lambda min": 3600.0,
    "lambda min rest frame": 1040.0,
    "mode": "spplate",
    "rebin": 3,
}

class SdssData(Data):
    """Reads the spectra from SDSS and formats its data as a list of
    Forest instances.

    Methods
    -------
    get_forest_list (from Data)
    __init__
    _parse_config


    Attributes
    ----------
    forests: list of Forest (from Data)
    A list of Forest from which to compute the deltas.

    min_num_pix: int (from Data)
    Minimum number of rebinned pixels to accept a forest.

    delta_log_lambda: float
    Variation of the logarithm of the wavelength (in Angs) between two pixels.

    in_dir: str
    Directory to spectra files.

    log_lambda_max: float
    Logarithm of the maximum wavelength (in Angs) to be considered in a forest.

    log_lambda_min: float
    Logarithm of the minimum wavelength (in Angs) to be considered in a forest.

    log_lambda_max_rest_frame: float
    As log_lambda_max but for rest-frame wavelength.

    log_lambda_min_rest_frame: float
    As log_lambda_min but for rest-frame wavelength.

    mode: str
    Reading mode. Currently supported reading modes are "spplate" and "spec"
    """
    def __init__(self, config):
        """Initialize class instance

        Arguments
        ---------
        config: configparser.SectionProxy
        Parsed options to initialize class

        Raises
        ------
        DataError if the selected reading mode is not supported
        """
        super().__init__(config)

        # load variables from config
        self.input_directory = None
        self.mode = None
        self._parse_config(config)

        # load DRQ Catalogue
        catalogue = DrqCatalogue(config).catalogue

        # setup SdssForest class variables
        SdssForest.delta_log_lambda = self.delta_log_lambda
        SdssForest.log_lambda_max = self.log_lambda_max
        SdssForest.log_lambda_max_rest_frame = self.log_lambda_max_rest_frame
        SdssForest.log_lambda_min = self.log_lambda_min
        SdssForest.log_lambda_min_rest_frame = self.log_lambda_min_rest_frame

        # read data
        if self.mode == "spplate":
            self.read_from_spplate(catalogue)
        elif self.mode == "spec":
            self.read_from_spec(catalogue)
        else:
            raise DataError(f"Error reading data in SdssData. Mode {self.mode} "
                            "is not supported.")

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
        rebin = config.get("rebin")
        if rebin is None:
            rebin = defaults.get("rebin")
        self.delta_log_lambda = rebin * 1e-4
        self.input_directory = config.get("input directory")
        if self.input_directory is None:
            raise DataError("Missing argument 'input directory' required by SdssData")
        lambda_max = config.get("lambda max")
        if lambda_max is None:
            lambda_max = defaults.get("lambda max")
        self.log_lambda_max = np.log10(lambda_max)
        lambda_max_rest_frame = config.get("lambda max rest frame")
        if lambda_max_rest_frame is None:
            lambda_max_rest_frame = defaults.get("lambda max rest frame")
        self.log_lambda_max_rest_frame = np.log10(lambda_max_rest_frame)
        lambda_min = config.get("lambda min")
        if lambda_min is None:
            lambda_min = defaults.get("lambda min")
        self.log_lambda_min = np.log10(lambda_min)
        lambda_min_rest_frame = config.get("lambda min rest frame")
        if lambda_min_rest_frame is None:
            lambda_min_rest_frame = defaults.get("lambda min rest frame")
        self.log_lambda_min_rest_frame = np.log10(lambda_min_rest_frame)
        self.mode = config.get("mode")
        if self.input_directory is None:
            self.mode = defaults.get("mode")

    def read_from_spec(self, catalogue):
        """Reads the spectra and formats its data as Forest instances.

        Arguments
        ---------
        catalogue: list of DrqObject
        Table with the DRQ catalogue
        """
        userprint(f"Reading {len(catalogue)} objects")

        forests_by_thingid = {}
        #-- Loop over unique objects
        for row in catalogue:
            thingid = row['THING_ID']
            plate = row["PLATE"]
            mjd = row["MJD"]
            fiberid = row["FIBERID"]

            filename = (f"{self.input_directory}/{plate}/spec-{plate}-{mjd}-"
                        f"{fiberid:04d}.fits")
            try:
                hdul = fitsio.FITS(filename)
            except IOError:
                warnings.warn(f"error reading {filename}. Ignoring file",
                              DataWarning)
                continue
            userprint("Read {}".format(filename))

            log_lambda = hdul[1]["loglam"][:]
            flux = hdul[1]["flux"][:]
            ivar = hdul[1]["ivar"][:] * (hdul[1]["and_mask"][:] == 0)

            forest = SdssForest(**{"log_lambda": log_lambda,
                                   "flux": flux,
                                   "ivar": ivar,
                                   "thingid": thingid,
                                   "ra": row["RA"],
                                   "dec": row["DEC"],
                                   "z": row["Z"],
                                   "plate": plate,
                                   "mjd": mjd,
                                   "fiberid": fiberid})

            if thingid in forests_by_thingid:
                forests_by_thingid[thingid].coadd(forest)
            else:
                forests_by_thingid[thingid] = forest

        self.forests = list(forests_by_thingid.values())

    def read_from_spplate(self, catalogue):
        """Reads the spectra and formats its data as Forest instances.

        Arguments
        ---------
        catalogue: astropy.Table
        Table with the DRQ catalogue
        """
        grouped_catalogue = catalogue.group_by("PLATE", "MJD")
        userprint("reading {} plates".format(len(grouped_catalogue.groups)))

        forests_by_thingid = {}
        for (plate, mjd), group in zip(grouped_catalogue.groups.keys,
                                       grouped_catalogue.groups):
            spplate = f"{self.input_directory}/{plate}/spPlate-{plate:04d}-{mjd}.fits"

            try:
                hdul = fitsio.FITS(spplate)
                header = hdul[0].read_header()
            except IOError:
                warnings.warn(f"error reading {spplate}. Ignoring file",
                              DataWarning)
                continue

            t0 = time.time()

            coeff0 = header["COEFF0"]
            coeff1 = header["COEFF1"]

            flux = hdul[0].read()
            ivar = hdul[1].read() * (hdul[2].read() == 0)
            log_lambda = coeff0 + coeff1 * np.arange(flux.shape[1])

            # Loop over all objects inside this spPlate file
            # and create the SdssForest objects
            for row in group:
                thingid = row["THING_ID"]
                fiberid = row["FIBERID"]
                array_index = fiberid - 1
                forest = SdssForest(**{"log_lambda": log_lambda,
                                       "flux": flux[array_index],
                                       "ivar": ivar[array_index],
                                       "thingid": row["THING_ID"],
                                       "ra": row["RA"],
                                       "dec": row["DEC"],
                                       "z": row["Z"],
                                       "plate": row["PLATE"],
                                       "mjd": row["MJD"],
                                       "fiberid": row["FIBERID"]})
                if thingid in forests_by_thingid:
                    forests_by_thingid[thingid].coadd(forest)
                else:
                    forests_by_thingid[thingid] = forest
                userprint(f"{thingid} read from file {spplate} and fiberid {fiberid}\n")

            num_read = len(group)
            time_read = (time.time() - t0) / (num_read + num_read == 0)
            userprint(f"INFO: read {num_read} from {os.path.basename(spplate)}"
                      f" in {time_read:.3f} per spec. Progress: "
                      f"{len(forests_by_thingid)} of {len(catalogue)}")
            hdul.close()

        self.forests = list(forests_by_thingid.values())
