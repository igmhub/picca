"""This module defines the class SdssData to read SDSS data"""
import os
import logging
import time

import numpy as np
import fitsio

from picca.delta_extraction.astronomical_objects.sdss_forest import SdssForest
from picca.delta_extraction.astronomical_objects.sdss_pk1d_forest import SdssPk1dForest
from picca.delta_extraction.data import Data, defaults, accepted_options
from picca.delta_extraction.errors import DataError
from picca.delta_extraction.quasar_catalogues.drq_catalogue import DrqCatalogue
from picca.delta_extraction.quasar_catalogues.drq_catalogue import defaults as defaults_drq
from picca.delta_extraction.quasar_catalogues.drq_catalogue import (
    accepted_options as accepted_options_quasar_catalogue)
from picca.delta_extraction.utils_pk1d import exp_diff, spectral_resolution
from picca.delta_extraction.utils import update_accepted_options, update_default_options

accepted_options = update_accepted_options(accepted_options, accepted_options_quasar_catalogue)
accepted_options = update_accepted_options(accepted_options, ["rebin", "mode"])
accepted_options = update_accepted_options(
    accepted_options,
    ["delta lambda", "delta log lambda", "delta lambda rest frame"],
    remove=True)

defaults = update_default_options(defaults, {
    "mode": "spplate",
    "rebin": 3,
})
defaults = update_default_options(defaults, defaults_drq)


class SdssData(Data):
    """Reads the spectra from SDSS and formats its data as a list of
    Forest instances.

    Methods
    -------
    (see Data in py/picca/delta_extraction/data.py)
    __init__
    __parse_config
    read_from_spec
    read_from_spplate

    Attributes
    ----------
    (see Data in py/picca/delta_extraction/data.py)

    logger: logging.Logger
    Logger object

    mode: str
    Reading mode. Currently supported reading modes are "spplate" and "spec"
    """

    def __init__(self, config):
        """Initialize class instance

        Arguments
        ---------
        config: configparser.SectionProxy
        Parsed options to initialize class

        Raise
        -----
        DataError if the selected reading mode is not supported
        """
        self.logger = logging.getLogger(__name__)

        # load variables from config
        self.mode = None
        self.__parse_config(config)

        super().__init__(config)

        # load DRQ Catalogue
        catalogue = DrqCatalogue(config).catalogue

        # read data
        if self.mode == "spplate":
            self.read_from_spplate(catalogue)
        elif self.mode == "spec":
            self.read_from_spec(catalogue)
        else:
            raise DataError(f"Error reading data in SdssData. Mode {self.mode} "
                            "is not supported.")

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
        self.mode = config.get("mode")
        if self.mode is None:
            raise DataError("Missing argument 'mode' required by SdssData")

        rebin = config.getint("rebin")
        if rebin is None:
            raise DataError("Missing argument 'rebin' required by SdssData")
        config["delta log lambda"] = str(rebin * 1e-4)
        del config["rebin"]

        config["wave solution"] = "log"

    def read_from_spec(self, catalogue):
        """Read the spectra and formats its data as Forest instances.

        Arguments
        ---------
        catalogue: astropy.table.Table
        Table with the DRQ catalogue
        """
        self.logger.progress(f"Reading {len(catalogue)} objects")

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
                self.logger.warning(f"Error reading {filename}. Ignoring file")
                continue
            self.logger.progress(f"Read {filename}")

            log_lambda = np.array(hdul[1]["loglam"][:], dtype=np.float64)
            flux = np.array(hdul[1]["flux"][:], dtype=np.float64)
            ivar = (np.array(hdul[1]["ivar"][:], dtype=np.float64) *
                    hdul[1]["and_mask"][:] == 0)

            if self.analysis_type == "BAO 3D":
                forest = SdssForest(
                    **{
                        "log_lambda": log_lambda,
                        "flux": flux,
                        "ivar": ivar,
                        "thingid": thingid,
                        "ra": row["RA"],
                        "dec": row["DEC"],
                        "z": row["Z"],
                        "plate": plate,
                        "mjd": mjd,
                        "fiberid": fiberid
                    })
            elif self.analysis_type == "PK 1D":
                # compute difference between exposure
                exposures_diff = exp_diff(hdul, log_lambda)
                # compute spectral resolution
                wdisp = hdul[1]["wdisp"][:]
                reso = spectral_resolution(wdisp, True, fiberid, log_lambda)

                forest = SdssPk1dForest(
                    **{
                        "log_lambda": log_lambda,
                        "flux": flux,
                        "ivar": ivar,
                        "thingid": thingid,
                        "ra": row["RA"],
                        "dec": row["DEC"],
                        "z": row["Z"],
                        "plate": plate,
                        "mjd": mjd,
                        "fiberid": fiberid,
                        "exposures_diff": exposures_diff,
                        "reso": reso,
                        "reso_pix": wdisp
                    })
            else:
                raise DataError(f"analysis_type = {self.analysis_type}")

            forest.rebin()
            if thingid in forests_by_thingid:
                forests_by_thingid[thingid].coadd(forest)
            else:
                forests_by_thingid[thingid] = forest

        self.forests = list(forests_by_thingid.values())

    def read_from_spplate(self, catalogue):
        """Read the spectra and formats its data as Forest instances.

        Arguments
        ---------
        catalogue: astropy.table.Table
        Table with the DRQ catalogue
        """
        grouped_catalogue = catalogue.group_by(["PLATE", "MJD"])
        num_objects = catalogue["THING_ID"].size
        self.logger.progress(f"reading {len(grouped_catalogue.groups)} plates")

        forests_by_thingid = {}
        num_read_total = 0
        for (plate, mjd), group in zip(grouped_catalogue.groups.keys,
                                       grouped_catalogue.groups):
            spplate = f"{self.input_directory}/{plate}/spPlate-{plate:04d}-{mjd}.fits"
            try:
                hdul = fitsio.FITS(spplate)
                header = hdul[0].read_header()
            except IOError:
                self.logger.warning(f"Error reading {spplate}. Ignoring file")
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
                if self.analysis_type == "BAO 3D":
                    forest = SdssForest(
                        **{
                            "log_lambda": log_lambda,
                            "flux": flux[array_index],
                            "ivar": ivar[array_index],
                            "thingid": thingid,
                            "ra": row["RA"],
                            "dec": row["DEC"],
                            "z": row["Z"],
                            "plate": plate,
                            "mjd": mjd,
                            "fiberid": fiberid
                        })
                elif self.analysis_type == "PK 1D":
                    # compute difference between exposure
                    exposures_diff = exp_diff(hdul, log_lambda)
                    # compute spectral resolution
                    wdisp = hdul[1]["wdisp"][:]
                    reso = spectral_resolution(wdisp, True, fiberid, log_lambda)

                    forest = SdssPk1dForest(
                        **{
                            "log_lambda": log_lambda,
                            "flux": flux[array_index],
                            "ivar": ivar[array_index],
                            "thingid": thingid,
                            "ra": row["RA"],
                            "dec": row["DEC"],
                            "z": row["Z"],
                            "plate": plate,
                            "mjd": mjd,
                            "fiberid": fiberid,
                            "exposures_diff": exposures_diff,
                            "reso": reso
                        })

                # rebin arrays
                # this needs to happen after all arrays are initialized by
                # Forest constructor
                forest.rebin()

                # keep the forest
                if thingid in forests_by_thingid:
                    existing_forest = forests_by_thingid[thingid]
                    existing_forest.coadd(forest)
                    forests_by_thingid[thingid] = existing_forest
                else:
                    forests_by_thingid[thingid] = forest
                self.logger.debug(
                    f"{thingid} read from file {spplate} and fiberid {fiberid}")

            num_read = len(group)
            num_read_total += num_read
            if num_read > 0.0:
                time_read = (time.time() - t0) / num_read
            else:
                time_read = np.nan
            self.logger.progress(
                f"read {num_read} from {os.path.basename(spplate)}"
                f" in {time_read:.3f} per spec. Progress: "
                f"{num_read_total} of {num_objects}")
            hdul.close()

        self.forests = list(forests_by_thingid.values())
