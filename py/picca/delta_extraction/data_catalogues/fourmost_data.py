"""This module defines the class FourmostData to load 4MOST data"""
import logging
import os

import numpy as np
import fitsio

from picca.delta_extraction.astronomical_objects.forest import Forest
from picca.delta_extraction.data import Data, defaults, accepted_options
from picca.delta_extraction.errors import DataError
from picca.delta_extraction.quasar_catalogues.fourmost_quasar_catalogue import (
    FourmostQuasarCatalogue)
from picca.delta_extraction.quasar_catalogues.fourmost_quasar_catalogue import (
    accepted_options as accepted_options_quasar_catalogue)
from picca.delta_extraction.quasar_catalogues.fourmost_quasar_catalogue import (
    defaults as defaults_quasar_catalogue)
from picca.delta_extraction.utils import (update_accepted_options,
                                          update_default_options)

accepted_options = update_accepted_options(accepted_options,
                                           accepted_options_quasar_catalogue)
accepted_options = update_accepted_options(accepted_options, ["catalogue"])
# the wavelength grid is a property of the instrument, not a user choice
accepted_options = update_accepted_options(
    accepted_options, ["wave solution", "delta lambda rest frame"])

defaults = update_default_options(
    defaults, {
        "delta lambda": 0.8,
        "lambda min rest frame": 1040.0,
        "lambda max rest frame": 1200.0,
        "wave solution": "lin",
    })
defaults = update_default_options(defaults, defaults_quasar_catalogue)

# The observed-frame range is a property of the instrument, so it deliberately
# overrides the SDSS-era defaults of the parent class. 4MOST LRS covers
# 3700-9500 Angstrom, but the first ~200 Angstrom carry 10-50x the noise of
# the plateau and the red end degrades beyond 9300 Angstrom.
# These are assigned directly rather than through update_default_options: its
# `force_overwrite` argument suppresses the conflict error but never performs
# the assignment (the write sits in the `else` branch of the key-exists test),
# so the parent value would survive.
defaults["lambda min"] = 3950.0
defaults["lambda max"] = 9300.0

# The 4MOST LRS wavelength solution. It is NOT stored in the delivered files:
# there is no WAVE column and no spectral WCS. The grid is linear, runs from
# 3700 Angstrom to 9500 Angstrom inclusive, with a step of 0.25 Angstrom.
FOURMOST_LAMBDA_MIN = 3700.0
FOURMOST_LAMBDA_MAX = 9500.0
FOURMOST_PIXEL_STEP = 0.25
FOURMOST_NUM_PIXELS = 23201

FOURMOST_FLUX_COLUMN = "FLUX"
FOURMOST_ERR_COLUMN = "ERR"

# 4MOST stores flux in raw erg/(s cm2 Angstrom), so values are of order
# 1e-17 and inverse variances of order 1e34. The continuum fit starts from a
# mean continuum of order unity, and Minuit does not converge across that gap:
# fitting the delivery unscaled yields "minuit didn't converge" for essentially
# every forest. SDSS and DESI both store pre-scaled flux for the same reason.
# Scaling here brings the flux to order unity and is declared in 'flux units'.
FOURMOST_FLUX_SCALE = 1e17
FOURMOST_FLUX_UNITS = "10**-17 erg/(s cm2 Angstrom)"


def get_fourmost_wavelength_grid():
    """Build the 4MOST LRS wavelength grid

    Return
    ------
    lambda_: array of float
    Wavelength of each pixel, in Angstroms

    Raise
    -----
    DataError if the hardcoded constants are mutually inconsistent
    """
    lambda_ = np.arange(FOURMOST_LAMBDA_MIN,
                        FOURMOST_LAMBDA_MAX + FOURMOST_PIXEL_STEP / 2,
                        FOURMOST_PIXEL_STEP)
    if lambda_.size != FOURMOST_NUM_PIXELS:
        raise DataError(
            "Inconsistent 4MOST wavelength grid constants. Grid built from "
            f"lambda min {FOURMOST_LAMBDA_MIN}, lambda max "
            f"{FOURMOST_LAMBDA_MAX} and pixel step {FOURMOST_PIXEL_STEP} has "
            f"{lambda_.size} pixels, but FOURMOST_NUM_PIXELS is "
            f"{FOURMOST_NUM_PIXELS}")
    return lambda_


class FourmostData(Data):
    """Read 4MOST spectra and format them as a list of Forest instances.

    The 4MOST delivery is a single BINTABLE with one row per spectrum. FLUX
    and ERR are stored as float32[1, 23201] columns and the wavelength grid is
    not stored at all; it is hardcoded here and validated against the array
    length on every read.

    Only the BAO 3D analysis type is supported: the delivery carries neither a
    resolution matrix nor the exposure differences that PK 1D requires.

    Methods
    -------
    (see Data in py/picca/delta_extraction/data.py)
    __init__
    __parse_config
    read_data

    Attributes
    ----------
    (see Data in py/picca/delta_extraction/data.py)

    catalogue: astropy.table.Table
    The quasar catalogue, one row per object

    filename: str
    Path to the 4MOST spectra file

    logger: logging.Logger
    Logger object
    """
    def __init__(self, config):
        """Initialize class instance

        Arguments
        ---------
        config: configparser.SectionProxy
        Parsed options to initialize class

        Raise
        -----
        DataError upon invalid configuration
        """
        self.logger = logging.getLogger(__name__)

        self.filename = None
        self.__parse_config(config)

        config["flux units"] = FOURMOST_FLUX_UNITS
        super().__init__(config)

        if self.analysis_type != "BAO 3D":
            raise DataError(
                "Invalid argument 'analysis type' for FourmostData. Only "
                "'BAO 3D' is supported, as the 4MOST delivery contains no "
                f"resolution information. Found: '{self.analysis_type}'")

        self.catalogue = FourmostQuasarCatalogue(config).catalogue

        self.read_data()

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
        self.filename = config.get("catalogue")
        if self.filename is None:
            raise DataError(
                "Missing argument 'catalogue' required by FourmostData")

        wave_solution = config.get("wave solution")
        if wave_solution is not None and wave_solution != "lin":
            raise DataError(
                "Invalid argument 'wave solution' for FourmostData. The 4MOST "
                "grid is linear, so only 'lin' is supported. Found: "
                f"'{wave_solution}'")
        config["wave solution"] = "lin"

        # Data requires an input directory; everything lives in one file here,
        # so point it at the directory holding that file.
        if config.get("input directory") is None:
            config["input directory"] = os.path.dirname(
                os.path.abspath(self.filename))

    def read_data(self):
        """Read the spectra and format them as Forest instances.

        Raise
        -----
        DataError if the file cannot be read, if the arrays do not match the
        hardcoded grid, or if no forest survives
        """
        lambda_ = get_fourmost_wavelength_grid()
        log_lambda_all = np.log10(lambda_)

        if not os.path.isfile(self.filename):
            raise DataError(
                f"Error reading 4MOST data. File not found: {self.filename}")

        row_index = np.asarray(self.catalogue["ROW_INDEX"], dtype=np.int64)
        self.logger.progress(
            f"Reading {len(row_index)} spectra from {self.filename}")

        with fitsio.FITS(self.filename) as hdul:
            hdu = hdul["Joined"]
            for column in (FOURMOST_FLUX_COLUMN, FOURMOST_ERR_COLUMN):
                if column not in hdu.get_colnames():
                    raise DataError(
                        "Error reading 4MOST data. Missing column "
                        f"'{column}' in {self.filename}")
            rows = hdu[sorted(row_index.tolist())]

        flux_all = np.asarray(rows[FOURMOST_FLUX_COLUMN],
                              dtype=np.float64) * FOURMOST_FLUX_SCALE
        err_all = np.asarray(rows[FOURMOST_ERR_COLUMN],
                             dtype=np.float64) * FOURMOST_FLUX_SCALE
        # columns are stored as [1, num_pixels]; drop the leading axis
        flux_all = flux_all.reshape(flux_all.shape[0], -1)
        err_all = err_all.reshape(err_all.shape[0], -1)

        if flux_all.shape[1] != FOURMOST_NUM_PIXELS:
            raise DataError(
                "Error reading 4MOST data. Expected "
                f"{FOURMOST_NUM_PIXELS} pixels per spectrum, matching the "
                f"hardcoded grid {FOURMOST_LAMBDA_MIN}-{FOURMOST_LAMBDA_MAX} "
                f"Angstrom at {FOURMOST_PIXEL_STEP} Angstrom. Found "
                f"{flux_all.shape[1]} in {self.filename}")
        if err_all.shape != flux_all.shape:
            raise DataError(
                "Error reading 4MOST data. FLUX and ERR have different "
                f"shapes: {flux_all.shape} and {err_all.shape}")

        # the catalogue was sorted by object, the rows were read in file order
        order = np.argsort(np.argsort(row_index))
        flux_all = flux_all[order]
        err_all = err_all[order]

        forests = []
        num_empty = 0
        for index, row in enumerate(self.catalogue):
            flux = flux_all[index]
            err = err_all[index]

            good = np.isfinite(flux) & np.isfinite(err) & (err > 0.0)
            ivar = np.zeros_like(flux)
            ivar[good] = 1.0 / err[good]**2
            flux = np.where(good, flux, 0.0)

            if not np.any(ivar > 0.0):
                num_empty += 1
                continue

            forest = Forest(**{
                "log_lambda": log_lambda_all.copy(),
                "flux": flux,
                "ivar": ivar,
                "ra": row["RA"],
                "dec": row["DEC"],
                "z": row["Z"],
                "los_id": int(row["LOS_ID"]),
            })

            # Rebin onto Forest.log_lambda_grid. This also applies the
            # rest-frame trimming, so it must run before the forest is used.
            # It has to happen after the constructor, which initialises the
            # arrays rebin operates on.
            forest.rebin()

            forests.append(forest)

        if len(forests) == 0:
            raise DataError(
                "Error reading 4MOST data. No spectra were read from "
                f"{self.filename}")

        self.forests = forests
        if num_empty > 0:
            self.logger.progress(
                f"Discarded {num_empty} spectra with no valid pixel")
        self.logger.progress(f"Read {len(self.forests)} forests")
