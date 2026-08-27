"""This module defines the class FourmostData to load 4MOST data"""
import logging
import os

import numpy as np
import fitsio

from picca.delta_extraction.astronomical_objects.forest import Forest
from picca.delta_extraction.astronomical_objects.fourmost_pk1d_forest import (
    FourmostPk1dForest)
from picca.delta_extraction.data import Data, defaults, accepted_options
from picca.delta_extraction.errors import DataError
from picca.delta_extraction.quasar_catalogues.fourmost_quasar_catalogue import (
    FourmostQuasarCatalogue)
from picca.delta_extraction.quasar_catalogues.fourmost_quasar_catalogue import (
    accepted_options as accepted_options_quasar_catalogue)
from picca.delta_extraction.quasar_catalogues.fourmost_quasar_catalogue import (
    defaults as defaults_quasar_catalogue)
from picca.delta_extraction.utils import (SPEED_LIGHT, update_accepted_options,
                                          update_default_options)

# The 4MOST LRS wavelength solution. It is NOT stored in the delivered files:
# there is no WAVE column and no spectral WCS. The grid is linear, runs from
# 3700 Angstrom to 9500 Angstrom inclusive, with a step of 0.25 Angstrom.
FOURMOST_LAMBDA_MIN = 3700.0
FOURMOST_LAMBDA_MAX = 9500.0
FOURMOST_PIXEL_STEP = 0.25
FOURMOST_NUM_PIXELS = 23201

accepted_options = update_accepted_options(accepted_options,
                                           accepted_options_quasar_catalogue)
accepted_options = update_accepted_options(accepted_options, ["catalogue"])
# the wavelength grid is a property of the instrument, not a user choice
accepted_options = update_accepted_options(
    accepted_options, ["wave solution", "delta lambda rest frame"])

defaults = update_default_options(
    defaults, {
        "delta lambda": FOURMOST_PIXEL_STEP,
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


# ---------------------------------------------------------------------------
# 4MOST LRS spectral resolution
# ---------------------------------------------------------------------------
# The delivered files carry no resolution information of any kind: no RESO
# matrix, no wdisp-like column, no LSF keyword. R(lambda) is therefore
# hardcoded here from the published instrument description.
#
# Sources:
#   de Jong et al. 2019, The Messenger 175, 3, "4MOST: Project overview and
#     information for the First Call for Proposals", DOI 10.18727/0722-6691/5117
#     - Table 1: LRS passband 3700-9500 Angstrom, <R> = 6500
#     - Figure 2: spectral resolution of the three LRS channels
#   4MOST User Manual, VIS-MAN-4MOST-47110-9800-0001, Issue 2.00, 2019-09-26
#     - Section 6.1.2: arm passbands; "The resolution is defined as
#       lambda/Delta-lambda, where Delta-lambda is the width of a resolution
#       element (given by the FWHM of a custom fit function to an unresolved
#       line)"; "The spectra are sampled with ~3 pixels per resolution element"
#     - Figure 6: LRS spectral resolution, "Average over the slit" curves
#
# The tables below are the "Average over the slit" curves of User Manual
# Figure 6, digitised from the released PDF at 220 dpi by colour extraction
# and axis-gridline calibration. Anchors are spaced 10 nm; linear
# interpolation between them reproduces the digitised curves to better than
# 0.5 per cent. They are *design/model* curves including CCD flatness, charge
# diffusion, scattered light and thermal effects -- not on-sky measurements.
#
# R here is FWHM-based, per the User Manual definition above. Converting to a
# Gaussian sigma therefore divides by 2.3548, not by 1.
#
# Note the arms overlap. Because the delivered spectrum is a single merged
# product spanning 3700-9500 Angstrom with no discontinuity in its error
# array, the effective LSF in an overlap is a weighted mixture of the two arm
# LSFs. A mixture of two Gaussians has second moment
#     sigma_eff^2 = w * sigma_1^2 + (1 - w) * sigma_2^2
# so the blend below is linear in sigma^2, ramped across the overlap. The
# ramp is a stand-in for the true merge weights, which are a property of the
# 4MOST L1 pipeline and are not published; replace it if they become
# available.
#
# Cross-check: the mean of the merged curve over 3700-9500 Angstrom is
# R = 6077, against the <R> = 6500 quoted in Table 1 of de Jong et al. That
# 6.5 per cent gap is expected -- 6500 is a round design figure, not the mean
# of the Figure 6 curves.
# ---------------------------------------------------------------------------

# Arm passbands in Angstrom (User Manual section 6.1.2: blue 370-554 nm,
# green 524-721 nm, red 691-950 nm)
FOURMOST_ARM_SPANS = {
    "blue": (3700.0, 5540.0),
    "green": (5240.0, 7210.0),
    "red": (6910.0, 9500.0),
}

# (wavelength in Angstrom, resolving power R = lambda / FWHM) per arm
FOURMOST_ARM_RESOLUTION = {
    "blue": (
        (3700, 3829), (3800, 3994), (3900, 4168), (4000, 4344),
        (4100, 4516), (4200, 4644), (4300, 4718), (4400, 4792),
        (4500, 4867), (4600, 4941), (4700, 5061), (4800, 5185),
        (4900, 5309), (5000, 5435), (5100, 5561), (5200, 5719),
        (5300, 5868), (5400, 6017), (5500, 6171), (5540, 6221),
    ),
    "green": (
        (5240, 5166), (5300, 5251), (5400, 5395), (5500, 5543),
        (5600, 5686), (5700, 5835), (5800, 5979), (5900, 6092),
        (6000, 6211), (6100, 6323), (6200, 6438), (6300, 6557),
        (6400, 6667), (6500, 6784), (6600, 6901), (6700, 7011),
        (6800, 7135), (6900, 7268), (7000, 7402), (7100, 7536),
        (7200, 7669), (7210, 7673),
    ),
    "red": (
        (6910, 5412), (7000, 5490), (7100, 5583), (7200, 5674),
        (7300, 5763), (7400, 5853), (7500, 5947), (7600, 6038),
        (7700, 6131), (7800, 6226), (7900, 6324), (8000, 6417),
        (8100, 6511), (8200, 6604), (8300, 6678), (8400, 6757),
        (8500, 6832), (8600, 6906), (8700, 6985), (8800, 7055),
        (8900, 7139), (9000, 7226), (9100, 7308), (9200, 7392),
        (9300, 7477), (9400, 7561), (9500, 7638),
    ),
}

# FWHM -> Gaussian sigma
FOURMOST_FWHM_TO_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))

# Overlap regions, blended linearly in sigma^2 from the first arm to the second
FOURMOST_ARM_BLENDS = (
    ("blue", "green", 5240.0, 5540.0),
    ("green", "red", 6910.0, 7210.0),
)


def _arm_sigma(arm, lambda_):
    """Gaussian LSF sigma of one LRS arm, in Angstrom

    Arguments
    ---------
    arm: str
    One of "blue", "green", "red"

    lambda_: array of float
    Wavelength in Angstrom

    Return
    ------
    sigma: array of float
    LSF sigma in Angstrom. Outside the arm passband the edge value of R is
    held constant; callers must only use the result inside the passband.
    """
    table = np.array(FOURMOST_ARM_RESOLUTION[arm], dtype=np.float64)
    resolving_power = np.interp(lambda_, table[:, 0], table[:, 1])
    return lambda_ / (resolving_power * FOURMOST_FWHM_TO_SIGMA)


def get_fourmost_sigma(lambda_):
    """Gaussian LSF sigma of the merged 4MOST LRS spectrum, in Angstrom

    Single arms outside the overlaps; inside an overlap the two arm LSFs are
    combined as a mixture, which adds in sigma^2 (see the note above).

    Arguments
    ---------
    lambda_: array of float
    Wavelength in Angstrom

    Return
    ------
    sigma: array of float
    LSF sigma in Angstrom
    """
    lambda_ = np.atleast_1d(np.asarray(lambda_, dtype=np.float64))
    variance = np.full(lambda_.shape, np.nan)

    # single-arm regions
    variance[lambda_ < 5240.0] = _arm_sigma("blue", lambda_[lambda_ < 5240.0])**2
    mid = (lambda_ > 5540.0) & (lambda_ < 6910.0)
    variance[mid] = _arm_sigma("green", lambda_[mid])**2
    variance[lambda_ > 7210.0] = _arm_sigma("red", lambda_[lambda_ > 7210.0])**2

    # overlaps
    for first, second, start, stop in FOURMOST_ARM_BLENDS:
        within = (lambda_ >= start) & (lambda_ <= stop)
        if not np.any(within):
            continue
        weight = (stop - lambda_[within]) / (stop - start)
        variance[within] = (weight * _arm_sigma(first, lambda_[within])**2 +
                            (1.0 - weight) * _arm_sigma(second, lambda_[within])**2)

    return np.sqrt(variance)


def get_fourmost_resolution(lambda_, pixel_step=FOURMOST_PIXEL_STEP):
    """Resolution arrays for a Pk1dForest

    picca reduces these to the single scalar MEANRESO_PIX, which it multiplies
    by the pixel step of the delta file. reso_pix is therefore expressed in
    units of pixel_step, which must be the step of the delta file rather than
    the native 0.25 Angstrom step whenever the two differ.

    Arguments
    ---------
    lambda_: array of float
    Wavelength in Angstrom

    pixel_step: float
    Wavelength step of the delta file, in Angstrom

    Return
    ------
    reso: array of float
    LSF sigma in km/s, used by picca only for the --reso-max cut

    reso_pix: array of float
    LSF sigma in units of pixel_step, used for the Gaussian correction
    """
    sigma = get_fourmost_sigma(lambda_)
    reso = SPEED_LIGHT * sigma / np.asarray(lambda_, dtype=np.float64)
    return reso, sigma / pixel_step

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

        if self.analysis_type == "PK 1D":
            # The per-pixel DIFF, RESO and RESO_PIX columns are produced by
            # Forest.get_data, which only the BinTableHDU writer calls; the
            # ImageHDU writer emits LAMBDA, METADATA, DELTA, WEIGHT and CONT
            # only. Refuse rather than silently drop the resolution arrays.
            if self.save_format != "BinTableHDU":
                raise DataError(
                    "Invalid argument 'save format' for FourmostData with "
                    "'analysis type' = 'PK 1D'. The per-pixel DIFF, RESO and "
                    "RESO_PIX arrays are only written by the 'BinTableHDU' "
                    f"writer. Found: '{self.save_format}'")
            FourmostPk1dForest.update_class_variables()

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

        if self.analysis_type == "PK 1D":
            # Resolution depends only on wavelength, so it is the same array
            # for every forest and is built once here, on the native grid.
            # rebin() carries it onto the delta grid with ivar weighting.
            reso_all, reso_pix_all = get_fourmost_resolution(
                lambda_, pixel_step=FOURMOST_PIXEL_STEP)
            # The repeat rows of an object are nested cumulative stacks of the
            # same photons, not independent sub-exposures, so no half-difference
            # noise realisation can be formed. DIFF is therefore written as
            # zeros, exactly as DesiData does when non-coadded spectra are not
            # available. Downstream this means only
            # 'picca_Pk1D.py --noise-estimate pipeline' (or 'mean_pipeline') is
            # meaningful; the diff-based estimators would read these zeros as a
            # noise-free spectrum. picca_Pk1D defaults to 'mean_diff', so the
            # flag must be set explicitly.
            exposures_diff_all = np.zeros_like(lambda_)

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

            args = {
                "log_lambda": log_lambda_all.copy(),
                "flux": flux,
                "ivar": ivar,
                "ra": row["RA"],
                "dec": row["DEC"],
                "z": row["Z"],
                "los_id": int(row["LOS_ID"]),
            }

            if self.analysis_type == "BAO 3D":
                forest = Forest(**args)
            else:
                args["exposures_diff"] = exposures_diff_all.copy()
                args["reso"] = reso_all.copy()
                args["reso_pix"] = reso_pix_all.copy()
                forest = FourmostPk1dForest(**args)

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
