"""This module defines the class DesiData to load DESI data
"""
import os
import logging
import glob

import fitsio
import healpy
import numpy as np

from picca.delta_extraction.astronomical_objects.desi_forest import DesiForest
from picca.delta_extraction.astronomical_objects.desi_pk1d_forest import DesiPk1dForest
from picca.delta_extraction.astronomical_objects.forest import Forest
from picca.delta_extraction.data import Data, defaults
from picca.delta_extraction.errors import DataError
from picca.delta_extraction.quasar_catalogues.desi_quasar_catalogue import DesiQuasarCatalogue
from picca.delta_extraction.utils import ACCEPTED_BLINDING_STRATEGIES
from picca.delta_extraction.utils_pk1d import spectral_resolution_desi

defaults.update({
    "delta lambda": 1.0,  # TODO: update this value to the read from DESI files
    "lambda max": 5500.0,
    "lambda max rest frame": 1200.0,
    "lambda min": 3600.0,
    "lambda min rest frame": 1040.0,
    "blinding": "corr_yshift",
    # TODO: update this to "lin" when we are sure that the linear binning work
    "wave solution": "log",
    "rebin": 3,
})

class DesiData(Data):
    """Abstract class to read DESI data and format it as a list of
    Forest instances.

    Methods
    -------
    filter_forests (from Data)
    __init__
    _parse_config
    read_data
    set_blinding

    Attributes
    ----------
    analysis_type: str (from Data)
    Selected analysis type. Current options are "BAO 3D" or "PK 1D"

    forests: list of Forest (from Data)
    A list of Forest from which to compute the deltas.

    min_num_pix: int (from Data)
    Minimum number of pixels in a forest. Forests with less pixels will be dropped.

    blinding: str
    A string specifying the chosen blinding strategies. Must be one of the
    accepted values in ACCEPTED_BLINDING_STRATEGIES

    catalogue: astropy.table.Table
    The quasar catalogue

    input_directory: str
    Directory to spectra files.

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

        super().__init__(config)

        # load variables from config
        self.input_directory = None
        self.blinding = None
        self._parse_config(config)

        # load z_truth catalogue
        self.catalogue = DesiQuasarCatalogue(config).catalogue

        # read data
        is_mock, is_sv = self.read_data()

        # set blinding
        self.set_blinding(is_mock, is_sv)

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
        # setup Forest class variables
        wave_solution = config.get("wave solution")
        if wave_solution is None:
            raise DataError("Missing argument 'wave solution' required by DesiData")
        if wave_solution not in ["lin", "log"]:
            raise DataError("Unrecognised value for 'wave solution'. Expected either "
                            f"'lin' or 'lof'. Found {wave_solution}")
        Forest.wave_solution = wave_solution

        if Forest.wave_solution == "log":
            rebin = config.getint("rebin")
            if rebin is None:
                raise DataError("Missing argument 'rebin' required by DesiData when "
                                "'wave solution' is set to 'log'")
            Forest.delta_log_lambda = rebin * 1e-4

            lambda_max = config.getfloat("lambda max")
            if lambda_max is None:
                raise DataError("Missing argument 'lambda max' required by DesiData")
            Forest.log_lambda_max = np.log10(lambda_max)
            lambda_max_rest_frame = config.getfloat("lambda max rest frame")
            if lambda_max_rest_frame is None:
                raise DataError("Missing argument 'lambda max rest frame' required by DesiData")
            Forest.log_lambda_max_rest_frame = np.log10(lambda_max_rest_frame)
            lambda_min = config.getfloat("lambda min")
            if lambda_min is None:
                raise DataError("Missing argument 'lambda min' required by DesiData")
            Forest.log_lambda_min = np.log10(lambda_min)
            lambda_min_rest_frame = config.getfloat("lambda min rest frame")
            if lambda_min_rest_frame is None:
                raise DataError("Missing argument 'lambda min rest frame' required by DesiData")
            Forest.log_lambda_min_rest_frame = np.log10(lambda_min_rest_frame)

        elif Forest.wave_solution == "lin":
            Forest.delta_lambda = config.getfloat("delta lambda")
            if Forest.delta_lambda is None:
                raise DataError("Missing argument 'delta lambda' required by DesiData")
            Forest.lambda_max = config.getfloat("lambda max")
            if Forest.lambda_max is None:
                raise DataError("Missing argument 'lambda max' required by DesiData")
            Forest.lambda_max_rest_frame = config.getfloat("lambda max rest frame")
            if Forest.lambda_max_rest_frame is None:
                raise DataError("Missing argument 'lambda max rest frame' required by DesiData")
            Forest.lambda_min = config.getfloat("lambda min")
            if Forest.lambda_min is None:
                raise DataError("Missing argument 'lambda min' required by DesiData")
            Forest.lambda_min_rest_frame = config.getfloat("lambda min rest frame")
            if Forest.lambda_min_rest_frame is None:
                raise DataError("Missing argument 'lambda min rest frame' required by DesiData")
        else:
            raise DataError("Forest.wave_solution must be either "
                            "'log' or 'lin'")

        # instance variables
        self.blinding = config.get("blinding")
        if self.blinding is None:
            raise DataError("Missing argument 'blinding' required by DesiData")
        if self.blinding not in ACCEPTED_BLINDING_STRATEGIES:
            raise DataError("Unrecognized blinding strategy. Accepted strategies "
                            f"are {ACCEPTED_BLINDING_STRATEGIES}. Found {self.blinding}")

        self.input_directory = config.get("input directory")
        if self.input_directory is None:
            raise DataError(
                "Missing argument 'input directory' required by DesiData")

    # pylint: disable=no-self-use
    # this method should use self in child classes
    def read_data(self):
        """Read the spectra and formats its data as Forest instances.

        Method to be implemented by child classes.

        Return
        ------
        is_mock: bool
        True if mocks are read, False otherwise

        is_sv: bool
        True if all the read data belong to SV. False otherwise

        Raise
        -----
        DataError if no quasars were found
        """
<<<<<<< HEAD
        in_nside = 16
        is_mock = True

        healpix = [
            healpy.ang2pix(in_nside, np.pi / 2 - row["DEC"], row["RA"], nest=True)
            for row in catalogue
        ]
        catalogue["HEALPIX"] = healpix
        catalogue.sort("HEALPIX")

        if not "SURVEY" in catalogue.colnames:
            catalogue["SURVEY"]=np.ma.masked

        grouped_catalogue = catalogue.group_by(["HEALPIX", "SURVEY"])

        forests_by_targetid = {}

        for (index,
             (healpix, survey)), group in zip(enumerate(grouped_catalogue.groups.keys),
                                    grouped_catalogue.groups):
            input_directory = f'{self.input_directory}/'
            filename_truth = (
                f"{input_directory}/{healpix//100}/{healpix}/truth-{in_nside}-"
                f"{healpix}.fits")
            if not os.path.isfile(filename_truth):
                is_mock = False
                input_directory = f'{self.input_directory}/{survey}/dark'
                filename = (
                f"{input_directory}/{healpix//100}/{healpix}/coadd-{survey}-"
                f"dark-{healpix}.fits")
            else:
                filename = (
                f"{input_directory}/{healpix//100}/{healpix}/spectra-"
                f"{in_nside}-{healpix}.fits")

            # the truth file is used to check if we are reading in mocks
            # in case we are, and we are computing pk1d, we also use them to load
            # the resolution matrix


            self.logger.progress(
                f"Read {index} of {len(grouped_catalogue.groups.keys)}. "
                f"num_data: {len(forests_by_targetid)}")
            try:
                hdul = fitsio.FITS(filename)
            except IOError:
                self.logger.warning(f"Error reading pix {healpix}. Ignoring file")
                continue

            # Read targetid from fibermap to match to catalogue later
            fibermap = hdul['FIBERMAP'].read()
            targetid_spec = fibermap["TARGETID"]

            # First read all wavelength, flux, ivar, mask, and resolution
            # from this file
            spectrographs_data = {}
            colors = ["B", "R"]
            if "Z_FLUX" in hdul:
                colors.append("Z")
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
                            raise DataError(
                                "Error while reading {color} band from "
                                "{filename}. Analysis type is  'PK 1D', "
                                "but file does not contain HDU "
                                f"'{color}_RESOLUTION' ")
                    spectrographs_data[color] = spec
                except OSError:
                    self.logger.warning(
                        f"Error while reading {color} band from {filename}."
                        "Ignoring color.")
            hdul.close()

            # Loop over quasars in catalogue inside this healpixel
            for row in group:
                # Find which row in tile contains this quasar
                # It should be there by construction
                targetid = row["TARGETID"]
                w_t = np.where(targetid_spec == targetid)[0]
                if len(w_t) == 0:
                    self.logger.warning(
                        f"Error reading {targetid}. Ignoring object")
                    continue
                if len(w_t) > 1:
                    self.logger.warning(
                        "Warning: more than one spectrum in this file "
                        f"for {targetid}")
                else:
                    w_t = w_t[0]

                # Construct DesiForest instance
                # Fluxes from the different spectrographs will be coadded
                for spec in spectrographs_data.values():
                    ivar = spec['IVAR'][w_t].copy()
                    flux = spec['FLUX'][w_t].copy()

                    args = {
                        "flux": flux,
                        "ivar": ivar,
                        "targetid": targetid,
                        "ra": row['RA'],
                        "dec": row['DEC'],
                        "z": row['Z'],
                    }
                    if Forest.wave_solution == "log":
                        args["log_lambda"] = np.log10(spec['WAVELENGTH'])
                    elif Forest.wave_solution == "lin":
                        args["lambda"] = spec['WAVELENGTH']
                    else:
                        raise DataError("Forest.wave_solution must be either "
                                        "'log' or 'lin'")

                    if self.analysis_type == "BAO 3D":
                        forest = DesiForest(**args)
                    elif self.analysis_type == "PK 1D":
                        reso_sum = spec['RESO'][w_t].copy()
                        reso_in_km_per_s = spectral_resolution_desi(
                            reso_sum, spec['WAVELENGTH'])
                        exposures_diff = np.zeros(spec['WAVELENGTH'].shape)

                        args["exposures_diff"] = exposures_diff
                        args["reso"] = reso_in_km_per_s
                        forest = DesiPk1dForest(**args)
                    else:
                        raise DataError("Unkown analysis type. Expected 'BAO 3D'"
                                        f"or 'PK 1D'. Found '{self.analysis_type}'")

                    if targetid in forests_by_targetid:
                        forests_by_targetid[targetid].coadd(forest)
                    else:
                        forests_by_targetid[targetid] = forest

        self.forests = list(forests_by_targetid.values())

        return is_mock

    def read_from_tile(self, catalogue):
        """Read the spectra and formats its data as Forest instances.
=======
        raise DataError("Function 'read_data' was not overloaded by child class")
>>>>>>> de403740532fd59da45a945b3bd9575ddc69f447

    def set_blinding(self, is_mock, is_sv):
        """Set the blinding in Forest.

        Update the stored value if necessary.

        Attributes
        ----------
        is_mock: boolean
        True if reading mocks, False otherwise

        is_sv: boolean
        True if reading SV data only, False otherwise
        """
        # blinding checks
        if is_mock:
            if self.blinding != "none":
                self.logger.warning(f"Selected blinding, {self.blinding} is "
                                    "being ignored as mocks should not be "
                                    "blinded. 'none' blinding engaged")
                self.blinding = "none"
        if is_sv:
            if self.blinding != "none":
                self.logger.warning(f"Selected blinding, {self.blinding} is "
                                    "being ignored as SV data should not be "
                                    "blinded. 'none' blinding engaged")
                self.blinding = "none"
        # TODO: remove this when we are ready to unblind
        else:
            if self.blinding != "corr_yshift":
                self.logger.warning(f"Selected blinding, {self.blinding} is "
                                    "being ignored as data should be blinded. "
                                    "'corr_yshift' blinding engaged")
                self.blinding = "corr_yshift"

        # set blinding strategy
        Forest.blinding = self.blinding
