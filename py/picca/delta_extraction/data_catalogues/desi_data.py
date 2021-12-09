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
from picca.delta_extraction.quasar_catalogues.ztruth_catalogue import ZtruthCatalogue
from picca.delta_extraction.utils import ACCEPTED_BLINDING_STRATEGIES
from picca.delta_extraction.utils_pk1d import spectral_resolution_desi


defaults.update({
    "delta lambda": 1.0,  # TODO: update this value to the read from DESI files
    "lambda max": 5500.0,
    "lambda max rest frame": 1200.0,
    "lambda min": 3600.0,
    "lambda min rest frame": 1040.0,
    "mode": 'healpix',
    "blinding": "corr_yshift",
    # TODO: update this to "lin" when we are sure that the linear binning work
    "wave solution": "log",
    "rebin": 3,
    "use all": False,
    "use single nights": False,
})

class DesiData(Data):
    """Reads the spectra from Quickquasars and formats its data as a list of
    Forest instances.

    Methods
    -------
    filter_forests (from Data)
    __init__
    _parse_config
    read_from_desi
    read_from_minisv_desi

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

    input_directory: str
    Directory to spectra files.

    logger: logging.Logger
    Logger object

    mini_sv: bool
    Read data in Mini SV format.
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
        self.mini_sv = None
        self.use_all = None
        self.use_single_nights = None
        self._parse_config(config)

        # load z_truth catalogue
        catalogue = ZtruthCatalogue(config).catalogue

        # read data
        if self.mode == "healpix":
            is_mock = self.read_from_healpix(catalogue)
        elif self.mode == "tile":
            self.read_from_tile(catalogue)
            is_mock = True


        if is_mock:
            if self.blinding != "none":
                self.logger.warning(f"Selected blinding, {self.blinding} is "
                                    "being ignored as mocks should not be "
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
            Forest.delta_lambda = config.get("delta lambda")
            if Forest.delta_lambda is None:
                raise DataError("Missing argument 'delta lambda' required by DesiData")
            Forest.lambda_max = config.get("lambda max")
            if Forest.lambda_max is None:
                raise DataError("Missing argument 'lambda max' required by DesiData")
            Forest.lambda_max_rest_frame = config.get("lambda max rest frame")
            if Forest.lambda_max_rest_frame is None:
                raise DataError("Missing argument 'lambda max rest frame' required by DesiData")
            Forest.lambda_min = config.get("lambda min")
            if Forest.lambda_min is None:
                raise DataError("Missing argument 'lambda min' required by DesiData")
            Forest.lambda_min_rest_frame = config.get("lambda min rest frame")
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

        self.mode = config.get("mode")
        if self.mode is None:
            raise DataError("Missing argument 'mode' required by DesiData")
        if self.mode not in ["healpix", "tile"]:
            raise DataError("Invalid argument 'mode'. Expected: 'healpix' or 'tile',"
                            f"Found: {self.mode}")

        self.use_all = config.getboolean("use all")
        if self.use_all is None:
            raise DataError("Missing argument 'use all' required by DesiData")

        self.use_single_nights = config.getboolean("use single nights")
        if self.use_single_nights is None:
            raise DataError("Missing argument 'use single nights' required by DesiData")

    def read_from_healpix(self, catalogue):
        """Read the spectra and formats its data as Forest instances.

        Method used to read healpix-based survey data.

        Arguments
        ---------
        catalogue: astropy.Table
        Table with the quasar catalogue

        Return
        ------
        is_mock: bool
        True if mocks were loaded (i.e. there is a truth file in the folder) and
        Flase otherwise

        Raise
        -----
        DataError if the analysis type is PK 1D and resolution data is not present
        """
        in_nside = 64

        healpix = [
            healpy.ang2pix(in_nside, np.pi / 2 - row["DEC"], row["RA"], nest=True)
            for row in catalogue
        ]
        catalogue["HEALPIX"] = healpix
        catalogue.sort("HEALPIX")
        grouped_catalogue = catalogue.group_by(["HEALPIX", "SURVEY"])

        forests_by_targetid = {}
        is_mock = True
        for (index,
             (healpix, survey)), group in zip(enumerate(grouped_catalogue.groups.keys),
                                    grouped_catalogue.groups):

            input_directory = f'{self.input_directory}/{survey}/dark'
            filename = (
                f"{input_directory}/{healpix//100}/{healpix}/coadd-{survey}-"
                f"dark-{healpix}.fits")

            # the truth file is used to check if we are reading in mocks
            # in case we are, and we are computing pk1d, we also use them to load
            # the resolution matrix
            filename_truth = (
                f"{input_directory}/{healpix//100}/{healpix}/truth-{in_nside}-"
                f"{healpix}.fits")
            if not os.path.isfile(filename_truth):
                is_mock = False

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

        Method used to read tile-based survey data.

        Arguments
        ---------
        catalogue: astropy.Table
        Table with the quasar catalogue

        Raise
        -----
        DataError if the analysis type is PK 1D and resolution data is not present
        DataError if no quasars were found
        """
        forests_by_targetid = {}
        num_data = 0

        if self.use_single_nights or "cumulative" in self.input_directory:
            files_in = sorted(glob.glob(os.path.join(self.input_directory, "**/coadd-*.fits"),
                              recursive=True))

            if "cumulative" in self.input_directory:
                petal_tile_night = [
                    f"{entry['PETAL_LOC']}-{entry['TILEID']}-thru{entry['LAST_NIGHT']}"
                    for entry in catalog
                ]
            else:
                petal_tile_night = [
                    f"{entry['PETAL_LOC']}-{entry['TILEID']}-{entry['NIGHT']}"
                    for entry in catalogue
                ]
        else:
            if self.use_all:
                files_in = sorted(glob.glob(os.path.join(self.input_directory, "**/all/**/coadd-*.fits"),
                             recursive=True))
            else:
                files_in = sorted(glob.glob(os.path.join(self.input_directory, "**/deep/**/coadd-*.fits"),
                             recursive=True))
            petal_tile = [
                f"{entry['PETAL_LOC']}-{entry['TILEID']}"
                for entry in catalogue
            ]
        # this uniqueness check is to ensure each petal/tile/night combination
        # only appears once in the filelist
        petal_tile_night_unique = np.unique(petal_tile_night)

        filenames = []
        for f_in in files_in:
            for ptn in petal_tile_night_unique:
                if ptn in os.path.basename(f_in):
                    filenames.append(f_in)
        filenames = np.unique(filenames)

        for index, filename in enumerate(filenames):
            self.logger.progress("read tile {} of {}. ndata: {}".format(
                index, len(filenames), num_data))
            try:
                hdul = fitsio.FITS(filename)
            except IOError:
                self.logger.warning(f"Error reading file {filename}. Ignoring file")
                continue

            fibermap = hdul['FIBERMAP'].read()
            fibermap_colnames = hdul["FIBERMAP"].get_colnames()
            # pre-Andes
            if 'TARGET_RA' in fibermap_colnames:
                ra = fibermap['TARGET_RA']
                dec = fibermap['TARGET_DEC']
                tile_spec = fibermap['TILEID'][0]
                night_spec = fibermap['NIGHT'][0]
                colors = ['BRZ']
                if index == 0:
                    self.logger.warning(
                        "Reading all-band coadd as in minisv pre-Andes "
                        "dataset")
            # Andes
            elif 'RA_TARGET' in fibermap_colnames:
                ra = fibermap['RA_TARGET']
                dec = fibermap['DEC_TARGET']
                tile_spec = filename.split('-')[-2]
                night_spec = int(filename.split('-')[-1].split('.')[0])
                colors = ['B', 'R', 'Z']
                if index == 0:
                    self.logger.warning(
                        "Couldn't read the all band-coadd, trying "
                        "single band as introduced in Andes reduction")
            ra = np.radians(ra)
            dec = np.radians(dec)

            petal_spec = fibermap['PETAL_LOC'][0]

            targetid_spec = fibermap['TARGETID']

            spectrographs_data = {}
            for color in colors:
                try:
                    spec = {}
                    spec['WAVELENGTH'] = hdul[f'{color}_WAVELENGTH'].read()
                    spec['FLUX'] = hdul[f'{color}_FLUX'].read()
                    spec['IVAR'] = (hdul[f'{color}_IVAR'].read() *
                                    (hdul[f'{color}_MASK'].read() == 0))
                    if self.analysis_type == "PK 1D":
                        if f"{color}_RESOLUTION" in hdul:
                            spec["RESO"] = hdul[f"{color}_RESOLUTION"].read()
                        else:
                            raise DataError(
                                "Error while reading {color} band from "
                                "{filename}. Analysis type is  'PK 1D', "
                                "but file does not contain HDU "
                                f"'{color}_RESOLUTION' ")
                    w = np.isnan(spec['FLUX']) | np.isnan(spec['IVAR'])
                    for key in ['FLUX', 'IVAR']:
                        spec[key][w] = 0.
                    spectrographs_data[color] = spec
                except OSError:
                    self.logger.warning(
                        f"Error while reading {color} band from {filename}."
                        "Ignoring color.")

            hdul.close()

            select = ((catalogue['TILEID'] == tile_spec) &
                      (catalogue['PETAL_LOC'] == petal_spec) &
                      (catalogue['NIGHT'] == night_spec))
            self.logger.progress(
                f'This is tile {tile_spec}, petal {petal_spec}, night {night_spec}'
            )

            # Loop over quasars in catalog inside this tile-petal
            for entry in catalogue[select]:

                # Find which row in tile contains this quasar
                targetid = entry['TARGETID']
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

                for spec in spectrographs_data.values():
                    ivar = spec['IVAR'][w_t].copy()
                    flux = spec['FLUX'][w_t].copy()

                    rgs = {
                        "flux": flux,
                        "ivar": ivar,
                        "targetid": targetid,
                        "ra": entry['RA'],
                        "dec": entry['DEC'],
                        "z": entry['Z'],
                        "petal": entry["PETAL_LOC"],
                        "tile": entry["TILEID"],
                        "night": entry["NIGHT"],
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
                        reso_in_km_per_s = np.real(
                            spectral_resolution_desi(reso_sum,
                                                     spec['WAVELENGTH']))
                        exposures_diff = np.zeros(spec['log_lambda'].shape)

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

                num_data += 1
        self.logger.progress("Found {} quasars in input files".format(num_data))

        if num_data == 0:
            raise DataError("No Quasars found, stopping here")

        self.forests = list(forests_by_targetid.values())
