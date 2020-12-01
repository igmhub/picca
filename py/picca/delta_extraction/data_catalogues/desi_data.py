"""This module defines the abstract class Data from which all
classes loading data must inherit
"""
import warnings
import numpy as np
import fitsio
import healpy

from picca.delta_extraction.data import Data
from picca.delta_extraction.errors import DataError, DataWarning
from picca.delta_extraction.userprint import userprint

from picca.delta_extraction.astronomical_objects.desi_forest import DesiForest

from picca.delta_extraction.quasar_catalogues.ztruth_catalogue import ZtruthCatalogue

defaults = {
    "delta lambda": 1.0,
    "lambda max": 5500.0,
    "lambda max rest frame": 1200.0,
    "lambda min": 3600.0,
    "lambda min rest frame": 1040.0,
}

class DesiData(Data):
    """Reads the spectra from Quickquasars and formats its data as a list of
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

    delta_lambda: float
    Variation of the wavelength (in Angs) between two pixels.

    in_dir: str
    Directory to spectra files.

    lambda_max: float
    Maximum wavelength (in Angs) to be considered in a forest.

    lambda_min: float
    Minimum wavelength (in Angs) to be considered in a forest.

    lambda_max_rest_frame: float
    As lambda_max but for rest-frame wavelength.

    lambda_min_rest_frame: float
    As lambda_min but for rest-frame wavelength.
    """
    def __init__(self, config):
        """Initialize class instance

        Arguments
        ---------
        config: configparser.SectionProxy
        Parsed options to initialize class
        """
        super().__init__(config)

        # load variables from config
        self.input_directory = None
        self.mode = None
        self._parse_config(config)

        # load z_truth catalogue
        catalogue = ZtruthCatalogue(config)

        # setup DesiForest class variables
        DesiForest.delta_lambda = self.delta_lambda
        DesiForest.lambda_max = self.lambda_max
        DesiForest.lambda_max_rest_frame = self.lambda_max_rest_frame
        DesiForest.lambda_min = self.lambda_min
        DesiForest.lambda_min_rest_frame = self.lambda_min_rest_frame

        # read data
        self.read_from_desi(catalogue)

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
        self.delta_lambda = config.get("delta lambda")
        if self.delta_lambda is None:
            self.delta_lambda = defaults.get("delta lambda")
        self.input_directory = config.get("input directory")
        if self.input_directory is None:
            raise DataError("Missing argument 'input directory' required by SdssData")
        self.lambda_max = config.get("lambda max")
        if self.lambda_max is None:
            self.lambda_max = defaults.get("lambda max")
        self.lambda_max_rest_frame = config.get("lambda max rest frame")
        if self.lambda_max_rest_frame is None:
            self.lambda_max_rest_frame = defaults.get("lambda max rest frame")
        self.lambda_min = config.get("lambda min")
        if self.lambda_min is None:
            self.lambda_min = defaults.get("lambda min")
        self.lambda_min_rest_frame = config.get("lambda min rest frame")
        if self.lambda_min_rest_frame is None:
            self.lambda_min_rest_frame = defaults.get("lambda min rest frame")


    def read_from_desi(self, catalogue):
        """Reads the spectra and formats its data as Forest instances.

        Arguments
        ---------
        catalogue: astropy.Table
        Table with the quasar catalogue
        """
        in_nside = int(self.input_directory.split('spectra-')[-1].replace('/', ''))

        ra = catalogue['RA'].data
        dec = catalogue['DEC'].data
        in_healpixs = healpy.ang2pix(in_nside, np.pi / 2. - dec, ra, nest=True)
        unique_in_healpixs = np.unique(in_healpixs)

        forests_by_targetid = {}
        for index, healpix in enumerate(unique_in_healpixs):
            filename = (f"{self.input_directory}/{healpix//100}/{healpix}/spectra"
                        f"-{in_nside}-{healpix}.fits")

            userprint(f"Read {index} of {len(unique_in_healpixs)}. "
                      f"num_data: {len(self.forests)}")
            try:
                hdul = fitsio.FITS(filename)
            except IOError:
                warnings.warn(f"Error reading pix {healpix}. Ignoring file",
                              DataWarning)
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
                    spec["WAVELENGTHL"] = hdul[f"{color}_WAVELENGTH"].read()
                    spec["FLUX"] = hdul[f"{color}_FLUX"].read()
                    spec["IVAR"] = (hdul[f"{color}_IVAR"].read() *
                                    (hdul[f"{color}_MASK"].read() == 0))
                    w = np.isnan(spec["FLUX"]) | np.isnan(spec["IVAR"])
                    for key in ["FLUX", "IVAR"]:
                        spec[key][w] = 0.
                    spectrographs_data[color] = spec
                except OSError:
                    warnings.warn(f"Error while reading {color} band from {filename}."
                                  "Ignoring color.",
                                  DataWarning)
            hdul.close()

            # Get the quasars in this healpix pixel
            select = np.where(in_healpixs == healpix)[0]

            # Loop over quasars in catalogue inside this healpixel
            for entry in catalogue[select]:
                # Find which row in tile contains this quasar
                # It should be there by construction
                targetid = entry["TARGETID"]
                w_t = np.where(targetid_spec == targetid)[0]
                if len(w_t) == 0:
                    warnings.warn(f"Error reading {targetid}. Ignoring file", DataWarning)
                    continue
                if len(w_t) > 1:
                    warnings.warn("Warning: more than one spectrum in this file "
                                  f"for {targetid}", DataWarning)
                else:
                    w_t = w_t[0]

                # Construct DesiForest instance
                # Fluxes from the different spectrographs will be coadded
                for spec in spectrographs_data.values():
                    ivar = spec['IV'][w_t].copy()
                    flux = spec['FL'][w_t].copy()

                    forest = DesiForest(**{"lambda": spec['WAVELENGTH'],
                                           "flux": flux,
                                           "ivar": ivar,
                                           "targetid": entry["TARGETID"],
                                           "ra": entry['RA'],
                                           "dec": entry['DEC'],
                                           "z": entry['Z'],
                                           "spectrograph": entry["SPECTROGRAPH"],
                                           "fiber": entry["FIBER"],})

                    if entry["TARGETID"] in forests_by_targetid:
                        forests_by_targetid[entry["TARGETID"]].coadd(forest)
                    else:
                        forests_by_targetid[entry["TARGETID"]] = forest

        self.forests = list(forests_by_targetid.values())
