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
from picca.delta_extraction.desi_data import DesiData, defaults
from picca.delta_extraction.errors import DataError
from picca.delta_extraction.quasar_catalogues.ztruth_catalogue import ZtruthCatalogue
from picca.delta_extraction.utils import ACCEPTED_BLINDING_STRATEGIES
from picca.delta_extraction.utils_pk1d import spectral_resolution_desi

defaults.update({
    "mode": "data",
})

class DesiHealpix(DesiData):
    """Reads the spectra from DESI using healpix mode and formats its data as a
    list of Forest instances.

    Should work for both data and mocks. This is specified using the 'mode'
    keyword. It is required to set the in_nside member.

    Methods
    -------
    filter_forests (from Data)
    set_blinding (from Data)
    __init__
    _parse_config
    read_data

    Attributes
    ----------
    analysis_type: str (from Data)
    Selected analysis type. Current options are "BAO 3D" or "PK 1D"

    forests: list of Forest (from Data)
    A list of Forest from which to compute the deltas.

    min_num_pix: int (from Data)
    Minimum number of pixels in a forest. Forests with less pixels will be dropped.

    blinding: str (from DesiData)
    A string specifying the chosen blinding strategies. Must be one of the
    accepted values in ACCEPTED_BLINDING_STRATEGIES

    input_directory: str (from DesiData)
    Directory to spectra files.

    in_nside: 64 or 16
    Nside used in the folder structure (64 for data and 16 for mocks)

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
        self.in_nside = None
        self._parse_config(config)

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


        # instance variables
        mode = config.get("mode")
        if mode is not None:
            raise DataError("Missing argument 'mode' required by DesiHealpix")
        if mode is not in ["data", "mocks"]:
            raise DataError("Unrecognized mode strategy. Accepted modes "
                            f"are 'data' and 'mocks'. Found {mode}")
        else:
            if mode == "data":
                self.in_nside = 64
            if mode == "mock"::
                self.in_nside = 16

    def read_data(self):
        """Read the spectra and formats its data as Forest instances.

        Method used to read healpix-based survey data.

        Return
        ------
        is_mock: bool
        True if mocks were loaded (i.e. there is a truth file in the folder) and
        Flase otherwise

        is_sv: bool
        True if all the read data belong to SV. False otherwise

        Raise
        -----
        DataError if the analysis type is PK 1D and resolution data is not present
        """
        healpix = [
            healpy.ang2pix(self.in_nside, np.pi / 2 - row["DEC"], row["RA"], nest=True)
            for row in self.catalogue
        ]
        self.catalogue["HEALPIX"] = healpix
        self.catalogue.sort("HEALPIX")
        grouped_catalogue = self.catalogue.group_by(["HEALPIX", "SURVEY"])

        forests_by_targetid = {}
        is_mock = True
        is_sv = True
        for (index,
             (healpix, survey)), group in zip(enumerate(grouped_catalogue.groups.keys),
                                    grouped_catalogue.groups):

            if 'main' in survey:
                is_sv = False

            input_directory = f'{self.input_directory}/{survey}/dark'
            filename = (
                f"{input_directory}/{healpix//100}/{healpix}/coadd-{survey}-"
                f"dark-{healpix}.fits")

            # the truth file is used to check if we are reading in mocks
            # in case we are, and we are computing pk1d, we also use them to load
            # the resolution matrix
            filename_truth = (
                f"{input_directory}/{healpix//100}/{healpix}/truth-{self.in_nside}-"
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

        return is_mock, is_sv
