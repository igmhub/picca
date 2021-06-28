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
from picca.delta_extraction.data import Data
from picca.delta_extraction.errors import DataError
from picca.delta_extraction.quasar_catalogues.ztruth_catalogue import ZtruthCatalogue
from picca.delta_extraction.utils_pk1d import spectral_resolution_desi

defaults = {
    "delta lambda": 1.0,  # TODO: update this value to the read from DESI files
    "lambda max": 5500.0,
    "lambda max rest frame": 1200.0,
    "lambda min": 3600.0,
    "lambda min rest frame": 1040.0,
    "mini SV": False,
}

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

        # setup Forest class variables
        Forest.wave_solution = "lin"

        # load variables from config
        self.input_directory = None
        self.mini_sv = None
        self._parse_config(config)

        # load z_truth catalogue
        catalogue = ZtruthCatalogue(config).catalogue

        # read data
        if self.mini_sv:
            self.read_from_minisv_desi(catalogue)
        else:
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
        # Forest class variables
        Forest.delta_lambda = config.get("delta lambda")
        if Forest.delta_lambda is None:
            Forest.delta_lambda = defaults.get("delta lambda")
        else:
            Forest.delta_lambda = config.getfloat("delta lambda")
        Forest.lambda_max = config.get("lambda max")
        if Forest.lambda_max is None:
            Forest.lambda_max = defaults.get("lambda max")
        Forest.lambda_max_rest_frame = config.get("lambda max rest frame")
        if Forest.lambda_max_rest_frame is None:
            Forest.lambda_max_rest_frame = defaults.get("lambda max rest frame")
        Forest.lambda_min = config.get("lambda min")
        if Forest.lambda_min is None:
            Forest.lambda_min = defaults.get("lambda min")
        Forest.lambda_min_rest_frame = config.get("lambda min rest frame")
        if Forest.lambda_min_rest_frame is None:
            Forest.lambda_min_rest_frame = defaults.get("lambda min rest frame")

        # instance variables
        self.input_directory = config.get("input directory")
        if self.input_directory is None:
            raise DataError(
                "Missing argument 'input directory' required by SdssData")

        self.mini_sv = config.getboolean("mini SV")
        if self.mini_sv is None:
            self.mini_sv = defaults.get("mini SV")

    def read_from_desi(self, catalogue):
        """Read the spectra and formats its data as Forest instances.

        Arguments
        ---------
        catalogue: astropy.Table
        Table with the quasar catalogue

        Raise
        -----
        DataError if the analysis type is PK 1D and resolution data is not present
        """
        in_nside = int(
            self.input_directory.split('spectra-')[-1].replace('/', ''))

        healpix = [
            healpy.ang2pix(16, np.pi / 2 - row["DEC"], row["RA"])
            for row in catalogue
        ]
        catalogue["healpix"] = healpix
        catalogue.sort("healpix")
        grouped_catalogue = catalogue.group_by("healpix")

        forests_by_targetid = {}
        for (index,
             healpix), group in zip(enumerate(grouped_catalogue.groups.keys),
                                    grouped_catalogue.groups):
            filename = (
                f"{self.input_directory}/{healpix//100}/{healpix}/spectra"
                f"-{in_nside}-{healpix}.fits")

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
                    ivar = spec['IV'][w_t].copy()
                    flux = spec['FL'][w_t].copy()

                    if self.analysis_type == "BAO 3D":
                        forest = DesiForest(
                            **{
                                "lambda": spec['WAVELENGTH'],
                                "flux": flux,
                                "ivar": ivar,
                                "targetid": targetid,
                                "ra": row['RA'],
                                "dec": row['DEC'],
                                "z": row['Z'],
                                "petal": row["PETAL_LOC"],
                                "tile": row["TILEID"],
                                "night": row["NIGHT"]
                            })
                    elif self.analysis_type == "PK 1D":
                        reso_sum = spec['RESO'][w_t].copy()
                        reso_in_km_per_s = spectral_resolution_desi(
                            reso_sum, spec['WAVELENGTH'])
                        exposures_diff = np.zeros(spec['WAVELENGTH'].shape)

                        forest = DesiPk1dForest(
                            **{
                                "lambda": spec['WAVELENGTH'],
                                "flux": flux,
                                "ivar": ivar,
                                "targetid": targetid,
                                "ra": row['RA'],
                                "dec": row['DEC'],
                                "z": row['Z'],
                                "petal": row["PETAL_LOC"],
                                "tile": row["TILEID"],
                                "night": row["NIGHT"],
                                "exposures_diff": exposures_diff,
                                "reso": reso_in_km_per_s
                            })

                    if targetid in forests_by_targetid:
                        forests_by_targetid[targetid].coadd(forest)
                    else:
                        forests_by_targetid[targetid] = forest

        self.forests = list(forests_by_targetid.values())

    def read_from_minisv_desi(self, catalogue):
        """Read the spectra and formats its data as Forest instances.
        Unlike the read_from_desi routine, this orders things by tile/petal
        Routine used to treat the DESI mini-SV data.

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

        nightcol='LAST_NIGHT' if 'LAST_NIGHT' in catalogue.colnames else 'NIGHT'
        thrustr='thru' if 'cumulative' in self.input_directory else ''

        files_in = glob.glob(os.path.join(self.input_directory,
                                          "**/coadd-*.fits"),
                             recursive=True)
        petal_tile_night = [
            f"{entry['PETAL_LOC']}-{entry['TILEID']}-{thrustr}{entry[nightcol]}"
            for entry in catalogue
        ]
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
            # SV releases
            ra_spec = fibermap['TARGET_RA']
            dec_spec = fibermap['TARGET_DEC']

            tile_spec = fibermap['TILEID'][0]
            night_spec = fibermap[nightcol][0]
            try:
                hdul['BRZ_WAVELENGTH']
            except OSError:
                colors = ['B','R','Z']
                if index == 0:
                    self.logger.warning(
                        "Reading single band coadds, picca will coadd the bands,"
                        "as typical since SV")
            else:
                colors = ['BRZ']
                if index == 0:
                    self.logger.warning(
                        "Reading all-band coadd as in minisv pre-Andes "
                        "dataset")
            ra_spec = np.radians(ra_spec)
            dec_spec = np.radians(dec_spec)

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
                      (catalogue[nightcol] == night_spec))
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

                    if self.analysis_type == "BAO 3D":
                        forest = DesiForest(
                            **{
                                "lambda": spec['WAVELENGTH'],
                                "flux": flux,
                                "ivar": ivar,
                                "targetid": targetid,
                                "ra": entry["RA"],
                                "dec": entry["DEC"],
                                "z": entry['Z'],
                                "petal": entry["PETAL_LOC"],
                                "tile": entry["TILEID"],
                                "night": entry[nightcol]
                            })
                    elif self.analysis_type == "PK 1D":
                        reso_sum = spec['RESO'][w_t].copy()
                        reso_in_km_per_s, reso_in_AA = spectral_resolution_desi(
                            reso_sum,spec['WAVELENGTH'])
                        exposures_diff = np.zeros(spec['WAVELENGTH'].shape)

                        forest = DesiPk1dForest(
                            **{
                                "lambda": spec['WAVELENGTH'],
                                "flux": flux,
                                "ivar": ivar,
                                "targetid": targetid,
                                "ra": entry['RA'],
                                "dec": entry['DEC'],
                                "z": entry['Z'],
                                "petal": entry["PETAL_LOC"],
                                "tile": entry["TILEID"],
                                "night": entry[nightcol],
                                "exposures_diff": exposures_diff,
                                "reso": np.real(reso_in_km_per_s),
                                "reso_AA": np.real(reso_in_AA),
                                "reso_matrix": reso_sum
                            })

                    if targetid in forests_by_targetid:
                        forests_by_targetid[targetid].coadd(forest)
                    else:
                        forests_by_targetid[targetid] = forest

                num_data += 1
        self.logger.progress("Found {} quasars in input files".format(num_data))

        if num_data == 0:
            raise DataError("No Quasars found, stopping here")
        #breakpoint()
        self.forests = list(forests_by_targetid.values())
