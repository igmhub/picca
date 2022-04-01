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
from picca.delta_extraction.data_catalogues.desi_data import DesiData, defaults, accepted_options
from picca.delta_extraction.errors import DataError
from picca.delta_extraction.utils_pk1d import spectral_resolution_desi, exp_diff_desi

accepted_options = sorted(
    list(
        set(accepted_options +
            ["use all", "use single nights", "use non-coadded spectra"])))

defaults.update({
    "use all": False,
    "use single nights": False,
    "use non-coadded spectra": False,
})


class DesiTile(DesiData):
    """Reads the spectra from DESI using tile mode and formats its data as a
    list of Forest instances.

    Methods
    -------
    filter_forests (from Data)
    set_blinding (from Data)
    __init__
    __parse_config
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

    catalogue: astropy.table.Table (from DesiData)
    The quasar catalogue

    input_directory: str (from DesiData)
    Directory to spectra files.

    use_all: bool
    If True, read using the all directory.

    use_single_nights: bool
    If True,  read using only nights specified within the cat

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

        # load variables from config
        self.use_all = None
        self.use_single_nights = None
        self.use_non_coadded_spectra = None

        self.__parse_config(config)
        #init of DesiData needs to come last, as it contains the actual data reading and thus needs all config
        super().__init__(config)


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
        self.use_all = config.getboolean("use all")
        if self.use_all is None:
            raise DataError("Missing argument 'use all' required by DesiTile")

        self.use_single_nights = config.getboolean("use single nights")
        if self.use_single_nights is None:
            raise DataError(
                "Missing argument 'use single nights' required by DesiTile")

        self.use_non_coadded_spectra = config.getboolean(
            "use non-coadded spectra")
        if self.use_non_coadded_spectra is None:
            raise DataError(
                "Missing argument 'use non-coadded spectra' required by DesiTile"
            )

    def read_data(self):
        """Read the spectra and formats its data as Forest instances.

        Return
        ------
        is_mock: bool
        False as DESI data are not mocks

        is_sv: bool
        True if all the read data belong to SV. False otherwise

        Raise
        -----
        DataError if the analysis type is PK 1D and resolution data is not present
        DataError if no quasars were found
        """
        if np.any((self.catalogue['TILEID'] < 60000)
                  & (self.catalogue['TILEID'] >= 1000)):
            is_sv = False
        else:
            is_sv = True

        forests_by_targetid = {}
        num_data = 0

        coadd_name = "spectra" if self.use_non_coadded_spectra else "coadd"

        if self.use_single_nights or "cumulative" in self.input_directory:
            files_in = sorted(
                glob.glob(os.path.join(self.input_directory,
                                       f"**/{coadd_name}-*.fits"),
                          recursive=True))

            if "cumulative" in self.input_directory:
                petal_tile_night = [
                    f"{entry['PETAL_LOC']}-{entry['TILEID']}-thru{entry['LAST_NIGHT']}"
                    for entry in self.catalogue
                ]
            else:
                petal_tile_night = [
                    f"{entry['PETAL_LOC']}-{entry['TILEID']}-{entry['NIGHT']}"
                    for entry in self.catalogue
                ]
        else:
            if self.use_all:
                files_in = sorted(
                    glob.glob(os.path.join(self.input_directory,
                                           f"**/all/**/{coadd_name}-*.fits"),
                              recursive=True))
            else:
                files_in = sorted(
                    glob.glob(os.path.join(self.input_directory,
                                           f"**/deep/**/{coadd_name}-*.fits"),
                              recursive=True))

            petal_tile = [
                f"{entry['PETAL_LOC']}-{entry['TILEID']}"
                for entry in self.catalogue
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
                self.logger.warning(
                    f"Error reading file {filename}. Ignoring file")
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
            no_scores_available = False

            spectrographs_data = {}
            for color in colors:
                try:
                    spec = {}
                    spec['WAVELENGTH'] = hdul[f'{color}_WAVELENGTH'].read()
                    spec['FLUX'] = hdul[f'{color}_FLUX'].read()
                    spec['IVAR'] = (hdul[f'{color}_IVAR'].read() *
                                    (hdul[f'{color}_MASK'].read() == 0))
                    if self.analysis_type == "PK 1D":
                        if self.use_non_coadded_spectra and "SCORES" in hdul:
                            spec['TEFF_LYA'] = 11.80090901380597 * hdul[
                                'SCORES'][f'TSNR2_LYA_{color}'].read()
                        else:
                            spec['TEFF_LYA'] = np.ones(spec["FLUX"].shape[0])
                            if self.use_non_coadded_spectra:
                                self.logger.warning(
                                    "SCORES are missing, Teff information (and thus DIFF) will be garbage"
                                )
                            no_scores_available = True

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

            select = ((self.catalogue['TILEID'] == tile_spec) &
                      (self.catalogue['PETAL_LOC'] == petal_spec) &
                      (self.catalogue['NIGHT'] == night_spec))
            self.logger.progress(
                f'This is tile {tile_spec}, petal {petal_spec}, night {night_spec}'
            )

            # Loop over quasars in catalog inside this tile-petal
            for entry in self.catalogue[select]:

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
                    if self.use_non_coadded_spectra:
                        ivar = np.atleast_2d(spec['IVAR'][w_t])
                        ivar_coadded_flux = np.atleast_2d(
                            ivar * spec['FLUX'][w_t]).sum(axis=0)
                        ivar = ivar.sum(axis=0)
                        flux = (ivar_coadded_flux / ivar)
                    else:
                        flux = spec['FLUX'][w_t].copy()
                        ivar = spec['IVAR'][w_t].copy()
                    args = {
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
                        if self.use_non_coadded_spectra and not no_scores_available:
                            exposures_diff = exp_diff_desi(spec, w_t)
                        else:
                            exposures_diff = np.zeros(spec['WAVELENGTH'].shape)
                        if len(spec['RESO'][w_t].shape) < 3:
                            reso_sum = spec['RESO'][w_t].copy()
                        else:
                            reso_sum = spec['RESO'][w_t].sum(axis=0)
                        reso_in_pix, reso_in_km_per_s = np.real(
                            spectral_resolution_desi(reso_sum,
                                                     spec['WAVELENGTH']))

                        args["exposures_diff"] = exposures_diff
                        args["reso"] = reso_in_km_per_s
                        args["resolution_matrix"] = reso_sum
                        args["reso_pix"] = reso_in_pix

                        forest = DesiPk1dForest(**args)
                    else:
                        raise DataError(
                            "Unkown analysis type. Expected 'BAO 3D'"
                            f"or 'PK 1D'. Found '{self.analysis_type}'")

                    # rebin arrays
                    # this needs to happen after all arrays are initialized by
                    # Forest constructor
                    forest.rebin()

                    # add the forest to list
                    if targetid in forests_by_targetid:
                        existing_forest = forests_by_targetid[targetid]
                        existing_forest.coadd(forest)
                        forests_by_targetid[targetid] = existing_forest
                    else:
                        forests_by_targetid[targetid] = forest

                num_data += 1
        self.logger.progress(
            "Found {} quasars in input files".format(num_data))

        if num_data == 0:
            raise DataError("No Quasars found, stopping here")

        self.forests = list(forests_by_targetid.values())

        return False, is_sv
