"""This module defines the class DesiData to load DESI data
"""
import os
import logging
import glob

import fitsio
import numpy as np

from picca.delta_extraction.data_catalogues.desi_data import DesiData
from picca.delta_extraction.data_catalogues.desi_data import(# pylint: disable=unused-import
    defaults, accepted_options)
from picca.delta_extraction.errors import DataError


class DesiTile(DesiData):
    """Reads the spectra from DESI using tile mode and formats its data as a
    list of Forest instances.

    Methods
    -------
    (see DesiData in py/picca/delta_extraction/data_catalogues/desi_data.py)
    __init__
    read_data

    Attributes
    ----------
    (see DesiData in py/picca/delta_extraction/data_catalogues/desi_data.py)

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
        if np.any((self.catalogue['TILEID'] < 60000) &
                  (self.catalogue['TILEID'] >= 1000)):
            is_sv = False
        else:
            is_sv = True

        forests_by_targetid = {}
        coadd_name = "spectra" if self.use_non_coadded_spectra else "coadd"

        files_in = sorted(
            glob.glob(os.path.join(self.input_directory,
                                   f"**/{coadd_name}-*.fits"),
                      recursive=True))

        if "cumulative" in self.input_directory:
            petal_tile_night = [
                f"{entry['PETAL_LOC']}-{entry['TILEID']}-thru{entry['LASTNIGHT']}"
                for entry in self.catalogue
            ]
        else:
            petal_tile_night = [
                f"{entry['PETAL_LOC']}-{entry['TILEID']}-{entry['NIGHT']}"
                for entry in self.catalogue
            ]

        # this uniqueness check is to ensure each petal/tile/night combination
        # only appears once in the filelist
        petal_tile_night_unique = np.unique(petal_tile_night)

        filenames = []
        for file_in in files_in:
            for petal_tile_night in petal_tile_night_unique:
                if petal_tile_night in os.path.basename(file_in):
                    filenames.append(file_in)
        filenames = np.unique(filenames)

        num_data = 0
        for index, filename in enumerate(filenames):
            self.logger.progress(
                f"read tile {index} of {len(filename)}. ndata: {num_data}")
            try:
                hdul = fitsio.FITS(filename)
            except IOError:
                self.logger.warning(
                    f"Error reading file {filename}. Ignoring file")
                continue

            fibermap = hdul['FIBERMAP'].read()

            ra = fibermap['TARGET_RA']
            dec = fibermap['TARGET_DEC']
            tile_spec = fibermap['TILEID'][0]
            if "cumulative" in self.input_directory:
                night_spec = int(filename.split('thru')[-1].split('.')[0])
            else:
                night_spec = int(filename.split('-')[-1].split('.')[0])

            colors = ['B', 'R', 'Z']
            ra = np.radians(ra)
            dec = np.radians(dec)

            petal_spec = fibermap['PETAL_LOC'][0]

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

            select = ((self.catalogue['TILEID'] == tile_spec) &
                      (self.catalogue['PETAL_LOC'] == petal_spec) &
                      (self.catalogue['NIGHT'] == night_spec))
            self.logger.progress(
                f'This is tile {tile_spec}, petal {petal_spec}, night {night_spec}'
            )

            num_data += self.format_data(self.catalogue[select],
                                        spectrographs_data,
                                        fibermap["TARGETID"],
                                        forests_by_targetid)
        self.logger.progress(f"Found {num_data} quasars in input files")

        if num_data == 0:
            raise DataError("No Quasars found, stopping here")

        self.forests = list(forests_by_targetid.values())

        return False, is_sv
