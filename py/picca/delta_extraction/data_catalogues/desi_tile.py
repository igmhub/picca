"""This module defines the class DesiData to load DESI data
"""
import os
import logging
import glob
import multiprocessing

import fitsio
import numpy as np

from picca.delta_extraction.astronomical_objects.desi_pk1d_forest import DesiPk1dForest
from picca.delta_extraction.data_catalogues.desi_data import (
    DesiData, DesiDataFileHandler, merge_new_forest, verify_exposures_shape)
from picca.delta_extraction.data_catalogues.desi_data import (  # pylint: disable=unused-import
    defaults, accepted_options)
from picca.delta_extraction.errors import DataError

accepted_options = sorted(list(set(accepted_options + ["num processors"])))


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

        if self.analysis_type == "PK 1D":
            DesiPk1dForest.update_class_variables()


    def read_data(self):
        """Read the spectra and formats its data as Forest instances.

        Return
        ------
        is_mock: bool
        False as DESI data are not mocks

        Raise
        -----
        DataError if the analysis type is PK 1D and resolution data is not present
        DataError if no quasars were found
        """
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
        forests_by_targetid = {}
        for file_in in files_in:
            for petal_tile_night in petal_tile_night_unique:
                if petal_tile_night in os.path.basename(file_in):
                    filenames.append(file_in)
        filenames = np.unique(filenames)

        if self.num_processors > 1:
            arguments = [(filename, self.catalogue) for filename in filenames]
            context = multiprocessing.get_context('fork')
            with context.Pool(processes=self.num_processors) as pool:
                imap_it = pool.imap(
                    DesiTileFileHandler(self.analysis_type,
                                        self.use_non_coadded_spectra,
                                        self.uniquify_night_targetid,
                                        self.keep_single_exposures,
                                        self.logger, self.input_directory),
                    arguments)
                for index, output_imap in enumerate(imap_it):
                    forests_by_targetid_aux = output_imap[0]
                    if self.use_non_coadded_spectra & self.keep_single_exposures:
                        # Change the dictionary key to prevent coadding.
                        # exposures on different files.
                        forests_by_targetid_aux= {f"{index}_{key}": items
                                                  for key, items in forests_by_targetid_aux.items()}
                    # Merge each dict to master forests_by_targetid
                    merge_new_forest(forests_by_targetid,
                                     forests_by_targetid_aux)
        else:
            num_data = 0
            reader = DesiTileFileHandler(self.analysis_type,
                                         self.use_non_coadded_spectra,
                                         self.uniquify_night_targetid,
                                         self.keep_single_exposures,
                                         self.logger, self.input_directory)
            for index, filename in enumerate(filenames):
                forests_by_targetid_aux, num_data_aux = reader(
                    (filename, self.catalogue))
                if self.use_non_coadded_spectra & self.keep_single_exposures:
                    # Change the dictionary key to prevent coadding.
                    # exposures on different files.
                    forests_by_targetid_aux= {f"{index}_{key}": items
                                              for key, items in forests_by_targetid_aux.items()}
                merge_new_forest(forests_by_targetid, forests_by_targetid_aux)
                num_data += num_data_aux
                self.logger.progress(
                    f"read tile {index} of {len(filename)}. ndata: {num_data}")

                self.logger.progress(f"Found {num_data} quasars in input files")

        if self.use_non_coadded_spectra & self.keep_single_exposures:
            verify_exposures_shape(forests_by_targetid)

        if len(forests_by_targetid) == 0:
            raise DataError("No Quasars found, stopping here")

        self.forests = list(forests_by_targetid.values())

        return False


# Class to read in parallel
# Seems lightweight to copy all these 3 arguments
class DesiTileFileHandler(DesiDataFileHandler):
    """File handler for class DesiTile

    Methods
    -------
    (see DesiDataFileHandler in py/picca/delta_extraction/data_catalogues/desi_data.py)
    read_file

    Attributes
    ----------
    (see DesiDataFileHandler in py/picca/delta_extraction/data_catalogues/desi_data.py)
    """

    def __init__(self,
                 analysis_type,
                 use_non_coadded_spectra,
                 uniquify_night_targetid,
                 keep_single_exposures,
                 logger,
                 input_directory):
        """Initialize file handler

        Arguments
        ---------
        analysis_type: str
        Selected analysis type. See class Data from py/picca/delta_extraction/data.py
        for details

        use_non_coadded_spectra: bool
        If True, load data from non-coadded spectra. Otherwise,
        load coadded data

        uniquify_night_targetid: bool
        If True, remove the quasars taken on the same night.

        keep_single_exposures: bool
        If True, the date loadded from non-coadded spectra are not coadded. 
        Otherwise, coadd the spectra here.

        logger: logging.Logger
        Logger object

        input_directory: str
        Directory where input data is stored.
        """
        self.input_directory = input_directory
        super().__init__(analysis_type,
                         use_non_coadded_spectra,
                         uniquify_night_targetid,
                         keep_single_exposures,
                         logger)

    def read_file(self, filename, catalogue):
        """Read the spectra and formats its data as Forest instances.

        Arguments
        ---------
        filename: str
        Name of the file to read

        catalogue: astropy.table.Table
        The quasar catalogue fragment associated with this file

        Returns:
        ---------
        forests_by_targetid: dict
        Dictionary were forests are stored.

        num_data: int
        The number of instances loaded

        Raise
        -----
        DataError if the analysis type is PK 1D and resolution data is not present
        """
        try:
            hdul = fitsio.FITS(filename)
        except IOError:
            self.logger.warning(f"Error reading file {filename}. Ignoring file")
            return {}, 0

        fibermap = hdul['FIBERMAP'].read()

        ra = fibermap['TARGET_RA']
        dec = fibermap['TARGET_DEC']
        tile_spec = fibermap['TILEID'][0]
        try:
            if "cumulative" in self.input_directory:
                night_spec = int(filename.split('thru')[-1].split('.')[0])
            else:
                night_spec = int(filename.split('-')[-1].split('.')[0])
        except ValueError:
            self.logger.warning(f"In file {filename}, error reading night. Ignoring file")
            return {}, 0

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
                            f"Error while reading {color} band from "
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

        if "cumulative" in self.input_directory:
            select = ((catalogue['TILEID'] == tile_spec) &
                      (catalogue['PETAL_LOC'] == petal_spec) &
                      (catalogue['LASTNIGHT'] == night_spec))
        else:
            select = ((catalogue['TILEID'] == tile_spec) &
                      (catalogue['PETAL_LOC'] == petal_spec) &
                      (catalogue['NIGHT'] == night_spec))
        self.logger.progress(
            f'This is tile {tile_spec}, petal {petal_spec}, night {night_spec}')

        forests_by_targetid, num_data = self.format_data(
            catalogue[select],
            spectrographs_data,
            fibermap["TARGETID"],
        )

        return forests_by_targetid, num_data
