"""This module defines the class DesiQuasarCatalogue to read z_truth
files from DESI
"""
import logging

from astropy.table import Table
import fitsio
import numpy as np

from picca.delta_extraction.errors import QuasarCatalogueError
from picca.delta_extraction.quasar_catalogue import QuasarCatalogue, accepted_options

accepted_options = sorted(list(set(accepted_options + [
    "catalogue", "keep surveys"])))

defaults = {
    "keep surveys": "all"
}

accepted_surveys = ["sv1", "sv2", "sv3", "main", "special", "all"]

class DesiQuasarCatalogue(QuasarCatalogue):
    """Reads the z_truth catalogue from DESI

    Methods
    -------
    trim_catalogue (from QuasarCatalogue)
    __init__
    _parse_config
    filter_surveys
    read_catalogue

    Attributes
    ----------
    catalogue: astropy.table.Table (from QuasarCatalogue)
    The quasar catalogue

    max_num_spec: int or None (from QuasarCatalogue)
    Maximum number of spectra to read. None for no maximum

    z_max: float (from QuasarCatalogue)
    Maximum redshift. Quasars with redshifts higher than or equal to
    z_max will be discarded

    z_min: float (from QuasarCatalogue)
    Minimum redshift. Quasars with redshifts lower than z_min will be
    discarded

    filename: str
    Filename of the z_truth catalogue

    keep surveys: list
    Only keep the entries in the catalogue that have a "SURVEY" specified in
    this list. Ignored if "SURVEY" column is not present in the catalogue.
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
        self.filename = None
        self._parse_config(config)

        # read quasar catalogue
        self.read_catalogue()

        # if not all surveys are specified, then filter the catalogue
        self.filter_surveys()

        # if there is a maximum number of spectra, make sure they are selected
        # in a contiguous regions
        if self.max_num_spec is not None:
            super().trim_catalogue()

    def _parse_config(self, config):
        """Parse the configuration options

        Arguments
        ---------
        config: configparser.SectionProxy
        Parsed options to initialize class

        Raise
        -----
        QuasarCatalogueError upon missing required variables
        """
        keep_surveys = config.get("keep surveys")
        if keep_surveys is None:
            raise QuasarCatalogueError(
                "Missing argument 'keep surveys' required by DesiQuasarCatalogue")
        self.keep_surveys = keep_surveys.split()
        for survey in self.keep_surveys:
            if survey not in accepted_surveys:
                raise QuasarCatalogueError(
                    f"Unrecognised survey. Expected one of {accepted_surveys}. "
                    f"Found: {survey}")
        # if "all" is given, then make sure "sv1", "sv2", "sv3" and "main" are present
        if "all" in self.keep_surveys:
            for survey in ["sv1", "sv2", "sv3", "main"]:
                if survey not in self.keep_surveys:
                    self.keep_surveys.append(survey)

        self.filename = config.get("catalogue")
        if self.filename is None:
            raise QuasarCatalogueError("Missing argument 'catalogue' required "
                                       "by DesiQuasarCatalogue")

    def filter_surveys(self):
        """Filter all the objects in the catalogue not belonging to the specified
        surveys.
        """
        if 'SURVEY' in self.catalogue.colnames:
            mask = np.isin(self.catalogue["SURVEY"], self.keep_surveys)
            self.catalogue = self.catalogue[mask]
            self.logger.progress(f"and in selected surveys {self.keep_surveys}         : nb object in cat = {len(self.catalogue)}")

    def read_catalogue(self):
        """Read the z_truth catalogue

        Return
        ------
        catalogue: Astropy.table.Table
        Table with the catalogue
        """
        self.logger.progress(f'Reading catalogue from {self.filename}')
        catalogue = Table(fitsio.read(self.filename, ext=1))

        if 'TARGET_RA' in catalogue.colnames:
            catalogue.rename_column('TARGET_RA', 'RA')
            catalogue.rename_column('TARGET_DEC', 'DEC')

        keep_columns = ['RA', 'DEC', 'Z', 'TARGETID']
        if 'TILEID' in catalogue.colnames:
            keep_columns += ['TILEID', 'PETAL_LOC', 'FIBER']
        if 'SURVEY' in catalogue.colnames:
            keep_columns += ['SURVEY']
        if 'DESI_TARGET' in catalogue.colnames:
            keep_columns += ['DESI_TARGET']
        if 'SV1_DESI_TARGET' in catalogue.colnames:
            keep_columns += ['SV1_DESI_TARGET']
        if 'SV3_DESI_TARGET' in catalogue.colnames:
            keep_columns += ['SV3_DESI_TARGET']

        ## Sanity checks
        self.logger.progress('')
        w = np.ones(len(catalogue), dtype=bool)
        self.logger.progress(f"start                 : nb object in cat = {np.sum(w)}")

        ## Redshift range
        w &= catalogue['Z'] >= self.z_min
        self.logger.progress(f"and z >= {self.z_min}        : nb object in cat = {np.sum(w)}")
        w &= catalogue['Z'] < self.z_max
        self.logger.progress(f"and z < {self.z_max}         : nb object in cat = {np.sum(w)}")

        # Convert angles to radians
        np.radians(catalogue['RA'], out=catalogue['RA'])
        np.radians(catalogue['DEC'], out=catalogue['DEC'])

        catalogue.keep_columns(keep_columns)
        w = np.where(w)[0]
        self.catalogue = catalogue[w]
