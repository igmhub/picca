"""This module defines the class DesiQuasarCatalogue to read z_truth
files from DESI
"""
import logging

from astropy.table import Table
import fitsio
import healpy
import numpy as np

from picca.delta_extraction.errors import QuasarCatalogueError
from picca.delta_extraction.quasar_catalogue import QuasarCatalogue, accepted_options
from picca.delta_extraction.utils import update_accepted_options

accepted_options = update_accepted_options(
    accepted_options,
    ["catalogue", "in_nside", "keep surveys"])

defaults = {
    "keep surveys": "all",
    "in_nside": "64",
}

accepted_surveys = ["sv1", "sv2", "sv3", "main", "special", "all"]

class DesiQuasarCatalogue(QuasarCatalogue):
    """Reads the z_truth catalogue from DESI

    Methods
    -------
    (see QuasarCatalogue in py/picca/delta_extraction/quasar_catalogue.py)
    __init__
    __parse_config
    filter_surveys
    read_catalogue

    Attributes
    ----------
    (see QuasarCatalogue in py/picca/delta_extraction/quasar_catalogue.py)

    filename: str
    Filename of the z_truth catalogue

    in_nside: int
    Parameter in_nside to compute the healpix indexes

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
        self.in_nside = None
        self.keep_surveys = None
        self.__parse_config(config)

        # read quasar catalogue
        self.read_catalogue()

        # add healpix info
        self.add_healpix()

        # if there is a maximum number of spectra, make sure they are selected
        # in a contiguous regions
        if self.max_num_spec is not None:
            super().trim_catalogue()

    def __parse_config(self, config):
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

        self.in_nside = config.getint("in_nside")
        if self.in_nside is None:
            raise QuasarCatalogueError("Missing argument 'in_nside' required "
                                       "by DesiQuasarCatalogue")

    def add_healpix(self):
        """Add healpix information to the catalogue"""
        healpix = [
            healpy.ang2pix(self.in_nside,
                           np.pi / 2 - row["DEC"],
                           row["RA"],
                           nest=True) for row in self.catalogue
        ]
        self.catalogue["HEALPIX"] = healpix
        self.catalogue.sort("HEALPIX")

    def read_catalogue(self):
        """Read the DESI quasar catalogue"""
        self.logger.progress(f'Reading catalogue from {self.filename}')
        extnames = [ext.get_extname() for ext in fitsio.FITS(self.filename)]
        if "QSO_CAT" in extnames:
            extension = "QSO_CAT"
        elif "ZCATALOG" in extnames:
            extension = "ZCATALOG"
        else:
            raise QuasarCatalogueError(
                f"Could not find valid quasar catalog extension in fits file: {self.filename}")
        catalogue = Table(fitsio.read(self.filename, ext=extension))

        if 'TARGET_RA' in catalogue.colnames:
            catalogue.rename_column('TARGET_RA', 'RA')
            catalogue.rename_column('TARGET_DEC', 'DEC')

        keep_columns = ['RA', 'DEC', 'Z', 'TARGETID']
        if 'TILEID' in catalogue.colnames:
            keep_columns += ['TILEID', 'PETAL_LOC']
        if 'NIGHT' in catalogue.colnames:
            keep_columns += ['NIGHT']
        if 'LAST_NIGHT' in catalogue.colnames:
            keep_columns += ['LAST_NIGHT']
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

        # Filter all the objects in the catalogue not belonging to the specified
        # surveys.
        if 'SURVEY' in keep_columns:
            w &= np.isin(catalogue["SURVEY"], self.keep_surveys)
            self.logger.progress(f"and in selected surveys {self.keep_surveys}  "
                                 f"       : nb object in cat = {np.sum(w)}")

        # make sure we do not have an empty catalogue
        if np.sum(w) == 0:
            raise QuasarCatalogueError("Empty quasar catalogue. Revise filtering "
                                       "choices")

        # Convert angles to radians
        np.radians(catalogue['RA'], out=catalogue['RA'])
        np.radians(catalogue['DEC'], out=catalogue['DEC'])

        catalogue.keep_columns(keep_columns)
        w = np.where(w)[0]
        self.catalogue = catalogue[w]

        # add column SURVEY if not present
        # necessary for current mock catalogues
        if not "SURVEY" in self.catalogue.colnames:
            self.catalogue["SURVEY"] = np.ma.masked
