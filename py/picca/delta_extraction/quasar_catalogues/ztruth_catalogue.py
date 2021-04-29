"""This module defines the class ZtruthCatalogue to read z_truth
files from DESI
"""
import logging

from astropy.table import Table
import numpy as np

from picca.delta_extraction.errors import QuasarCatalogueError
from picca.delta_extraction.quasar_catalogue import QuasarCatalogue

class ZtruthCatalogue(QuasarCatalogue):
    """Reads the z_truth catalogue from DESI

    Methods
    -------
    trim_catalogue (from QuasarCatalogue)
    __init__
    _parse_config
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

        # read DRQ Catalogue
        catalogue = self.read_catalogue()

        self.catalogue = catalogue

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
        self.filename = config.get("catalogue")
        if self.filename is None:
            raise QuasarCatalogueError("Missing argument 'catalogue' required "
                                       "by ZtruthCatalogue")

    def read_catalogue(self):
        """Read the z_truth catalogue

        Return
        ------
        catalogue: Astropy.table.Table
        Table with the catalogue
        """
        self.logger.progress('Reading catalogue from ', self.filename)
        catalogue = Table.read(self.filename, ext=1)

        keep_columns = ['RA', 'DEC', 'Z', 'TARGETID', 'FIBER', 'SPECTROGRAPH']

        ## Sanity checks
        self.logger.progress('')
        w = np.ones(len(catalogue), dtype=bool)
        self.logger.progress(f"start                 : nb object in cat = {np.sum(w)}")

        ## Redshift range
        w &= catalogue['Z'] >= self.z_min
        self.logger.progress(f"and z >= {self.z_min}        : nb object in cat = {np.sum(w)}")
        w &= catalogue['Z'] < self.z_max
        self.logger.progress(f"and z < {self.z_max}         : nb object in cat = {np.sum(w)}")

        catalogue.keep_columns(keep_columns)
        w = np.where(w)[0]
        catalogue = catalogue[w]

        return catalogue
