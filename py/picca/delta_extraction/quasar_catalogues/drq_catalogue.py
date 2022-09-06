"""This module defines the class DrqCatalogue to read SDSS
DRQX Catalogues
"""
import glob
import logging

from astropy.table import Table, join
import numpy as np

from picca.delta_extraction.errors import QuasarCatalogueError
from picca.delta_extraction.quasar_catalogue import QuasarCatalogue, accepted_options

from picca.delta_extraction.utils import update_accepted_options

accepted_options = update_accepted_options(
    accepted_options,
    ["best obs", "BI max", "drq catalogue", "input directory", "keep BAL", "spAll"])

defaults = {
    "best obs": False,
    "keep BAL": False,
}

class DrqCatalogue(QuasarCatalogue):
    """Read the DRQ quasar catalogue SDSS

    Methods
    -------
    (see QuasarCatalogue in py/picca/delta_extraction/quasar_catalogue.py)
    __init__
    __parse_config
    read_drq
    read_spall

    Attributes
    ----------
    (see QuasarCatalogue in py/picca/delta_extraction/quasar_catalogue.py)

    best_obs: bool
    If True, reads only the best observation for objects with repeated
    observations

    bi_max: float or None
    Maximum value allowed for the Balnicity Index to keep the quasar.
    None for no maximum

    drq_filename: str
    Filename of the DRQ catalogue

    keep_bal: bool
    If False, remove the quasars flagged as having a Broad Absorption
    Line. Ignored if bi_max is not None

    logger: logging.Logger
    Logger object

    spall: str
    Path to the spAll file required for multiple observations
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
        self.best_obs = None
        self.bi_max = None
        self.drq_filename = None
        self.keep_bal = None
        self.spall = None
        self.__parse_config(config)

        # read DRQ Catalogue
        catalogue = self.read_drq()

        # if using multiple observations load the information from spAll file
        if not self.best_obs:
            catalogue = self.read_spall(catalogue)

        self.catalogue = catalogue

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
        self.best_obs = config.getboolean("best obs")
        if self.best_obs is None:
            raise QuasarCatalogueError("Missing argument 'best obs' "
                                       "required by DrqCatalogue")
        self.bi_max = config.getfloat("BI max")
        self.drq_filename = config.get("drq catalogue")
        if self.drq_filename is None:
            raise QuasarCatalogueError("Missing argument 'drq catalogue' "
                                       "required by DrqCatalogue")
        self.keep_bal = config.getboolean("keep BAL")
        if self.keep_bal is None:
            raise QuasarCatalogueError("Missing argument 'keep BAL' "
                                       "required by DrqCatalogue")

        if self.best_obs:
            self.spall = None
        else:
            self.spall = config.get("spAll")
            if self.spall is None:
                self.logger.warning("Missing argument 'spAll' required by "
                                    "DrqCatalogue. Looking for spAll in input "
                                    "directory...")

                if config.get("input directory") is None:
                    self.logger.error("'spAll' file not found. If you didn't "
                                      "want to load the spAll file you should "
                                      "pass the option 'best obs = True'. "
                                      "Quiting...")
                    raise QuasarCatalogueError("Missing argument 'spAll' "
                                               "required by DrqCatalogue")
                folder = config.get("input directory")
                folder = folder.replace("spectra",
                                        "").replace("lite",
                                                    "").replace("full", "")
                filenames = glob.glob(f"{folder}/spAll-*.fits")
                if len(filenames) > 1:
                    self.logger.error("Found multiple 'spAll' files. Quiting...")
                    for filename in filenames:
                        self.logger.error(f"found: {filename}")
                    raise QuasarCatalogueError("Missing argument 'spAll' "
                                               "required by DrqCatalogue")
                if len(filenames) == 0:
                    self.logger.error("'spAll' file not found. If you didn't "
                                      "want to load the spAll file you should "
                                      "pass the option 'best obs = True'. "
                                      "Quiting...")
                    raise QuasarCatalogueError("Missing argument 'spAll' "
                                               "required by DrqCatalogue")
                self.spall = filenames[0]
                self.logger.ok_warning("'spAll' file found. Contining with "
                                       "normal execution")

    def read_drq(self):
        """Read the DRQ Catalogue

        Return
        ------
        catalogue: Astropy.table.Table
        Table with the DRQ catalogue

        Raise
        -----
        QuasarCatalogueError when no valid column for redshift is found when
        reading the catalogue
        QuasarCatalogue when 'BI max' is passed but HDU does not contain BI_CIV
        field
        """
        self.logger.progress(f"Reading DRQ catalogue from {self.drq_filename}")
        catalogue = Table.read(self.drq_filename, hdu="CATALOG")

        keep_columns = ['RA', 'DEC', 'Z', 'THING_ID', 'PLATE', 'MJD', 'FIBERID']

        # Redshift
        if 'Z' not in catalogue.colnames:
            if 'Z_VI' in catalogue.colnames:
                catalogue.rename_column('Z_VI', 'Z')
                self.logger.progress(
                    "Z not found (new DRQ >= DRQ14 style), using Z_VI (DRQ <= DRQ12)"
                )
            else:
                raise QuasarCatalogueError("Error in reading DRQ Catalogue. No "
                                           "valid column for redshift found in "
                                           f"{self.drq_filename}")

        ## Sanity checks
        w = np.ones(len(catalogue), dtype=bool)
        self.logger.progress(f"start                 : nb object in cat = {np.sum(w)}")
        w &= catalogue["THING_ID"] > 0
        self.logger.progress(f"and THING_ID > 0      : nb object in cat = {np.sum(w)}")
        w &= catalogue['RA'] != catalogue['DEC']
        self.logger.progress(f"and ra != dec         : nb object in cat = {np.sum(w)}")
        w &= catalogue['RA'] != 0.
        self.logger.progress(f"and ra != 0.          : nb object in cat = {np.sum(w)}")
        w &= catalogue['DEC'] != 0.
        self.logger.progress(f"and dec != 0.         : nb object in cat = {np.sum(w)}")

        ## Redshift range
        w &= catalogue['Z'] >= self.z_min
        self.logger.progress(f"and z >= {self.z_min}        : nb object in cat = {np.sum(w)}")
        w &= catalogue['Z'] < self.z_max
        self.logger.progress(f"and z < {self.z_max}         : nb object in cat = {np.sum(w)}")

        ## BAL visual
        if not self.keep_bal and self.bi_max is None:
            if 'BAL_FLAG_VI' in catalogue.colnames:
                self.bal_flag = catalogue['BAL_FLAG_VI']
                w &= self.bal_flag == 0
                self.logger.progress(
                    f"and BAL_FLAG_VI == 0  : nb object in cat = {np.sum(w)}")
                keep_columns += ['BAL_FLAG_VI']
            else:
                self.logger.warning(f"BAL_FLAG_VI not found in {self.drq_filename}.")
                self.logger.ok_warning("Ignoring")

        ## BAL CIV
        if self.bi_max is not None:
            if 'BI_CIV' in catalogue.colnames:
                bi = catalogue['BI_CIV']
                w &= bi <= self.bi_max
                self.logger.progress(
                    f"and BI_CIV <= {self.bi_max}  : nb object in cat = {np.sum(w)}")
                keep_columns += ['BI_CIV']
            else:
                raise QuasarCatalogueError("Error in reading DRQ Catalogue. "
                                           "'BI max' was passed but field BI_CIV "
                                           "was not present in the HDU")

        # DLA Column density
        if 'NHI' in catalogue.colnames:
            keep_columns += ['NHI']

        # Convert angles to radians
        np.radians(catalogue['RA'], out=catalogue['RA'])
        np.radians(catalogue['DEC'], out=catalogue['DEC'])

        catalogue.keep_columns(keep_columns)
        w = np.where(w)[0]
        catalogue = catalogue[w]

        return catalogue

    def read_spall(self, drq_catalogue):
        """Read the spAll file and adds

        Arguments
        ---------
        drq_catalogue: astropy.Table
        Table with the DRQ catalogue

        Return
        ------
        catalogue: Astropy.table.Table
        Table with the spAll + DRQ catalogue

        Raise
        -----
        QuasarCatalogueError if spAll file is not found
        """
        self.logger.progress(f"reading spAll from {self.spall}")
        try:
            catalogue = Table.read(self.spall, hdu=1)
            catalogue.keep_columns(["THING_ID", "PLATE", "MJD",
                                    "FIBERID", "PLATEQUALITY",
                                    "ZWARNING"])
        except IOError as error:
            raise QuasarCatalogueError(
                "Error in reading DRQ Catalogue. Error "
                f"reading file {self.spall}. IOError "
                f"message: {str(error)}"
            ) from error

        w = np.in1d(catalogue["THING_ID"], drq_catalogue["THING_ID"])
        self.logger.progress(f"Found {np.sum(w)} spectra with required THING_ID")
        w &= catalogue["PLATEQUALITY"] == "good"
        self.logger.progress(f"Found {np.sum(w)} spectra with 'good' plate")
        ## Removing spectra with the following ZWARNING bits set:
        ## SKY, LITTLE_COVERAGE, UNPLUGGED, BAD_TARGET, NODATA
        ## https://www.sdss.org/dr14/algorithms/bitmasks/#ZWARNING
        bad_z_warn_bit = {
            0: 'SKY',
            1: 'LITTLE_COVERAGE',
            7: 'UNPLUGGED',
            8: 'BAD_TARGET',
            9: 'NODATA'
        }
        for z_warn_bit, z_warn_bit_name in bad_z_warn_bit.items():
            wbit = (catalogue["ZWARNING"] & 2**z_warn_bit == 0)
            w &= wbit
            self.logger.progress(f"Found {np.sum(w)} spectra without {z_warn_bit} "
                                 f"bit set: {z_warn_bit_name}")
        self.logger.progress(f"# unique objs: {len(drq_catalogue)}")
        self.logger.progress(f"# spectra: {w.sum()}")
        catalogue = catalogue[w]

        # merge redshift information from DRQ catalogue
        # columns are discarded on DRQ catalogues to
        # avoid conflicts with PLATE, FIBERID, MJD when assigning
        # DRQ properies and on spAll catalogue to avoid
        # conflicts with the creation of DrqObjects at a later
        # stage
        select_cols = [name for name in catalogue.colnames
                       if name not in ["PLATEQUALITY", "ZWARNING"]]
        select_cols_drq = [name for name in drq_catalogue.colnames
                           if name not in ["PLATE", "FIBERID", "MJD"]]
        catalogue = join(catalogue[select_cols],
                         drq_catalogue[select_cols_drq],
                         join_type="left")

        return catalogue
