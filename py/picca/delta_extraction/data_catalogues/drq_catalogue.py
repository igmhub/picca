"""This module defines the abstract class Data from which all
classes loading data must inherit
"""
import glob
import warnings
from astropy.table import Table, join
import healpy
import numpy as np

from picca.delta_extraction.data import Data
from picca.delta_extraction.errors import DataError, DataWarning
from picca.delta_extraction.userprint import userprint

from picca.delta_extraction.astronomical_objects.drq_object import DrqObject

defaults = {
    "best obs": False,
    "keep BAL": False,
    "z max": 3.5,
    "z min": 2.1,
}

class DrqCatalogue:
    """Reads the DRQ quasar catalogue SDSS

    Methods
    -------
    get_forest_list (from Data)
    __init__
    _parse_config


    Attributes
    ----------
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

    max_num_spec: int or None
    Maximum number of spectra to read. None for no maximum

    spall: str
    Path to the spAll file required for multiple observations

    z_min: float
    Minimum redshift. Quasars with redshifts lower than z_min will be
    discarded

    z_max: float
    Maximum redshift. Quasars with redshifts higher than or equal to
    z_max will be discarded
    """
    def __init__(self, config):
        """Initialize class instance

        Arguments
        ---------
        config: configparser.SectionProxy
        Parsed options to initialize class
        """

        # load variables from config
        self.best_obs = None
        self.bi_max = None
        self.drq_filename = None
        self.keep_bal = None
        self.max_num_spec = None
        self.spall = None
        self.z_max = None
        self.z_min = None
        self._parse_config(config)

        # read DRQ Catalogue
        catalogue = self.read_drq()

        # if using multiople observations load the information from spAll file
        if not self.best_obs:
            catalogue = self.read_spall(catalogue)

        # format catalogue into a list of objects
        self.catalogue = catalogue

        # if there is a maximum number of spectra, make sure they are selected
        # in a contiguous regions
        if self.max_num_spec is not None:
            # sort forests by healpix
            healpix = [healpy.ang2pix(16, np.pi / 2 - row["DEC"], row["RA"])
                       for row in self.catalogue]
            self.catalogue["healpix"] = healpix
            self.catalogue.sort("healpix")
            self.catalogue = self.catalogue[:self.max_num_spec]

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
        self.best_obs = config.getboolean("best obs")
        if self.best_obs is None:
            self.best_obs = defaults.get("best obs")
        self.bi_max = config.getfloat("BI max")
        self.drq_filename = config.get("drq catalogue")
        if self.drq_filename is None:
            raise DataError("Missing argument 'drq catalogue' required by SdssData")
        self.keep_bal = config.getboolean("keep BAL")
        if self.keep_bal is None:
            self.keep_bal = defaults.get("keep BAL")
        self.max_num_spec = config.getint("max num spectra")
        self.z_min = config.getfloat("z min")
        if self.z_min is None:
            self.z_min = defaults.get("z min")
        self.z_max = config.getfloat("z max")
        if self.z_max is None:
            self.z_max = defaults.get("z max")

        if self.best_obs:
            self.spall = None
        else:
            self.spall = config.get("spAll")
            if self.spall is None:
                warnings.warn("Missing argument 'spAll' required by DrqCatalogue. "
                              "Looking for spAll in input directory...", DataWarning)

            if config.get("input directory") is None:
                warnings.warn("'spAll' file not found. If you didn't want to load "
                              "the spAll file you should pass the option "
                              "'best obs = True'. Quiting...", DataWarning)
                raise DataError("Missing argument 'spAll' required by DrqCatalogue.")
            folder = config.get("spAll").replace("spectra/",
                                                 "").replace("lite",
                                                             "").replace("full",
                                                                         "")
            filenames = glob.glob(folder + "/spAll-*.fits")

            if len(filenames) > 1:
                warnings.warn("Found multiple 'spAll' files not found. Quiting...",
                              DataWarning)
                for filename in filenames:
                    warnings.warn(f"found: {filename}", DataWarning)
                raise DataError("Missing argument 'spAll' required by DrqCatalogue.")
            if len(filenames) == 0:
                warnings.warn("'spAll' file not found. If you didn't want to load "
                              "the spAll file you should pass the option "
                              "'best obs = True'. Quiting...", DataWarning)
                raise DataError("Missing argument 'spAll' required by DrqCatalogue.")
            self.spall = filenames[0]
            warnings.warn("'spAll' file found. Contining with normal execution.")


    def read_drq(self):
        """Read the DRQ Catalogue

        Returns
        -------
        catalogue: Astropy.table.Table
        Table with the DRQ catalogue
        """
        userprint('Reading DRQ catalogue from ', self.drq_filename)
        catalogue = Table.read(self.drq_filename, ext=1)

        keep_columns = ['RA', 'DEC', 'Z', 'THING_ID', 'PLATE', 'MJD', 'FIBERID']

        # Redshift
        if 'Z' not in catalogue.colnames:
            if 'Z_VI' in catalogue.colnames:
                catalogue.rename_column('Z_VI', 'Z')
                userprint(
                    "Z not found (new DRQ >= DRQ14 style), using Z_VI (DRQ <= DRQ12)"
                )
            else:
                raise DataError("Error in reading DRQ Catalogue. No valid column "
                                f"for redshift found in {self.drq_filename}")

        ## Sanity checks
        userprint('')
        w = np.ones(len(catalogue), dtype=bool)
        userprint(f"start                 : nb object in cat = {np.sum(w)}")
        w &= catalogue["THING_ID"] > 0
        userprint(f"and THING_ID > 0      : nb object in cat = {np.sum(w)}")
        w &= catalogue['RA'] != catalogue['DEC']
        userprint(f"and ra != dec         : nb object in cat = {np.sum(w)}")
        w &= catalogue['RA'] != 0.
        userprint(f"and ra != 0.          : nb object in cat = {np.sum(w)}")
        w &= catalogue['DEC'] != 0.
        userprint(f"and dec != 0.         : nb object in cat = {np.sum(w)}")

        ## Redshift range
        w &= catalogue['Z'] >= self.z_min
        userprint(f"and z >= {self.z_min}        : nb object in cat = {np.sum(w)}")
        w &= catalogue['Z'] < self.z_max
        userprint(f"and z < {self.z_max}         : nb object in cat = {np.sum(w)}")

        ## BAL visual
        if not self.keep_bal and self.bi_max is None:
            if 'BAL_FLAG_VI' in catalogue.colnames:
                self.bal_flag = catalogue['BAL_FLAG_VI']
                w &= self.bal_flag == 0
                userprint(
                    f"and BAL_FLAG_VI == 0  : nb object in cat = {np.sum(w)}")
                keep_columns += ['BAL_FLAG_VI']
            else:
                warnings.warn(f"BAL_FLAG_VI not found in {self.drq_filename}",
                              DataWarning)

        ## BAL CIV
        if self.bi_max is not None:
            if 'BI_CIV' in catalogue.colnames:
                bi = catalogue['BI_CIV']
                w &= bi <= self.bi_max
                userprint(
                    f"and BI_CIV <= {self.bi_max}  : nb object in cat = {np.sum(w)}")
                keep_columns += ['BI_CIV']
            else:
                raise DataError("Error in reading DRQ Catalogue. 'BI max' "
                                "was passed but field BI_CIV was not present "
                                "in the HDU")

        # DLA Column density
        if 'NHI' in catalogue.colnames:
            keep_columns += ['NHI']

        # Convert angles to radians
        catalogue['RA'] = np.radians(catalogue['RA'])
        catalogue['DEC'] = np.radians(catalogue['DEC'])

        catalogue.keep_columns(keep_columns)
        w = np.where(w)[0]
        catalogue = catalogue[w]

        return catalogue

    def read_spall(self, drq_catalogue):
        """Reads the spAll file and adds

        Arguments
        ---------
        drq_catalogue: astropy.Table
        Table with the DRQ catalogue

        Returns
        -------
        catalogue: Astropy.table.Table
        Table with the spAll + DRQ catalogue

        Raise
        -----
        DataError if spAll file is not found
        """
        userprint(f"INFO: reading spAll from {self.spall}")
        try:
            catalogue = Table.read(self.spall, ext=1,
                                   keep_columns=["THING_ID", "PLATE", "MJD",
                                                 "FIBERID", "PLATEQUALITY",
                                                 "ZWARNING", "RA", "DEC"])
        except IOError as error:
            raise DataError("Error in reading DRQ Catalogue. Error reading "
                            f"file {self.spall}. IOError message: {str(error)}")

        w = np.in1d(catalogue["THING_ID"], drq_catalogue["THING_ID"])
        userprint(f"INFO: Found {np.sum(w)} spectra with required THING_ID")
        w &= catalogue["PLATEQUALITY"] == "good"
        userprint(f"INFO: Found {np.sum(w)} spectra with 'good' plate")
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
            userprint(f"INFO: Found {np.sum(w)} spectra without {z_warn_bit} "
                      f"bit set: {z_warn_bit_name}")
        userprint(f"INFO: # unique objs: {len(drq_catalogue)}")
        userprint(f"INFO: # spectra: {w.sum()}")
        catalogue = catalogue[w]

        # merge redshift information from DRQ catalogue
        # columns are discarded on DRQ catalogues to
        # avoid conflicts with PLATE, FIBERID, MJD when assigning
        # DRQ properies and on spAll catalogue to avoid
        # conflicts with the creation of DrqObjects at a later
        # stage
        select_cols = [name for name in catalogue.colnames
                       if name not in ["PLATEQUALITY", "ZWARNING"]]
        select_cols_drq = [name for name in catalogue.colnames
                           if name not in ["PLATE", "FIBERID", "MJD"]]
        catalogue = join(catalogue[select_cols],
                         drq_catalogue[select_cols_drq],
                         join_type="left")

        return catalogue
