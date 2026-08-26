"""This module defines the class FourmostQuasarCatalogue to read 4MOST
quasar catalogues
"""
import logging
import os

from astropy.table import Table
import numpy as np

from picca.delta_extraction.errors import QuasarCatalogueError
from picca.delta_extraction.quasar_catalogue import QuasarCatalogue, accepted_options
from picca.delta_extraction.utils import update_accepted_options

accepted_options = update_accepted_options(accepted_options, ["catalogue"])

defaults = {}

# Columns read from the 4MOST file. The delivery joins the L1 spectra table
# with the target catalogue, so these names come from the joined product.
FOURMOST_RA = "RA"
FOURMOST_DEC = "DEC"
FOURMOST_Z = "zBest"
FOURMOST_OBJECT_ID = "OBJ_UID"
FOURMOST_SPECTRUM_ID = "specuid"
FOURMOST_EXPTIME = "exptime"
FOURMOST_SNR = "snr"

REQUIRED_COLUMNS = [
    FOURMOST_RA, FOURMOST_DEC, FOURMOST_Z, FOURMOST_OBJECT_ID,
    FOURMOST_SPECTRUM_ID, FOURMOST_EXPTIME, FOURMOST_SNR,
]


class FourmostQuasarCatalogue(QuasarCatalogue):
    """Read the 4MOST quasar catalogue.

    The 4MOST delivery ships the catalogue and the spectra in the same file,
    one row per *spectrum*. Objects observed more than once appear several
    times, and those repeats are nested cumulative stacks of the same photons
    rather than independent exposures: they share MJD-OBS and their exposure
    times form a chain. Only the longest stack of each object is kept.

    Methods
    -------
    (see QuasarCatalogue in py/picca/delta_extraction/quasar_catalogue.py)
    __init__
    __parse_config
    read_catalogue
    select_longest_stacks

    Attributes
    ----------
    (see QuasarCatalogue in py/picca/delta_extraction/quasar_catalogue.py)

    filename: str
    Path to the 4MOST spectra file

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

        self.filename = None
        self.__parse_config(config)

        self.read_catalogue()

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
        self.filename = config.get("catalogue")
        if self.filename is None:
            raise QuasarCatalogueError(
                "Missing argument 'catalogue' required by "
                "FourmostQuasarCatalogue")

    def read_catalogue(self):
        """Read the catalogue and store it in self.catalogue

        Raise
        -----
        QuasarCatalogueError if the file is missing or a column is absent
        """
        if not os.path.isfile(self.filename):
            raise QuasarCatalogueError(
                "Error reading the 4MOST catalogue. File not found: "
                f"{self.filename}")

        self.logger.progress(f"Reading 4MOST catalogue from {self.filename}")
        raw = Table.read(self.filename, hdu="Joined")

        for column in REQUIRED_COLUMNS:
            if column not in raw.colnames:
                raise QuasarCatalogueError(
                    "Error reading the 4MOST catalogue. Missing column "
                    f"'{column}' in {self.filename}")

        # Build a fresh table rather than renaming in place. The delivery is a
        # join of several catalogues and carries case-variant duplicates of
        # several columns ('specuid' and 'SPECUID', 'snr' and 'SNR', 'ra' and
        # 'RA'), so an in-place rename collides.
        catalogue = Table()
        catalogue["RA"] = np.asarray(raw[FOURMOST_RA], dtype=np.float64)
        catalogue["DEC"] = np.asarray(raw[FOURMOST_DEC], dtype=np.float64)
        catalogue["Z"] = np.asarray(raw[FOURMOST_Z], dtype=np.float64)
        catalogue["LOS_ID"] = np.asarray(raw[FOURMOST_OBJECT_ID],
                                         dtype=np.int64)
        catalogue["SPECUID"] = np.asarray(raw[FOURMOST_SPECTRUM_ID],
                                          dtype=np.int64)
        catalogue["EXPTIME"] = np.asarray(raw[FOURMOST_EXPTIME],
                                          dtype=np.float64)
        catalogue["SNR"] = np.asarray(raw[FOURMOST_SNR], dtype=np.float64)
        # remember where each entry sits in the file, so that FourmostData
        # can read the matching FLUX and ERR rows
        catalogue["ROW_INDEX"] = np.arange(len(raw), dtype=np.int64)

        # redshift cuts
        num_before = len(catalogue)
        keep = np.isfinite(catalogue["Z"])
        keep &= catalogue["Z"] >= self.z_min
        keep &= catalogue["Z"] < self.z_max
        catalogue = catalogue[keep]
        self.logger.progress(
            f"Redshift cuts: kept {len(catalogue)} of {num_before} spectra")

        catalogue = self.select_longest_stacks(catalogue)

        # Convert angles to radians. 4MOST stores decimal degrees (J2000);
        # picca works in radians everywhere downstream.
        catalogue["RA"] = np.radians(np.asarray(catalogue["RA"],
                                                dtype=np.float64))
        catalogue["DEC"] = np.radians(np.asarray(catalogue["DEC"],
                                                 dtype=np.float64))

        self.catalogue = catalogue

    def select_longest_stacks(self, catalogue):
        """Keep one row per object: the longest stack.

        Repeated rows of the same object are nested cumulative stacks, so
        using more than one would reuse the same photons. Ties on exposure
        time are broken by signal-to-noise.

        Arguments
        ---------
        catalogue: astropy.table.Table
        Catalogue with one row per spectrum

        Return
        ------
        catalogue: astropy.table.Table
        Catalogue with one row per object
        """
        num_before = len(catalogue)

        # sort so that, within each object, the wanted row comes last
        catalogue.sort(["LOS_ID", "EXPTIME", "SNR"])
        los_id = np.asarray(catalogue["LOS_ID"])
        # the last row of each run of equal LOS_ID is the one to keep
        keep = np.ones(len(catalogue), dtype=bool)
        keep[:-1] = los_id[:-1] != los_id[1:]
        catalogue = catalogue[keep]

        self.logger.progress(
            f"Repeat handling: kept the longest stack of each object, "
            f"{len(catalogue)} objects from {num_before} spectra")
        return catalogue
