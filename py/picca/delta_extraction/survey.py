"""This module defines the Survey class.
This class is responsible for managing the workflow of the computation
of deltas. It thus manages the interactions between the different
objects.
"""
import time
import logging
from numba import prange#, jit

from picca.delta_extraction.astronomical_objects.forest import Forest
from picca.delta_extraction.config import Config
from picca.delta_extraction.correction import Correction
from picca.delta_extraction.data import Data
from picca.delta_extraction.errors import DeltaExtractionError
from picca.delta_extraction.mask import Mask
from picca.delta_extraction.expected_flux import ExpectedFlux

# create logger
module_logger = logging.getLogger(__name__)

class Survey:
    """Class to manage the computation of deltas

    Methods
    -------
    __init__
    apply_corrections
    apply_masks
    compute_expected_flux
    extract_deltas
    filter_forests
    initialize_folders
    load_config
    read_corrections
    read_data
    read_masks
    save_deltas

    Attributes
    ----------
    config: Config
    A Config instance

    corrections: list of Correction
    Spectral corrections to be applied to all spectra. This includes things
    like calibration correction, ivar correction, dust correction, ...

    data: Data
    A Data instance containing the loaded Forests

    expected_flux: ExpectedFlux
    An ExpectedFlux instance to compute the expected flux in each Forest and
    extract the deltas

    logger: logging.Logger
    Logger object

    masks: list of Mask
    Mask corrections to be applied to individual spectra. This includes things
    like absorber mask, DLA mask, ...
    """
    def __init__(self):
        """Initialize class instance"""
        self.logger = logging.getLogger('picca.delta_extraction.survey.Survey')
        self.config = None
        self.corrections = None
        self.masks = None
        self.data = None
        self.expected_flux = None

    #@jit(nopython=True, parallel=True)
    def apply_corrections(self):
        """Apply the corrections. To be run after self.read_corrections()"""
        t0 = time.time()
        self.logger.info("Applying corrections")

        # pylint: disable=not-an-iterable
        # prange is used to signal jit of parallelisation but is otherwise
        # equivalent to range
        for forest_index in prange(len(self.data.forests)):
            for correction_index in range(len(self.corrections)):
                self.corrections[correction_index].apply_correction(self.data.forests[forest_index])

        t1 = time.time()
        self.logger.info(f"Time spent applying corrections: {t1-t0}")

    #@jit(nopython=True, parallel=True)
    def apply_masks(self):
        """Apply the corrections. To be run after self.read_corrections()"""
        t0 = time.time()
        self.logger.info("Applying masks")

        # pylint: disable=not-an-iterable
        # prange is used to signal jit of parallelisation but is otherwise
        # equivalent to range
        for forest_index in prange(len(self.data.forests)):
            for mask_index in range(len(self.masks)):
                self.masks[mask_index].apply_mask(self.data.forests[forest_index])

        t1 = time.time()
        self.logger.info(f"Time spent applying masks: {t1-t0}")

    def compute_expected_flux(self):
        """Compute the expected flux.
        This includes the quasar continua and the mean transimission.
        """
        t0 = time.time()
        self.logger.info("Computing mean expected flux.")

        ExpectedFluxType = self.config.expected_flux[0]
        expected_flux_arguments = self.config.expected_flux[1]
        self.expected_flux = ExpectedFluxType(expected_flux_arguments)
        self.expected_flux.compute_expected_flux(self.data.forests)
        t1 = time.time()
        self.logger.info(f"Time spent computing the mean expected flux: {t1-t0}")

    #@jit(nopython=True, parallel=True)
    def extract_deltas(self):
        """Compute the delta fields"""
        t0 = time.time()
        self.logger.info("Extracting deltas")

        # filter bad_continuum forests
        self.data.filter_bad_cont_forests()

        # pylint: disable=not-an-iterable
        # prange is used to signal jit of parallelisation but is otherwise
        # equivalent to range
        for forest_index in prange(len(self.data.forests)):
            self.expected_flux.extract_deltas(self.data.forests[forest_index])

        t1 = time.time()


        self.logger.info(f"Time spent extracting deltas: {t1-t0}")



    def filter_forests(self):
        """Remove forests that do not meet quality standards"""
        self.data.filter_forests()

    def load_config(self, config_file):
        """Load the configuration of the run, sets up the print function
        that will be used to print, and initializes the saving folders

        Arguments
        ---------
        config_file: str
        Name of the file specifying the configuration
        """
        # load configuration
        self.config = Config(config_file)

    def read_corrections(self):
        """Read the spectral corrections."""
        self.corrections = []
        t0 = time.time()
        num_corrections = self.config.num_corrections
        self.logger.info(f"Reading corrections. There are {num_corrections} corrections")

        for CorrectionType, correction_arguments in self.config.corrections:
            correction = CorrectionType(correction_arguments)
            self.corrections.append(correction)

        t1 = time.time()
        self.logger.info(f"Time spent reading Corrections: {t1-t0}")

    def read_data(self):
        """Read the data.

        Raise
        -----
        DeltaExtractionError when instances from data.forests are not from type Forests
        """
        t0 = time.time()
        self.logger.info("Reading data")

        DataType, data_arguments = self.config.data
        self.data = DataType(data_arguments)
        # we should never enter this block unless DataType is not correctly
        # writen
        if not all([isinstance(forest, Forest) for forest in self.data.forests]): # pragma: no cover
            raise DeltaExtractionError("Error reading data.\n At least one of "
                                       "the elements in variable 'forest' is "
                                       "not of class Forest. This can happen if "
                                       "the Data object responsible for reading "
                                       "the data did not define the correct data "
                                       "type. Please check for correct "
                                       "inheritance pattern.")
        self.data.find_nside()

        t1 = time.time()
        self.logger.info(f"Time spent reading data: {t1-t0}")

    def read_masks(self):
        """Read the spectral masks."""
        self.masks = []
        t0 = time.time()
        num_masks = self.config.num_masks
        self.logger.info(f"Reading masks. There are {num_masks} masks")

        for MaskType, mask_arguments in self.config.masks:
            mask = MaskType(mask_arguments)
            self.masks.append(mask)

        t1 = time.time()
        self.logger.info(f"Time spent reading masks: {t1-t0}")

    def save_deltas(self):
        """Save the deltas."""
        t0 = time.time()
        self.logger.info("Saving deltas")
        self.data.save_deltas()
        t1 = time.time()
        self.logger.info(f"Time spent saving deltas: {t1-t0}")
