"""This module defines the Survey class.
This class is responsible for managing the workflow of the computation
of deltas. It thus manages the interactions between the different
objects.
"""
import os
import time
from numba import jit, prange

from picca.delta_extraction.astronomical_objects.forest import Forest
from picca.delta_extraction.config import Config
from picca.delta_extraction.correction import Correction
from picca.delta_extraction.data import Data
from picca.delta_extraction.errors import DeltaExtractionError
from picca.delta_extraction.mask import Mask
from picca.delta_extraction.mean_expected_flux import MeanExpectedFlux
from picca.delta_extraction.userprint import UserPrint, userprint

class Survey:
    """Class to manage the computation of deltas

    Methods
    -------
    __init__
    apply_corrections
    apply_masks
    load_config
    print
    read_corrections
    read_data
    read_masks

    Attributes
    ----------
    config: Config
    A Config instance

    corrections: list of Correction
    Spectral corrections to be applied to all spectra. This includes things
    like calibration correction, ivar correction, dust correction, ...

    forests: List of Forest
    A list of Forest from which to compute the deltas.

    masks: list of Mask
    Mask corrections to be applied to individual spectra. This includes things
    like absorber mask, DLA mask, ...

    print: function
    Print function with the level of verbosity specified in the coniguration
    """
    def __init__(self):
        """Initializes class instance"""
        self.config = None
        self.corrections = None
        self.masks = None
        self.forests = None
        self.mean_expected_flux = None

    @jit(nopython=True, parallel=True)
    def extract_deltas(self):
        """Computes the delta fields"""
        t0 = time.time()
        # pylint: disable=not-an-iterable
        # prange is used to signal jit of parallelisation but is otherwise
        # equivalent to range
        for forest_index in prange(len(self.forests)):
            self.mean_expected_flux.extract_delta(self.forests[forest_index])

        t1 = time.time()
        userprint(f"Time spent extracting deltas: {t1-t0}")

    @jit(nopython=True, parallel=True)
    def apply_corrections(self):
        """Applies the corrections. To be run after self.read_corrections()"""
        t0 = time.time()

        # pylint: disable=not-an-iterable
        # prange is used to signal jit of parallelisation but is otherwise
        # equivalent to range
        for forest_index in prange(len(self.forests)):
            for correction_index in range(len(self.corrections)):
                self.corrections[correction_index].apply_correction(self.forests[forest_index])

        t1 = time.time()
        userprint(f"Time spent applying corrections: {t1-t0}")

    @jit(nopython=True, parallel=True)
    def apply_masks(self):
        """Applies the corrections. To be run after self.read_corrections()"""
        t0 = time.time()

        # pylint: disable=not-an-iterable
        # prange is used to signal jit of parallelisation but is otherwise
        # equivalent to range
        for forest_index in prange(len(self.forests)):
            for mask_index in range(len(self.masks)):
                self.masks[mask_index].apply_mask(self.forests[forest_index])

        t1 = time.time()
        userprint(f"Time spent applying corrections: {t1-t0}")

    def compute_mean_expected_flux(self):
        """Computes the mean expected flux.
        This includes the quasar continua and the mean transimission.

        Raises
        ------
        DeltaExtractionError selected mean expected flux object does not have
        the correct type
        """
        t0 = time.time()
        userprint("Computing mean expected flux.")

        MeanExpectedFluxType = self.config.mean_expected_flux[0]
        mean_expected_flux_arguments = self.config.mean_expected_flux[1]
        self.mean_expected_flux = MeanExpectedFluxType(mean_expected_flux_arguments)
        if not isinstance(self.mean_expected_flux, MeanExpectedFlux):
            raise DeltaExtractionError("Error computing mean expected flux.\n"
                                       f"Type {MeanExpectedFluxType} with arguments "
                                       f"{mean_expected_flux_arguments} is "
                                       "not a correct type. Expected inheritance "
                                       "from 'MeanExpectedFlux'. Please check "
                                       "for correct inheritance pattern.")

        self.mean_expected_flux.compute_mean_expected_flux(self.forests)
        t1 = time.time()
        userprint(f"Time spent computing the mean expected flux: {t1-t0}")

    def initialize_folders(self):
        """Initialize output folders

        Raises
        ------
        DeltaExtractionError if the output path was already used and the
        overwrite is not selected
        """
        if not os.path.exists(self.config.out_dir):
            os.makedirs(self.config.out_dir)
            self.config.write_config()
        elif self.config.overwrite:
            self.config.write_config()
        else:
            raise DeltaExtractionError("Specified folder contains a previous run."
                                       "Pass overwrite option in configuration file"
                                       "in order to ignore the previous run or"
                                       "change the output path variable to point "
                                       "elsewhere")

    def load_config(self, config_file):
        """Loads the configuration of the run, sets up the print function
        that will be used to print, and initializes the saving folders

        Arguments
        ---------
        config_file: str
        Name of the file specifying the configuration
        """
        # load configuration
        self.config = Config(config_file)

        # printing setup
        if self.config.quiet:
            UserPrint.print_type = "quietprint"
        if self.config.log is not None:
            UserPrint.initialize_log(self.config.log)

    def read_corrections(self):
        """Reads the spectral corrections.

        Raises
        ------
        DeltaExtractionError when any of the read correction do not have the
        correct type.
        """
        self.corrections = []
        t0 = time.time()
        num_corrections = self.config.num_corrections
        userprint(f"Reading corrections. There are {num_corrections} corrections")

        for CorrectionType, correction_arguments in self.config.corrections:
            correction = CorrectionType(correction_arguments)
            if not isinstance(correction, Correction):
                raise DeltaExtractionError("Error reading correction\n"
                                           f"Type {CorrectionType} with arguments "
                                           f"{correction_arguments} is not a correct "
                                           "type. Corrections should inher from "
                                           "'Correction'. Please check for correct "
                                           "inheritance pattern.")
            self.corrections.append(correction)

        t1 = time.time()
        userprint(f"Time spent reading Corrections: {t1-t0}")

    def read_data(self):
        """Reads the data.

        Raises
        ------
        DeltaExtractionError when data could not be read
        """
        t0 = time.time()
        userprint("Reading data")

        DataType, data_arguments = self.config.data
        data = DataType(data_arguments)
        if not isinstance(data, Data):
            raise DeltaExtractionError("Error reading data\n"
                                       f"Type {DataType} with arguments "
                                       f"{data_arguments} is not a correct "
                                       "type. Data should inher from "
                                       "'Forest'. Please check for correct "
                                       "inheritance pattern.")
        self.forests = data.get_forest_list()
        if not all([isinstance(forest, Forest) for forest in self.forests]):
            raise DeltaExtractionError("Error reading data.\n At least one of "
                                       "the elements in variable 'forest' is "
                                       "not of class Forest. This can happen if "
                                       "the Data object responsible for reading "
                                       "the data did not define the correct data "
                                       "type. Please check for correct "
                                       "inheritance pattern.")

        t1 = time.time()
        userprint(f"Time spent reading data: {t1-t0}")

    def read_masks(self):
        """Reads the spectral masks.

        Raises
        ------
        DeltaExtractionError when any of the read correction do not have the
        correct type.
        """
        self.masks = []
        t0 = time.time()
        num_masks = self.config.num_masks
        userprint(f"Reading masks. There are {num_masks} corrections")

        for MaskType, mask_arguments in self.config.masks:
            mask = MaskType(mask_arguments)
            if not isinstance(mask, Mask):
                raise DeltaExtractionError("Error reading mask\n"
                                           f"Type {MaskType} with arguments "
                                           f"{mask_arguments} is not a correct "
                                           "type. Masks should inher from "
                                           "'Mask'. Please check for correct "
                                           "inheritance pattern.")
            self.masks.append(mask)

        t1 = time.time()
        userprint(f"Time spent reading masks: {t1-t0}")
