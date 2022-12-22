"""This file contains tests related to Data and its childs"""
from configparser import ConfigParser
import logging
import os
import unittest
import copy
import numpy as np

from picca.delta_extraction.astronomical_objects.desi_forest import DesiForest
from picca.delta_extraction.astronomical_objects.desi_pk1d_forest import DesiPk1dForest
from picca.delta_extraction.astronomical_objects.sdss_forest import SdssForest
from picca.delta_extraction.config import default_config
from picca.delta_extraction.data import Data
from picca.delta_extraction.data import defaults as defaults_data
from picca.delta_extraction.data import accepted_analysis_type, accepted_save_format
from picca.delta_extraction.data_catalogues.desi_data import DesiData
from picca.delta_extraction.data_catalogues.desi_data import defaults as defaults_desi_data
from picca.delta_extraction.data_catalogues.desi_data import accepted_options as accepted_options_desi_data
from picca.delta_extraction.data_catalogues.desi_healpix import DesiHealpix, DesiHealpixFileHandler
from picca.delta_extraction.data_catalogues.desi_healpix import defaults as defaults_desi_healpix
from picca.delta_extraction.data_catalogues.desi_tile import DesiTile
from picca.delta_extraction.data_catalogues.desi_tile import defaults as defaults_desi_tile
from picca.delta_extraction.data_catalogues.desisim_mocks import DesisimMocks
from picca.delta_extraction.data_catalogues.desisim_mocks import defaults as defaults_desisim_mocks
from picca.delta_extraction.data_catalogues.sdss_data import SdssData
from picca.delta_extraction.data_catalogues.sdss_data import defaults as defaults_sdss_data
from picca.delta_extraction.errors import DataError, QuasarCatalogueError
from picca.delta_extraction.utils import ABSORBER_IGM
from picca.delta_extraction.utils import ACCEPTED_BLINDING_STRATEGIES
from picca.delta_extraction.utils import setup_logger
from picca.tests.delta_extraction.abstract_test import AbstractTest
from picca.tests.delta_extraction.test_utils import reset_logger, setup_test_logger
from picca.tests.delta_extraction.test_utils import forest1
from picca.tests.delta_extraction.test_utils import sdss_data_kwargs
from picca.tests.delta_extraction.test_utils import sdss_data_kwargs_filter_forest

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class DataTest(AbstractTest):
    """Test class Data and its childs.

    Methods
    -------
    compare_ascii (from AbstractTest)
    compare_fits (from AbstractTest)
    setUp (from AbstractTest)
    test_data
    test_data_filter_forests
    test_desi_data
    test_desi_data_minisv
    test_sdss_data_spec
    test_sdss_data_spplate
    """

    def check_read_file_error(self, data, catalogue, filename, expected_message,
                              warnings=False):
        """Check the warning/error message when running data.read_file()

        Arguments
        ---------
        data: DesiData
        Data instance

        catalogue: astropy.table.Table
        Expected error message

        filename: str
        Filename to load

        expected_message: str
        Expected error message

        warnings: bool - Default: False
        If True, treat warnings as errors
        """
        if warnings:
            setup_test_logger("picca.delta_extraction.data.Data", DataError)
        with self.assertRaises(DataError) as context_manager:
            reader = DesiHealpixFileHandler(data.analysis_type,
                                            data.use_non_coadded_spectra,
                                            data.logger)

            reader((filename, catalogue))
        self.compare_error_message(context_manager, expected_message)
        if warnings:
            setup_test_logger("picca.delta_extraction.data.Data", DataError,
                              reset=True)

    def test_data(self):
        """Test Abstract class Data

        Load a Data instace.
        """
        config = ConfigParser()
        config.read_dict({"data": {
                            "out dir": f"{THIS_DIR}/results/",
                            "rejection log file": "rejection_log.fits.gz",
                            "wave solution": "log",
                            "delta log lambda": 3e-4,
                            "input directory": f"{THIS_DIR}/data/",
                            "num processors": 1,
                         }})
        for key, value in defaults_data.items():
            if key not in config["data"]:
                config["data"][key] = str(value)

        data = Data(config["data"])

        self.assertTrue(len(data.forests) == 0)
        self.assertTrue(data.min_num_pix == 50)

        config = ConfigParser()
        config.read_dict({"data": {"minimum number pixels in forest": 40,
                                   "out dir": f"{THIS_DIR}/results/",
                                   "rejection log file": "rejection_log.fits.gz",
                                   "wave solution": "log",
                                   "delta log lambda": 3e-4,
                                   "input directory": f"{THIS_DIR}/data/",
                                   "num processors": 1,
                         }})
        for key, value in defaults_data.items():
            if key not in config["data"]:
                config["data"][key] = str(value)
        data = Data(config["data"])

        self.assertTrue(len(data.forests) == 0)
        self.assertTrue(data.min_num_pix == 40)
        self.assertTrue(data.analysis_type == "BAO 3D")

    def test_data_parse_config(self):
        """Test method __parse_config from Data"""
        # create a Data instance with missing wave_solution
        config = ConfigParser()
        config.read_dict({"data": {
        }})
        expected_message = (
            "Missing argument 'wave solution' required by Data"
        )
        with self.assertRaises(DataError) as context_manager:
            Data(config["data"])
        self.compare_error_message(context_manager, expected_message)

        # create a Data instance with invalid wave_solution
        config = ConfigParser()
        config.read_dict({"data": {
            "wave solution": "wrong",
        }})
        expected_message = (
            "Unrecognised value for 'wave solution'. Expected either 'lin' or "
            "'log'. Found 'wrong'."
        )
        with self.assertRaises(DataError) as context_manager:
            Data(config["data"])
        self.compare_error_message(context_manager, expected_message)

        # create a Data instance with missing delta_log_lambda
        config = ConfigParser()
        config.read_dict({"data": {
            "wave solution": "log",
        }})
        expected_message = (
            "Missing argument 'delta log lambda' required by Data when "
            "'wave solution' is set to 'log'"
        )
        with self.assertRaises(DataError) as context_manager:
            Data(config["data"])
        self.compare_error_message(context_manager, expected_message)

        # create a Data instance with missing delta_lambda
        config = ConfigParser()
        config.read_dict({"data": {
            "wave solution": "lin",
        }})
        expected_message = (
            "Missing argument 'delta lambda' required by Data when "
            "'wave solution' is set to 'lin'"
        )
        with self.assertRaises(DataError) as context_manager:
            Data(config["data"])
        self.compare_error_message(context_manager, expected_message)

        # create a Data instance with missing lambda_max
        config = ConfigParser()
        config.read_dict({"data": {
            "wave solution": "lin",
            "delta lambda": 0.8,
        }})
        expected_message = (
            "Missing argument 'lambda max' required by Data"
        )
        with self.assertRaises(DataError) as context_manager:
            Data(config["data"])
        self.compare_error_message(context_manager, expected_message)

        # create a Data instance with missing lambda_max_rest_frame
        config = ConfigParser()
        config.read_dict({"data": {
            "wave solution": "lin",
            "delta lambda": 0.8,
            "lambda max": 5500.0,
        }})
        expected_message = (
            "Missing argument 'lambda max rest frame' required by Data"
        )
        with self.assertRaises(DataError) as context_manager:
            Data(config["data"])
        self.compare_error_message(context_manager, expected_message)

        # create a Data instance with missing lambda_min
        config = ConfigParser()
        config.read_dict({"data": {
            "wave solution": "lin",
            "delta lambda": 0.8,
            "lambda max": 5500.0,
            "lambda max rest frame": 1200.0,
        }})
        expected_message = (
            "Missing argument 'lambda min' required by Data"
        )
        with self.assertRaises(DataError) as context_manager:
            Data(config["data"])
        self.compare_error_message(context_manager, expected_message)

        # create a Data instance with missing lambda_min_rest_frame
        config = ConfigParser()
        config.read_dict({"data": {
            "wave solution": "lin",
            "delta lambda": 0.8,
            "lambda max": 5500.0,
            "lambda max rest frame": 1200.0,
            "lambda min": 3600.0,
        }})
        expected_message = (
            "Missing argument 'lambda min rest frame' required by Data"
        )
        with self.assertRaises(DataError) as context_manager:
            Data(config["data"])
        self.compare_error_message(context_manager, expected_message)

        # create a Data instance with missing analysis_type
        config = ConfigParser()
        config.read_dict({"data": {
            "wave solution": "lin",
            "delta lambda": 0.8,
            "lambda max": 5500.0,
            "lambda max rest frame": 1200.0,
            "lambda min": 3600.0,
            "lambda min rest frame": 1040.0,
        }})
        expected_message = (
            "Missing argument 'analysis type' required by Data"
        )
        with self.assertRaises(DataError) as context_manager:
            Data(config["data"])
        self.compare_error_message(context_manager, expected_message)

        # create a Data instance with wrong analysis_type
        config = ConfigParser()
        config.read_dict({"data": {
            "wave solution": "lin",
            "delta lambda": 0.8,
            "lambda max": 5500.0,
            "lambda max rest frame": 1200.0,
            "lambda min": 3600.0,
            "lambda min rest frame": 1040.0,
            "analysis type": "INVALID"
        }})
        expected_message = (
            "Invalid argument 'analysis type' required by Data. "
            "Found: 'INVALID'. Accepted "
            "values: " + ",".join(accepted_analysis_type)
        )
        with self.assertRaises(DataError) as context_manager:
            Data(config["data"])
        self.compare_error_message(context_manager, expected_message)

        # create a Data instance with missing lambda_abs_igm
        config = ConfigParser()
        config.read_dict({"data": {
            "wave solution": "lin",
            "delta lambda": 0.8,
            "lambda max": 5500.0,
            "lambda max rest frame": 1200.0,
            "lambda min": 3600.0,
            "lambda min rest frame": 1040.0,
            "analysis type": "PK 1D",
        }})
        expected_message = (
            "Missing argument 'lambda abs IGM' required by Data when "
            "'analysys type' is 'PK 1D'"
        )
        with self.assertRaises(DataError) as context_manager:
            Data(config["data"])
        self.compare_error_message(context_manager, expected_message)

        # create a Data instance with invallid lambda_abs_igm
        config = ConfigParser()
        config.read_dict({"data": {
            "wave solution": "lin",
            "delta lambda": 0.8,
            "lambda max": 5500.0,
            "lambda max rest frame": 1200.0,
            "lambda min": 3600.0,
            "lambda min rest frame": 1040.0,
            "analysis type": "PK 1D",
            "lambda abs IGM": "wrong",
        }})
        keys = [key for key in ABSORBER_IGM.keys()]
        expected_message = (
            "Invalid argument 'lambda abs IGM' required by Data. Found: "
            "'wrong'. Accepted values: " + ", ".join(keys)
        )
        with self.assertRaises(DataError) as context_manager:
            Data(config["data"])
        self.compare_error_message(context_manager, expected_message)

        # create a Data instance with missing input_directory
        config = ConfigParser()
        config.read_dict({"data": {
            "wave solution": "lin",
            "delta lambda": 0.8,
            "lambda max": 5500.0,
            "lambda max rest frame": 1200.0,
            "lambda min": 3600.0,
            "lambda min rest frame": 1040.0,
            "analysis type": "BAO 3D",
        }})
        expected_message = (
            "Missing argument 'input directory' required by Data"
        )
        with self.assertRaises(DataError) as context_manager:
            Data(config["data"])
        self.compare_error_message(context_manager, expected_message)

        # create a Data instance with missing min_num_pix
        config = ConfigParser()
        config.read_dict({"data": {
            "wave solution": "lin",
            "delta lambda": 0.8,
            "lambda max": 5500.0,
            "lambda max rest frame": 1200.0,
            "lambda min": 3600.0,
            "lambda min rest frame": 1040.0,
            "analysis type": "BAO 3D",
            "input directory": f"{THIS_DIR}/data",
        }})
        expected_message = (
            "Missing argument 'minimum number pixels in forest' required by Data"
        )
        with self.assertRaises(DataError) as context_manager:
            Data(config["data"])
        self.compare_error_message(context_manager, expected_message)

        # create a Data instance with missing num_processors
        config = ConfigParser()
        config.read_dict({"data": {
            "wave solution": "lin",
            "delta lambda": 0.8,
            "lambda max": 5500.0,
            "lambda max rest frame": 1200.0,
            "lambda min": 3600.0,
            "lambda min rest frame": 1040.0,
            "analysis type": "BAO 3D",
            "input directory": f"{THIS_DIR}/data",
            "minimum number pixels in forest": 50,
        }})
        expected_message = (
            "Missing argument 'num processors' required by Data"
        )
        with self.assertRaises(DataError) as context_manager:
            Data(config["data"])
        self.compare_error_message(context_manager, expected_message)

        # create a Data instance with missing out_dir
        config = ConfigParser()
        config.read_dict({"data": {
            "wave solution": "lin",
            "delta lambda": 0.8,
            "lambda max": 5500.0,
            "lambda max rest frame": 1200.0,
            "lambda min": 3600.0,
            "lambda min rest frame": 1040.0,
            "analysis type": "BAO 3D",
            "input directory": f"{THIS_DIR}/data",
            "minimum number pixels in forest": 50,
            "num processors": 1,
        }})
        expected_message = (
            "Missing argument 'out dir' required by Data"
        )
        with self.assertRaises(DataError) as context_manager:
            Data(config["data"])
        self.compare_error_message(context_manager, expected_message)

        # create a Data instance with missing rejection_log_file
        config = ConfigParser()
        config.read_dict({"data": {
            "wave solution": "lin",
            "delta lambda": 0.8,
            "lambda max": 5500.0,
            "lambda max rest frame": 1200.0,
            "lambda min": 3600.0,
            "lambda min rest frame": 1040.0,
            "analysis type": "BAO 3D",
            "input directory": f"{THIS_DIR}/data",
            "minimum number pixels in forest": 50,
            "num processors": 1,
            "out dir": f"{THIS_DIR}/results",
        }})
        expected_message = (
            "Missing argument 'rejection log file' required by Data"
        )
        with self.assertRaises(DataError) as context_manager:
            Data(config["data"])
        self.compare_error_message(context_manager, expected_message)

        # create a Data instance with invalid rejection_log_file (including folders)
        config = ConfigParser()
        config.read_dict({"data": {
            "wave solution": "lin",
            "delta lambda": 0.8,
            "lambda max": 5500.0,
            "lambda max rest frame": 1200.0,
            "lambda min": 3600.0,
            "lambda min rest frame": 1040.0,
            "analysis type": "BAO 3D",
            "input directory": f"{THIS_DIR}/data",
            "minimum number pixels in forest": 50,
            "num processors": 1,
            "out dir": f"{THIS_DIR}/results",
            "rejection log file": "results/rejection_log.fits.gz",
        }})
        expected_message = (
            "Error constructing Data. 'rejection log file' should not "
            f"incude folders. Found: {THIS_DIR}/results/rejection_log.fits.gz"
        )
        with self.assertRaises(DataError) as context_manager:
            Data(config["data"])
        self.compare_error_message(context_manager, expected_message)

        # create a Data instance with invalid rejection_log_file (wrong extension)
        config = ConfigParser()
        config.read_dict({"data": {
            "wave solution": "lin",
            "delta lambda": 0.8,
            "lambda max": 5500.0,
            "lambda max rest frame": 1200.0,
            "lambda min": 3600.0,
            "lambda min rest frame": 1040.0,
            "analysis type": "BAO 3D",
            "input directory": f"{THIS_DIR}/data",
            "minimum number pixels in forest": 50,
            "num processors": 1,
            "out dir": f"{THIS_DIR}/results",
            "rejection log file": "rejection_log.txt",
        }})
        expected_message = (
            "Error constructing Data. Invalid extension for "
            "'rejection log file'. Filename "
            "should en with '.fits' or '.fits.gz'. Found "
            "'rejection_log.txt'"
        )
        with self.assertRaises(DataError) as context_manager:
            Data(config["data"])
        self.compare_error_message(context_manager, expected_message)

        # create a Data instance with missing save format
        config = ConfigParser()
        config.read_dict({"data": {
            "wave solution": "lin",
            "delta lambda": 0.8,
            "lambda max": 5500.0,
            "lambda max rest frame": 1200.0,
            "lambda min": 3600.0,
            "lambda min rest frame": 1040.0,
            "analysis type": "BAO 3D",
            "input directory": f"{THIS_DIR}/data",
            "minimum number pixels in forest": 50,
            "num processors": 1,
            "out dir": f"{THIS_DIR}/results",
            "rejection log file": "rejection_log.fits.gz",
        }})
        expected_message = (
            "Missing argument 'save format' required by Data"
        )
        with self.assertRaises(DataError) as context_manager:
            Data(config["data"])
        self.compare_error_message(context_manager, expected_message)

        # create a Data instance with invalid save format
        config = ConfigParser()
        config.read_dict({"data": {
            "wave solution": "lin",
            "delta lambda": 0.8,
            "lambda max": 5500.0,
            "lambda max rest frame": 1200.0,
            "lambda min": 3600.0,
            "lambda min rest frame": 1040.0,
            "analysis type": "BAO 3D",
            "input directory": f"{THIS_DIR}/data",
            "minimum number pixels in forest": 50,
            "num processors": 1,
            "out dir": f"{THIS_DIR}/results",
            "rejection log file": "rejection_log.fits.gz",
            "save format": "InvalidFormat",
        }})
        expected_message = (
            "Invalid argument 'save format' required by Data. Found: 'InvalidFormat'. Accepted values: " + ",".join(accepted_save_format)
        )
        with self.assertRaises(DataError) as context_manager:
            Data(config["data"])
        self.compare_error_message(context_manager, expected_message)

        # create a Data instance with missing minimal snr bao3d
        config = ConfigParser()
        config.read_dict({"data": {
            "wave solution": "lin",
            "delta lambda": 0.8,
            "lambda max": 5500.0,
            "lambda max rest frame": 1200.0,
            "lambda min": 3600.0,
            "lambda min rest frame": 1040.0,
            "analysis type": "BAO 3D",
            "input directory": f"{THIS_DIR}/data",
            "minimum number pixels in forest": 50,
            "num processors": 1,
            "out dir": f"{THIS_DIR}/results",
            "rejection log file": "rejection_log.fits.gz",
            "save format": "BinTableHDU",
        }})
        expected_message = (
            "Missing argument 'minimal snr bao3d' (if 'analysis type' = "
            "'BAO 3D') or ' minimal snr pk1d' (if 'analysis type' = 'Pk1d') "
            "required by Data"
        )
        with self.assertRaises(DataError) as context_manager:
            Data(config["data"])
        self.compare_error_message(context_manager, expected_message)

        # create a Data instance with missing minimal snr bao3d
        config = ConfigParser()
        config.read_dict({"data": {
            "wave solution": "lin",
            "delta lambda": 0.8,
            "lambda max": 5500.0,
            "lambda max rest frame": 1200.0,
            "lambda min": 3600.0,
            "lambda min rest frame": 1040.0,
            "analysis type": "PK 1D",
            "lambda abs IGM": "LYA",
            "input directory": f"{THIS_DIR}/data",
            "minimum number pixels in forest": 50,
            "num processors": 1,
            "out dir": f"{THIS_DIR}/results",
            "rejection log file": "rejection_log.fits.gz",
            "save format": "BinTableHDU",
        }})
        expected_message = (
            "Missing argument 'minimal snr bao3d' (if 'analysis type' = "
            "'BAO 3D') or ' minimal snr pk1d' (if 'analysis type' = 'Pk1d') "
            "required by Data"
        )
        with self.assertRaises(DataError) as context_manager:
            Data(config["data"])
        self.compare_error_message(context_manager, expected_message)

    def test_data_filter_forests(self):
        """Test method filter_forests from Abstract Class Data"""
        # create Data instance
        config = ConfigParser()
        config.read_dict({"data": {
                            "out dir": f"{THIS_DIR}/results/",
                            "rejection log file": "rejection_log.fits.gz",
                            "save format": "BinTableHDU",
                            "wave solution": "log",
                            "delta log lambda": 3e-4,
                            "delta log lambda rest frame": 3e-4,
                            "input directory": f"{THIS_DIR}/data/",
                            "num processors": 1,
                        }})
        for key, value in defaults_data.items():
            if key not in config["data"]:
                config["data"][key] = str(value)
        data = Data(config["data"])

        # add dummy forests
        data.forests.append(forest1)
        self.assertTrue(len(data.forests) == 1)

        # filter forests
        data.filter_forests()
        self.assertTrue(len(data.forests) == 1)

        # reset data
        data = Data(config["data"])

        # add nan forest
        forest_nan = copy.deepcopy(forest1)
        forest_nan.flux *= np.nan
        data.forests.append(forest_nan)
        self.assertTrue(len(data.forests) == 1)

        # filter forests
        data.filter_forests()
        self.assertTrue(len(data.forests) == 0)
        self.assertTrue(len(data.rejection_log.cols[0]) == 1)
        self.assertTrue(len(data.rejection_log.cols[1]) == 1)
        self.assertTrue(data.rejection_log.cols[1][0] == "nan_forest")

        # create Data instance with insane forest length requirements
        config = ConfigParser()
        config.read_dict({"data": {"minimum number pixels in forest": 10000,
                                   "out dir": f"{THIS_DIR}/results/",
                                   "rejection log file": "rejection_log.fits.gz",
                                    "save format": "BinTableHDU",
                                   "wave solution": "log",
                                   "delta log lambda": 3e-4,
                                   "delta log lambda rest frame": 3e-4,
                                   "input directory": f"{THIS_DIR}/data/",
                                   "num processors": 1,
                         }})
        for key, value in defaults_data.items():
            if key not in config["data"]:
                config["data"][key] = str(value)
        data = Data(config["data"])

        # add dummy forest
        data.forests.append(forest1)
        self.assertTrue(len(data.forests) == 1)

        # filter forests
        data.filter_forests()
        self.assertTrue(len(data.forests) == 0)
        self.assertTrue(len(data.rejection_log.cols[0]) == 1)
        self.assertTrue(len(data.rejection_log.cols[1]) == 1)
        self.assertTrue(data.rejection_log.cols[1][0] == "short_forest")

        # create Data instance with insane forest s/n requirements
        config = ConfigParser()
        config.read_dict({"data": {"out dir": f"{THIS_DIR}/results/",
                                   "rejection log file": "rejection_log.fits.gz",
                                    "save format": "BinTableHDU",
                                   "wave solution": "log",
                                   "delta log lambda": 3e-4,
                                   "delta log lambda rest frame": 3e-4,
                                   "input directory": f"{THIS_DIR}/data/",
                                   "minimal snr bao3d": 100000000,
                                   "num processors": 1,
                         }})
        for key, value in defaults_data.items():
            if key not in config["data"]:
                config["data"][key] = str(value)
        data = Data(config["data"])

        # add dummy forest
        data.forests.append(forest1)
        self.assertTrue(len(data.forests) == 1)

        # filter forests
        data.filter_forests()
        self.assertTrue(len(data.forests) == 0)
        self.assertTrue(len(data.rejection_log.cols[0]) == 1)
        self.assertTrue(len(data.rejection_log.cols[1]) == 1)
        self.assertTrue(data.rejection_log.cols[1][0] == f"low SNR ({forest1.mean_snr})")

    def test_desi_data(self):
        """Test DesiData

        It should not be possible to create a DesiData class
        since it is an abstract class.
        """
        config = ConfigParser()
        config.read_dict({"data": {
            "catalogue": f"{THIS_DIR}/data/dummy_desi_quasar_catalogue.fits",
            "keep surveys": "all special",
            "input directory": f"{THIS_DIR}/data/",
            "out dir": f"{THIS_DIR}/results/",
            "num processors": 1,
        }})
        for key, value in defaults_desi_data.items():
            if key not in config["data"]:
                config["data"][key] = str(value)

        expected_message = ("Function 'read_data' was not overloaded by "
                            "child class")
        with self.assertRaises(DataError) as context_manager:
            DesiData(config["data"])
        self.compare_error_message(context_manager, expected_message)

    def test_desi_data_parse_config(self):
        """Test method __parse_config from DesiData

        In particular test error reporting and defaults loading
        """
        # create a DesiData instance
        # since DesiData is an abstract class, we create a DesiHealpix instance
        config = ConfigParser()
        config.read_dict({"data": {
            "catalogue": f"{THIS_DIR}/data/QSO_cat_fuji_dark_healpix.fits",
            "keep surveys": "all special",
            "input directory": f"{THIS_DIR}/data/",
            "out dir": f"{THIS_DIR}/results/",
            "num processors": 1,
        }})
        for key, value in defaults_desi_healpix.items():
            if key not in config["data"]:
                config["data"][key] = str(value)

        data = DesiHealpix(config["data"])

        # run __parse_config with missing 'blinding'
        config = ConfigParser()
        config.read_dict({"data": {
                        }})
        expected_message = "Missing argument 'unblind' required by DesiData"
        with self.assertRaises(DataError) as context_manager:
            data._DesiData__parse_config(config["data"])
        self.compare_error_message(context_manager, expected_message)

        # run __parse_config with missing 'use_non_coadded_spectra'
        config = ConfigParser()
        config.read_dict({"data": {
            "unblind": "True",
                        }})
        expected_message = (
            "Missing argument 'use non-coadded spectra' required by DesiData"
        )
        with self.assertRaises(DataError) as context_manager:
            data._DesiData__parse_config(config["data"])
        self.compare_error_message(context_manager, expected_message)

        # check the defaults loading
        config = ConfigParser()
        config.read_dict({"data": {
                        }})
        for key, value in defaults_desi_data.items():
            if key not in config["data"]:
                config["data"][key] = str(value)
        for key, value in default_config.get("general").items():
            if key in accepted_options_desi_data and key not in config["data"]:
                config["data"][key] = str(value)
        data._DesiData__parse_config(config["data"])

    def test_desi_data_set_blinding(self):
        """Test method set_blinding of DesiData"""
        # create a DesiData instance with sv data only and blinding = none
        # since DesiData is an abstract class, we create a DesiHealpix instance
        config = ConfigParser()
        config.read_dict({"data": {
            "catalogue": f"{THIS_DIR}/data/QSO_cat_fuji_dark_healpix.fits",
            "keep surveys": "all special",
            "input directory": f"{THIS_DIR}/data/",
            "out dir": f"{THIS_DIR}/results/",
            "num processors": 1,
            "blinding": "none",
        }})
        for key, value in defaults_desi_healpix.items():
            if key not in config["data"]:
                config["data"][key] = str(value)

        data = DesiHealpix(config["data"])
        self.assertTrue(data.blinding == "none")

        # create a DesiData instance with sv data only and blinding = desi_m2
        # since DesiData is an abstract class, we create a DesiHealpix instance
        config = ConfigParser()
        config.read_dict({"data": {
            "catalogue": f"{THIS_DIR}/data/QSO_cat_fuji_dark_healpix.fits",
            "keep surveys": "all special",
            "input directory": f"{THIS_DIR}/data/",
            "out dir": f"{THIS_DIR}/results/",
            "num processors": 1,
            "blinding": "desi_m2",
        }})
        for key, value in defaults_desi_healpix.items():
            if key not in config["data"]:
                config["data"][key] = str(value)

        data = DesiHealpix(config["data"])
        self.assertTrue(data.blinding == "none")

        # create a DesiData instance with main data and blinding = none
        # since DesiData is an abstract class, we create a DesiHealpix instance
        config = ConfigParser()
        config.read_dict({"data": {
            "catalogue": f"{THIS_DIR}/data/QSO_cat_fuji_dark_healpix_with_main.fits",
            "keep surveys": "all special",
            "input directory": f"{THIS_DIR}/data/",
            "out dir": f"{THIS_DIR}/results/",
            "num processors": 1,
            "blinding": "none",
        }})
        for key, value in defaults_desi_healpix.items():
            if key not in config["data"]:
                config["data"][key] = str(value)

        data = DesiHealpix(config["data"])
        self.assertTrue(data.blinding == "none")

        # create a DesiData instance with main data and blinding = desi_m2
        # since DesiData is an abstract class, we create a DesiHealpix instance
        config = ConfigParser()
        config.read_dict({"data": {
            "catalogue": f"{THIS_DIR}/data/QSO_cat_fuji_dark_healpix_with_main.fits",
            "keep surveys": "all special",
            "input directory": f"{THIS_DIR}/data/",
            "out dir": f"{THIS_DIR}/results/",
            "num processors": 1,
            "blinding": "desi_m2",
        }})
        for key, value in defaults_desi_healpix.items():
            if key not in config["data"]:
                config["data"][key] = str(value)

        data = DesiHealpix(config["data"])
        self.assertTrue(data.blinding == "none")

        # create a DesiData instance with mock data and blinding = desi_m2
        # since DesiData is an abstract class, we create a DesisimMocks instance
        config = ConfigParser()
        config.read_dict({"data": {
            "catalogue": f"{THIS_DIR}/data/desi_mock_test_catalogue.fits",
            "input directory": f"{THIS_DIR}/data/",
            "out dir": f"{THIS_DIR}/results/",
            "num processors": 1,
            "blinding": "desi_m2",
        }})
        for key, value in defaults_desisim_mocks.items():
            if key not in config["data"]:
                config["data"][key] = str(value)

        data = DesisimMocks(config["data"])
        self.assertTrue(data.blinding == "none")

        # create a DesiData instance with main data and blinding = none
        # since DesiData is an abstract class, we create a DesiHealpix instance
        config = ConfigParser()
        config.read_dict({"data": {
            "catalogue": f"{THIS_DIR}/data/desi_mock_test_catalogue.fits",
            "input directory": f"{THIS_DIR}/data/",
            "out dir": f"{THIS_DIR}/results/",
            "num processors": 1,
            "blinding": "none",
        }})
        for key, value in defaults_desisim_mocks.items():
            if key not in config["data"]:
                config["data"][key] = str(value)

        data = DesisimMocks(config["data"])
        self.assertTrue(data.blinding == "none")

    def test_desi_healpix(self):
        """Test DesiHealpix"""
        # case: BAO 3D
        config = ConfigParser()
        config.read_dict({"data": {
            "catalogue": f"{THIS_DIR}/data/QSO_cat_fuji_dark_healpix.fits.gz",
            "keep surveys": "all",
            "input directory": f"{THIS_DIR}/data/",
            "out dir": f"{THIS_DIR}/results/",
            "num processors": 1,
            "analysis type": "BAO 3D"
        }})
        for key, value in defaults_desi_healpix.items():
            if key not in config["data"]:
                config["data"][key] = str(value)

        data = DesiHealpix(config["data"])

        self.assertTrue(len(data.forests) == 63)
        self.assertTrue(data.min_num_pix == 50)
        self.assertTrue(data.analysis_type == "BAO 3D")
        self.assertTrue(
            all(isinstance(forest, DesiForest) for forest in data.forests))

        # case: Pk 1D
        config = ConfigParser()
        config.read_dict({"data": {
            "catalogue": f"{THIS_DIR}/data/QSO_cat_fuji_dark_healpix.fits.gz",
            "keep surveys": "all",
            "input directory": f"{THIS_DIR}/data/",
            "out dir": f"{THIS_DIR}/results/",
            "num processors": 1,
            "analysis type": "PK 1D"
        }})
        for key, value in defaults_desi_healpix.items():
            if key not in config["data"]:
                config["data"][key] = str(value)

        data = DesiHealpix(config["data"])

        self.assertTrue(len(data.forests) == 63)
        self.assertTrue(data.min_num_pix == 50)
        self.assertTrue(data.analysis_type == "PK 1D")
        self.assertTrue(
            all(isinstance(forest, DesiPk1dForest) for forest in data.forests))

    def test_desi_healpix_parse_config(self):
        """Test method __parse_config from DesiHealpix"""

        # create a DesiHealpix with missing Data options
        config = ConfigParser()
        config.read_dict({"data": {
            "catalogue": f"{THIS_DIR}/data/QSO_cat_fuji_dark_healpix.fits.gz",
            "keep surveys": "all",
            "input directory": f"{THIS_DIR}/data/",
            "out dir": f"{THIS_DIR}/results/",
            "use non-coadded spectra": False,
            "num processors": 1,
        }})
        expected_message = (
            "Missing argument 'wave solution' required by Data"
        )
        with self.assertRaises(DataError) as context_manager:
            DesiHealpix(config["data"])
        self.compare_error_message(context_manager, expected_message)

        # create a DesiHealpix with missing DesiData options
        config = ConfigParser()
        config.read_dict({"data": {
            "catalogue": f"{THIS_DIR}/data/QSO_cat_fuji_dark_healpix.fits.gz",
            "keep surveys": "all",
            "input directory": f"{THIS_DIR}/data/",
            "out dir": f"{THIS_DIR}/results/",
            "use non-coadded spectra": False,
            "num processors": 1,
            "wave solution": "lin",
            "delta lambda": 0.8,
            "lambda max": 5500.0,
            "lambda max rest frame": 1200.0,
            "lambda min": 3600.0,
            "lambda min rest frame": 1040.0,
            "analysis type": "BAO 3D",
            "minimum number pixels in forest": 50,
            "rejection log file": "rejection.fits",
            "minimal snr bao3d": 0.0,
            "save format": "BinTableHDU",
        }})
        expected_message = (
            "Missing argument 'unblind' required by DesiData"
        )
        with self.assertRaises(DataError) as context_manager:
            DesiHealpix(config["data"])
        self.compare_error_message(context_manager, expected_message)

    def test_desi_healpix_read_data(self):
        """Test method read_data from DesiHealpix"""
        # run with one processor; case: only sv data
        config = ConfigParser()
        config.read_dict({"data": {
            "catalogue": f"{THIS_DIR}/data/QSO_cat_fuji_dark_healpix.fits.gz",
            "keep surveys": "all",
            "input directory": f"{THIS_DIR}/data/",
            "out dir": f"{THIS_DIR}/results/",
            "num processors": 1,
        }})
        for key, value in defaults_desi_healpix.items():
            if key not in config["data"]:
                config["data"][key] = str(value)

        data = DesiHealpix(config["data"])

        self.assertTrue(len(data.forests) == 63)

        # run with 0 processors; case: only sv data
        config = ConfigParser()
        config.read_dict({"data": {
            "catalogue": f"{THIS_DIR}/data/QSO_cat_fuji_dark_healpix.fits.gz",
            "keep surveys": "all",
            "input directory": f"{THIS_DIR}/data/",
            "out dir": f"{THIS_DIR}/results/",
            "num processors": 0,
        }})
        for key, value in defaults_desi_healpix.items():
            if key not in config["data"]:
                config["data"][key] = str(value)

        data = DesiHealpix(config["data"])

        self.assertTrue(len(data.forests) == 63)

        # run with 2 processors; case: only sv data
        config = ConfigParser()
        config.read_dict({"data": {
            "catalogue": f"{THIS_DIR}/data/QSO_cat_fuji_dark_healpix.fits.gz",
            "keep surveys": "all",
            "input directory": f"{THIS_DIR}/data/",
            "out dir": f"{THIS_DIR}/results/",
            "num processors": 2,
        }})
        for key, value in defaults_desi_healpix.items():
            if key not in config["data"]:
                config["data"][key] = str(value)

        data = DesiHealpix(config["data"])

        self.assertTrue(len(data.forests) == 63)

        # run with one processor; case: only sv data, select sv2 (no quasars)
        config = ConfigParser()
        config.read_dict({"data": {
            "catalogue": f"{THIS_DIR}/data/QSO_cat_fuji_dark_healpix.fits.gz",
            "keep surveys": "sv2",
            "input directory": f"{THIS_DIR}/data/",
            "out dir": f"{THIS_DIR}/results/",
            "num processors": 1,
        }})
        for key, value in defaults_desi_healpix.items():
            if key not in config["data"]:
                config["data"][key] = str(value)

        expected_message = "Empty quasar catalogue. Revise filtering choices"
        with self.assertRaises(QuasarCatalogueError) as context_manager:
            data = DesiHealpix(config["data"])
        self.compare_error_message(context_manager, expected_message)

        # run with one processor; case: only sv data, select sv1
        config = ConfigParser()
        config.read_dict({"data": {
            "catalogue": f"{THIS_DIR}/data/QSO_cat_fuji_dark_healpix.fits.gz",
            "keep surveys": "sv1",
            "input directory": f"{THIS_DIR}/data/",
            "out dir": f"{THIS_DIR}/results/",
            "num processors": 1,
        }})
        for key, value in defaults_desi_healpix.items():
            if key not in config["data"]:
                config["data"][key] = str(value)

        data = DesiHealpix(config["data"])

        self.assertTrue(len(data.forests) == 62)

        # run with one processor; case: main data present
        config = ConfigParser()
        config.read_dict({"data": {
            "catalogue": f"{THIS_DIR}/data/QSO_cat_fuji_dark_healpix_with_main.fits.gz",
            "keep surveys": "all",
            "input directory": f"{THIS_DIR}/data/",
            "out dir": f"{THIS_DIR}/results/",
            "num processors": 1,
        }})
        for key, value in defaults_desi_healpix.items():
            if key not in config["data"]:
                config["data"][key] = str(value)

        data = DesiHealpix(config["data"])

        self.assertTrue(len(data.forests) == 63)

        # run with 0 processors; case: main data present
        config = ConfigParser()
        config.read_dict({"data": {
            "catalogue": f"{THIS_DIR}/data/QSO_cat_fuji_dark_healpix_with_main.fits.gz",
            "keep surveys": "all",
            "input directory": f"{THIS_DIR}/data/",
            "out dir": f"{THIS_DIR}/results/",
            "num processors": 0,
        }})
        for key, value in defaults_desi_healpix.items():
            if key not in config["data"]:
                config["data"][key] = str(value)

        data = DesiHealpix(config["data"])

        self.assertTrue(len(data.forests) == 63)

        # run with 2 processors; case: main data present
        config = ConfigParser()
        config.read_dict({"data": {
            "catalogue": f"{THIS_DIR}/data/QSO_cat_fuji_dark_healpix_with_main.fits.gz",
            "keep surveys": "all",
            "input directory": f"{THIS_DIR}/data/",
            "out dir": f"{THIS_DIR}/results/",
            "num processors": 2,
        }})
        for key, value in defaults_desi_healpix.items():
            if key not in config["data"]:
                config["data"][key] = str(value)

        data = DesiHealpix(config["data"])

        self.assertTrue(len(data.forests) == 63)

        # case: data missing
        config = ConfigParser()
        config.read_dict({"data": {
            "catalogue": f"{THIS_DIR}/data/QSO_cat_fuji_dark_healpix_with_main.fits.gz",
            "keep surveys": "all",
            "input directory": f"{THIS_DIR}/data/fake/",
            "out dir": f"{THIS_DIR}/results/",
            "num processors": 1,
        }})
        for key, value in defaults_desi_healpix.items():
            if key not in config["data"]:
                config["data"][key] = str(value)

        expected_message = "No quasars found, stopping here"
        with self.assertRaises(DataError) as context_manager:
            data = DesiHealpix(config["data"])
        self.compare_error_message(context_manager, expected_message)

    def test_desi_healpix_read_file(self):
        """Test method read_file from DesiHealpix"""
        # first load create a data instance
        config = ConfigParser()
        config.read_dict({"data": {
            "catalogue": f"{THIS_DIR}/data/QSO_cat_fuji_dark_healpix.fits.gz",
            "keep surveys": "all",
            "input directory": f"{THIS_DIR}/data/",
            "out dir": f"{THIS_DIR}/results/",
            "num processors": 1,
            "analysis type": "PK 1D",
        }})
        for key, value in defaults_desi_healpix.items():
            if key not in config["data"]:
                config["data"][key] = str(value)

        data = DesiHealpix(config["data"])
        catalogue = data.catalogue
        pos = catalogue["TARGETID"] == "39632936152072660"

        # case: data without color Z and missing R_RESOLUTION
        filename = f"{THIS_DIR}/data/bad_format/spectra-main-dark-9144.fits"
        expected_message = (
            "Error while reading R band from /Users/iperez/Documents/GitHub/"
            "picca/py/picca/tests/delta_extraction/data/bad_format/spectra-main-"
            "dark-9144.fits. Analysis type is 'PK 1D', but file does not "
            "contain HDU 'R_RESOLUTION'"
        )
        self.check_read_file_error(data, catalogue[pos], filename,
                                   expected_message)

        # case: missing Z color
        filename = f"{THIS_DIR}/data/bad_format/missing_z_color.fits"
        expected_message = (
            "Missing Z band from /Users/iperez/Documents/GitHub/"
            "picca/py/picca/tests/delta_extraction/data/bad_format/"
            "missing_z_color.fits. Ignoring color."
        )
        self.check_read_file_error(data, catalogue[pos], filename,
                                   expected_message, warnings=True)

        # case: error reading B color
        filename = f"{THIS_DIR}/data/bad_format/missing_b_color.fits"
        expected_message = (
            "Error while reading B band from /Users/iperez/Documents/GitHub/"
            "picca/py/picca/tests/delta_extraction/data/bad_format/"
            "missing_b_color.fits. Ignoring color."
        )
        self.check_read_file_error(data, catalogue[pos], filename,
                                   expected_message, warnings=True)

        # case: missing file
        filename = "missing.fits"
        expected_message = "Error reading 'missing.fits'. Ignoring file"
        self.check_read_file_error(data, catalogue[pos], filename,
                                   expected_message, warnings=True)

        # TODO: Add tests for use_non_coadded_spectra=True

    def test_desi_tile(self):
        """Test DesiTile"""
        # load DesiTile using coadds
        config = ConfigParser()
        config.read_dict({"data": {
            "catalogue": f"{THIS_DIR}/data/QSO_cat_fuji_dark_tile.fits.gz",
            "input directory": f"{THIS_DIR}/data/tile/cumulative",
            "out dir": f"{THIS_DIR}/results/",
            "num processors": 1,
        }})
        for key, value in defaults_desi_tile.items():
            if key not in config["data"]:
                config["data"][key] = str(value)

        data = DesiTile(config["data"])

        self.assertTrue(len(data.forests) == 10)

        # load DesiTile using spectra
        config = ConfigParser()
        config.read_dict({"data": {
            "catalogue": f"{THIS_DIR}/data/QSO_cat_fuji_dark_tile.fits.gz",
            "input directory": f"{THIS_DIR}/data/tile/cumulative",
            "out dir": f"{THIS_DIR}/results/",
            "use non-coadded spectra": False,
            "num processors": 1,
        }})
        for key, value in defaults_desi_tile.items():
            if key not in config["data"]:
                config["data"][key] = str(value)

        data = DesiTile(config["data"])

        self.assertTrue(len(data.forests) == 10)

    def test_desi_tile_parse_config(self):
        """Test method __parse_config from DesiTile"""
        # create a DesiTile with missing Data options
        config = ConfigParser()
        config.read_dict({"data": {
            "catalogue": f"{THIS_DIR}/data/QSO_cat_fuji_dark_healpix.fits.gz",
            "keep surveys": "all",
            "input directory": f"{THIS_DIR}/data/",
            "out dir": f"{THIS_DIR}/results/",
            "use non-coadded spectra": False,
            "num processors": 1,
        }})
        expected_message = (
            "Missing argument 'wave solution' required by Data"
        )
        with self.assertRaises(DataError) as context_manager:
            DesiTile(config["data"])
        self.compare_error_message(context_manager, expected_message)

    def test_desi_tile_read_data(self):
        """Test method read_data from DesiTile"""
        # run with one processor; case: using coadds
        config = ConfigParser()
        config.read_dict({"data": {
            "catalogue": f"{THIS_DIR}/data/QSO_cat_fuji_dark_tile.fits.gz",
            "input directory": f"{THIS_DIR}/data/tile/cumulative",
            "out dir": f"{THIS_DIR}/results/",
            "num processors": 1,
        }})
        for key, value in defaults_desi_tile.items():
            if key not in config["data"]:
                config["data"][key] = str(value)

        data = DesiTile(config["data"])

        self.assertTrue(len(data.forests) == 10)

        # run with 0 processors; case: using coadds
        config = ConfigParser()
        config.read_dict({"data": {
            "catalogue": f"{THIS_DIR}/data/QSO_cat_fuji_dark_tile.fits.gz",
            "input directory": f"{THIS_DIR}/data/tile/cumulative",
            "out dir": f"{THIS_DIR}/results/",
            "num processors": 0,
        }})
        for key, value in defaults_desi_tile.items():
            if key not in config["data"]:
                config["data"][key] = str(value)

        data = DesiTile(config["data"])

        self.assertTrue(len(data.forests) == 10)

        # run with 2 processors; case: using coadds
        config = ConfigParser()
        config.read_dict({"data": {
            "catalogue": f"{THIS_DIR}/data/QSO_cat_fuji_dark_tile.fits.gz",
            "input directory": f"{THIS_DIR}/data/tile/cumulative",
            "out dir": f"{THIS_DIR}/results/",
            "num processors": 2,
        }})
        for key, value in defaults_desi_tile.items():
            if key not in config["data"]:
                config["data"][key] = str(value)

        data = DesiTile(config["data"])

        self.assertTrue(len(data.forests) == 10)

        # run with one processor; case: using individual spectra
        config = ConfigParser()
        config.read_dict({"data": {
            "catalogue": f"{THIS_DIR}/data/QSO_cat_fuji_dark_tile.fits.gz",
            "input directory": f"{THIS_DIR}/data/tile/cumulative",
            "out dir": f"{THIS_DIR}/results/",
            "use non-coadded spectra": False,
            "num processors": 1,
        }})
        for key, value in defaults_desi_tile.items():
            if key not in config["data"]:
                config["data"][key] = str(value)

        data = DesiTile(config["data"])

        self.assertTrue(len(data.forests) == 10)

        # run with 0 processors; case: using individual spectra
        config = ConfigParser()
        config.read_dict({"data": {
            "catalogue": f"{THIS_DIR}/data/QSO_cat_fuji_dark_tile.fits.gz",
            "input directory": f"{THIS_DIR}/data/tile/cumulative",
            "out dir": f"{THIS_DIR}/results/",
            "use non-coadded spectra": False,
            "num processors": 0,
        }})
        for key, value in defaults_desi_tile.items():
            if key not in config["data"]:
                config["data"][key] = str(value)

        data = DesiTile(config["data"])

        self.assertTrue(len(data.forests) == 10)

        # run with 2 processors; case: using individual spectra
        config = ConfigParser()
        config.read_dict({"data": {
            "catalogue": f"{THIS_DIR}/data/QSO_cat_fuji_dark_tile.fits.gz",
            "input directory": f"{THIS_DIR}/data/tile/cumulative",
            "out dir": f"{THIS_DIR}/results/",
            "use non-coadded spectra": False,
            "num processors": 2,
        }})
        for key, value in defaults_desi_tile.items():
            if key not in config["data"]:
                config["data"][key] = str(value)

        data = DesiTile(config["data"])

        self.assertTrue(len(data.forests) == 10)

    def test_desisim_mocks(self):
        """Test DesisimMocks"""
        # case: BAO 3D
        config = ConfigParser()
        config.read_dict({"data": {
            "catalogue": f"{THIS_DIR}/data/desi_mock_test_catalogue.fits",
            "keep surveys": "all",
            "input directory": f"{THIS_DIR}/data/",
            "out dir": f"{THIS_DIR}/results/",
            "num processors": 1,
            "analysis type": "BAO 3D"
        }})
        for key, value in defaults_desisim_mocks.items():
            if key not in config["data"]:
                config["data"][key] = str(value)

        data = DesisimMocks(config["data"])

        self.assertTrue(len(data.forests) == 194)
        self.assertTrue(data.min_num_pix == 50)
        self.assertTrue(data.analysis_type == "BAO 3D")
        self.assertTrue(
            all(isinstance(forest, DesiForest) for forest in data.forests))

        # case: Pk 1D
        config = ConfigParser()
        config.read_dict({"data": {
            "catalogue": f"{THIS_DIR}/data/desi_mock_test_catalogue.fits",
            "keep surveys": "all",
            "input directory": f"{THIS_DIR}/data/",
            "out dir": f"{THIS_DIR}/results/",
            "num processors": 1,
            "analysis type": "PK 1D"
        }})
        for key, value in defaults_desisim_mocks.items():
            if key not in config["data"]:
                config["data"][key] = str(value)

        data = DesisimMocks(config["data"])

        self.assertTrue(len(data.forests) == 194)
        self.assertTrue(data.min_num_pix == 50)
        self.assertTrue(data.analysis_type == "PK 1D")
        self.assertTrue(
            all(isinstance(forest, DesiPk1dForest) for forest in data.forests))

    def test_desisim_mocks_read_data(self):
        """Test method read_data from DesisimMocks"""
        # run with one processor
        config = ConfigParser()
        config.read_dict({"data": {
            "catalogue": f"{THIS_DIR}/data/desi_mock_test_catalogue.fits",
            "keep surveys": "all",
            "input directory": f"{THIS_DIR}/data/",
            "out dir": f"{THIS_DIR}/results/",
            "num processors": 1,
            "analysis type": "BAO 3D"
        }})
        for key, value in defaults_desisim_mocks.items():
            if key not in config["data"]:
                config["data"][key] = str(value)

        data = DesisimMocks(config["data"])

        self.assertTrue(len(data.forests) == 194)

        # run with 0 processors
        config = ConfigParser()
        config.read_dict({"data": {
            "catalogue": f"{THIS_DIR}/data/desi_mock_test_catalogue.fits",
            "keep surveys": "all",
            "input directory": f"{THIS_DIR}/data/",
            "out dir": f"{THIS_DIR}/results/",
            "num processors": 0,
            "analysis type": "BAO 3D"
        }})
        for key, value in defaults_desisim_mocks.items():
            if key not in config["data"]:
                config["data"][key] = str(value)

        data = DesisimMocks(config["data"])

        self.assertTrue(len(data.forests) == 194)

        # run with 2 processors
        config = ConfigParser()
        config.read_dict({"data": {
            "catalogue": f"{THIS_DIR}/data/desi_mock_test_catalogue.fits",
            "keep surveys": "all",
            "input directory": f"{THIS_DIR}/data/",
            "out dir": f"{THIS_DIR}/results/",
            "num processors": 2,
            "analysis type": "BAO 3D"
        }})
        for key, value in defaults_desisim_mocks.items():
            if key not in config["data"]:
                config["data"][key] = str(value)

        data = DesisimMocks(config["data"])

        self.assertTrue(len(data.forests) == 194)

        # case: data missing
        config = ConfigParser()
        config.read_dict({"data": {
            "catalogue": f"{THIS_DIR}/data/desi_mock_test_catalogue.fits",
            "keep surveys": "all",
            "input directory": f"{THIS_DIR}/data/fake/",
            "out dir": f"{THIS_DIR}/results/",
            "num processors": 1,
        }})
        for key, value in defaults_desisim_mocks.items():
            if key not in config["data"]:
                config["data"][key] = str(value)

        expected_message = "No quasars found, stopping here"
        with self.assertRaises(DataError) as context_manager:
            data = DesisimMocks(config["data"])
        self.compare_error_message(context_manager, expected_message)

    def test_sdss_data_filter_forest(self):
        """Test SdssData when run in spec mode"""
        config = ConfigParser()
        data_kwargs = sdss_data_kwargs_filter_forest.copy()
        config.read_dict({
            "data": data_kwargs
        })
        for key, value in defaults_sdss_data.items():
            if key not in config["data"]:
                config["data"][key] = str(value)
        data = SdssData(config["data"])

        self.assertTrue(len(data.forests) == 24)
        self.assertTrue(data.min_num_pix == 50)
        self.assertTrue(data.analysis_type == "BAO 3D")
        self.assertTrue(
            all(isinstance(forest, SdssForest) for forest in data.forests))

        # filter forests
        data.filter_forests()
        self.assertTrue(len(data.forests) == 22)

    def test_sdss_data_parse_config(self):
        """Test method __parse_config from SdssData

        In particular test error reporting and defaults loading
        """
        config = ConfigParser()
        data_kwargs = sdss_data_kwargs.copy()
        data_kwargs.update({"mode": "spec"})
        config.read_dict({
            "data": data_kwargs
        })
        for key, value in defaults_sdss_data.items():
            if key not in config["data"]:
                config["data"][key] = str(value)
        data = SdssData(config["data"])

        # create a config file with missing options
        config = ConfigParser()
        config.read_dict({"data": {
                        }})

        # run __parse_config with missing 'mode'
        expected_message = "Missing argument 'mode' required by SdssData"
        with self.assertRaises(DataError) as context_manager:
            data._SdssData__parse_config(config["data"])
        self.compare_error_message(context_manager, expected_message)

        # run __parse_config with missing 'mode'
        config["data"]["mode"] = "spec"
        expected_message = "Missing argument 'rebin' required by SdssData"
        with self.assertRaises(DataError) as context_manager:
            data._SdssData__parse_config(config["data"])
        self.compare_error_message(context_manager, expected_message)

        # check the defaults loading
        config = ConfigParser()
        config.read_dict({"data": {
                        }})
        for key, value in defaults_sdss_data.items():
            if key not in config["data"]:
                config["data"][key] = str(value)
        data._SdssData__parse_config(config["data"])

    def test_sdss_data_spec(self):
        """Test SdssData when run in spec mode"""
        config = ConfigParser()
        data_kwargs = sdss_data_kwargs.copy()
        data_kwargs.update({"mode": "spec"})
        config.read_dict({
            "data": data_kwargs
        })
        for key, value in defaults_sdss_data.items():
            if key not in config["data"]:
                config["data"][key] = str(value)
        data = SdssData(config["data"])

        self.assertTrue(len(data.forests) == 43)
        self.assertTrue(data.min_num_pix == 50)
        self.assertTrue(data.analysis_type == "BAO 3D")
        self.assertTrue(
            all(isinstance(forest, SdssForest) for forest in data.forests))

    def test_sdss_data_spplate(self):
        """Tests SdssData when run in spplate mode"""
        # using default  value for 'mode'
        config = ConfigParser()
        config.read_dict({
            "data": sdss_data_kwargs
        })
        for key, value in defaults_sdss_data.items():
            if key not in config["data"]:
                config["data"][key] = str(value)
        data = SdssData(config["data"])

        self.assertTrue(len(data.forests) == 43)
        self.assertTrue(data.min_num_pix == 50)
        self.assertTrue(data.analysis_type == "BAO 3D")
        self.assertTrue(
            all(isinstance(forest, SdssForest) for forest in data.forests))

        # specifying 'mode'
        config = ConfigParser()
        data_kwargs = sdss_data_kwargs.copy()
        data_kwargs.update({"mode": "spplate"})
        config.read_dict({
            "data": data_kwargs
        })
        for key, value in defaults_sdss_data.items():
            if key not in config["data"]:
                config["data"][key] = str(value)
        data = SdssData(config["data"])

        self.assertTrue(len(data.forests) == 43)
        self.assertTrue(data.min_num_pix == 50)
        self.assertTrue(
            all(isinstance(forest, SdssForest) for forest in data.forests))


if __name__ == '__main__':
    unittest.main()
