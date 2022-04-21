"""This file contains tests related to Data and its childs"""
from configparser import ConfigParser
import logging
import os
import unittest

from picca.delta_extraction.astronomical_objects.desi_forest import DesiForest
from picca.delta_extraction.astronomical_objects.desi_pk1d_forest import DesiPk1dForest
from picca.delta_extraction.astronomical_objects.sdss_forest import SdssForest
from picca.delta_extraction.data import Data
from picca.delta_extraction.data import defaults as defaults_data
from picca.delta_extraction.data_catalogues.desi_data import DesiData
from picca.delta_extraction.data_catalogues.desi_data import defaults as defaults_desi_data
from picca.delta_extraction.data_catalogues.desi_healpix import DesiHealpix
from picca.delta_extraction.data_catalogues.desi_healpix import defaults as defaults_desi_healpix
from picca.delta_extraction.data_catalogues.desi_tile import DesiTile
from picca.delta_extraction.data_catalogues.desi_tile import defaults as defaults_desi_tile
from picca.delta_extraction.data_catalogues.sdss_data import SdssData
from picca.delta_extraction.data_catalogues.sdss_data import defaults as defaults_sdss_data
from picca.delta_extraction.errors import DataError, QuasarCatalogueError
from picca.delta_extraction.utils import ACCEPTED_BLINDING_STRATEGIES
from picca.delta_extraction.utils import setup_logger
from picca.tests.delta_extraction.abstract_test import AbstractTest
from picca.tests.delta_extraction.test_utils import reset_logger
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
                         }})
        for key, value in defaults_data.items():
            if key not in config["data"]:
                config["data"][key] = str(value)
        data = Data(config["data"])

        self.assertTrue(len(data.forests) == 0)
        self.assertTrue(data.min_num_pix == 40)
        self.assertTrue(data.analysis_type == "BAO 3D")

    def test_data_filter_forests(self):
        """Test method filter_forests from Abstract Class Data"""
        out_file = f"{THIS_DIR}/results/data_filter_forests_print.txt"
        test_file = f"{THIS_DIR}/data/data_filter_forests_print.txt"

        # setup printing
        setup_logger(logging_level_console=logging.ERROR, log_file=out_file)

        # create Data instance
        config = ConfigParser()
        config.read_dict({"data": {
                            "out dir": f"{THIS_DIR}/results/",
                            "rejection log file": "rejection_log.fits.gz",
                            "wave solution": "log",
                            "delta log lambda": 3e-4,
                            "delta log lambda rest frame": 3e-4,
                            "input directory": f"{THIS_DIR}/data/",
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
        self.assertTrue(len(data.forests) == 1)

        # create Data instance with insane forest requirements
        config = ConfigParser()
        config.read_dict({"data": {"minimum number pixels in forest": 10000,
                                   "out dir": f"{THIS_DIR}/results/",
                                   "rejection log file": "rejection_log.fits.gz",
                                   "wave solution": "log",
                                   "delta log lambda": 3e-4,
                                   "delta log lambda rest frame": 3e-4,
                                   "input directory": f"{THIS_DIR}/data/",
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

        # reset printing
        reset_logger()
        self.compare_ascii(test_file, out_file)

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

        # create a config file with missing options
        config = ConfigParser()
        config.read_dict({"data": {
                        }})

        # run __parse_config with missing 'blinding'
        expected_message = "Missing argument 'blinding' required by DesiData"
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
        data._DesiData__parse_config(config["data"])

        # check loading with the wrong blinding
        config["data"]["blinding"] = "invalid"
        expected_message = (
            "Unrecognized blinding strategy. Accepted strategies "
            f"are {ACCEPTED_BLINDING_STRATEGIES}. Found 'invalid'")
        with self.assertRaises(DataError) as context_manager:
            data._DesiData__parse_config(config["data"])
        self.compare_error_message(context_manager, expected_message)

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

        # create a DesiData instance with sv data only and blinding = corr_yshift
        # since DesiData is an abstract class, we create a DesiHealpix instance
        config = ConfigParser()
        config.read_dict({"data": {
            "catalogue": f"{THIS_DIR}/data/QSO_cat_fuji_dark_healpix.fits",
            "keep surveys": "all special",
            "input directory": f"{THIS_DIR}/data/",
            "out dir": f"{THIS_DIR}/results/",
            "num processors": 1,
            "blinding": "corr_yshift",
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
        self.assertTrue(data.blinding == "corr_yshift")

        # create a DesiData instance with main data and blinding = corr_yshift
        # since DesiData is an abstract class, we create a DesiHealpix instance
        config = ConfigParser()
        config.read_dict({"data": {
            "catalogue": f"{THIS_DIR}/data/QSO_cat_fuji_dark_healpix_with_main.fits",
            "keep surveys": "all special",
            "input directory": f"{THIS_DIR}/data/",
            "out dir": f"{THIS_DIR}/results/",
            "num processors": 1,
            "blinding": "corr_yshift",
        }})
        for key, value in defaults_desi_healpix.items():
            if key not in config["data"]:
                config["data"][key] = str(value)

        data = DesiHealpix(config["data"])
        self.assertTrue(data.blinding == "corr_yshift")

        # TODO: add tests when loading mocks

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
        # create a DesiHealpix with missing use_non_coadded_spectra
        config = ConfigParser()
        config.read_dict({"data": {
            "catalogue": f"{THIS_DIR}/data/QSO_cat_fuji_dark_healpix.fits.gz",
            "keep surveys": "all",
            "input directory": f"{THIS_DIR}/data/",
            "out dir": f"{THIS_DIR}/results/",
            "num processors": 1,
        }})
        expected_message = (
            "Missing argument 'use non-coadded spectra' required by DesiHealpix"
        )
        with self.assertRaises(DataError) as context_manager:
            DesiHealpix(config["data"])
        self.compare_error_message(context_manager, expected_message)

        # create a DesiHealpix with missing num_processors
        config = ConfigParser()
        config.read_dict({"data": {
            "catalogue": f"{THIS_DIR}/data/QSO_cat_fuji_dark_healpix.fits.gz",
            "keep surveys": "all",
            "input directory": f"{THIS_DIR}/data/",
            "out dir": f"{THIS_DIR}/results/",
            "use non-coadded spectra": False,
        }})
        expected_message = (
            "Missing argument 'num processors' required by DesiHealpix"
        )
        with self.assertRaises(DataError) as context_manager:
            DesiHealpix(config["data"])
        self.compare_error_message(context_manager, expected_message)

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
        }})
        expected_message = (
            "Missing argument 'blinding' required by DesiData"
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
        # case: data without color Z and missing R_RESOLUTION
        config = ConfigParser()
        config.read_dict({"data": {
            "catalogue": f"{THIS_DIR}/data/QSO_cat_fuji_dark_healpix_with_main.fits.gz",
            "keep surveys": "all",
            "input directory": f"{THIS_DIR}/data/bad_format/",
            "out dir": f"{THIS_DIR}/results/",
            "num processors": 2,
            "analysis type": "PK 1D",
            "use non-coadded spectra": True,
        }})
        for key, value in defaults_desi_healpix.items():
            if key not in config["data"]:
                config["data"][key] = str(value)

        expected_message = (
            "Error while reading R band from /Users/iperez/Documents/GitHub/"
            "picca/py/picca/tests/delta_extraction/data/bad_format//main/"
            "dark/91/9144/spectra-main-dark-9144.fits. Analysis type is "
            "'PK 1D', but file does not contain HDU 'R_RESOLUTION'"
        )
        with self.assertRaises(DataError) as context_manager:
            data = DesiHealpix(config["data"])
        self.compare_error_message(context_manager, expected_message)

        # TODO: test Pk1d and mocks (R_RESOLUTION is looked for in truth file)

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
            "use non-coadded spectra": "False",
        }})
        for key, value in defaults_desi_tile.items():
            if key not in config["data"]:
                config["data"][key] = str(value)

        data = DesiTile(config["data"])

        self.assertTrue(len(data.forests) == 10)

    def test_desisim_mocks(self):
        """Test DesisimMocks"""
        # TODO: add test

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
