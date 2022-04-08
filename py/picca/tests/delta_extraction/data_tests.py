"""This file contains tests related to Data and its childs"""
from configparser import ConfigParser
import os
import unittest

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
from picca.delta_extraction.errors import DataError
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
        setup_logger(log_file=out_file)

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
        # since it is an abstract class.
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

        with self.assertRaises(DataError):
            try:
                DesiData(config["data"])
            except DataError as error:
                self.assertTrue(str(error) ==
                    "Function 'read_data' was not overloaded by child class")
                raise error

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
        with self.assertRaises(DataError):
            try:
                data._DesiData__parse_config(config["data"])
            except DataError as error:
                self.assertTrue(str(error) ==
                    "Missing argument 'blinding' required by DesiData")
                raise error

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
        with self.assertRaises(DataError):
            try:
                data._DesiData__parse_config(config["data"])
            except DataError as error:
                print(error)
                self.assertTrue(str(error) == (
                    "Unrecognized blinding strategy. Accepted strategies "
                    f"are {ACCEPTED_BLINDING_STRATEGIES}. Found 'invalid'"))
                raise error

    def test_desi_healpix(self):
        """Test DesiHealpix"""
        # setup printing
        setup_logger()

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

    def test_desi_tile(self):
        """Test DesiTile"""
        # setup printing
        setup_logger()

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
        with self.assertRaises(DataError):
            try:
                data._SdssData__parse_config(config["data"])
            except DataError as error:
                self.assertTrue(str(error) ==
                    "Missing argument 'mode' required by SdssData")
                raise error

        # run __parse_config with missing 'mode'
        config["data"]["mode"] = "spec"
        with self.assertRaises(DataError):
            try:
                data._SdssData__parse_config(config["data"])
            except DataError as error:
                self.assertTrue(str(error) ==
                    "Missing argument 'rebin' required by SdssData")
                raise error

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
