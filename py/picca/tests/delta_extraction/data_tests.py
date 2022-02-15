"""This file contains tests related to Data and its childs"""
from configparser import ConfigParser
import os
import unittest

from picca.delta_extraction.astronomical_objects.sdss_forest import SdssForest
from picca.delta_extraction.data import Data
from picca.delta_extraction.data import defaults as defaults_data
from picca.delta_extraction.data_catalogues.sdss_data import SdssData
from picca.delta_extraction.data_catalogues.sdss_data import defaults as defaults_sdss_data
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
        """Test DesiData"""
        # TODO: add test

    def test_desi_healpix(self):
        """Test DesiHealpix"""
        # TODO: add test

    def test_desi_tile(self):
        """Test DesiTile"""
        # TODO: add test

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
