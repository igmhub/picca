"""This file contains tests related to Data and its childs"""
import os
import unittest
from configparser import ConfigParser

from picca.delta_extraction.astronomical_objects.sdss_forest import SdssForest
from picca.delta_extraction.data import Data
from picca.delta_extraction.data_catalogues.sdss_data import SdssData
from picca.delta_extraction.userprint import UserPrint
from picca.delta_extraction.tests.abstract_test import AbstractTest
from picca.delta_extraction.tests.test_utils import forest1

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class DataTest(AbstractTest):
    """Test class Data and its childs."""

    def test_data(self):
        """Test Abstract class Data

        Load a Data instace.
        """
        config = ConfigParser()
        config.read_dict({"data": {}})
        data = Data(config["data"])

        self.assertTrue(len(data.forests) == 0)
        self.assertTrue(data.min_num_pix == 50)

        config = ConfigParser()
        config.read_dict({"data": {"minimum number pixels in forest": 40}})
        data = Data(config["data"])

        self.assertTrue(len(data.forests) == 0)
        self.assertTrue(data.min_num_pix == 40)

    def test_data_filter_forests(self):
        """Test method filter_forests from Abstract Class Data"""
        out_file = f"{THIS_DIR}/results/data_filter_forests_print.txt"
        test_file = f"{THIS_DIR}/data/data_filter_forests_print.txt"

        # setup printing
        UserPrint.initialize_log(out_file)

        # create Data instance
        config = ConfigParser()
        config.read_dict({"data": {}})
        data = Data(config["data"])

        # add dummy forest
        data.forests.append(forest1)
        self.assertTrue(len(data.forests) == 1)

        # filter forests
        data.filter_forests()
        self.assertTrue(len(data.forests) == 1)

        # create Data instance with insane forest requirements
        config = ConfigParser()
        config.read_dict({"data": {"minimum number pixels in forest": 10000}})
        data = Data(config["data"])

        # add dummy forest
        data.forests.append(forest1)
        self.assertTrue(len(data.forests) == 1)

        # filter forests
        data.filter_forests()
        self.assertTrue(len(data.forests) == 0)

        # reset printing
        self.compare_ascii(test_file, out_file, expand_dir=True)
        UserPrint.reset_log()

    def test_sdss_data_spec(self):
        """Tests SdssData when run in spec mode"""
        config = ConfigParser()
        config.read_dict({
            "data": {
                "input directory":
                    f"{THIS_DIR}/data",
                "drq catalogue":
                    f"{THIS_DIR}/data/cat_for_clustering_plate3655.fits.gz",
                "mode":
                    "spec"
            }
        })
        data = SdssData(config["data"])

        self.assertTrue(len(data.forests) == 43)
        self.assertTrue(data.min_num_pix == 50)
        self.assertTrue(
            all(isinstance(forest, SdssForest) for forest in data.forests))

    def test_sdss_data_spplate(self):
        """Tests SdssData when run in spplate mode"""
        # using default  value for 'mode'
        config = ConfigParser()
        config.read_dict({
            "data": {
                "input directory":
                    f"{THIS_DIR}/data",
                "drq catalogue":
                    f"{THIS_DIR}/data/cat_for_clustering_plate3655.fits.gz",
            }
        })
        data = SdssData(config["data"])

        self.assertTrue(len(data.forests) == 43)
        self.assertTrue(data.min_num_pix == 50)
        self.assertTrue(
            all(isinstance(forest, SdssForest) for forest in data.forests))

        # specifying 'mode'
        config = ConfigParser()
        config.read_dict({
            "data": {
                "input directory":
                    f"{THIS_DIR}/data",
                "drq catalogue":
                    f"{THIS_DIR}/data/cat_for_clustering_plate3655.fits.gz",
                "mode":
                    "spplate",
            }
        })
        data = SdssData(config["data"])

        self.assertTrue(len(data.forests) == 43)
        self.assertTrue(data.min_num_pix == 50)
        self.assertTrue(
            all(isinstance(forest, SdssForest) for forest in data.forests))


if __name__ == '__main__':
    unittest.main()
