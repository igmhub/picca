"""This file contains tests related to Data and its childs"""
from configparser import ConfigParser
import os
import unittest

from astropy.table import Table

from picca.delta_extraction.errors import QuasarCatalogueError
from picca.delta_extraction.quasar_catalogue import QuasarCatalogue
from picca.delta_extraction.quasar_catalogues.drq_catalogue import DrqCatalogue
from picca.delta_extraction.quasar_catalogues.drq_catalogue import defaults as defaults_drq
from picca.delta_extraction.tests.abstract_test import AbstractTest
from picca.delta_extraction.tests.test_utils import reset_logger
from picca.delta_extraction.utils import setup_logger

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class QuasarCatalogueTest(AbstractTest):
    """Test the quasar catalogue.

    Methods
    -------
    compare_ascii (from AbstractTest)
    compare_fits (from AbstractTest)
    setUp (from AbstractTest)
    test_drq_catalogue
    test_quasar_catalogue
    test_quasar_catalogue_trim_catalogue
    test_ztruth_catalogue
    """
    def test_drq_catalogue(self):
        """Load a DrqCatalogue"""
        out_file = f"{THIS_DIR}/results/drq_catalogue_print.txt"
        test_file = f"{THIS_DIR}/data/drq_catalogue_print.txt"

        # setup printing
        setup_logger(log_file=out_file)

        # Case 0: missing redshift variables
        config = ConfigParser()
        config.read_dict({
            "data": {
                "drq catalogue":
                    f"{THIS_DIR}/data/cat_for_clustering_plate3655.fits.gz",
            }
        })
        for key, value in defaults_drq.items():
            if key not in config["data"]:
                config["data"][key] = str(value)

        with self.assertRaises(QuasarCatalogueError):
            quasar_catalogue = DrqCatalogue(config["data"])

        # Case 1: missing spAll file
        config = ConfigParser()
        config.read_dict({
            "data": {
                "drq catalogue":
                    f"{THIS_DIR}/data/cat_for_clustering_plate3655.fits.gz",
                "z max": 3.5,
                "z min": 2.1,
            }
        })
        for key, value in defaults_drq.items():
            if key not in config["data"]:
                config["data"][key] = str(value)
        with self.assertRaises(QuasarCatalogueError):
            quasar_catalogue = DrqCatalogue(config["data"])

        # case 2: finding spAll file
        config = ConfigParser()
        config.read_dict({
            "data": {
                "drq catalogue":
                    f"{THIS_DIR}/data/cat_for_clustering_plate3655.fits.gz",
                "input directory":
                    f"{THIS_DIR}/data/",
                "z max": 3.5,
                "z min": 2.1,
            }
        })
        for key, value in defaults_drq.items():
            if key not in config["data"]:
                config["data"][key] = str(value)
        quasar_catalogue = DrqCatalogue(config["data"])

        # case 3: with spAll file
        config = ConfigParser()
        config.read_dict({
            "data": {
                "drq catalogue":
                    f"{THIS_DIR}/data/cat_for_clustering_plate3655.fits.gz",
                "spAll":
                    f"{THIS_DIR}/data/spAll-plate3655.fits",
                "z max": 3.5,
                "z min": 2.1,
            }
        })
        for key, value in defaults_drq.items():
            if key not in config["data"]:
                config["data"][key] = str(value)
        quasar_catalogue = DrqCatalogue(config["data"])

        # reset printing
        reset_logger()
        self.compare_ascii(test_file, out_file)

        self.assertTrue(quasar_catalogue.catalogue is not None)
        self.assertTrue(quasar_catalogue.z_min == 2.1)
        self.assertTrue(quasar_catalogue.z_max == 3.5)
        self.assertTrue(quasar_catalogue.max_num_spec is None)


    def test_quasar_catalogue(self):
        """Test Abstract class QuasarCatalogue

        Load a QuasarCatalogue instace.
        """
        config = ConfigParser()
        config.read_dict({"data": {}})
        with self.assertRaises(QuasarCatalogueError):
            quasar_catalogue = QuasarCatalogue(config["data"])

        config = ConfigParser()
        config.read_dict(
            {"data": {
                "z min": 2.15,
                "z max": 3.2,
                "max num spec": 2
            }})
        quasar_catalogue = QuasarCatalogue(config["data"])

        self.assertTrue(quasar_catalogue.catalogue is None)
        self.assertTrue(quasar_catalogue.z_min == 2.15)
        self.assertTrue(quasar_catalogue.z_max == 3.2)
        self.assertTrue(quasar_catalogue.max_num_spec == 2)

    def test_quasar_catalogue_trim_catalogue(self):
        """Test method trim_catalogue from Abstract Class QuasarCatalogue"""
        ra = [0.15, 0.0]
        dec = [0.0, 0.0]
        catalogue = Table(data=[ra, dec], names=["RA", "DEC"])

        # load instance without maximum number of objects
        config = ConfigParser()
        config.read_dict(
            {"data": {
                "z min": 2.15,
                "z max": 3.2,
            }})
        quasar_catalogue = QuasarCatalogue(config["data"])
        quasar_catalogue.catalogue = catalogue.copy()
        # trimming function does nothing
        quasar_catalogue.trim_catalogue()
        self.assertTrue(all(quasar_catalogue.catalogue["RA"] == ra))
        self.assertTrue(all(quasar_catalogue.catalogue["DEC"] == dec))

        # load instance with a large maximum number of objects
        config = ConfigParser()
        config.read_dict(
            {"data": {
                "z min": 2.15,
                "z max": 3.2,
                "max num spec": 5
            }})
        quasar_catalogue = QuasarCatalogue(config["data"])
        quasar_catalogue.catalogue = catalogue.copy()
        # trimming function sorts the catalogue
        # max_num_spec is too large and it does not trim it
        quasar_catalogue.trim_catalogue()
        self.assertTrue(all(quasar_catalogue.catalogue["RA"] == ra[::-1]))
        self.assertTrue(all(quasar_catalogue.catalogue["DEC"] == dec[::-1]))

        # load instance with maximum number of objects
        config = ConfigParser()
        config.read_dict(
            {"data": {
                "z min": 2.15,
                "z max": 3.2,
                "max num spec": 1
            }})
        quasar_catalogue = QuasarCatalogue(config["data"])
        quasar_catalogue.catalogue = catalogue.copy()
        # trimming function sorts the catalogue
        # max_num_spec is too large and it does not trim it
        quasar_catalogue.trim_catalogue()
        self.assertTrue(all(quasar_catalogue.catalogue["RA"] == ra[::-1][:1]))
        self.assertTrue(all(quasar_catalogue.catalogue["DEC"] == dec[::-1][:1]))

    def test_ztruth_catalogue(self):
        """Load a ZtruthCatalogue"""
        # TODO: add test
        with self.assertRaises(Exception):
            raise NotImplementedError()

if __name__ == '__main__':
    unittest.main()
