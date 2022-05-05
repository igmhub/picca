"""This file contains tests related to Data and its childs"""
from configparser import ConfigParser
import os
import unittest

from astropy.table import Table

from picca.delta_extraction.errors import QuasarCatalogueError
from picca.delta_extraction.quasar_catalogue import QuasarCatalogue
from picca.delta_extraction.quasar_catalogues.desi_quasar_catalogue import DesiQuasarCatalogue
from picca.delta_extraction.quasar_catalogues.desi_quasar_catalogue import defaults as defaults_desi_quasar_cat
from picca.delta_extraction.quasar_catalogues.drq_catalogue import DrqCatalogue
from picca.delta_extraction.quasar_catalogues.drq_catalogue import defaults as defaults_drq
from picca.tests.delta_extraction.abstract_test import AbstractTest
from picca.tests.delta_extraction.test_utils import reset_logger
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

        expected_message = (
            "Missing argument 'z min' required by QuasarCatalogue")
        with self.assertRaises(QuasarCatalogueError) as context_manager:
            quasar_catalogue = DrqCatalogue(config["data"])
        self.compare_error_message(context_manager, expected_message)

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
        expected_message = (
            "Missing argument 'spAll' required by DrqCatalogue")
        with self.assertRaises(QuasarCatalogueError) as context_manager:
            quasar_catalogue = DrqCatalogue(config["data"])
        self.compare_error_message(context_manager, expected_message)

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
        expected_message = (
            "Missing argument 'z min' required by QuasarCatalogue")
        with self.assertRaises(QuasarCatalogueError) as context_manager:
            quasar_catalogue = QuasarCatalogue(config["data"])
        self.compare_error_message(context_manager, expected_message)

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

    def test_quasar_catalogue_missing_options(self):
        """Test Abstract class QuasarCatalogue

        Load a QuasarCatalogue instace with missing options
        """
        # case: no zmin, but we can compute it
        config = ConfigParser()
        config.read_dict(
            {"data": {
                "z max": 3.2,
                "lambda min": 3600.0,
                "lambda min rest frame": 1040.0,
                "lambda max": 5500.0,
                "lambda max rest frame": 1200.0,
                "max num spec": 2,
            }})
        quasar_catalogue = QuasarCatalogue(config["data"])

        self.assertTrue(quasar_catalogue.catalogue is None)
        self.assertTrue(quasar_catalogue.z_min == 2.0)
        self.assertTrue(quasar_catalogue.z_max == 3.2)
        self.assertTrue(quasar_catalogue.max_num_spec == 2)

        # case: no zmin, cannot compute it
        config = ConfigParser()
        config.read_dict({"data": {}})
        expected_message = (
            "Missing argument 'z min' required by QuasarCatalogue")
        with self.assertRaises(QuasarCatalogueError) as context_manager:
            quasar_catalogue = QuasarCatalogue(config["data"])
        self.compare_error_message(context_manager, expected_message)

        # case: no zmax, but we can compute it
        config = ConfigParser()
        config.read_dict(
            {"data": {
                "z min": 2.0,
                "lambda min": 3600.0,
                "lambda min rest frame": 1040.0,
                "lambda max": 5500.0,
                "lambda max rest frame": 1200.0,
                "max num spec": 2,
            }})
        quasar_catalogue = QuasarCatalogue(config["data"])

        self.assertTrue(quasar_catalogue.catalogue is None)
        self.assertTrue(quasar_catalogue.z_min == 2.0)
        self.assertTrue(quasar_catalogue.z_max == 4.288461538461538)
        self.assertTrue(quasar_catalogue.max_num_spec == 2)

        # case: no zmin, cannot compute it
        config = ConfigParser()
        config.read_dict({"data": {
            "z min": 2.0,
        }})
        expected_message = (
            "Missing argument 'z max' required by QuasarCatalogue")
        with self.assertRaises(QuasarCatalogueError) as context_manager:
            quasar_catalogue = QuasarCatalogue(config["data"])
        self.compare_error_message(context_manager, expected_message)

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

    def test_desi_quasar_catalogue(self):
        """Load a DesiQuasarCatalogue"""
        out_file = f"{THIS_DIR}/results/desi_catalogue_print.txt"
        test_file = f"{THIS_DIR}/data/desi_catalogue_print.txt"

        # setup printing
        setup_logger(log_file=out_file)

        config = ConfigParser()
        config.read_dict(
            {"data": {
                "z min": 2.15,
                "z max": 3.2,
                "catalogue": f"{THIS_DIR}/data/dummy_desi_quasar_catalogue.fits",
                "keep surveys": "all special"
            }})

        for key, value in defaults_desi_quasar_cat.items():
            if key not in config["data"]:
                config["data"][key] = str(value)

        quasar_catalogue = DesiQuasarCatalogue(config["data"])

        # reset printing
        reset_logger()
        self.compare_ascii(test_file, out_file)

        self.assertTrue(quasar_catalogue.catalogue is not None)
        self.assertTrue(len(quasar_catalogue.catalogue) == 3)
        self.assertTrue(quasar_catalogue.z_min == 2.15)
        self.assertTrue(quasar_catalogue.z_max == 3.2)
        self.assertTrue(quasar_catalogue.max_num_spec is None)

    def test_desi_quasar_catalogue_filter_surveys(self):
        """Load a DesiQuasarCatalogue filtering by survey"""
        # setup printing
        setup_logger(log_file=None)

        # test filtering with one survey
        config = ConfigParser()
        config.read_dict(
            {"data": {
                "z min": 2.15,
                "z max": 3.2,
                "max_num_spec": 1,
                "catalogue": f"{THIS_DIR}/data/dummy_desi_quasar_catalogue.fits.gz",
                "keep surveys": "sv3"
            }})

        for key, value in defaults_desi_quasar_cat.items():
            if key not in config["data"]:
                config["data"][key] = str(value)

        quasar_catalogue = DesiQuasarCatalogue(config["data"])

        self.assertTrue(quasar_catalogue.catalogue is not None)
        self.assertTrue(len(quasar_catalogue.catalogue) == 1)
        self.assertTrue(quasar_catalogue.z_min == 2.15)
        self.assertTrue(quasar_catalogue.z_max == 3.2)
        self.assertTrue(quasar_catalogue.max_num_spec is None)
        self.assertTrue(len(quasar_catalogue.keep_surveys) == 1)
        self.assertTrue(quasar_catalogue.keep_surveys[0] == "sv3")

        # test filtering with two survey
        config = ConfigParser()
        config.read_dict(
            {"data": {
                "z min": 2.15,
                "z max": 3.2,
                "max_num_spec": 1,
                "catalogue": f"{THIS_DIR}/data/dummy_desi_quasar_catalogue.fits.gz",
                "keep surveys": "main sv3"
            }})

        for key, value in defaults_desi_quasar_cat.items():
            if key not in config["data"]:
                config["data"][key] = str(value)

        quasar_catalogue = DesiQuasarCatalogue(config["data"])

        self.assertTrue(quasar_catalogue.catalogue is not None)
        self.assertTrue(len(quasar_catalogue.catalogue) == 2)
        self.assertTrue(quasar_catalogue.z_min == 2.15)
        self.assertTrue(quasar_catalogue.z_max == 3.2)
        self.assertTrue(quasar_catalogue.max_num_spec is None)
        self.assertTrue(len(quasar_catalogue.keep_surveys) == 2)
        self.assertTrue(quasar_catalogue.keep_surveys[0] == "main")
        self.assertTrue(quasar_catalogue.keep_surveys[1] == "sv3")

        # now test the behaviour of all
        config = ConfigParser()
        config.read_dict(
            {"data": {
                "z min": 2.15,
                "z max": 3.2,
                "max_num_spec": 1,
                "catalogue": f"{THIS_DIR}/data/dummy_desi_quasar_catalogue.fits.gz",
                "keep surveys": "all"
            }})

        for key, value in defaults_desi_quasar_cat.items():
            if key not in config["data"]:
                config["data"][key] = str(value)

        quasar_catalogue = DesiQuasarCatalogue(config["data"])

        # reset printing after test
        reset_logger()

        self.assertTrue(quasar_catalogue.catalogue is not None)
        self.assertTrue(len(quasar_catalogue.catalogue) == 2)
        self.assertTrue(quasar_catalogue.z_min == 2.15)
        self.assertTrue(quasar_catalogue.z_max == 3.2)
        self.assertTrue(quasar_catalogue.max_num_spec is None)
        self.assertTrue(len(quasar_catalogue.keep_surveys) == 5)
        self.assertTrue(quasar_catalogue.keep_surveys[0] == "all")
        self.assertTrue(quasar_catalogue.keep_surveys[1] == "sv1")
        self.assertTrue(quasar_catalogue.keep_surveys[2] == "sv2")
        self.assertTrue(quasar_catalogue.keep_surveys[3] == "sv3")
        self.assertTrue(quasar_catalogue.keep_surveys[4] == "main")

        # now test the behaviour of sv1 + all
        config = ConfigParser()
        config.read_dict(
            {"data": {
                "z min": 2.15,
                "z max": 3.2,
                "max_num_spec": 1,
                "catalogue": f"{THIS_DIR}/data/dummy_desi_quasar_catalogue.fits.gz",
                "keep surveys": "sv1 all"
            }})

        for key, value in defaults_desi_quasar_cat.items():
            if key not in config["data"]:
                config["data"][key] = str(value)

        quasar_catalogue = DesiQuasarCatalogue(config["data"])

        # reset printing after test
        reset_logger()

        self.assertTrue(quasar_catalogue.catalogue is not None)
        self.assertTrue(len(quasar_catalogue.catalogue) == 2)
        self.assertTrue(quasar_catalogue.z_min == 2.15)
        self.assertTrue(quasar_catalogue.z_max == 3.2)
        self.assertTrue(quasar_catalogue.max_num_spec is None)
        self.assertTrue(len(quasar_catalogue.keep_surveys) == 5)
        self.assertTrue(quasar_catalogue.keep_surveys[0] == "sv1")
        self.assertTrue(quasar_catalogue.keep_surveys[1] == "all")
        self.assertTrue(quasar_catalogue.keep_surveys[2] == "sv2")
        self.assertTrue(quasar_catalogue.keep_surveys[3] == "sv3")
        self.assertTrue(quasar_catalogue.keep_surveys[4] == "main")

        # now test the behaviour of all + special
        config = ConfigParser()
        config.read_dict(
            {"data": {
                "z min": 2.15,
                "z max": 3.2,
                "max_num_spec": 1,
                "catalogue": f"{THIS_DIR}/data/dummy_desi_quasar_catalogue.fits.gz",
                "keep surveys": "all special"
            }})

        for key, value in defaults_desi_quasar_cat.items():
            if key not in config["data"]:
                config["data"][key] = str(value)

        quasar_catalogue = DesiQuasarCatalogue(config["data"])

        # reset printing after test
        reset_logger()

        self.assertTrue(quasar_catalogue.catalogue is not None)
        self.assertTrue(len(quasar_catalogue.catalogue) == 3)
        self.assertTrue(quasar_catalogue.z_min == 2.15)
        self.assertTrue(quasar_catalogue.z_max == 3.2)
        self.assertTrue(quasar_catalogue.max_num_spec is None)
        self.assertTrue(len(quasar_catalogue.keep_surveys) == 6)
        self.assertTrue(quasar_catalogue.keep_surveys[0] == "all")
        self.assertTrue(quasar_catalogue.keep_surveys[1] == "special")
        self.assertTrue(quasar_catalogue.keep_surveys[2] == "sv1")
        self.assertTrue(quasar_catalogue.keep_surveys[3] == "sv2")
        self.assertTrue(quasar_catalogue.keep_surveys[4] == "sv3")
        self.assertTrue(quasar_catalogue.keep_surveys[5] == "main")



    def test_desi_quasar_catalogue_trim_catalogue(self):
        """Load a DesiQuasarCatalogue trimming the catalogue"""
        # setup printing
        setup_logger(log_file=None)

        config = ConfigParser()
        config.read_dict(
            {"data": {
                "z min": 2.15,
                "z max": 3.2,
                "max num spec": 1,
                "catalogue": f"{THIS_DIR}/data/dummy_desi_quasar_catalogue.fits.gz",
                "keep surveys": "all"
            }})

        for key, value in defaults_desi_quasar_cat.items():
            if key not in config["data"]:
                config["data"][key] = str(value)

        quasar_catalogue = DesiQuasarCatalogue(config["data"])

        # reset printing
        reset_logger()

        self.assertTrue(quasar_catalogue.catalogue is not None)
        print(len(quasar_catalogue.catalogue))
        self.assertTrue(len(quasar_catalogue.catalogue) == 1)
        self.assertTrue(quasar_catalogue.z_min == 2.15)
        self.assertTrue(quasar_catalogue.z_max == 3.2)
        self.assertTrue(quasar_catalogue.max_num_spec == 1)

if __name__ == '__main__':
    unittest.main()
