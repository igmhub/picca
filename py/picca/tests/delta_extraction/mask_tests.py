"""This file contains tests related to Mask and its childs"""
from configparser import ConfigParser
import os
import unittest

import numpy as np

from picca.delta_extraction.mask import Mask
from picca.delta_extraction.masks.lines_mask import LinesMask
from picca.delta_extraction.masks.dla_mask import DlaMask
from picca.delta_extraction.masks.dla_mask import defaults as defaults_dla_mask
from picca.delta_extraction.masks.absorber_mask import AbsorberMask
from picca.delta_extraction.masks.absorber_mask import (
    defaults as defaults_absorber_mask)
from picca.delta_extraction.errors import MaskError
from picca.delta_extraction.utils import setup_logger
from picca.tests.delta_extraction.abstract_test import AbstractTest
from picca.tests.delta_extraction.test_utils import reset_logger
from picca.tests.delta_extraction.test_utils import setup_forest, reset_forest
from picca.tests.delta_extraction.test_utils import forest1_log_lambda, forest1
from picca.tests.delta_extraction.test_utils import forest2_log_lambda, forest2
from picca.tests.delta_extraction.test_utils import forest3_log_lambda, forest3

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class MaskTest(AbstractTest):
    """Test class Mask and its childs.

    Methods
    -------
    compare_ascii (from AbstractTest)
    compare_fits (from AbstractTest)
    setUp (from AbstractTest)
    test_absorber_mask
    test_dla_mask
    test_mask
    """
    def setUp(self):
        """ Actions done at test startup
        Initialize Forest class variables
        """
        super().setUp()
        setup_forest("log")

    def test_absorber_mask(self):
        """Test correct initialisation and inheritance for class
        AbsorberMask

        Load a AbsorberMask instace and check that it is
        correctly initialized.
        """
        in_file = f"{THIS_DIR}/data/dummy_absorbers_cat.fits.gz"
        out_file = f"{THIS_DIR}/results/absorber_mask_print.txt"
        test_file = f"{THIS_DIR}/data/absorber_mask_print.txt"

        # setup printing
        setup_logger(log_file=out_file)

        # initialize mask
        config = ConfigParser()
        config.read_dict({"mask": {"filename": in_file,
                                   "los_id name": "THING_ID"}
                         })
        for key, value in defaults_absorber_mask.items():
            if key not in config["mask"]:
                config["mask"][key] = str(value)
        mask = AbsorberMask(config["mask"])
        self.assertTrue(isinstance(mask, Mask))
        self.assertTrue(mask.absorber_mask_width == 2.5)


        # apply mask to forest with 1 absorber
        mask.apply_mask(forest1)

        w = np.ones(forest1_log_lambda.size, dtype=bool)
        w &= np.fabs(1.e4 * (forest1_log_lambda - np.log10(5600))) > 2.5
        self.assertTrue(np.allclose(forest1.flux, np.ones_like(forest1_log_lambda[w])))
        self.assertTrue(np.allclose(forest1.log_lambda, forest1_log_lambda[w]))
        self.assertTrue(np.allclose(forest1.ivar, np.ones_like(forest1_log_lambda[w])*4))
        self.assertTrue(np.allclose(forest1.transmission_correction,
                                    np.ones_like(forest1_log_lambda[w])))

        # apply mask to forest with 2 absorbers
        mask.apply_mask(forest2)

        w = np.ones(forest2_log_lambda.size, dtype=bool)
        w &= np.fabs(1.e4 * (forest2_log_lambda - np.log10(5600))) > 2.5
        w &= np.fabs(1.e4 * (forest2_log_lambda - np.log10(5650))) > 2.5
        self.assertTrue(np.allclose(forest2.flux, np.ones_like(forest2_log_lambda[w])))
        self.assertTrue(np.allclose(forest2.log_lambda, forest2_log_lambda[w]))
        self.assertTrue(np.allclose(forest2.ivar, np.ones_like(forest2_log_lambda[w])*4))
        self.assertTrue(np.allclose(forest2.transmission_correction,
                                    np.ones_like(forest2_log_lambda[w])))

        # apply mask to forest without absorbers
        mask.apply_mask(forest3)
        self.assertTrue(np.allclose(forest2.flux, np.ones_like(forest3_log_lambda)))
        self.assertTrue(np.allclose(forest2.log_lambda, forest3_log_lambda))
        self.assertTrue(np.allclose(forest2.ivar, np.ones_like(forest3_log_lambda)*4))
        self.assertTrue(np.allclose(forest2.transmission_correction,
                                    np.ones_like(forest3_log_lambda)))

        # initialize mask specifying variables
        config = ConfigParser()
        config.read_dict({"mask": {"filename": in_file,
                                   "absorber mask width": 1.5,
                                   "los_id name": "THING_ID"}})
        for key, value in defaults_absorber_mask.items():
            if key not in config["mask"]:
                config["mask"][key] = str(value)
        mask = AbsorberMask(config["mask"])
        self.assertTrue(mask.absorber_mask_width == 1.5)

        reset_logger()
        self.compare_ascii(test_file, out_file)

    def test_dla_mask(self):
        """Test correct initialisation and inheritance for class
        DlaMask

        Load a DlaMask instace and check that it is
        correctly initialized.
        """
        in_file = f"{THIS_DIR}/data/dummy_absorbers_cat.fits.gz"
        out_file = f"{THIS_DIR}/results/dla_mask_print.txt"
        test_file = f"{THIS_DIR}/data/dla_mask_print.txt"

        # setup printing
        setup_logger(log_file=out_file)

        # initialize mask
        config = ConfigParser()
        config.read_dict({"mask": {"filename": in_file,
                                   "los_id name": "THING_ID"}
                        })
        for key, value in defaults_dla_mask.items():
            if key not in config["mask"]:
                config["mask"][key] = str(value)
        mask = DlaMask(config["mask"])
        self.assertTrue(isinstance(mask, Mask))

        # apply mask to forest with 1 DLA
        mask.apply_mask(forest1)
        # TODO: check that the profile is correct

        # apply mask to forest with 2 DLAs
        mask.apply_mask(forest2)
        # TODO: check that the profile is correct

        # apply mask to forest without DLAs
        mask.apply_mask(forest3)
        self.assertTrue(np.allclose(forest3.flux, np.ones_like(forest3_log_lambda)))
        self.assertTrue(np.allclose(forest3.log_lambda, forest3_log_lambda))
        self.assertTrue(np.allclose(forest3.ivar, np.ones_like(forest3_log_lambda)*4))
        self.assertTrue(np.allclose(forest3.transmission_correction,
                                    np.ones_like(forest3_log_lambda)))

        reset_logger()
        self.compare_ascii(test_file, out_file)

    def test_lines_mask(self):
        """Test LinesMask"""
        # TODO: add test

    def test_lines_mask_missing_options(self):
            """Test correct error reporting when initializing with missing options
            for class LinesMask
            """
            # create LinesMask instance with missing options
            config = ConfigParser()
            config.read_dict({"masks": {}})
            expected_message = (
                "Missing argument 'filename' required by LinesMask")
            with self.assertRaises(MaskError) as context_manager:
                correction = LinesMask(config["masks"])
            self.compare_error_message(context_manager, expected_message)

    def test_mask(self):
        """Test Abstract class Mask

        Load a Mask instace and check that method apply_mask is not initialized.
        """
        config = ConfigParser()
        config.read_dict({"masks": {}})
        mask = Mask(config['masks'])
        expected_message = (
            "Function 'apply_mask' was not overloaded by child class")
        with self.assertRaises(MaskError) as context_manager:
            mask.apply_mask(forest1)
        self.compare_error_message(context_manager, expected_message)

if __name__ == '__main__':
    unittest.main()
