"""This file contains tests related to Mask and its childs"""
from configparser import ConfigParser
import copy
import os
import unittest

import numpy as np

from picca.delta_extraction.mask import Mask
from picca.delta_extraction.masks.bal_mask import BalMask
from picca.delta_extraction.masks.bal_mask import defaults as defaults_bal_mask
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

    def test_bal_mask_ai_remove(self):
        """Test correct initialisation and inheritance for class
        BalMask

        Load a BlaMask instace and check that it is
        correctly initialized.
        """
        in_file = f"{THIS_DIR}/data/baltestcat.fits.gz"
        out_file = f"{THIS_DIR}/results/bal_mask_print.txt"
        test_file = f"{THIS_DIR}/data/bal_mask_print.txt"
        test_file_forest1 = f"{THIS_DIR}/data/bal_mask_ai_forest1_remove.txt"
        test_file_forest2 = f"{THIS_DIR}/data/bal_mask_ai_forest2_remove.txt"

        # setup printing
        setup_logger(log_file=out_file)

        # initialize mask using AI to select BALs
        config = ConfigParser()
        config.read_dict({"mask": {"filename": in_file,
                                   "los_id name": "TARGETID",
                                   "bal index type": "ai",
                                   "keep pixels": False}
                        })
        for key, value in defaults_bal_mask.items():
            if key not in config["mask"]:
                config["mask"][key] = str(value)
        mask = BalMask(config["mask"])
        self.assertTrue(isinstance(mask, Mask))

        # apply mask to forests with a BAL
        forest = copy.deepcopy(forest1)
        mask.apply_mask(forest)
        forest_masked = np.genfromtxt(test_file_forest1, names=True)
        self.assertEqual(forest.flux.size, forest_masked["flux"].size)
        self.assertTrue(np.allclose(forest.flux, forest_masked["flux"]))
        self.assertTrue(np.allclose(forest.log_lambda, forest_masked["log_lambda"]))
        self.assertTrue(np.allclose(forest.ivar, forest_masked["ivar"]))
        self.assertTrue(np.allclose(forest.transmission_correction,
                                    forest_masked["transmission_correction"]))

        forest = copy.deepcopy(forest2)
        mask.apply_mask(forest)
        forest_masked = np.genfromtxt(test_file_forest2, names=True)
        self.assertEqual(forest.flux.size, forest_masked["flux"].size)
        self.assertTrue(np.allclose(forest.flux, forest_masked["flux"]))
        self.assertTrue(np.allclose(forest.log_lambda, forest_masked["log_lambda"]))
        self.assertTrue(np.allclose(forest.ivar, forest_masked["ivar"]))
        self.assertTrue(np.allclose(forest.transmission_correction,
                                    forest_masked["transmission_correction"]))


        # apply mask to forest without BALs
        mask.apply_mask(forest3)
        self.assertTrue(np.allclose(forest3.flux, np.ones_like(forest3_log_lambda)))
        self.assertTrue(np.allclose(forest3.log_lambda, forest3_log_lambda))
        self.assertTrue(np.allclose(forest3.ivar, np.ones_like(forest3_log_lambda)*4))
        self.assertTrue(np.allclose(forest3.transmission_correction,
                                    np.ones_like(forest3_log_lambda)))

        reset_logger()
        self.compare_ascii(test_file, out_file)

    def test_bal_mask_ai_set_ivar(self):
        """Test correct initialisation and inheritance for class
        BalMask

        Load a BlaMask instace and check that it is
        correctly initialized.
        """
        in_file = f"{THIS_DIR}/data/baltestcat.fits.gz"
        out_file = f"{THIS_DIR}/results/bal_mask_print.txt"
        test_file = f"{THIS_DIR}/data/bal_mask_print.txt"
        test_file_forest1 = f"{THIS_DIR}/data/bal_mask_ai_forest1_set_ivar.txt"
        test_file_forest2 = f"{THIS_DIR}/data/bal_mask_ai_forest2_set_ivar.txt"

        # setup printing
        setup_logger(log_file=out_file)

        # initialize mask using AI to select BALs
        config = ConfigParser()
        config.read_dict({"mask": {"filename": in_file,
                                   "los_id name": "TARGETID",
                                   "bal index type": "ai",
                                   "keep pixels": True}
                        })
        for key, value in defaults_bal_mask.items():
            if key not in config["mask"]:
                config["mask"][key] = str(value)
        mask = BalMask(config["mask"])
        self.assertTrue(isinstance(mask, Mask))

        # apply mask to forests with a BAL
        forest = copy.deepcopy(forest1)
        mask.apply_mask(forest)
        forest_masked = np.genfromtxt(test_file_forest1, names=True)
        self.assertEqual(forest.flux.size, forest_masked["flux"].size)
        self.assertTrue(np.allclose(forest.flux, forest_masked["flux"]))
        self.assertTrue(np.allclose(forest.log_lambda, forest_masked["log_lambda"]))
        self.assertTrue(np.allclose(forest.ivar, forest_masked["ivar"]))
        self.assertTrue(np.allclose(forest.transmission_correction,
                                    forest_masked["transmission_correction"]))

        forest = copy.deepcopy(forest2)
        mask.apply_mask(forest)
        forest_masked = np.genfromtxt(test_file_forest2, names=True)
        self.assertEqual(forest.flux.size, forest_masked["flux"].size)
        self.assertTrue(np.allclose(forest.flux, forest_masked["flux"]))
        self.assertTrue(np.allclose(forest.log_lambda, forest_masked["log_lambda"]))
        self.assertTrue(np.allclose(forest.ivar, forest_masked["ivar"]))
        self.assertTrue(np.allclose(forest.transmission_correction,
                                    forest_masked["transmission_correction"]))


        # apply mask to forest without BALs
        mask.apply_mask(forest3)
        self.assertTrue(np.allclose(forest3.flux, np.ones_like(forest3_log_lambda)))
        self.assertTrue(np.allclose(forest3.log_lambda, forest3_log_lambda))
        self.assertTrue(np.allclose(forest3.ivar, np.ones_like(forest3_log_lambda)*4))
        self.assertTrue(np.allclose(forest3.transmission_correction,
                                    np.ones_like(forest3_log_lambda)))

        reset_logger()
        self.compare_ascii(test_file, out_file)

    def test_bal_mask_bi_remove(self):
        """Test correct initialisation and inheritance for class
        BalMask

        Load a BlaMask instace and check that it is
        correctly initialized.
        """
        in_file = f"{THIS_DIR}/data/baltestcat.fits.gz"
        out_file = f"{THIS_DIR}/results/bal_mask_print.txt"
        test_file = f"{THIS_DIR}/data/bal_mask_print.txt"
        test_file_forest1 = f"{THIS_DIR}/data/bal_mask_bi_forest1_remove.txt"
        test_file_forest2 = f"{THIS_DIR}/data/bal_mask_bi_forest2_remove.txt"

        # setup printing
        setup_logger(log_file=out_file)

        # initialize mask using BI to select BALs
        config = ConfigParser()
        config.read_dict({"mask": {"filename": in_file,
                                   "los_id name": "TARGETID",
                                   "bal index type": "bi",
                                   "keep pixels": False}
                        })
        for key, value in defaults_bal_mask.items():
            if key not in config["mask"]:
                config["mask"][key] = str(value)
        mask = BalMask(config["mask"])
        self.assertTrue(isinstance(mask, Mask))

        # apply mask to forests with a BAL
        forest = copy.deepcopy(forest1)
        mask.apply_mask(forest)
        forest_masked = np.genfromtxt(test_file_forest1, names=True)
        self.assertEqual(forest.flux.size, forest_masked["flux"].size)
        self.assertTrue(np.allclose(forest.flux, forest_masked["flux"]))
        self.assertTrue(np.allclose(forest.log_lambda, forest_masked["log_lambda"]))
        self.assertTrue(np.allclose(forest.ivar, forest_masked["ivar"]))
        self.assertTrue(np.allclose(forest.transmission_correction,
                                    forest_masked["transmission_correction"]))

        forest = copy.deepcopy(forest2)
        mask.apply_mask(forest)
        forest_masked = np.genfromtxt(test_file_forest2, names=True)
        self.assertEqual(forest.flux.size, forest_masked["flux"].size)
        self.assertTrue(np.allclose(forest.flux, forest_masked["flux"]))
        self.assertTrue(np.allclose(forest.log_lambda, forest_masked["log_lambda"]))
        self.assertTrue(np.allclose(forest.ivar, forest_masked["ivar"]))
        self.assertTrue(np.allclose(forest.transmission_correction,
                                    forest_masked["transmission_correction"]))


        # apply mask to forest without BALs
        mask.apply_mask(forest3)
        self.assertTrue(np.allclose(forest3.flux, np.ones_like(forest3_log_lambda)))
        self.assertTrue(np.allclose(forest3.log_lambda, forest3_log_lambda))
        self.assertTrue(np.allclose(forest3.ivar, np.ones_like(forest3_log_lambda)*4))
        self.assertTrue(np.allclose(forest3.transmission_correction,
                                    np.ones_like(forest3_log_lambda)))

        reset_logger()
        self.compare_ascii(test_file, out_file)

    def test_bal_mask_bi_set_ivar(self):
        """Test correct initialisation and inheritance for class
        BalMask

        Load a BlaMask instace and check that it is
        correctly initialized.
        """
        in_file = f"{THIS_DIR}/data/baltestcat.fits.gz"
        out_file = f"{THIS_DIR}/results/bal_mask_print.txt"
        test_file = f"{THIS_DIR}/data/bal_mask_print.txt"
        test_file_forest1 = f"{THIS_DIR}/data/bal_mask_bi_forest1_set_ivar.txt"
        test_file_forest2 = f"{THIS_DIR}/data/bal_mask_bi_forest2_set_ivar.txt"

        # setup printing
        setup_logger(log_file=out_file)

        # initialize mask using AI to select BALs
        config = ConfigParser()
        config.read_dict({"mask": {"filename": in_file,
                                   "los_id name": "TARGETID",
                                   "bal index type": "bi",
                                   "keep pixels": True}
                        })
        for key, value in defaults_bal_mask.items():
            if key not in config["mask"]:
                config["mask"][key] = str(value)
        mask = BalMask(config["mask"])
        self.assertTrue(isinstance(mask, Mask))

        # apply mask to forests with a BAL
        forest = copy.deepcopy(forest1)
        mask.apply_mask(forest)
        forest_masked = np.genfromtxt(test_file_forest1, names=True)
        self.assertEqual(forest.flux.size, forest_masked["flux"].size)
        self.assertTrue(np.allclose(forest.flux, forest_masked["flux"]))
        self.assertTrue(np.allclose(forest.log_lambda, forest_masked["log_lambda"]))
        self.assertTrue(np.allclose(forest.ivar, forest_masked["ivar"]))
        self.assertTrue(np.allclose(forest.transmission_correction,
                                    forest_masked["transmission_correction"]))

        forest = copy.deepcopy(forest2)
        mask.apply_mask(forest)
        forest_masked = np.genfromtxt(test_file_forest2, names=True)
        self.assertEqual(forest.flux.size, forest_masked["flux"].size)
        self.assertTrue(np.allclose(forest.flux, forest_masked["flux"]))
        self.assertTrue(np.allclose(forest.log_lambda, forest_masked["log_lambda"]))
        self.assertTrue(np.allclose(forest.ivar, forest_masked["ivar"]))
        self.assertTrue(np.allclose(forest.transmission_correction,
                                    forest_masked["transmission_correction"]))


        # apply mask to forest without BALs
        mask.apply_mask(forest3)
        self.assertTrue(np.allclose(forest3.flux, np.ones_like(forest3_log_lambda)))
        self.assertTrue(np.allclose(forest3.log_lambda, forest3_log_lambda))
        self.assertTrue(np.allclose(forest3.ivar, np.ones_like(forest3_log_lambda)*4))
        self.assertTrue(np.allclose(forest3.transmission_correction,
                                    np.ones_like(forest3_log_lambda)))

        reset_logger()
        self.compare_ascii(test_file, out_file)

    def test_bal_mask_missing_options(self):
            """Test correct error reporting when initializing with missing options
            for class BalMask
            """
            # create BalMask instance with missing Mask options
            config = ConfigParser()
            config.read_dict({"masks": {}})
            expected_message = (
                "Missing argument 'keep pixels' required by Mask")
            with self.assertRaises(MaskError) as context_manager:
                mask = BalMask(config["masks"])
            self.compare_error_message(context_manager, expected_message)

            # create BalMask instance with missing "filename"
            config = ConfigParser()
            config.read_dict({"masks": {"keep pixels": True}})
            expected_message = (
                "Missing argument 'filename' required by BalMask")
            with self.assertRaises(MaskError) as context_manager:
                mask = BalMask(config["masks"])
            self.compare_error_message(context_manager, expected_message)

            # create BalMask instance with missing "los_id"
            config = ConfigParser()
            config.read_dict({"masks": {
                "keep pixels": True,
                "filename": f"{THIS_DIR}/data/baltestcat.fits.gz",
            }})
            expected_message = (
                "Missing argument 'los_id name' required by BalMask")
            with self.assertRaises(MaskError) as context_manager:
                mask = BalMask(config["masks"])
            self.compare_error_message(context_manager, expected_message)

            # create BalMask instance with missing "bal index type"
            config = ConfigParser()
            config.read_dict({"masks": {
                "keep pixels": True,
                "filename": f"{THIS_DIR}/data/baltestcat.fits.gz",
                "los_id name": "TARGETID",
            }})
            expected_message = (
                "Missing argument 'bal index type' required by BalMask")
            with self.assertRaises(MaskError) as context_manager:
                mask = BalMask(config["masks"])
            self.compare_error_message(context_manager, expected_message)

    def test_dla_mask_missing_options(self):
            """Test correct error reporting when initializing with missing options
            for class DlaMask
            """
            # create DlaMask instance with missing Mask options
            config = ConfigParser()
            config.read_dict({"masks": {}})
            expected_message = (
                "Missing argument 'keep pixels' required by Mask")
            with self.assertRaises(MaskError) as context_manager:
                mask = DlaMask(config["masks"])
            self.compare_error_message(context_manager, expected_message)

            # create DlaMask instance with missing "filename"
            config = ConfigParser()
            config.read_dict({"masks": {"keep pixels": True}})
            expected_message = (
                "Missing argument 'filename' required by DlaMask")
            with self.assertRaises(MaskError) as context_manager:
                mask = DlaMask(config["masks"])
            self.compare_error_message(context_manager, expected_message)

            # create DlaMask instance with missing "los_id"
            config = ConfigParser()
            config.read_dict({"masks": {
                "keep pixels": True,
                "filename": f"{THIS_DIR}/data/dummy_absorbers_cat.fits.gz",
            }})
            expected_message = (
                "Missing argument 'los_id name' required by DlaMask")
            with self.assertRaises(MaskError) as context_manager:
                mask = DlaMask(config["masks"])
            self.compare_error_message(context_manager, expected_message)

            # create DlaMask instance with missing "dla mask limit"
            config = ConfigParser()
            config.read_dict({"masks": {
                "keep pixels": True,
                "filename": f"{THIS_DIR}/data/dummy_absorbers_cat.fits.gz",
                "los_id name": "THING_ID",
            }})
            expected_message = (
                "Missing argument 'dla mask limit' required by DlaMask")
            with self.assertRaises(MaskError) as context_manager:
                mask = DlaMask(config["masks"])
            self.compare_error_message(context_manager, expected_message)

    def test_dla_mask_remove(self):
        """Test correct initialisation and inheritance for class
        DlaMask when masked pixels are removed

        Load a DlaMask instace and check that it is
        correctly initialized.
        """
        in_file = f"{THIS_DIR}/data/dummy_absorbers_cat.fits.gz"
        out_file = f"{THIS_DIR}/results/dla_mask_print.txt"
        out_file_forest1 = f"{THIS_DIR}/results/dla_mask_forest1_remove.txt"
        out_file_forest2 = f"{THIS_DIR}/results/dla_mask_forest2_remove.txt"
        test_file = f"{THIS_DIR}/data/dla_mask_print.txt"
        test_file_forest1 = f"{THIS_DIR}/data/dla_mask_forest1_remove.txt"
        test_file_forest2 = f"{THIS_DIR}/data/dla_mask_forest2_remove.txt"

        # setup printing
        setup_logger(log_file=out_file)

        # initialize mask
        config = ConfigParser()
        config.read_dict({"mask": {"filename": in_file,
                                   "los_id name": "THING_ID",
                                   "keep pixels": False}
                        })
        for key, value in defaults_dla_mask.items():
            if key not in config["mask"]:
                config["mask"][key] = str(value)
        mask = DlaMask(config["mask"])
        self.assertTrue(isinstance(mask, Mask))

        # apply mask to forest with 1 DLA
        forest = copy.deepcopy(forest1)
        mask.apply_mask(forest)

        # save the results
        f = open(out_file_forest1, "w")
        f.write("# log_lambda flux ivar transmission_correction\n")
        for log_lambda, flux, ivar, transmission_correction in zip(
            forest.log_lambda, forest.flux, forest.ivar, forest.transmission_correction):
                f.write(f"{log_lambda} {flux} {ivar} {transmission_correction}\n")
        f.close()

        # load expected values and compare
        forest_masked = np.genfromtxt(test_file_forest1, names=True)
        self.assertEqual(forest.flux.size, forest_masked["flux"].size)
        self.assertTrue(np.allclose(forest.flux, forest_masked["flux"]))
        self.assertTrue(np.allclose(forest.log_lambda, forest_masked["log_lambda"]))
        self.assertTrue(np.allclose(forest.ivar, forest_masked["ivar"]))
        self.assertTrue(np.allclose(forest.transmission_correction,
                                    forest_masked["transmission_correction"]))


        # apply mask to forest with 2 DLAs
        forest = copy.deepcopy(forest2)
        mask.apply_mask(forest)

        # save the results
        f = open(out_file_forest2, "w")
        f.write("# log_lambda flux ivar transmission_correction\n")
        for log_lambda, flux, ivar, transmission_correction in zip(
            forest.log_lambda, forest.flux, forest.ivar, forest.transmission_correction):
                f.write(f"{log_lambda} {flux} {ivar} {transmission_correction}\n")
        f.close()

        # load expected values and compare
        forest_masked = np.genfromtxt(test_file_forest2, names=True)
        self.assertEqual(forest.flux.size, forest_masked["flux"].size)
        self.assertTrue(np.allclose(forest.flux, forest_masked["flux"]))
        self.assertTrue(np.allclose(forest.log_lambda, forest_masked["log_lambda"]))
        self.assertTrue(np.allclose(forest.ivar, forest_masked["ivar"]))
        self.assertTrue(np.allclose(forest.transmission_correction,
                                    forest_masked["transmission_correction"]))

        # apply mask to forest without DLAs
        mask.apply_mask(forest3)
        self.assertTrue(np.allclose(forest3.flux, np.ones_like(forest3_log_lambda)))
        self.assertTrue(np.allclose(forest3.log_lambda, forest3_log_lambda))
        self.assertTrue(np.allclose(forest3.ivar, np.ones_like(forest3_log_lambda)*4))
        self.assertTrue(np.allclose(forest3.transmission_correction,
                                    np.ones_like(forest3_log_lambda)))

        reset_logger()
        self.compare_ascii(test_file, out_file)

    def test_dla_mask_set_ivar(self):
        """Test correct initialisation and inheritance for class
        DlaMask when masked pixels are kept and masking is done by setting
        the inverse variance to zero

        Load a DlaMask instace and check that it is
        correctly initialized.
        """
        in_file = f"{THIS_DIR}/data/dummy_absorbers_cat.fits.gz"
        out_file = f"{THIS_DIR}/results/dla_mask_print.txt"
        out_file_forest1 = f"{THIS_DIR}/results/dla_mask_forest1_set_ivar.txt"
        out_file_forest2 = f"{THIS_DIR}/results/dla_mask_forest2_set_ivar.txt"
        test_file = f"{THIS_DIR}/data/dla_mask_print.txt"
        test_file_forest1 = f"{THIS_DIR}/data/dla_mask_forest1_set_ivar.txt"
        test_file_forest2 = f"{THIS_DIR}/data/dla_mask_forest2_set_ivar.txt"

        # setup printing
        setup_logger(log_file=out_file)

        # initialize mask
        config = ConfigParser()
        config.read_dict({"mask": {"filename": in_file,
                                   "los_id name": "THING_ID",
                                   "keep pixels": True}
                        })
        for key, value in defaults_dla_mask.items():
            if key not in config["mask"]:
                config["mask"][key] = str(value)
        mask = DlaMask(config["mask"])
        self.assertTrue(isinstance(mask, Mask))

        # apply mask to forest with 1 DLA
        forest = copy.deepcopy(forest1)
        mask.apply_mask(forest)

        # save the results
        f = open(out_file_forest1, "w")
        f.write("# log_lambda flux ivar transmission_correction\n")
        for log_lambda, flux, ivar, transmission_correction in zip(
            forest.log_lambda, forest.flux, forest.ivar, forest.transmission_correction):
                f.write(f"{log_lambda} {flux} {ivar} {transmission_correction}\n")
        f.close()

        # load expected values and compare
        forest_masked = np.genfromtxt(test_file_forest1, names=True)
        self.assertEqual(forest.flux.size, forest_masked["flux"].size)
        self.assertTrue(np.allclose(forest.flux, forest_masked["flux"]))
        self.assertTrue(np.allclose(forest.log_lambda, forest_masked["log_lambda"]))
        self.assertTrue(np.allclose(forest.ivar, forest_masked["ivar"]))
        self.assertTrue(np.allclose(forest.transmission_correction,
                                    forest_masked["transmission_correction"]))


        # apply mask to forest with 2 DLAs
        forest = copy.deepcopy(forest2)
        mask.apply_mask(forest)

        # save the results
        f = open(out_file_forest2, "w")
        f.write("# log_lambda flux ivar transmission_correction\n")
        for log_lambda, flux, ivar, transmission_correction in zip(
            forest.log_lambda, forest.flux, forest.ivar, forest.transmission_correction):
                f.write(f"{log_lambda} {flux} {ivar} {transmission_correction}\n")
        f.close()

        # load expected values and compare

        forest_masked = np.genfromtxt(test_file_forest2, names=True)
        self.assertEqual(forest.flux.size, forest_masked["flux"].size)
        self.assertTrue(np.allclose(forest.flux, forest_masked["flux"]))
        self.assertTrue(np.allclose(forest.log_lambda, forest_masked["log_lambda"]))
        self.assertTrue(np.allclose(forest.ivar, forest_masked["ivar"]))
        self.assertTrue(np.allclose(forest.transmission_correction,
                                    forest_masked["transmission_correction"]))

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
            # create LinesMask instance with missing Mask options
            config = ConfigParser()
            config.read_dict({"masks": {}})
            expected_message = (
                "Missing argument 'keep pixels' required by Mask")
            with self.assertRaises(MaskError) as context_manager:
                correction = LinesMask(config["masks"])
            self.compare_error_message(context_manager, expected_message)

            # create LinesMask instance with missing options
            config = ConfigParser()
            config.read_dict({"masks": {"keep pixels": True}})
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
        expected_message = (
            "Missing argument 'keep pixels' required by Mask")
        with self.assertRaises(MaskError) as context_manager:
            mask = Mask(config['masks'])


        config.read_dict({"masks": {"keep pixels": False}})
        mask = Mask(config['masks'])
        expected_message = (
            "Function 'apply_mask' was not overloaded by child class")
        with self.assertRaises(MaskError) as context_manager:
            mask.apply_mask(forest1)
        self.compare_error_message(context_manager, expected_message)

if __name__ == '__main__':
    unittest.main()
