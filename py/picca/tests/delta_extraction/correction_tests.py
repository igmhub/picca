"""This file contains tests related to Correction and its childs"""
from configparser import ConfigParser
import copy
import os
import unittest

import numpy as np

from picca.delta_extraction.astronomical_objects.forest import Forest
from picca.delta_extraction.correction import Correction
from picca.delta_extraction.corrections.calibration_correction import CalibrationCorrection
from picca.delta_extraction.corrections.dust_correction import DustCorrection
from picca.delta_extraction.corrections.dust_correction import (
    defaults as defaults_dust_correction)
from picca.delta_extraction.corrections.ivar_correction import IvarCorrection
from picca.delta_extraction.corrections.optical_depth_correction import (
    OpticalDepthCorrection)
from picca.delta_extraction.errors import CorrectionError
from picca.delta_extraction.utils import setup_logger
from picca.tests.delta_extraction.abstract_test import AbstractTest
from picca.tests.delta_extraction.test_utils import reset_logger
from picca.tests.delta_extraction.test_utils import forest1_log_lambda, forest1

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class CorrectionTest(AbstractTest):
    """Test Correction and its childs.

    Methods
    -------
    compare_ascii (from AbstractTest)
    compare_fits (from AbstractTest)
    setUp (from AbstractTest)
    test_correction
    test_calibration_correction
    test_dust_correction
    test_ivar_correction
    test_optical_depth_correction
    """

    def test_correction(self):
        """Test Abstract class Correction

        Load a Correction instace and check that method apply_correction is
        not initialized.
        """
        # create Correction instance
        correction = Correction()

        # run apply_correction, this should raise CorrectionError
        forest = copy.deepcopy(forest1)

        expected_message = ("Function 'apply_correction' was not overloaded by "
                            "child class")
        with self.assertRaises(CorrectionError) as context_manager:

            correction.apply_correction(forest)
        self.compare_error_message(context_manager, expected_message)

    def test_calibration_correction(self):
        """Test correct initialisation and inheritance for class
        CalibrationCorrection

        Load a CalibrationCorrection instace and check that it is
        correctly initialized.
        """
        in_file = f"{THIS_DIR}/data/dummy_corrections.fits.gz"

        # create CalibrationCorrection instance
        config = ConfigParser()
        config.read_dict({"corrections": {"filename": in_file}})
        correction = CalibrationCorrection(config["corrections"])
        self.assertTrue(isinstance(correction, Correction))

        # apply the correction
        forest = copy.deepcopy(forest1)
        correction.apply_correction(forest)

        self.assertTrue(
            np.allclose(forest.flux,
                        np.ones_like(forest1_log_lambda) * 0.5))
        self.assertTrue(np.allclose(forest.log_lambda, forest1_log_lambda))
        self.assertTrue(
            np.allclose(forest.ivar,
                        np.ones_like(forest1_log_lambda) * 16))
        self.assertTrue(
            np.allclose(forest.transmission_correction,
                        np.ones_like(forest1_log_lambda)))

    def test_calibration_correction_missing_options(self):
        """Test correct error reporting when initializing with missing options
        for class CalibrationCorrection
        """
        # create CalibrationCorrection instance with missing options
        config = ConfigParser()
        config.read_dict({"corrections": {}})

        expected_message = ("Missing argument 'filename' required by "
                            "SdssCalibrationCorrection")
        with self.assertRaises(CorrectionError) as context_manager:
            correction = CalibrationCorrection(config["corrections"])
        self.compare_error_message(context_manager, expected_message)

    def test_dust_correction(self):
        """Test correct initialisation and inheritance for class
        DustCorrection

        Load a DustCorrection instace and check that it is
        correctly initialized.

        #TODO:Check that the function apply_correction
        is correctly implemented in a dummy Forest instance

        """
        in_file = f"{THIS_DIR}/data/dummy_corrections.fits.gz"

        # create DustCorrection instance
        config = ConfigParser()
        config.read_dict({"corrections": {"filename": in_file}})
        for key, value in defaults_dust_correction.items():
            if key not in config["corrections"]:
                config["corrections"][key] = str(value)
        correction = DustCorrection(config["corrections"])
        self.assertTrue(isinstance(correction, Correction))

        # apply the correction
        forest = copy.deepcopy(forest1)
        correction.apply_correction(forest)

        # TODO: add checks in ivar and flux
        self.assertTrue(np.allclose(forest.log_lambda, forest1_log_lambda))
        self.assertTrue(
            np.allclose(forest.transmission_correction,
                        np.ones_like(forest1_log_lambda)))

        # create DustCorrection instance specifying the extinction conversion
        # factor
        config = ConfigParser()
        config.read_dict({
            "corrections": {
                "filename": in_file,
                "extinction_conversion_r": 3.5
            }
        })
        correction = DustCorrection(config["corrections"])
        self.assertTrue(len(correction.extinction_bv_map) == 1)
        self.assertTrue(correction.extinction_bv_map.get(100000) == 1 / 3.5)

    def test_dust_correction_missing_options(self):
        """Test correct error reporting when initializing with missing options
        for class DustCorrection
        """
        # create DustCorrection instance with missing options
        config = ConfigParser()
        config.read_dict({"corrections": {}})
        expected_message = ("Missing argument 'extinction_conversion_r' required "
                            "by DustCorrection")
        with self.assertRaises(CorrectionError) as context_manager:
            correction = DustCorrection(config["corrections"])
        self.compare_error_message(context_manager, expected_message)

    def test_ivar_correction(self):
        """Test correct initialisation and inheritance for class
        IvarCorrection

        Load a IvarCorrection instace and check that it is
        correctly initialized.
        """
        in_file = f"{THIS_DIR}/data/dummy_corrections.fits.gz"

        # create IvarCorrection instance
        config = ConfigParser()
        config.read_dict({"corrections": {"filename": in_file}})
        correction = IvarCorrection(config["corrections"])
        self.assertTrue(isinstance(correction, Correction))

        # apply the correction
        forest = copy.deepcopy(forest1)
        correction.apply_correction(forest)

        self.assertTrue(
            np.allclose(forest.flux, np.ones_like(forest1_log_lambda)))
        self.assertTrue(np.allclose(forest.log_lambda, forest1_log_lambda))
        self.assertTrue(
            np.allclose(forest.ivar,
                        np.ones_like(forest1_log_lambda) * 2))
        self.assertTrue(
            np.allclose(forest.transmission_correction,
                        np.ones_like(forest1_log_lambda)))

    def test_ivar_correction_missing_options(self):
        """Test correct error reporting when initializing with missing options
        for class IvarCorrection
        """
        # create IvarCorrection instance with missing options
        config = ConfigParser()
        config.read_dict({"corrections": {}})
        expected_message = ("Missing argument 'filename' required "
                            "by SdssIvarCorrection")
        with self.assertRaises(CorrectionError) as context_manager:
            correction = IvarCorrection(config["corrections"])
        self.compare_error_message(context_manager, expected_message)

    def test_optical_depth_correction(self):
        """Test correct initialisation and inheritance for class
        OpticalDepthCorrection

        Load a IvarCorrection instace and check that it is
        correctly initialized.
        """
        in_file = f"{THIS_DIR}/data/dummy_corrections.fits.gz"
        out_file = f"{THIS_DIR}/results/optical_depth_correction_print.txt"
        test_file = f"{THIS_DIR}/data/optical_depth_correction_print.txt"

        # setup printing
        setup_logger(log_file=out_file)
        Forest.wave_solution = "log"

        config = ConfigParser()
        config.read_dict({
            "corrections": {
                "filename": in_file,
                "optical depth tau": "1",
                "optical depth gamma": "0",
                "optical depth absorber": "LYA"
            }
        })
        correction = OpticalDepthCorrection(config["corrections"])
        self.assertTrue(isinstance(correction, Correction))

        # apply the correction
        forest = copy.deepcopy(forest1)
        correction.apply_correction(forest)

        self.assertTrue(
            np.allclose(forest.flux, np.ones_like(forest1_log_lambda)))
        self.assertTrue(np.allclose(forest.log_lambda, forest1_log_lambda))
        self.assertTrue(
            np.allclose(forest.ivar,
                        np.ones_like(forest1_log_lambda) * 4))
        self.assertTrue(
            np.allclose(forest.transmission_correction,
                        np.ones_like(forest1_log_lambda) * 0.36787944117144233))

        reset_logger()
        self.compare_ascii(test_file, out_file)

    def test_optical_depth_correction_missing_options(self):
        """Test correct error reporting when initializing with missing options
        for class OpticalDepthCorrection
        """
        # create OpticalDepthCorrection instance with missing options
        config = ConfigParser()
        config.read_dict({"corrections": {}})
        expected_message = ("Missing argument 'optical depth tau' required "
                            "by SdssOpticalDepthCorrection")
        with self.assertRaises(CorrectionError) as context_manager:
            correction = OpticalDepthCorrection(config["corrections"])
        self.compare_error_message(context_manager, expected_message)

if __name__ == '__main__':
    unittest.main()
