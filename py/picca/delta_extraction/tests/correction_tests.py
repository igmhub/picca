"""This file contains tests related to Correction and its childs"""
import os
import unittest
import copy
from configparser import ConfigParser
import numpy as np

from picca.delta_extraction.correction import Correction
from picca.delta_extraction.errors import CorrectionError
from picca.delta_extraction.corrections.sdss_calibration_correction import SdssCalibrationCorrection
from picca.delta_extraction.corrections.sdss_dust_correction import SdssDustCorrection
from picca.delta_extraction.corrections.sdss_ivar_correction import SdssIvarCorrection
from picca.delta_extraction.corrections.sdss_optical_depth_correction import (
    SdssOpticalDepthCorrection
)
from picca.delta_extraction.userprint import UserPrint

from picca.delta_extraction.tests.abstract_test import AbstractTest
from picca.delta_extraction.tests.test_utils import forest1_log_lambda, forest1

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class CorrectionTest(AbstractTest):
    """Test Correction and its childs."""
    def test_correction(self):
        """Tests Abstract class Correction

        Load a Correction instace and check that method apply_correction is
        not initialized.
        """
        with self.assertRaises(CorrectionError):
            # create Correction instance
            correction = Correction()

            # run apply_correction, this should raise CorrectionError
            forest = copy.deepcopy(forest1)
            correction.apply_correction(forest)

    def test_sdss_calibration_correction(self):
        """Tests correct initialisation and inheritance for class
        SdssCalibrationCorrection

        Load a SdssCalibrationCorrection instace and check that it is
        correctly initialized.
        """
        in_file = f"{THIS_DIR}/data/dummy_corrections.fits.gz"

        # create SdssCalibrationCorrection instance
        config = ConfigParser()
        config.read_dict({"corrections": {"filename": in_file}})
        correction = SdssCalibrationCorrection(config["corrections"])
        self.assertTrue(isinstance(correction, Correction))

        # apply the correction
        forest = copy.deepcopy(forest1)
        correction.apply_correction(forest)

        self.assertTrue(np.allclose(forest.flux, np.ones_like(forest1_log_lambda)*0.5))
        self.assertTrue(np.allclose(forest.log_lambda, forest1_log_lambda))
        self.assertTrue(np.allclose(forest.ivar, np.ones_like(forest1_log_lambda)*16))
        self.assertTrue(np.allclose(forest.transmission_correction,
                                    np.ones_like(forest1_log_lambda)))

    def test_sdss_dust_correction(self):
        """Tests correct initialisation and inheritance for class
        SdssDustCorrection

        Load a SdssDustCorrection instace and check that it is
        correctly initialized.

        #TODO:Check that the function apply_correction
        is correctly implemented in a dummy Forest instance

        """
        in_file = f"{THIS_DIR}/data/dummy_corrections.fits.gz"

        # create SdssDustCorrection instance
        config = ConfigParser()
        config.read_dict({"corrections": {"filename": in_file}})
        correction = SdssDustCorrection(config["corrections"])
        self.assertTrue(isinstance(correction, Correction))

        # apply the correction
        forest = copy.deepcopy(forest1)
        correction.apply_correction(forest)

        # TODO: add checks in ivar and flux
        self.assertTrue(np.allclose(forest.log_lambda, forest1_log_lambda))
        self.assertTrue(np.allclose(forest.transmission_correction,
                                    np.ones_like(forest1_log_lambda)))

        # create SdssDustCorrection instance specifying the extinction conversion
        # factor
        config = ConfigParser()
        config.read_dict({"corrections": {"filename": in_file,
                                          "extinction_conversion_r": 3.5}})
        correction = SdssDustCorrection(config["corrections"])
        self.assertTrue(len(correction.extinction_bv_map) == 1)
        self.assertTrue(correction.extinction_bv_map.get(100000) == 1/3.5)

    def test_sdss_ivar_correction(self):
        """Tests correct initialisation and inheritance for class
        SdssIvarCorrection

        Load a SdssIvarCorrection instace and check that it is
        correctly initialized.
        """
        in_file = f"{THIS_DIR}/data/dummy_corrections.fits.gz"

        # create SdssIvarCorrection instance
        config = ConfigParser()
        config.read_dict({"corrections": {"filename": in_file}})
        correction = SdssIvarCorrection(config["corrections"])
        self.assertTrue(isinstance(correction, Correction))

        # apply the correction
        forest = copy.deepcopy(forest1)
        correction.apply_correction(forest)

        self.assertTrue(np.allclose(forest.flux, np.ones_like(forest1_log_lambda)))
        self.assertTrue(np.allclose(forest.log_lambda, forest1_log_lambda))
        self.assertTrue(np.allclose(forest.ivar, np.ones_like(forest1_log_lambda)*2))
        self.assertTrue(np.allclose(forest.transmission_correction,
                                    np.ones_like(forest1_log_lambda)))

    def test_sdss_optical_depth_correction(self):
        """Tests correct initialisation and inheritance for class
        OpticalDepthCorrection

        Load a SdssIvarCorrection instace and check that it is
        correctly initialized.
        """
        in_file = f"{THIS_DIR}/data/dummy_corrections.fits.gz"
        out_file = f"{THIS_DIR}/results/sdss_optical_depth_correction_print.txt"
        test_file = f"{THIS_DIR}/data/sdss_optical_depth_correction_print.txt"

        # setup printing
        UserPrint.initialize_log(out_file)

        config = ConfigParser()
        config.read_dict({"corrections": {"filename": in_file,
                                          "optical depth tau": "1",
                                          "optical depth gamma": "0",
                                          "optical depth absorber": "LYA"}})
        correction = SdssOpticalDepthCorrection(config["corrections"])
        self.assertTrue(isinstance(correction, Correction))
        self.compare_ascii(test_file, out_file, expand_dir=True)

        # reset printing
        UserPrint.reset_log()

        # apply the correction
        forest = copy.deepcopy(forest1)
        correction.apply_correction(forest)

        self.assertTrue(np.allclose(forest.flux, np.ones_like(forest1_log_lambda)))
        self.assertTrue(np.allclose(forest.log_lambda, forest1_log_lambda))
        self.assertTrue(np.allclose(forest.ivar, np.ones_like(forest1_log_lambda)*4))
        self.assertTrue(np.allclose(forest.transmission_correction,
                                    np.ones_like(forest1_log_lambda)*0.36787944117144233))


if __name__ == '__main__':
    unittest.main()
