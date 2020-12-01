"""This file contains configuration tests"""
import os
import unittest

from picca.delta_extraction.correction import Correction
from picca.delta_extraction.errors import CorrectionError

from picca.delta_extraction.corrections.sdss_calibration_correction import SdssCalibrationCorrection
from picca.delta_extraction.corrections.sdss_dust_correction import SdssDustCorrection
from picca.delta_extraction.corrections.sdss_ivar_correction import SdssIvarCorrection

from picca.delta_extraction.tests.abstract_test import AbstractTest

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class TestConfiguration(AbstractTest):
    """Test the configuration."""

    def test_correction(self):
        """Tests Abstract class Correction

        Load a Correction instace and check that it cannot be
        initialized.

        #TODO:Check that the function apply_correction
        is correctly implemented in a dummy Forest instance

        """
        with self.assertRaises(CorrectionError):
            correction = Correction()
            correction.apply_correction("fake data")

    def test_sdss_calibration_correction(self):
        """Tests correct initialisation and inheritance for class
        SdssCalibrationCorrection

        Load a SdssCalibrationCorrection instace and check that it is
        correctly initialized.

        #TODO:Check that the function apply_correction
        is correctly implemented in a dummy Forest instance

        """
        in_file = f"{THIS_DIR}/data/delta_attributes.fits.gz"

        correction = SdssCalibrationCorrection({"filename": in_file})
        self.assertTrue(isinstance(correction, Correction))

    def test_sdss_dust_correction(self):
        """Tests correct initialisation and inheritance for class
        SdssDustCorrection

        Load a SdssDustCorrection instace and check that it is
        correctly initialized.

        #TODO:Check that the function apply_correction
        is correctly implemented in a dummy Forest instance

        """
        in_file = f"{THIS_DIR}/data/delta_attributes.fits.gz"

        correction = SdssDustCorrection({"filename": in_file})
        self.assertTrue(isinstance(correction, Correction))

    def test_sdss_ivar_correction(self):
        """Tests correct initialisation and inheritance for class
        SdssIvarCorrection

        Load a SdssIvarCorrection instace and check that it is
        correctly initialized.

        #TODO:Check that the function apply_correction
        is correctly implemented in a dummy Forest instance

        """
        in_file = f"{THIS_DIR}/data/delta_attributes.fits.gz"

        correction = SdssIvarCorrection({"filename": in_file})
        self.assertTrue(isinstance(correction, Correction))

if __name__ == '__main__':
    unittest.main()
