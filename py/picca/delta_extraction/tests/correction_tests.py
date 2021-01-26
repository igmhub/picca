"""This file contains configuration tests"""
import os
import unittest
import numpy as np

from picca.delta_extraction.astronomical_objects.forest import Forest
from picca.delta_extraction.astronomical_objects.sdss_forest import SdssForest
from picca.delta_extraction.correction import Correction
from picca.delta_extraction.errors import CorrectionError
from picca.delta_extraction.corrections.sdss_calibration_correction import SdssCalibrationCorrection
from picca.delta_extraction.corrections.sdss_dust_correction import SdssDustCorrection
from picca.delta_extraction.corrections.sdss_ivar_correction import SdssIvarCorrection
from picca.delta_extraction.corrections.sdss_optical_depth_correction import SdssOpticalDepthCorrection

from picca.delta_extraction.tests.abstract_test import AbstractTest

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class TestCorrection(AbstractTest):
    """Test Correction and its childs."""

    def test_correction(self):
        """Tests Abstract class Correction

        Load a Correction instace and check that it cannot be
        initialized.
        """
        with self.assertRaises(CorrectionError):
            # create Correction instance
            correction = Correction()
            # create forest instance
            kwargs = {
                "los_id": 9999,
                "ra": 0.15,
                "dec": 0.0,
                "z": 2.1,
                "flux": np.ones(15),
                "ivar": np.ones(15)*4,
            }
            forest = Forest(**kwargs)
            # run apply_correction, this should raise CorrectionError
            correction.apply_correction(forest)

    def test_sdss_calibration_correction(self):
        """Tests correct initialisation and inheritance for class
        SdssCalibrationCorrection

        Load a SdssCalibrationCorrection instace and check that it is
        correctly initialized.
        """
        in_file = f"{THIS_DIR}/data/dummy_corrections.fits.gz"

        # create SdssCorrection instance
        correction = SdssCalibrationCorrection({"filename": in_file})
        self.assertTrue(isinstance(correction, Correction))

        # setup SdssForest class variables
        SdssForest.delta_log_lambda = 1e-4
        SdssForest.log_lambda_max = np.log10(5500.0)
        SdssForest.log_lambda_max_rest_frame = np.log10(1200.0)
        SdssForest.log_lambda_min = np.log10(3600.0)
        SdssForest.log_lambda_min_rest_frame = np.log10(1040.0)

        # create SdssForest instance
        kwargs = {
            "los_id": 9999,
            "ra": 0.15,
            "dec": 0.0,
            "z": 2.1,
            "flux": np.ones(10),
            "ivar": np.ones(10)*4,
            "log_lambda": np.array([3.5565, 3.55655, 3.5567, 3.55675, 3.5569,
                                    3.55695, 3.5571, 3.55715, 3.5573, 3.55735]),
            "thingid": 100000000,
            "plate": 0,
            "fiberid": 0,
            "mjd": 0,
        }
        forest = SdssForest(**kwargs)
        self.assertTrue(np.allclose(forest.flux, np.ones(5)))
        self.assertTrue(np.allclose(forest.log_lambda, np.array([3.556525,
                                                                 3.556725,
                                                                 3.556925,
                                                                 3.557125,
                                                                 3.557325])))
        self.assertTrue(np.allclose(forest.ivar, np.ones(5)*8))

        # apply the correction
        correction.apply_correction(forest)

        self.assertTrue(np.allclose(forest.flux, np.ones(5)*0.5))
        self.assertTrue(np.allclose(forest.log_lambda, np.array([3.556525,
                                                                 3.556725,
                                                                 3.556925,
                                                                 3.557125,
                                                                 3.557325])))
        self.assertTrue(np.allclose(forest.ivar, np.ones(5)*32))

    def test_sdss_dust_correction(self):
        """Tests correct initialisation and inheritance for class
        SdssDustCorrection

        Load a SdssDustCorrection instace and check that it is
        correctly initialized.

        #TODO:Check that the function apply_correction
        is correctly implemented in a dummy Forest instance

        """
        in_file = f"{THIS_DIR}/data/dummy_corrections.fits.gz"

        correction = SdssDustCorrection({"filename": in_file})
        self.assertTrue(isinstance(correction, Correction))

        # setup SdssForest class variables
        SdssForest.delta_log_lambda = 1e-4
        SdssForest.log_lambda_max = np.log10(5500.0)
        SdssForest.log_lambda_max_rest_frame = np.log10(1200.0)
        SdssForest.log_lambda_min = np.log10(3600.0)
        SdssForest.log_lambda_min_rest_frame = np.log10(1040.0)

        # create SdssForest instance
        kwargs = {
            "los_id": 9999,
            "ra": 0.15,
            "dec": 0.0,
            "z": 2.1,
            "flux": np.ones(10),
            "ivar": np.ones(10)*4,
            "log_lambda": np.array([3.5565, 3.55655, 3.5567, 3.55675, 3.5569,
                                    3.55695, 3.5571, 3.55715, 3.5573, 3.55735]),
            "thingid": 100000,
            "plate": 0,
            "fiberid": 0,
            "mjd": 0,
        }
        forest = SdssForest(**kwargs)
        self.assertTrue(np.allclose(forest.flux, np.ones(5)))
        self.assertTrue(np.allclose(forest.log_lambda, np.array([3.556525,
                                                                 3.556725,
                                                                 3.556925,
                                                                 3.557125,
                                                                 3.557325])))
        self.assertTrue(np.allclose(forest.ivar, np.ones(5)*8))

        # apply the correction
        correction.apply_correction(forest)

        self.assertTrue(np.allclose(forest.flux, [3.20135542, 3.20003281,
                                                  3.19871122, 3.19739066,
                                                  3.19607111]))
        self.assertTrue(np.allclose(forest.log_lambda, np.array([3.556525,
                                                                 3.556725,
                                                                 3.556925,
                                                                 3.557125,
                                                                 3.557325])))
        self.assertTrue(np.allclose(forest.ivar, [0.78058859,
                                                  0.78123398,
                                                  0.78187967,
                                                  0.78252565,
                                                  0.78317194]))

    def test_sdss_ivar_correction(self):
        """Tests correct initialisation and inheritance for class
        SdssIvarCorrection

        Load a SdssIvarCorrection instace and check that it is
        correctly initialized.
        """
        in_file = f"{THIS_DIR}/data/dummy_corrections.fits.gz"

        correction = SdssIvarCorrection({"filename": in_file})
        self.assertTrue(isinstance(correction, Correction))

        # setup SdssForest class variables
        SdssForest.delta_log_lambda = 1e-4
        SdssForest.log_lambda_max = np.log10(5500.0)
        SdssForest.log_lambda_max_rest_frame = np.log10(1200.0)
        SdssForest.log_lambda_min = np.log10(3600.0)
        SdssForest.log_lambda_min_rest_frame = np.log10(1040.0)

        # create SdssForest instance
        kwargs = {
            "los_id": 9999,
            "ra": 0.15,
            "dec": 0.0,
            "z": 2.1,
            "flux": np.ones(10),
            "ivar": np.ones(10)*4,
            "log_lambda": np.array([3.5565, 3.55655, 3.5567, 3.55675, 3.5569,
                                    3.55695, 3.5571, 3.55715, 3.5573, 3.55735]),
            "thingid": 100000000,
            "plate": 0,
            "fiberid": 0,
            "mjd": 0,
        }
        forest = SdssForest(**kwargs)
        self.assertTrue(np.allclose(forest.flux, np.ones(5)))
        self.assertTrue(np.allclose(forest.log_lambda, np.array([3.556525,
                                                                 3.556725,
                                                                 3.556925,
                                                                 3.557125,
                                                                 3.557325])))
        self.assertTrue(np.allclose(forest.ivar, np.ones(5)*8))

        # apply the correction
        correction.apply_correction(forest)

        self.assertTrue(np.allclose(forest.flux, np.ones(5)))
        self.assertTrue(np.allclose(forest.log_lambda, np.array([3.556525,
                                                                 3.556725,
                                                                 3.556925,
                                                                 3.557125,
                                                                 3.557325])))
        self.assertTrue(np.allclose(forest.ivar, np.ones(5)*4))

    def test_sdss_optical_depth_correction(self):
        """Tests correct initialisation and inheritance for class
        OpticalDepthCorrection

        Load a SdssIvarCorrection instace and check that it is
        correctly initialized.
        """
        in_file = f"{THIS_DIR}/data/dummy_corrections.fits.gz"

        correction = SdssOpticalDepthCorrection({"filename": in_file,
                                                 "optical depth tau": "1",
                                                 "optical depth gamma": "0",
                                                 "optical depth absorber": "LYA",
                                                 })
        self.assertTrue(isinstance(correction, Correction))
        # setup SdssForest class variables
        SdssForest.delta_log_lambda = 1e-4
        SdssForest.log_lambda_max = np.log10(5500.0)
        SdssForest.log_lambda_max_rest_frame = np.log10(1200.0)
        SdssForest.log_lambda_min = np.log10(3600.0)
        SdssForest.log_lambda_min_rest_frame = np.log10(1040.0)

        # create SdssForest instance
        kwargs = {
            "los_id": 9999,
            "ra": 0.15,
            "dec": 0.0,
            "z": 2.1,
            "flux": np.ones(10),
            "ivar": np.ones(10)*4,
            "log_lambda": np.array([3.5565, 3.55655, 3.5567, 3.55675, 3.5569,
                                    3.55695, 3.5571, 3.55715, 3.5573, 3.55735]),
            "thingid": 100000000,
            "plate": 0,
            "fiberid": 0,
            "mjd": 0,
        }
        forest = SdssForest(**kwargs)
        self.assertTrue(np.allclose(forest.flux, np.ones(5)))
        self.assertTrue(np.allclose(forest.log_lambda, np.array([3.556525,
                                                                 3.556725,
                                                                 3.556925,
                                                                 3.557125,
                                                                 3.557325])))
        self.assertTrue(np.allclose(forest.ivar, np.ones(5)*8))
        self.assertTrue(np.allclose(forest.transmission_correction, np.ones(5)))

        # apply the correction
        correction.apply_correction(forest)

        self.assertTrue(np.allclose(forest.flux, np.ones(5)))
        self.assertTrue(np.allclose(forest.log_lambda, np.array([3.556525,
                                                                 3.556725,
                                                                 3.556925,
                                                                 3.557125,
                                                                 3.557325])))
        self.assertTrue(np.allclose(forest.ivar, np.ones(5)*8))
        self.assertTrue(np.allclose(forest.transmission_correction,
                                    np.ones(5)*0.36787944117144233))


if __name__ == '__main__':
    unittest.main()
