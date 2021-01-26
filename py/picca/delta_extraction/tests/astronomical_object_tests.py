"""This file contains configuration tests"""
import unittest
import numpy as np

from picca.delta_extraction.astronomical_object import AstronomicalObject
from picca.delta_extraction.errors import AstronomicalObjectError

from picca.delta_extraction.astronomical_objects.forest import Forest
from picca.delta_extraction.astronomical_objects.sdss_forest import SdssForest
from picca.delta_extraction.astronomical_objects.desi_forest import DesiForest

from picca.delta_extraction.tests.abstract_test import AbstractTest

class TestAstronomicalObject(AbstractTest):
    """Test AstronomicalObject and its childs."""

    def tearDown(self):
        # reset SdssForest class variables
        SdssForest.delta_log_lambda = None
        SdssForest.log_lambda_max = None
        SdssForest.log_lambda_max_rest_frame = None
        SdssForest.log_lambda_min = None
        SdssForest.log_lambda_min_rest_frame = None

        # reset DesiForest class variables
        DesiForest.delta_log_lambda = None
        DesiForest.log_lambda_max = None
        DesiForest.log_lambda_max_rest_frame = None
        DesiForest.log_lambda_min = None
        DesiForest.log_lambda_min_rest_frame = None

    def test_astronomical_object(self):
        """Test constructor for AstronomicalObject."""
        kwargs = {
            "los_id": 9999,
            "ra": 0.15,
            "dec": 0.0,
            "z": 2.1,
        }
        test_obj = AstronomicalObject(**kwargs)
        self.assertTrue(test_obj.los_id == 9999)
        self.assertTrue(test_obj.healpix == 1505)
        self.assertTrue(test_obj.z == 2.1)

    def test_astronomical_object_comparison(self):
        """Test comparison between instances of AstronomicalObject."""
        kwargs = {
            "los_id": 9999,
            "ra": 0.15,
            "dec": 0.0,
            "z": 2.1,
        }
        test_obj = AstronomicalObject(**kwargs)

        kwargs_gt = {
            "healpix_ordering": {"los_id": 9999, "ra": 0.0, "dec": 0.0, "z": 2.1},
            "ra_ordering": {"los_id": 9999, "ra": 0.1, "dec": 0.0, "z": 2.1},
            "dec_ordering": {"los_id": 9999, "ra": 0.15, "dec": -0.01, "z": 2.1},
            "z_ordering": {"los_id": 9999, "ra": 0.15, "dec": 0.0, "z": 2.0},
        }
        for kwargs_other in kwargs_gt.values():
            other = AstronomicalObject(**kwargs_other)
            self.assertTrue(test_obj > other)
            self.assertTrue(test_obj != other)
            self.assertFalse(test_obj == other)
            self.assertFalse(test_obj < other)

        # equal objects
        kwargs_other = {
            "los_id": 1234,
            "ra": 0.15,
            "dec": 0.0,
            "z": 2.1,
        }
        other = AstronomicalObject(**kwargs_other)
        self.assertFalse(test_obj > other)
        self.assertTrue(test_obj == other)
        self.assertFalse(test_obj < other)

    def test_forest(self):
        """Test constructor for Forest object."""
        # create a Forest
        kwargs = {
            "los_id": 9999,
            "ra": 0.15,
            "dec": 0.0,
            "z": 2.1,
            "flux": np.ones(15),
            "ivar": np.ones(15)*4,
        }
        test_obj = Forest(**kwargs)
        self.assertTrue(isinstance(test_obj, AstronomicalObject))
        self.assertTrue(test_obj.los_id == 9999)
        self.assertTrue(test_obj.healpix == 1505)
        self.assertTrue(test_obj.z == 2.1)
        self.assertTrue(test_obj.bad_continuum_reason is None)
        self.assertTrue(test_obj.continuum is None)
        self.assertTrue(test_obj.deltas is None)
        self.assertTrue(np.allclose(test_obj.flux, np.ones(15)))
        self.assertTrue(np.allclose(test_obj.ivar, np.ones(15)*4))
        self.assertTrue(len(test_obj.mask_fields) == 3)
        self.assertTrue(test_obj.mask_fields[0] == "flux")
        self.assertTrue(test_obj.mask_fields[1] == "ivar")
        self.assertTrue(test_obj.mask_fields[2] == "transmission_correction")
        self.assertTrue(np.allclose(test_obj.transmission_correction, np.ones(15)))
        self.assertTrue(np.allclose(test_obj.mean_snr, 2))

        # create a Forest specifying all variables
        kwargs = {
            "los_id": 9999,
            "ra": 0.15,
            "dec": 0.0,
            "z": 2.1,
            "flux": np.ones(15),
            "ivar": np.ones(15)*4,
            "continuum": np.ones(15),
            "deltas": np.zeros(15),
            "mask fields": ["flux"],
        }
        test_obj = Forest(**kwargs)
        self.assertTrue(isinstance(test_obj, AstronomicalObject))
        self.assertTrue(test_obj.los_id == 9999)
        self.assertTrue(test_obj.healpix == 1505)
        self.assertTrue(test_obj.z == 2.1)
        self.assertTrue(test_obj.bad_continuum_reason is None)
        self.assertTrue(np.allclose(test_obj.continuum, np.ones(15)))
        self.assertTrue(np.allclose(test_obj.deltas, np.zeros(15)))
        self.assertTrue(np.allclose(test_obj.flux, np.ones(15)))
        self.assertTrue(np.allclose(test_obj.ivar, np.ones(15)*4))
        self.assertTrue(len(test_obj.mask_fields) == 1)
        self.assertTrue(test_obj.mask_fields[0] == "flux")
        self.assertTrue(np.allclose(test_obj.transmission_correction, np.ones(15)))
        self.assertTrue(np.allclose(test_obj.mean_snr, 2))

        # create a Forest with extra variables
        kwargs = {
            "los_id": 9999,
            "ra": 0.15,
            "dec": 0.0,
            "z": 2.1,
            "flux": np.ones(15),
            "ivar": np.ones(15)*4,
            "test variable": "test",
        }
        test_obj = Forest(**kwargs)
        self.assertTrue(isinstance(test_obj, AstronomicalObject))
        self.assertTrue(test_obj.los_id == 9999)
        self.assertTrue(test_obj.healpix == 1505)
        self.assertTrue(test_obj.z == 2.1)
        self.assertTrue(test_obj.bad_continuum_reason is None)
        self.assertTrue(test_obj.continuum is None)
        self.assertTrue(test_obj.deltas is None)
        self.assertTrue(np.allclose(test_obj.flux, np.ones(15)))
        self.assertTrue(np.allclose(test_obj.ivar, np.ones(15)*4))
        self.assertTrue(len(test_obj.mask_fields) == 3)
        self.assertTrue(test_obj.mask_fields[0] == "flux")
        self.assertTrue(test_obj.mask_fields[1] == "ivar")
        self.assertTrue(test_obj.mask_fields[2] == "transmission_correction")
        self.assertTrue(np.allclose(test_obj.transmission_correction, np.ones(15)))
        self.assertTrue(np.allclose(test_obj.mean_snr, 2))

        # create a Forest with missing AstronomicalObject variables
        kwargs = {
            "flux": np.ones(15),
            "ivar": np.ones(15)*4,
        }
        with self.assertRaises(AstronomicalObjectError):
            Forest(**kwargs)

        # create a Forest with missing Forest variables
        kwargs = {
            "los_id": 9999,
            "ra": 0.15,
            "dec": 0.0,
            "z": 2.1,
            "ivar": np.ones(15)*4,
        }
        with self.assertRaises(AstronomicalObjectError):
            Forest(**kwargs)

    def test_forest_rebin(self):
        """Test rebin function in Forest."""
        # create a Forest
        kwargs = {
            "los_id": 9999,
            "ra": 0.15,
            "dec": 0.0,
            "z": 2.1,
            "flux": np.arange(10),
            "ivar": np.ones(10)*4,
        }
        test_obj = Forest(**kwargs)

        # rebin
        bins = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
        test_obj.rebin(bins)

        self.assertTrue(isinstance(test_obj, AstronomicalObject))
        self.assertTrue(np.allclose(test_obj.flux, np.array([0.5, 2.5, 4.5, 6.5, 8.5])))
        self.assertTrue(np.allclose(test_obj.ivar, np.ones(5)*8))
        self.assertTrue(np.allclose(test_obj.mean_snr, 12.727922061357855))

    def test_sdss_object(self):
        """Test constructor for SdssForest.
        This includes a test of function rebin.
        """
        # create a SdssForest
        kwargs = {
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
        # expected error as class variables are not yet set
        with self.assertRaises(AstronomicalObjectError):
            SdssForest(**kwargs)
            print(SdssForest.delta_log_lambda)
            print(SdssForest.log_lambda_max)
            print(SdssForest.log_lambda_min)

        # set class variables
        SdssForest.delta_log_lambda = 1e-4
        SdssForest.log_lambda_max = np.log10(5500.0)
        SdssForest.log_lambda_max_rest_frame = np.log10(1200.0)
        SdssForest.log_lambda_min = np.log10(3600.0)
        SdssForest.log_lambda_min_rest_frame = np.log10(1040.0)

        # create a SdssForest
        test_obj = SdssForest(**kwargs)

        self.assertTrue(isinstance(test_obj, AstronomicalObject))
        self.assertTrue(test_obj.los_id == 100000000)
        self.assertTrue(test_obj.healpix == 1505)
        self.assertTrue(test_obj.z == 2.1)
        self.assertTrue(test_obj.bad_continuum_reason is None)
        self.assertTrue(test_obj.continuum is None)
        self.assertTrue(test_obj.deltas is None)
        self.assertTrue(np.allclose(test_obj.flux, np.ones(5)))
        self.assertTrue(np.allclose(test_obj.log_lambda, np.array([3.556525,
                                                                   3.556725,
                                                                   3.556925,
                                                                   3.557125,
                                                                   3.557325])))
        self.assertTrue(np.allclose(test_obj.ivar, np.ones(5)*8))
        self.assertTrue(len(test_obj.mask_fields) == 4)
        self.assertTrue(test_obj.mask_fields[0] == "flux")
        self.assertTrue(test_obj.mask_fields[1] == "ivar")
        self.assertTrue(test_obj.mask_fields[2] == "transmission_correction")
        self.assertTrue(test_obj.mask_fields[3] == "log_lambda")
        self.assertTrue(np.allclose(test_obj.transmission_correction, np.ones(5)))
        self.assertTrue(np.allclose(test_obj.mean_snr, 2.8284271247461903))
        self.assertTrue(test_obj.plate == 0)
        self.assertTrue(test_obj.fiberid == 0)
        self.assertTrue(test_obj.mjd == 0)
        self.assertTrue(test_obj.thingid == 100000000)

        # create forest with extra variables
        kwargs = {
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
            "test variable": "test",
        }
        test_obj = SdssForest(**kwargs)
        self.assertTrue(isinstance(test_obj, AstronomicalObject))
        self.assertTrue(test_obj.los_id == 100000000)
        self.assertTrue(test_obj.healpix == 1505)
        self.assertTrue(test_obj.z == 2.1)
        self.assertTrue(test_obj.bad_continuum_reason is None)
        self.assertTrue(test_obj.continuum is None)
        self.assertTrue(test_obj.deltas is None)
        self.assertTrue(np.allclose(test_obj.flux, np.ones(5)))
        self.assertTrue(np.allclose(test_obj.log_lambda, np.array([3.556525,
                                                                   3.556725,
                                                                   3.556925,
                                                                   3.557125,
                                                                   3.557325])))
        self.assertTrue(np.allclose(test_obj.ivar, np.ones(5)*8))
        self.assertTrue(len(test_obj.mask_fields) == 4)
        self.assertTrue(test_obj.mask_fields[0] == "flux")
        self.assertTrue(test_obj.mask_fields[1] == "ivar")
        self.assertTrue(test_obj.mask_fields[2] == "transmission_correction")
        self.assertTrue(test_obj.mask_fields[3] == "log_lambda")
        self.assertTrue(np.allclose(test_obj.transmission_correction, np.ones(5)))
        self.assertTrue(np.allclose(test_obj.mean_snr, 2.8284271247461903))
        self.assertTrue(test_obj.plate == 0)
        self.assertTrue(test_obj.fiberid == 0)
        self.assertTrue(test_obj.mjd == 0)
        self.assertTrue(test_obj.thingid == 100000000)

        # create a SdssForest with missing SdssForest variables
        kwargs = {
            "ra": 0.15,
            "dec": 0.0,
            "z": 2.1,
            "flux": np.ones(15),
            "ivar": np.ones(15)*4,
        }
        with self.assertRaises(AstronomicalObjectError):
            SdssForest(**kwargs)

        # create forest with missing Forest variables
        kwargs = {
            "los_id": 9999,
            "ra": 0.15,
            "dec": 0.0,
            "z": 2.1,
            "ivar": np.ones(15)*4,
            "thingid": 100000000,
            "plate": 0,
            "fiberid": 0,
            "mjd": 0,
        }
        with self.assertRaises(AstronomicalObjectError):
            SdssForest(**kwargs)

    def test_sdss_object_coadd(self):
        """Test the coadd function in SdssForest"""
        # set class variables
        SdssForest.delta_log_lambda = 1e-4
        SdssForest.log_lambda_max = np.log10(5500.0)
        SdssForest.log_lambda_max_rest_frame = np.log10(1200.0)
        SdssForest.log_lambda_min = np.log10(3600.0)
        SdssForest.log_lambda_min_rest_frame = np.log10(1040.0)

        # create a SdssForest
        kwargs = {
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
        test_obj = SdssForest(**kwargs)

        # create a second SdssForest
        kwargs_other = {
            "ra": 0.1,
            "dec": 0.01,
            "z": 2.2,
            "flux": np.ones(10)*3,
            "ivar": np.ones(10)*4,
            "log_lambda": np.array([3.5565, 3.55655, 3.5567, 3.55675, 3.5569,
                                    3.55695, 3.5571, 3.55715, 3.5573, 3.55735]),
            "thingid": 1000002000,
            "plate": 1,
            "fiberid": 2,
            "mjd": 3,
        }
        test_obj_other = SdssForest(**kwargs_other)

        # coadd them
        test_obj.coadd(test_obj_other)

        self.assertTrue(test_obj.los_id == 100000000)
        self.assertTrue(test_obj.healpix == 1505)
        self.assertTrue(test_obj.z == 2.1)
        self.assertTrue(test_obj.bad_continuum_reason is None)
        self.assertTrue(test_obj.continuum is None)
        self.assertTrue(test_obj.deltas is None)
        self.assertTrue(np.allclose(test_obj.flux, np.ones(5)*2))
        self.assertTrue(np.allclose(test_obj.log_lambda, np.array([3.556525,
                                                                   3.556725,
                                                                   3.556925,
                                                                   3.557125,
                                                                   3.557325])))
        self.assertTrue(np.allclose(test_obj.ivar, np.ones(5)*16))
        self.assertTrue(len(test_obj.mask_fields) == 4)
        self.assertTrue(test_obj.mask_fields[0] == "flux")
        self.assertTrue(test_obj.mask_fields[1] == "ivar")
        self.assertTrue(test_obj.mask_fields[2] == "transmission_correction")
        self.assertTrue(test_obj.mask_fields[3] == "log_lambda")
        self.assertTrue(np.allclose(test_obj.transmission_correction, np.ones(5)))
        self.assertTrue(np.allclose(test_obj.mean_snr, 8))
        self.assertTrue(test_obj.plate == 0)
        self.assertTrue(test_obj.fiberid == 0)
        self.assertTrue(test_obj.mjd == 0)
        self.assertTrue(test_obj.thingid == 100000000)

    def test_desi_object(self):
        """Test constructor for DesiForest.
        This includes a test of function rebin.
        """
        # create a SdssForest
        kwargs = {
            "ra": 0.15,
            "dec": 0.0,
            "z": 2.1,
            "flux": np.ones(10),
            "ivar": np.ones(10)*4,
            "lambda": np.array([3610, 3610.4, 3650, 3650.4, 3670, 3670.4,
                                3680, 3680.4, 3700, 3700.4]),
            "targetid": 100000000,
            "night": 0,
            "petal": 0,
            "tile": 0,
        }
        # expected error as class variables are not yet set
        with self.assertRaises(AstronomicalObjectError):
            DesiForest(**kwargs)

        # set class variables
        DesiForest.delta_lambda = 1.
        DesiForest.lambda_max = 5500.0
        DesiForest.lambda_max_rest_frame = 1200.0
        DesiForest.lambda_min = 3600.0
        DesiForest.lambda_min_rest_frame = 1040.0

        # create a SdssForest
        test_obj = DesiForest(**kwargs)

        self.assertTrue(isinstance(test_obj, AstronomicalObject))
        self.assertTrue(test_obj.los_id == 100000000)
        self.assertTrue(test_obj.healpix == 1505)
        self.assertTrue(test_obj.z == 2.1)
        self.assertTrue(test_obj.bad_continuum_reason is None)
        self.assertTrue(test_obj.continuum is None)
        self.assertTrue(test_obj.deltas is None)
        self.assertTrue(np.allclose(test_obj.flux, np.ones(5)))
        self.assertTrue(np.allclose(test_obj.lambda_, np.array([3610,
                                                                3650,
                                                                3670,
                                                                3680,
                                                                3700])))
        self.assertTrue(np.allclose(test_obj.ivar, np.ones(5)*8))
        self.assertTrue(len(test_obj.mask_fields) == 4)
        self.assertTrue(test_obj.mask_fields[0] == "flux")
        self.assertTrue(test_obj.mask_fields[1] == "ivar")
        self.assertTrue(test_obj.mask_fields[2] == "transmission_correction")
        self.assertTrue(test_obj.mask_fields[3] == "lambda_")
        self.assertTrue(np.allclose(test_obj.transmission_correction, np.ones(5)))
        self.assertTrue(np.allclose(test_obj.mean_snr, 2.8284271247461903))
        self.assertTrue(test_obj.night == 0)
        self.assertTrue(test_obj.petal == 0)
        self.assertTrue(test_obj.tile == 0)
        self.assertTrue(test_obj.targetid == 100000000)

        # create forest with extra variables
        kwargs = {
            "ra": 0.15,
            "dec": 0.0,
            "z": 2.1,
            "flux": np.ones(10),
            "ivar": np.ones(10)*4,
            "lambda": np.array([3610, 3610.4, 3650, 3650.4, 3670, 3670.4,
                                3680, 3680.4, 3700, 3700.4]),
            "targetid": 100000000,
            "night": 0,
            "petal": 0,
            "tile": 0,
            "test variable": "test",
        }
        test_obj = DesiForest(**kwargs)
        self.assertTrue(isinstance(test_obj, AstronomicalObject))
        self.assertTrue(test_obj.los_id == 100000000)
        self.assertTrue(test_obj.healpix == 1505)
        self.assertTrue(test_obj.z == 2.1)
        self.assertTrue(test_obj.bad_continuum_reason is None)
        self.assertTrue(test_obj.continuum is None)
        self.assertTrue(test_obj.deltas is None)
        self.assertTrue(np.allclose(test_obj.flux, np.ones(5)))
        self.assertTrue(np.allclose(test_obj.lambda_, np.array([3610,
                                                                3650,
                                                                3670,
                                                                3680,
                                                                3700])))
        self.assertTrue(np.allclose(test_obj.ivar, np.ones(5)*8))
        self.assertTrue(len(test_obj.mask_fields) == 4)
        self.assertTrue(test_obj.mask_fields[0] == "flux")
        self.assertTrue(test_obj.mask_fields[1] == "ivar")
        self.assertTrue(test_obj.mask_fields[2] == "transmission_correction")
        self.assertTrue(test_obj.mask_fields[3] == "lambda_")
        self.assertTrue(np.allclose(test_obj.transmission_correction, np.ones(5)))
        self.assertTrue(np.allclose(test_obj.mean_snr, 2.8284271247461903))
        self.assertTrue(test_obj.night == 0)
        self.assertTrue(test_obj.petal == 0)
        self.assertTrue(test_obj.tile == 0)
        self.assertTrue(test_obj.targetid == 100000000)

        # create a SdssForest with missing SdssForest variables
        kwargs = {
            "ra": 0.15,
            "dec": 0.0,
            "z": 2.1,
            "flux": np.ones(15),
            "ivar": np.ones(15)*4,
        }
        with self.assertRaises(AstronomicalObjectError):
            SdssForest(**kwargs)

        # create forest with missing Forest variables
        kwargs = {
            "los_id": 9999,
            "ra": 0.15,
            "dec": 0.0,
            "z": 2.1,
            "ivar": np.ones(15)*4,
            "targetid": 100000000,
            "night": 0,
            "petal": 0,
            "fiber": 0,
        }
        with self.assertRaises(AstronomicalObjectError):
            SdssForest(**kwargs)

    def test_desi_object_coadd(self):
        """Test the coadd function in DesiForest"""
        # set class variables
        DesiForest.delta_log_lambda = 1e-4
        DesiForest.log_lambda_max = np.log10(5500.0)
        DesiForest.log_lambda_max_rest_frame = np.log10(1200.0)
        DesiForest.log_lambda_min = np.log10(3600.0)
        DesiForest.log_lambda_min_rest_frame = np.log10(1040.0)

        # create a DesiForest
        kwargs = {
            "ra": 0.15,
            "dec": 0.0,
            "z": 2.1,
            "flux": np.ones(10),
            "ivar": np.ones(10)*4,
            "lambda": np.array([3610, 3610.4, 3650, 3650.4, 3670, 3670.4,
                                3680, 3680.4, 3700, 3700.4]),
            "targetid": 100000000,
            "night": 0,
            "petal": 0,
            "tile": 0,
        }
        test_obj = DesiForest(**kwargs)

        # create a second DesiForest
        kwargs_other = {
            "ra": 0.1,
            "dec": 0.01,
            "z": 2.2,
            "flux": np.ones(10)*3,
            "ivar": np.ones(10)*4,
            "lambda": np.array([3610, 3610.4, 3650, 3650.4, 3670, 3670.4,
                                3680, 3680.4, 3700, 3700.4]),
            "targetid": 100010000,
            "night": 1,
            "petal": 2,
            "tile": 3,
        }
        test_obj_other = DesiForest(**kwargs_other)

        # coadd them
        test_obj.coadd(test_obj_other)

        self.assertTrue(test_obj.los_id == 100000000)
        self.assertTrue(test_obj.healpix == 1505)
        self.assertTrue(test_obj.z == 2.1)
        self.assertTrue(test_obj.bad_continuum_reason is None)
        self.assertTrue(test_obj.continuum is None)
        self.assertTrue(test_obj.deltas is None)
        self.assertTrue(np.allclose(test_obj.flux, np.ones(5)*2))
        self.assertTrue(np.allclose(test_obj.lambda_, np.array([3610,
                                                                3650,
                                                                3670,
                                                                3680,
                                                                3700])))
        self.assertTrue(np.allclose(test_obj.ivar, np.ones(5)*16))
        self.assertTrue(len(test_obj.mask_fields) == 4)
        self.assertTrue(test_obj.mask_fields[0] == "flux")
        self.assertTrue(test_obj.mask_fields[1] == "ivar")
        self.assertTrue(test_obj.mask_fields[2] == "transmission_correction")
        self.assertTrue(test_obj.mask_fields[3] == "lambda_")
        self.assertTrue(np.allclose(test_obj.transmission_correction, np.ones(5)))
        self.assertTrue(np.allclose(test_obj.mean_snr, 8))
        self.assertTrue(test_obj.night == 0)
        self.assertTrue(test_obj.petal == 0)
        self.assertTrue(test_obj.tile == 0)
        self.assertTrue(test_obj.targetid == 100000000)

if __name__ == '__main__':
    unittest.main()
