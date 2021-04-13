"""This file contains tests related to AstronomicalObject and its childs"""
import unittest
import numpy as np

from picca.delta_extraction.astronomical_object import AstronomicalObject
from picca.delta_extraction.astronomical_objects.forest import Forest
from picca.delta_extraction.astronomical_objects.desi_forest import DesiForest
from picca.delta_extraction.astronomical_objects.desi_pk1d_forest import DesiPk1dForest
from picca.delta_extraction.astronomical_objects.pk1d_forest import Pk1dForest
from picca.delta_extraction.astronomical_objects.sdss_forest import SdssForest
from picca.delta_extraction.astronomical_objects.sdss_pk1d_forest import SdssPk1dForest
from picca.delta_extraction.errors import AstronomicalObjectError
from picca.delta_extraction.tests.abstract_test import AbstractTest
from picca.delta_extraction.tests.test_utils import (reset_forest, setup_forest,
                                                     setup_pk1d_forest)

class AstronomicalObjectTest(AbstractTest):
    """Test AstronomicalObject and its childs."""

    def setUp(self):
        reset_forest()

    def tearDown(self):
        reset_forest()

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

    def test_astronomical_object_get_header(self):
        """Test method get_header for AstronomicalObject."""
        kwargs = {
            "los_id": 9999,
            "ra": 0.15,
            "dec": 0.0,
            "z": 2.1,
        }
        test_obj = AstronomicalObject(**kwargs)

        header = test_obj.get_header()
        self.assertTrue(len(header) == 4)
        self.assertTrue(header[0].get("name") == "LOS_ID")
        self.assertTrue(header[0].get("value") == 9999)
        self.assertTrue(header[1].get("name") == "RA")
        self.assertTrue(header[1].get("value") == 0.15)
        self.assertTrue(header[2].get("name") == "DEC")
        self.assertTrue(header[2].get("value") == 0.0)
        self.assertTrue(header[3].get("name") == "Z")
        self.assertTrue(header[3].get("value") == 2.1)


    def test_desi_forest(self):
        """Test constructor for DesiForest.
        This includes a test of function rebin.
        """
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
        # expected error as class variables are not yet set
        with self.assertRaises(AstronomicalObjectError):
            DesiForest(**kwargs)

        # set Forest class variables
        setup_forest(wave_solution="lin")

        # create a DesiForest
        test_obj = DesiForest(**kwargs)

        self.assertTrue(isinstance(test_obj, Forest))
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
        self.assertTrue(len(Forest.mask_fields) == 4)
        self.assertTrue(Forest.mask_fields[0] == "flux")
        self.assertTrue(Forest.mask_fields[1] == "ivar")
        self.assertTrue(Forest.mask_fields[2] == "transmission_correction")
        self.assertTrue(Forest.mask_fields[3] == "lambda_")
        self.assertTrue(np.allclose(test_obj.transmission_correction, np.ones(5)))
        self.assertTrue(np.allclose(test_obj.mean_snr, 2.8284271247461903))
        self.assertTrue(isinstance(test_obj.night, list))
        self.assertTrue(test_obj.night[0] == 0)
        self.assertTrue(isinstance(test_obj.petal, list))
        self.assertTrue(test_obj.petal[0] == 0)
        self.assertTrue(isinstance(test_obj.tile, list))
        self.assertTrue(test_obj.tile[0] == 0)
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
        self.assertTrue(isinstance(test_obj, Forest))
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
        self.assertTrue(len(Forest.mask_fields) == 4)
        self.assertTrue(Forest.mask_fields[0] == "flux")
        self.assertTrue(Forest.mask_fields[1] == "ivar")
        self.assertTrue(Forest.mask_fields[2] == "transmission_correction")
        self.assertTrue(Forest.mask_fields[3] == "lambda_")
        self.assertTrue(np.allclose(test_obj.transmission_correction, np.ones(5)))
        self.assertTrue(np.allclose(test_obj.mean_snr, 2.8284271247461903))
        self.assertTrue(isinstance(test_obj.night, list))
        self.assertTrue(test_obj.night[0] == 0)
        self.assertTrue(isinstance(test_obj.petal, list))
        self.assertTrue(test_obj.petal[0] == 0)
        self.assertTrue(isinstance(test_obj.tile, list))
        self.assertTrue(test_obj.tile[0] == 0)
        self.assertTrue(test_obj.targetid == 100000000)

        # create a DesiForest with missing night, petal and tile
        kwargs = {
            "ra": 0.15,
            "dec": 0.0,
            "z": 2.1,
            "flux": np.ones(10),
            "ivar": np.ones(10)*4,
            "lambda": np.array([3610, 3610.4, 3650, 3650.4, 3670, 3670.4,
                                3680, 3680.4, 3700, 3700.4]),
            "targetid": 100000000,
        }

        test_obj = DesiForest(**kwargs)
        self.assertTrue(isinstance(test_obj, Forest))
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
        self.assertTrue(len(Forest.mask_fields) == 4)
        self.assertTrue(Forest.mask_fields[0] == "flux")
        self.assertTrue(Forest.mask_fields[1] == "ivar")
        self.assertTrue(Forest.mask_fields[2] == "transmission_correction")
        self.assertTrue(Forest.mask_fields[3] == "lambda_")
        self.assertTrue(np.allclose(test_obj.transmission_correction, np.ones(5)))
        self.assertTrue(np.allclose(test_obj.mean_snr, 2.8284271247461903))
        self.assertTrue(isinstance(test_obj.night, list))
        self.assertTrue(len(test_obj.night) == 0)
        self.assertTrue(isinstance(test_obj.petal, list))
        self.assertTrue(len(test_obj.petal) == 0)
        self.assertTrue(isinstance(test_obj.tile, list))
        self.assertTrue(len(test_obj.tile) == 0)
        self.assertTrue(test_obj.targetid == 100000000)


        # create a DesiForest with missing DesiForest variables
        kwargs = {
            "ra": 0.15,
            "dec": 0.0,
            "z": 2.1,
            "flux": np.ones(15),
            "ivar": np.ones(15)*4,
            "lambda": np.array([3610, 3610.4, 3650, 3650.4, 3670, 3670.4,
                                3680, 3680.4, 3700, 3700.4]),
        }
        with self.assertRaises(AstronomicalObjectError):
            DesiForest(**kwargs)

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
            DesiForest(**kwargs)

    def test_desi_forest_coadd(self):
        """Test the coadd function in DesiForest"""
        # set class variables
        setup_forest(wave_solution="lin")

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
            "targetid": 100000000,
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
        self.assertTrue(len(Forest.mask_fields) == 4)
        self.assertTrue(Forest.mask_fields[0] == "flux")
        self.assertTrue(Forest.mask_fields[1] == "ivar")
        self.assertTrue(Forest.mask_fields[2] == "transmission_correction")
        self.assertTrue(Forest.mask_fields[3] == "lambda_")
        self.assertTrue(np.allclose(test_obj.transmission_correction, np.ones(5)))
        self.assertTrue(np.allclose(test_obj.mean_snr, 8))
        self.assertTrue(isinstance(test_obj.night, list))
        self.assertTrue(test_obj.night[0] == 0)
        self.assertTrue(test_obj.night[1] == 1)
        self.assertTrue(isinstance(test_obj.petal, list))
        self.assertTrue(test_obj.petal[0] == 0)
        self.assertTrue(test_obj.petal[1] == 2)
        self.assertTrue(isinstance(test_obj.tile, list))
        self.assertTrue(test_obj.tile[0] == 0)
        self.assertTrue(test_obj.tile[1] == 3)
        self.assertTrue(test_obj.targetid == 100000000)

        # create a third DesiForest with different targetid
        kwargs_other = {
            "ra": 0.1,
            "dec": 0.01,
            "z": 2.2,
            "flux": np.ones(10)*3,
            "ivar": np.ones(10)*4,
            "lambda": np.array([3610, 3610.4, 3650, 3650.4, 3670, 3670.4,
                                3680, 3680.4, 3700, 3700.4]),
            "targetid": 100000010,
            "night": 1,
            "petal": 2,
            "tile": 3,
        }
        test_obj_other = DesiForest(**kwargs_other)

        # coadding them whould raise an error
        with self.assertRaises(AstronomicalObjectError):
            test_obj.coadd(test_obj_other)

    def test_desi_forest_get_data(self):
        """Test method get_data for DesiForest."""
        # set class variables
        setup_forest(wave_solution="lin")

        # create a DesiForest
        kwargs = {
            "ra": 0.15,
            "dec": 0.0,
            "z": 2.1,
            "flux": np.ones(15),
            "ivar": np.ones(15)*4,
            "lambda": np.ones(15),
            "targetid": 100000000,
            "night": 1,
            "petal": 2,
            "tile": 3,
        }
        test_obj = DesiForest(**kwargs)

        cols, names, units, comments = test_obj.get_data()
        self.assertTrue(len(names) == 4)
        self.assertTrue(len(cols) == 4)
        self.assertTrue(len(units) == 4)
        self.assertTrue(len(comments) == 4)
        self.assertTrue(names[0] == "LAMBDA")
        self.assertTrue(np.allclose(cols[0], np.ones(15)))
        self.assertTrue(units[0] == "Angstrom")
        self.assertTrue(comments[0] == "Lambda")
        self.assertTrue(names[1] == "CONT")
        self.assertTrue(np.allclose(cols[1], np.zeros(15)))
        self.assertTrue(units[1] == "Flux units")
        self.assertTrue(comments[1] == ("Quasar continuum if BAD_CONT is 'None'. "
                                        "Check input spectra for units"))
        self.assertTrue(names[2] == "DELTA")
        self.assertTrue(np.allclose(cols[2], np.zeros(15)))
        self.assertTrue(units[2] == "")
        self.assertTrue(comments[2] == "Delta field")
        self.assertTrue(names[3] == "WEIGHT")
        self.assertTrue(np.allclose(cols[3], np.zeros(15)))
        self.assertTrue(units[3] == "")
        self.assertTrue(comments[3] == "Pixel weights")

    def test_desi_forest_get_header(self):
        """Test method get_header for DesiForest."""
        # set class variables
        setup_forest(wave_solution="lin")

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
            "night": 1,
            "petal": 2,
            "tile": 3,
        }
        test_obj = DesiForest(**kwargs)

        # get header and test
        header = test_obj.get_header()
        self.assertTrue(len(header) == 10)
        self.assertTrue(header[0].get("name") == "LOS_ID")
        self.assertTrue(header[0].get("value") == 100000000)
        self.assertTrue(header[1].get("name") == "RA")
        self.assertTrue(header[1].get("value") == 0.15)
        self.assertTrue(header[2].get("name") == "DEC")
        self.assertTrue(header[2].get("value") == 0.0)
        self.assertTrue(header[3].get("name") == "Z")
        self.assertTrue(header[3].get("value") == 2.1)
        self.assertTrue(header[4].get("name") == "BAD_CONT")
        self.assertTrue(header[4].get("value") == "None")
        self.assertTrue(header[5].get("name") == "MEANSNR")
        self.assertTrue(np.isclose(header[5].get("value"), 2.828427))
        self.assertTrue(header[6].get("name") == "TARGETID")
        self.assertTrue(header[6].get("value") == 100000000)
        self.assertTrue(header[7].get("name") == "NIGHT")
        self.assertTrue(header[7].get("value") == "1")
        self.assertTrue(header[8].get("name") == "PETAL")
        self.assertTrue(header[8].get("value") == "2")
        self.assertTrue(header[9].get("name") == "TILE")
        self.assertTrue(header[9].get("value") == "3")

        # create a second DesiForest and coadd it to the first
        kwargs_other = {
            "ra": 0.1,
            "dec": 0.01,
            "z": 2.2,
            "flux": np.ones(10),
            "ivar": np.ones(10)*4,
            "lambda": np.array([3610, 3610.4, 3650, 3650.4, 3670, 3670.4,
                                3680, 3680.4, 3700, 3700.4]),
            "targetid": 100000000,
            "night": 1,
            "petal": 2,
            "tile": 3,
        }
        test_obj_other = DesiForest(**kwargs_other)
        test_obj.coadd(test_obj_other)

        # get header and test
        header = test_obj.get_header()
        self.assertTrue(len(header) == 10)
        self.assertTrue(header[0].get("name") == "LOS_ID")
        self.assertTrue(header[0].get("value") == 100000000)
        self.assertTrue(header[1].get("name") == "RA")
        self.assertTrue(header[1].get("value") == 0.15)
        self.assertTrue(header[2].get("name") == "DEC")
        self.assertTrue(header[2].get("value") == 0.0)
        self.assertTrue(header[3].get("name") == "Z")
        self.assertTrue(header[3].get("value") == 2.1)
        self.assertTrue(header[4].get("name") == "BAD_CONT")
        self.assertTrue(header[4].get("value") == "None")
        self.assertTrue(header[5].get("name") == "MEANSNR")
        self.assertTrue(header[5].get("value") == 4)
        self.assertTrue(header[6].get("name") == "TARGETID")
        self.assertTrue(header[6].get("value") == 100000000)
        self.assertTrue(header[7].get("name") == "NIGHT")
        self.assertTrue(header[7].get("value") == "1-1")
        self.assertTrue(header[8].get("name") == "PETAL")
        self.assertTrue(header[8].get("value") == "2-2")
        self.assertTrue(header[9].get("name") == "TILE")
        self.assertTrue(header[9].get("value") == "3-3")

    def test_desi_pk1d_forest(self):
        """Test constructor for DesiPk1dForest.
        This includes a test of function rebin.
        """
        # create a DesiPk1dForest
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
            "exposures_diff": np.ones(10),
            "reso": np.ones(10),
        }

        # create a DesiPk1dForest class variables are not yet set
        with self.assertRaises(AstronomicalObjectError):
            DesiPk1dForest(**kwargs)

        # set class variables
        setup_pk1d_forest("LYA")

        # expected error as Forest class variables are not yet set
        with self.assertRaises(AstronomicalObjectError):
            DesiPk1dForest(**kwargs)

        # set class variables
        setup_forest(wave_solution="lin")

        test_obj = DesiPk1dForest(**kwargs)

        self.assertTrue(isinstance(test_obj, Forest))
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
        self.assertTrue(len(Forest.mask_fields) == 6)
        self.assertTrue(Forest.mask_fields[0] == "flux")
        self.assertTrue(Forest.mask_fields[1] == "ivar")
        self.assertTrue(Forest.mask_fields[2] == "transmission_correction")
        self.assertTrue(Forest.mask_fields[3] == "lambda_")
        self.assertTrue(Forest.mask_fields[4] == "exposures_diff")
        self.assertTrue(Forest.mask_fields[5] == "reso")
        self.assertTrue(np.allclose(test_obj.transmission_correction, np.ones(5)))
        self.assertTrue(np.allclose(test_obj.mean_snr, 2.8284271247461903))
        self.assertTrue(isinstance(test_obj.night, list))
        self.assertTrue(test_obj.night[0] == 0)
        self.assertTrue(isinstance(test_obj.petal, list))
        self.assertTrue(test_obj.petal[0] == 0)
        self.assertTrue(isinstance(test_obj.tile, list))
        self.assertTrue(test_obj.tile[0] == 0)
        self.assertTrue(test_obj.targetid == 100000000)
        self.assertTrue(Pk1dForest.lambda_abs_igm == 1215.67)
        self.assertTrue(np.allclose(test_obj.exposures_diff, np.ones(5)))
        self.assertTrue(np.allclose(test_obj.reso, np.ones(5)))
        self.assertTrue(np.isclose(test_obj.mean_z, 2.0065725073416303))
        self.assertTrue(np.isclose(test_obj.mean_reso, 1.0))

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
            "exposures_diff": np.ones(10),
            "reso": np.ones(10),
        }
        test_obj = DesiPk1dForest(**kwargs)
        self.assertTrue(isinstance(test_obj, Forest))
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
        self.assertTrue(len(Forest.mask_fields) == 6)
        self.assertTrue(Forest.mask_fields[0] == "flux")
        self.assertTrue(Forest.mask_fields[1] == "ivar")
        self.assertTrue(Forest.mask_fields[2] == "transmission_correction")
        self.assertTrue(Forest.mask_fields[3] == "lambda_")
        self.assertTrue(Forest.mask_fields[4] == "exposures_diff")
        self.assertTrue(Forest.mask_fields[5] == "reso")
        self.assertTrue(np.allclose(test_obj.transmission_correction, np.ones(5)))
        self.assertTrue(np.allclose(test_obj.mean_snr, 2.8284271247461903))
        self.assertTrue(isinstance(test_obj.night, list))
        self.assertTrue(test_obj.night[0] == 0)
        self.assertTrue(isinstance(test_obj.petal, list))
        self.assertTrue(test_obj.petal[0] == 0)
        self.assertTrue(isinstance(test_obj.tile, list))
        self.assertTrue(test_obj.tile[0] == 0)
        self.assertTrue(test_obj.targetid == 100000000)
        self.assertTrue(Pk1dForest.lambda_abs_igm == 1215.67)
        self.assertTrue(np.allclose(test_obj.exposures_diff, np.ones(5)))
        self.assertTrue(np.allclose(test_obj.reso, np.ones(5)))
        self.assertTrue(np.isclose(test_obj.mean_z, 2.0065725073416303))
        self.assertTrue(np.isclose(test_obj.mean_reso, 1.0))

        # create a DesiPk1dForest with missing night, petal and tile
        kwargs = {
            "ra": 0.15,
            "dec": 0.0,
            "z": 2.1,
            "flux": np.ones(10),
            "ivar": np.ones(10)*4,
            "lambda": np.array([3610, 3610.4, 3650, 3650.4, 3670, 3670.4,
                                3680, 3680.4, 3700, 3700.4]),
            "targetid": 100000000,
            "exposures_diff": np.ones(10),
            "reso": np.ones(10),
        }

        test_obj = DesiPk1dForest(**kwargs)
        self.assertTrue(isinstance(test_obj, Forest))
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
        self.assertTrue(len(Forest.mask_fields) == 6)
        self.assertTrue(Forest.mask_fields[0] == "flux")
        self.assertTrue(Forest.mask_fields[1] == "ivar")
        self.assertTrue(Forest.mask_fields[2] == "transmission_correction")
        self.assertTrue(Forest.mask_fields[3] == "lambda_")
        self.assertTrue(Forest.mask_fields[4] == "exposures_diff")
        self.assertTrue(Forest.mask_fields[5] == "reso")
        self.assertTrue(np.allclose(test_obj.transmission_correction, np.ones(5)))
        self.assertTrue(np.allclose(test_obj.mean_snr, 2.8284271247461903))
        self.assertTrue(isinstance(test_obj.night, list))
        self.assertTrue(len(test_obj.night) == 0)
        self.assertTrue(isinstance(test_obj.petal, list))
        self.assertTrue(len(test_obj.petal) == 0)
        self.assertTrue(isinstance(test_obj.tile, list))
        self.assertTrue(len(test_obj.tile) == 0)
        self.assertTrue(test_obj.targetid == 100000000)
        self.assertTrue(Pk1dForest.lambda_abs_igm == 1215.67)
        self.assertTrue(np.allclose(test_obj.exposures_diff, np.ones(5)))
        self.assertTrue(np.allclose(test_obj.reso, np.ones(5)))
        self.assertTrue(np.isclose(test_obj.mean_z, 2.0065725073416303))
        self.assertTrue(np.isclose(test_obj.mean_reso, 1.0))

        # create a DesiForest with missing DesiForest variables
        kwargs = {
            "ra": 0.15,
            "dec": 0.0,
            "z": 2.1,
            "flux": np.ones(15),
            "ivar": np.ones(15)*4,
            "lambda": np.array([3610, 3610.4, 3650, 3650.4, 3670, 3670.4,
                                3680, 3680.4, 3700, 3700.4]),
            "targetid": 100000000,
            "exposures_diff": np.ones(10),
            "reso": np.ones(10),
        }
        with self.assertRaises(AstronomicalObjectError):
            DesiPk1dForest(**kwargs)

        # create a DesiForest with missing Pk1dForest variables
        kwargs = {
            "ra": 0.15,
            "dec": 0.0,
            "z": 2.1,
            "flux": np.ones(15),
            "ivar": np.ones(15)*4,
            "lambda": np.array([3610, 3610.4, 3650, 3650.4, 3670, 3670.4,
                                3680, 3680.4, 3700, 3700.4]),
            "targetid": 100000000,
        }
        with self.assertRaises(AstronomicalObjectError):
            DesiPk1dForest(**kwargs)

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
            "exposures_diff": np.ones(10),
            "reso": np.ones(10),
        }
        with self.assertRaises(AstronomicalObjectError):
            DesiPk1dForest(**kwargs)

    def test_desi_pk1d_forest_coadd(self):
        """Test the coadd function in DesiPk1d_Forest"""
        # set class variables
        setup_forest(wave_solution="lin")
        setup_pk1d_forest("LYA")

        # create a DesiPk1dForest
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
            "exposures_diff": np.ones(10),
            "reso": np.ones(10),
        }
        test_obj = DesiPk1dForest(**kwargs)

        # create a second DesiPk1dForest
        kwargs_other = {
            "ra": 0.1,
            "dec": 0.01,
            "z": 2.2,
            "flux": np.ones(10)*3,
            "ivar": np.ones(10)*4,
            "lambda": np.array([3610, 3610.4, 3650, 3650.4, 3670, 3670.4,
                                3680, 3680.4, 3700, 3700.4]),
            "targetid": 100000000,
            "night": 1,
            "petal": 2,
            "tile": 3,
            "exposures_diff": np.ones(10)*3,
            "reso": np.ones(10),
        }
        test_obj_other = DesiPk1dForest(**kwargs_other)

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
        self.assertTrue(len(Forest.mask_fields) == 6)
        self.assertTrue(Forest.mask_fields[0] == "flux")
        self.assertTrue(Forest.mask_fields[1] == "ivar")
        self.assertTrue(Forest.mask_fields[2] == "transmission_correction")
        self.assertTrue(Forest.mask_fields[3] == "lambda_")
        self.assertTrue(Forest.mask_fields[4] == "exposures_diff")
        self.assertTrue(Forest.mask_fields[5] == "reso")
        self.assertTrue(np.allclose(test_obj.transmission_correction, np.ones(5)))
        self.assertTrue(np.allclose(test_obj.mean_snr, 8))
        self.assertTrue(isinstance(test_obj.night, list))
        self.assertTrue(test_obj.night[0] == 0)
        self.assertTrue(test_obj.night[1] == 1)
        self.assertTrue(isinstance(test_obj.petal, list))
        self.assertTrue(test_obj.petal[0] == 0)
        self.assertTrue(test_obj.petal[1] == 2)
        self.assertTrue(isinstance(test_obj.tile, list))
        self.assertTrue(test_obj.tile[0] == 0)
        self.assertTrue(test_obj.tile[1] == 3)
        self.assertTrue(test_obj.targetid == 100000000)
        self.assertTrue(Pk1dForest.lambda_abs_igm == 1215.67)
        self.assertTrue(np.allclose(test_obj.exposures_diff, np.ones(5)*2))
        self.assertTrue(np.allclose(test_obj.reso, np.ones(5)))
        self.assertTrue(np.isclose(test_obj.mean_z, 2.0065725073416303))

        # create a third DesiPk1dForest with different targetid
        kwargs_other = {
            "ra": 0.1,
            "dec": 0.01,
            "z": 2.2,
            "flux": np.ones(10)*3,
            "ivar": np.ones(10)*4,
            "lambda": np.array([3610, 3610.4, 3650, 3650.4, 3670, 3670.4,
                                3680, 3680.4, 3700, 3700.4]),
            "targetid": 100000010,
            "night": 1,
            "petal": 2,
            "tile": 3,
            "exposures_diff": np.ones(10)*3,
            "reso": np.ones(10),
        }
        test_obj_other = DesiPk1dForest(**kwargs_other)

        # coadding them whould raise an error
        with self.assertRaises(AstronomicalObjectError):
            test_obj.coadd(test_obj_other)

    def test_desi_pk1d_forest_get_data(self):
        """Test method get_data for DesiPk1dForest."""
        # set class variables
        setup_forest(wave_solution="lin")
        setup_pk1d_forest("LYA")

        # create a DesiPk1dForest
        kwargs = {
            "ra": 0.15,
            "dec": 0.0,
            "z": 2.1,
            "flux": np.ones(15),
            "ivar": np.ones(15)*4,
            "lambda": np.ones(15),
            "targetid": 100000000,
            "night": 1,
            "petal": 2,
            "tile": 3,
            "exposures_diff": np.ones(15),
            "reso": np.ones(15),
        }
        test_obj = DesiPk1dForest(**kwargs)

        cols, names, units, comments = test_obj.get_data()
        self.assertTrue(len(names) == 5)
        self.assertTrue(len(cols) == 5)
        self.assertTrue(len(units) == 5)
        self.assertTrue(len(comments) == 5)
        self.assertTrue(names[0] == "LAMBDA")
        self.assertTrue(np.allclose(cols[0], np.ones(15)))
        self.assertTrue(units[0] == "Angstrom")
        self.assertTrue(comments[0] == "Lambda")
        self.assertTrue(names[1] == "CONT")
        self.assertTrue(np.allclose(cols[1], np.zeros(15)))
        self.assertTrue(units[1] == "Flux units")
        self.assertTrue(comments[1] == ("Quasar continuum if BAD_CONT is 'None'. "
                                        "Check input spectra for units"))
        self.assertTrue(names[2] == "DELTA")
        self.assertTrue(np.allclose(cols[2], np.zeros(15)))
        self.assertTrue(units[2] == "")
        self.assertTrue(comments[2] == "Delta field")
        self.assertTrue(names[3] == "WEIGHT")
        self.assertTrue(np.allclose(cols[3], np.zeros(15)))
        self.assertTrue(units[3] == "")
        self.assertTrue(comments[3] == "Pixel weights")
        self.assertTrue(names[4] == "DIFF")
        self.assertTrue(np.allclose(cols[4], np.ones(15)))
        self.assertTrue(units[4] == "Flux units")
        self.assertTrue(comments[4] == "Difference. Check input spectra for units")


    def test_desi_pk1d_forest_get_header(self):
        """Test method get_header for DesiPk1dForest."""
        # set class variables
        setup_forest(wave_solution="lin")
        setup_pk1d_forest("LYA")

        # create a DesiPk1dForest
        kwargs = {
            "ra": 0.15,
            "dec": 0.0,
            "z": 2.1,
            "flux": np.ones(10),
            "ivar": np.ones(10)*4,
            "lambda": np.array([3610, 3610.4, 3650, 3650.4, 3670, 3670.4,
                                3680, 3680.4, 3700, 3700.4]),
            "targetid": 100000000,
            "night": 1,
            "petal": 2,
            "tile": 3,
            "exposures_diff": np.ones(10),
            "reso": np.ones(10),
        }
        test_obj = DesiPk1dForest(**kwargs)

        # get header and test
        header = test_obj.get_header()
        self.assertTrue(len(header) == 12)
        self.assertTrue(header[0].get("name") == "LOS_ID")
        self.assertTrue(header[0].get("value") == 100000000)
        self.assertTrue(header[1].get("name") == "RA")
        self.assertTrue(header[1].get("value") == 0.15)
        self.assertTrue(header[2].get("name") == "DEC")
        self.assertTrue(header[2].get("value") == 0.0)
        self.assertTrue(header[3].get("name") == "Z")
        self.assertTrue(header[3].get("value") == 2.1)
        self.assertTrue(header[4].get("name") == "BAD_CONT")
        self.assertTrue(header[4].get("value") == "None")
        self.assertTrue(header[5].get("name") == "MEANSNR")
        self.assertTrue(np.isclose(header[5].get("value"), 2.828427))
        self.assertTrue(header[6].get("name") == "MEANZ")
        self.assertTrue(header[6].get("value") == 2.0065725073416303)
        self.assertTrue(header[7].get("name") == "MEANRESO")
        self.assertTrue(header[7].get("value") == 1.0)
        self.assertTrue(header[8].get("name") == "TARGETID")
        self.assertTrue(header[8].get("value") == 100000000)
        self.assertTrue(header[9].get("name") == "NIGHT")
        self.assertTrue(header[9].get("value") == "1")
        self.assertTrue(header[10].get("name") == "PETAL")
        self.assertTrue(header[10].get("value") == "2")
        self.assertTrue(header[11].get("name") == "TILE")
        self.assertTrue(header[11].get("value") == "3")

        # create a second DesiPk1dForest and coadd it to the first
        kwargs_other = {
            "ra": 0.1,
            "dec": 0.01,
            "z": 2.2,
            "flux": np.ones(10),
            "ivar": np.ones(10)*4,
            "lambda": np.array([3610, 3610.4, 3650, 3650.4, 3670, 3670.4,
                                3680, 3680.4, 3700, 3700.4]),
            "targetid": 100000000,
            "night": 1,
            "petal": 2,
            "tile": 3,
            "exposures_diff": np.ones(10),
            "reso": np.ones(10),
        }
        test_obj_other = DesiPk1dForest(**kwargs_other)
        test_obj.coadd(test_obj_other)

        # get header and test
        header = test_obj.get_header()
        self.assertTrue(len(header) == 12)
        self.assertTrue(header[0].get("name") == "LOS_ID")
        self.assertTrue(header[0].get("value") == 100000000)
        self.assertTrue(header[1].get("name") == "RA")
        self.assertTrue(header[1].get("value") == 0.15)
        self.assertTrue(header[2].get("name") == "DEC")
        self.assertTrue(header[2].get("value") == 0.0)
        self.assertTrue(header[3].get("name") == "Z")
        self.assertTrue(header[3].get("value") == 2.1)
        self.assertTrue(header[4].get("name") == "BAD_CONT")
        self.assertTrue(header[4].get("value") == "None")
        self.assertTrue(header[5].get("name") == "MEANSNR")
        self.assertTrue(header[5].get("value") == 4)
        self.assertTrue(header[6].get("name") == "MEANZ")
        self.assertTrue(header[6].get("value") == 2.0065725073416303)
        self.assertTrue(header[7].get("name") == "MEANRESO")
        self.assertTrue(header[7].get("value") == 1.0)
        self.assertTrue(header[8].get("name") == "TARGETID")
        self.assertTrue(header[8].get("value") == 100000000)
        self.assertTrue(header[9].get("name") == "NIGHT")
        self.assertTrue(header[9].get("value") == "1-1")
        self.assertTrue(header[10].get("name") == "PETAL")
        self.assertTrue(header[10].get("value") == "2-2")
        self.assertTrue(header[11].get("name") == "TILE")
        self.assertTrue(header[11].get("value") == "3-3")

    def test_forest(self):
        """Test constructor for Forest object.
        This includes a test of function rebin."""
        # create a Forest with missing Forest variables
        kwargs = {
            "los_id": 9999,
            "ra": 0.15,
            "dec": 0.0,
            "z": 2.1,
            "flux": np.ones(15),
            "ivar": np.ones(15)*4,
        }
        with self.assertRaises(AstronomicalObjectError):
            Forest(**kwargs)

        # set class variables; case: logarithmic wavelength solution
        setup_forest(wave_solution="log")

        # create a Forest
        kwargs = {
            "los_id": 9999,
            "ra": 0.15,
            "dec": 0.0,
            "z": 2.1,
            "flux": np.ones(15),
            "ivar": np.ones(15)*4,
            "log_lambda": np.ones(15),
        }
        test_obj = Forest(**kwargs)
        self.assertTrue(isinstance(test_obj, AstronomicalObject))
        self.assertTrue(test_obj.los_id == 9999)
        self.assertTrue(test_obj.healpix == 1505)
        self.assertTrue(test_obj.z == 2.1)
        self.assertTrue(test_obj.bad_continuum_reason is None)
        self.assertTrue(test_obj.continuum is None)
        self.assertTrue(test_obj.deltas is None)
        self.assertTrue(np.allclose(test_obj.log_lambda, np.ones(15)))
        self.assertTrue(test_obj.lambda_ is None)
        self.assertTrue(np.allclose(test_obj.flux, np.ones(15)))
        self.assertTrue(np.allclose(test_obj.ivar, np.ones(15)*4))
        self.assertTrue(len(Forest.mask_fields) == 4)
        self.assertTrue(Forest.mask_fields[0] == "flux")
        self.assertTrue(Forest.mask_fields[1] == "ivar")
        self.assertTrue(Forest.mask_fields[2] == "transmission_correction")
        self.assertTrue(Forest.mask_fields[3] == "log_lambda")
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
            "log_lambda": np.ones(15)
        }
        test_obj = Forest(**kwargs)
        self.assertTrue(isinstance(test_obj, AstronomicalObject))
        self.assertTrue(test_obj.los_id == 9999)
        self.assertTrue(test_obj.healpix == 1505)
        self.assertTrue(test_obj.z == 2.1)
        self.assertTrue(test_obj.bad_continuum_reason is None)
        self.assertTrue(np.allclose(test_obj.continuum, np.ones(15)))
        self.assertTrue(np.allclose(test_obj.deltas, np.zeros(15)))
        self.assertTrue(np.allclose(test_obj.log_lambda, np.ones(15)))
        self.assertTrue(test_obj.lambda_ is None)
        self.assertTrue(np.allclose(test_obj.flux, np.ones(15)))
        self.assertTrue(np.allclose(test_obj.ivar, np.ones(15)*4))
        self.assertTrue(len(Forest.mask_fields) == 4)
        self.assertTrue(Forest.mask_fields[0] == "flux")
        self.assertTrue(Forest.mask_fields[1] == "ivar")
        self.assertTrue(Forest.mask_fields[2] == "transmission_correction")
        self.assertTrue(Forest.mask_fields[3] == "log_lambda")
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
            "test_variable": "test",
            "log_lambda": np.ones(15)
        }
        test_obj = Forest(**kwargs)
        self.assertTrue(isinstance(test_obj, AstronomicalObject))
        self.assertTrue(test_obj.los_id == 9999)
        self.assertTrue(test_obj.healpix == 1505)
        self.assertTrue(test_obj.z == 2.1)
        self.assertTrue(test_obj.bad_continuum_reason is None)
        self.assertTrue(test_obj.continuum is None)
        self.assertTrue(test_obj.deltas is None)
        self.assertTrue(np.allclose(test_obj.log_lambda, np.ones(15)))
        self.assertTrue(test_obj.lambda_ is None)
        self.assertTrue(np.allclose(test_obj.flux, np.ones(15)))
        self.assertTrue(np.allclose(test_obj.ivar, np.ones(15)*4))
        self.assertTrue(len(Forest.mask_fields) == 4)
        self.assertTrue(Forest.mask_fields[0] == "flux")
        self.assertTrue(Forest.mask_fields[1] == "ivar")
        self.assertTrue(Forest.mask_fields[2] == "transmission_correction")
        self.assertTrue(Forest.mask_fields[3] == "log_lambda")
        self.assertTrue(np.allclose(test_obj.transmission_correction, np.ones(15)))
        self.assertTrue(np.allclose(test_obj.mean_snr, 2))

        # create a Forest with missing AstronomicalObject variables
        kwargs = {
            "flux": np.ones(15),
            "ivar": np.ones(15)*4,
        }
        with self.assertRaises(AstronomicalObjectError):
            Forest(**kwargs)

        # set class variables; case: linear wavelength solution
        reset_forest()
        setup_forest(wave_solution="lin")

        # create a Forest
        kwargs = {
            "los_id": 9999,
            "ra": 0.15,
            "dec": 0.0,
            "z": 2.1,
            "flux": np.ones(15),
            "ivar": np.ones(15)*4,
            "lambda": np.ones(15),
        }
        test_obj = Forest(**kwargs)
        self.assertTrue(isinstance(test_obj, AstronomicalObject))
        self.assertTrue(test_obj.los_id == 9999)
        self.assertTrue(test_obj.healpix == 1505)
        self.assertTrue(test_obj.z == 2.1)
        self.assertTrue(test_obj.bad_continuum_reason is None)
        self.assertTrue(test_obj.continuum is None)
        self.assertTrue(test_obj.deltas is None)
        self.assertTrue(test_obj.log_lambda is None)
        self.assertTrue(np.allclose(test_obj.lambda_, np.ones(15)))
        self.assertTrue(np.allclose(test_obj.flux, np.ones(15)))
        self.assertTrue(np.allclose(test_obj.ivar, np.ones(15)*4))
        self.assertTrue(len(Forest.mask_fields) == 4)
        self.assertTrue(Forest.mask_fields[0] == "flux")
        self.assertTrue(Forest.mask_fields[1] == "ivar")
        self.assertTrue(Forest.mask_fields[2] == "transmission_correction")
        self.assertTrue(Forest.mask_fields[3] == "lambda_")
        self.assertTrue(np.allclose(test_obj.transmission_correction, np.ones(15)))
        self.assertTrue(np.allclose(test_obj.mean_snr, 2))

    def test_forest_coadd(self):
        """Test the coadd function in Forest"""
        # set class variables; case: logarithmic wavelength solution
        setup_forest(wave_solution="log")

        # create a Forest
        kwargs = {
            "los_id": 100000000,
            "ra": 0.15,
            "dec": 0.0,
            "z": 2.1,
            "flux": np.ones(10),
            "ivar": np.ones(10)*4,
            "log_lambda": np.array([3.5565, 3.55655, 3.5567, 3.55675, 3.5569,
                                    3.55695, 3.5571, 3.55715, 3.5573, 3.55735]),
        }
        test_obj = Forest(**kwargs)

        # create a second SdssForest
        kwargs_other = {
            "los_id": 100000000,
            "ra": 0.1,
            "dec": 0.01,
            "z": 2.2,
            "flux": np.ones(10)*3,
            "ivar": np.ones(10)*4,
            "log_lambda": np.array([3.5565, 3.55655, 3.5567, 3.55675, 3.5569,
                                    3.55695, 3.5571, 3.55715, 3.5573, 3.55735]),
        }
        test_obj_other = Forest(**kwargs_other)

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
        self.assertTrue(len(Forest.mask_fields) == 4)
        self.assertTrue(Forest.mask_fields[0] == "flux")
        self.assertTrue(Forest.mask_fields[1] == "ivar")
        self.assertTrue(Forest.mask_fields[2] == "transmission_correction")
        self.assertTrue(Forest.mask_fields[3] == "log_lambda")
        self.assertTrue(np.allclose(test_obj.transmission_correction, np.ones(5)))
        self.assertTrue(np.allclose(test_obj.mean_snr, 8))

        # create a third Forest with different los_id
        kwargs_other = {
            "los_id": 999,
            "ra": 0.1,
            "dec": 0.01,
            "z": 2.2,
            "flux": np.ones(10)*3,
            "ivar": np.ones(10)*4,
            "log_lambda": np.array([3.5565, 3.55655, 3.5567, 3.55675, 3.5569,
                                    3.55695, 3.5571, 3.55715, 3.5573, 3.55735]),
        }
        test_obj_other = Forest(**kwargs_other)

        # coadding them whould raise an error
        with self.assertRaises(AstronomicalObjectError):
            test_obj.coadd(test_obj_other)

        # set class variables; case: linear wavelength solution
        reset_forest()
        setup_forest(wave_solution="lin")

        # create a Forest
        kwargs = {
            "los_id": 100000000,
            "ra": 0.15,
            "dec": 0.0,
            "z": 2.1,
            "flux": np.ones(10),
            "ivar": np.ones(10)*4,
            "lambda": np.array([3610, 3610.4, 3650, 3650.4, 3670, 3670.4,
                                3680, 3680.4, 3700, 3700.4]),
        }
        test_obj = Forest(**kwargs)

        # create a second Forest
        kwargs_other = {
            "los_id": 100000000,
            "ra": 0.1,
            "dec": 0.01,
            "z": 2.2,
            "flux": np.ones(10)*3,
            "ivar": np.ones(10)*4,
            "lambda": np.array([3610, 3610.4, 3650, 3650.4, 3670, 3670.4,
                                3680, 3680.4, 3700, 3700.4]),
        }
        test_obj_other = Forest(**kwargs_other)

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
        self.assertTrue(len(Forest.mask_fields) == 4)
        self.assertTrue(Forest.mask_fields[0] == "flux")
        self.assertTrue(Forest.mask_fields[1] == "ivar")
        self.assertTrue(Forest.mask_fields[2] == "transmission_correction")
        self.assertTrue(Forest.mask_fields[3] == "lambda_")
        self.assertTrue(np.allclose(test_obj.transmission_correction, np.ones(5)))
        self.assertTrue(np.allclose(test_obj.mean_snr, 8))

    def test_forest_get_data(self):
        """Test method get_data for Forest."""
        # set class variables; case: logarithmic wavelength solution
        setup_forest(wave_solution="log")

        # create a Forest
        kwargs = {
            "los_id": 9999,
            "ra": 0.15,
            "dec": 0.0,
            "z": 2.1,
            "flux": np.ones(15),
            "ivar": np.ones(15)*4,
            "log_lambda": np.ones(15),
        }
        test_obj = Forest(**kwargs)

        cols, names, units, comments = test_obj.get_data()
        self.assertTrue(len(names) == 4)
        self.assertTrue(len(cols) == 4)
        self.assertTrue(len(units) == 4)
        self.assertTrue(len(comments) == 4)
        self.assertTrue(names[0] == "LOGLAM")
        self.assertTrue(np.allclose(cols[0], np.ones(15)))
        self.assertTrue(units[0] == "log Angstrom")
        self.assertTrue(comments[0] == "Log lambda")
        self.assertTrue(names[1] == "CONT")
        self.assertTrue(np.allclose(cols[1], np.zeros(15)))
        self.assertTrue(units[1] == "Flux units")
        self.assertTrue(comments[1] == ("Quasar continuum if BAD_CONT is 'None'. "
                                        "Check input spectra for units"))
        self.assertTrue(names[2] == "DELTA")
        self.assertTrue(np.allclose(cols[2], np.zeros(15)))
        self.assertTrue(units[2] == "")
        self.assertTrue(comments[2] == "Delta field")
        self.assertTrue(names[3] == "WEIGHT")
        self.assertTrue(np.allclose(cols[3], np.zeros(15)))
        self.assertTrue(units[3] == "")
        self.assertTrue(comments[3] == "Pixel weights")

        # set class variables; case: linear wavelength solution
        reset_forest()
        setup_forest(wave_solution="lin")

        # create a Forest
        kwargs = {
            "los_id": 9999,
            "ra": 0.15,
            "dec": 0.0,
            "z": 2.1,
            "flux": np.ones(15),
            "ivar": np.ones(15)*4,
            "lambda": np.ones(15),
        }
        test_obj = Forest(**kwargs)

        cols, names, units, comments = test_obj.get_data()
        self.assertTrue(len(names) == 4)
        self.assertTrue(len(cols) == 4)
        self.assertTrue(len(units) == 4)
        self.assertTrue(len(comments) == 4)
        self.assertTrue(names[0] == "LAMBDA")
        self.assertTrue(np.allclose(cols[0], np.ones(15)))
        self.assertTrue(units[0] == "Angstrom")
        self.assertTrue(comments[0] == "Lambda")
        self.assertTrue(names[1] == "CONT")
        self.assertTrue(np.allclose(cols[1], np.zeros(15)))
        self.assertTrue(units[1] == "Flux units")
        self.assertTrue(comments[1] == ("Quasar continuum if BAD_CONT is 'None'. "
                                        "Check input spectra for units"))
        self.assertTrue(names[2] == "DELTA")
        self.assertTrue(np.allclose(cols[2], np.zeros(15)))
        self.assertTrue(units[2] == "")
        self.assertTrue(comments[2] == "Delta field")
        self.assertTrue(names[3] == "WEIGHT")
        self.assertTrue(np.allclose(cols[3], np.zeros(15)))
        self.assertTrue(units[3] == "")
        self.assertTrue(comments[3] == "Pixel weights")

    def test_forest_get_header(self):
        """Test method get_header for Forest."""
        # set class variables; case: logarithmic wavelength solution
        setup_forest(wave_solution="log")

        # create a Forest
        kwargs = {
            "los_id": 9999,
            "ra": 0.15,
            "dec": 0.0,
            "z": 2.1,
            "flux": np.ones(15),
            "ivar": np.ones(15)*4,
            "log_lambda": np.ones(15),
        }
        test_obj = Forest(**kwargs)

        # get header and test
        header = test_obj.get_header()
        self.assertTrue(len(header) == 6)
        self.assertTrue(header[0].get("name") == "LOS_ID")
        self.assertTrue(header[0].get("value") == 9999)
        self.assertTrue(header[1].get("name") == "RA")
        self.assertTrue(header[1].get("value") == 0.15)
        self.assertTrue(header[2].get("name") == "DEC")
        self.assertTrue(header[2].get("value") == 0.0)
        self.assertTrue(header[3].get("name") == "Z")
        self.assertTrue(header[3].get("value") == 2.1)
        self.assertTrue(header[4].get("name") == "BAD_CONT")
        self.assertTrue(header[4].get("value") == "None")
        self.assertTrue(header[5].get("name") == "MEANSNR")
        self.assertTrue(header[5].get("value") == 2)

        # set class variables; case: linear wavelength solution
        setup_forest(wave_solution="lin")

        # create a Forest
        kwargs = {
            "los_id": 9999,
            "ra": 0.15,
            "dec": 0.0,
            "z": 2.1,
            "flux": np.ones(15),
            "ivar": np.ones(15)*4,
            "lambda": np.ones(15),
        }
        test_obj = Forest(**kwargs)

        # get header and test
        header = test_obj.get_header()
        self.assertTrue(len(header) == 6)
        self.assertTrue(header[0].get("name") == "LOS_ID")
        self.assertTrue(header[0].get("value") == 9999)
        self.assertTrue(header[1].get("name") == "RA")
        self.assertTrue(header[1].get("value") == 0.15)
        self.assertTrue(header[2].get("name") == "DEC")
        self.assertTrue(header[2].get("value") == 0.0)
        self.assertTrue(header[3].get("name") == "Z")
        self.assertTrue(header[3].get("value") == 2.1)
        self.assertTrue(header[4].get("name") == "BAD_CONT")
        self.assertTrue(header[4].get("value") == "None")
        self.assertTrue(header[5].get("name") == "MEANSNR")
        self.assertTrue(header[5].get("value") == 2)

    def test_pk1d_forest(self):
        """Test constructor for Pk1dForest object."""
        # create a Pk1dForest with missing Forest variables
        kwargs = {
            "los_id": 9999,
            "ra": 0.15,
            "dec": 0.0,
            "z": 2.1,
            "flux": np.ones(10),
            "ivar": np.ones(10)*4,
            "log_lambda": np.ones(10),
            "exposures_diff": np.ones(10),
            "reso": np.ones(10),
        }
        with self.assertRaises(AstronomicalObjectError):
            Pk1dForest(**kwargs)

        # set class variables; case: logarithmic wavelength solution
        setup_forest(wave_solution="log")

        # create a Pk1dForest with missing Pk1dForest variables
        kwargs = {
            "los_id": 9999,
            "ra": 0.15,
            "dec": 0.0,
            "z": 2.1,
            "flux": np.ones(10),
            "ivar": np.ones(10)*4,
            "log_lambda": np.ones(10),
            "exposures_diff": np.ones(10),
            "reso": np.ones(10),
        }
        with self.assertRaises(AstronomicalObjectError):
            Pk1dForest(**kwargs)

        # set class variables
        setup_pk1d_forest("LYA")

        # create a Pk1dForest
        kwargs = {
            "los_id": 9999,
            "ra": 0.15,
            "dec": 0.0,
            "z": 2.1,
            "flux": np.ones(10),
            "ivar": np.ones(10)*4,
            "log_lambda": np.ones(10),
            "exposures_diff": np.ones(10),
            "reso": np.ones(10),
        }
        test_obj = Pk1dForest(**kwargs)
        self.assertTrue(isinstance(test_obj, AstronomicalObject))
        self.assertTrue(test_obj.los_id == 9999)
        self.assertTrue(test_obj.healpix == 1505)
        self.assertTrue(test_obj.z == 2.1)
        self.assertTrue(test_obj.bad_continuum_reason is None)
        self.assertTrue(test_obj.continuum is None)
        self.assertTrue(test_obj.deltas is None)
        self.assertTrue(np.allclose(test_obj.flux, np.ones(10)))
        self.assertTrue(np.allclose(test_obj.ivar, np.ones(10)*4))
        self.assertTrue(np.allclose(test_obj.log_lambda, np.ones(10)))
        self.assertTrue(test_obj.lambda_ is None)
        self.assertTrue(len(Forest.mask_fields) == 6)
        self.assertTrue(Forest.mask_fields[0] == "flux")
        self.assertTrue(Forest.mask_fields[1] == "ivar")
        self.assertTrue(Forest.mask_fields[2] == "transmission_correction")
        self.assertTrue(Forest.mask_fields[3] == "log_lambda")
        self.assertTrue(Forest.mask_fields[4] == "exposures_diff")
        self.assertTrue(Forest.mask_fields[5] == "reso")
        self.assertTrue(np.allclose(test_obj.transmission_correction, np.ones(10)))
        self.assertTrue(np.allclose(test_obj.mean_snr, 2))
        self.assertTrue(Pk1dForest.lambda_abs_igm == 1215.67)
        self.assertTrue(np.allclose(test_obj.exposures_diff, np.ones(10)))
        self.assertTrue(np.allclose(test_obj.reso, np.ones(10)))
        self.assertTrue(np.isclose(test_obj.mean_z, -0.9917740834272459))
        self.assertTrue(np.isclose(test_obj.mean_reso, 1.0))

        # set class variables; case: linear wavelength solution
        reset_forest()
        setup_forest(wave_solution="lin")
        setup_pk1d_forest("LYA")

        # create a Forest
        kwargs = {
            "los_id": 9999,
            "ra": 0.15,
            "dec": 0.0,
            "z": 2.1,
            "flux": np.ones(10),
            "ivar": np.ones(10)*4,
            "lambda": np.ones(10),
            "exposures_diff": np.ones(10),
            "reso": np.ones(10),
        }
        test_obj = Pk1dForest(**kwargs)
        self.assertTrue(isinstance(test_obj, AstronomicalObject))
        self.assertTrue(test_obj.los_id == 9999)
        self.assertTrue(test_obj.healpix == 1505)
        self.assertTrue(test_obj.z == 2.1)
        self.assertTrue(test_obj.bad_continuum_reason is None)
        self.assertTrue(test_obj.continuum is None)
        self.assertTrue(test_obj.deltas is None)
        self.assertTrue(test_obj.log_lambda is None)
        self.assertTrue(np.allclose(test_obj.lambda_, np.ones(10)))
        self.assertTrue(np.allclose(test_obj.flux, np.ones(10)))
        self.assertTrue(np.allclose(test_obj.ivar, np.ones(10)*4))
        self.assertTrue(len(Forest.mask_fields) == 6)
        self.assertTrue(Forest.mask_fields[0] == "flux")
        self.assertTrue(Forest.mask_fields[1] == "ivar")
        self.assertTrue(Forest.mask_fields[2] == "transmission_correction")
        self.assertTrue(Forest.mask_fields[3] == "lambda_")
        self.assertTrue(Forest.mask_fields[4] == "exposures_diff")
        self.assertTrue(Forest.mask_fields[5] == "reso")
        self.assertTrue(np.allclose(test_obj.transmission_correction, np.ones(10)))
        self.assertTrue(np.allclose(test_obj.mean_snr, 2))
        self.assertTrue(Pk1dForest.lambda_abs_igm == 1215.67)
        self.assertTrue(np.allclose(test_obj.exposures_diff, np.ones(10)))
        self.assertTrue(np.allclose(test_obj.reso, np.ones(10)))
        self.assertTrue(np.isclose(test_obj.mean_z, -0.9991774083427246))
        self.assertTrue(np.isclose(test_obj.mean_reso, 1.0))

    def test_pk1d_forest_coadd(self):
        """Test the coadd function in Pk1dForest"""
        # set class variables; case: logarithmic wavelength solution
        setup_forest(wave_solution="log")
        setup_pk1d_forest("LYA")

        # create a Pk1dForest
        kwargs = {
            "los_id": 9999,
            "ra": 0.15,
            "dec": 0.0,
            "z": 2.1,
            "flux": np.ones(10),
            "ivar": np.ones(10)*4,
            "log_lambda": np.array([3.5565, 3.55655, 3.5567, 3.55675, 3.5569,
                                    3.55695, 3.5571, 3.55715, 3.5573, 3.55735]),
            "exposures_diff": np.ones(10),
            "reso": np.ones(10),
        }
        test_obj = Pk1dForest(**kwargs)

        # create a second Pk1dForest
        kwargs_other = {
            "los_id": 9999,
            "ra": 0.15,
            "dec": 0.0,
            "z": 2.1,
            "flux": np.ones(10)*3,
            "ivar": np.ones(10)*4,
            "log_lambda": np.array([3.5565, 3.55655, 3.5567, 3.55675, 3.5569,
                                    3.55695, 3.5571, 3.55715, 3.5573, 3.55735]),
            "exposures_diff": np.ones(10)*3,
            "reso": np.ones(10),
        }
        test_obj_other = Pk1dForest(**kwargs_other)

        # coadd them
        test_obj.coadd(test_obj_other)

        self.assertTrue(test_obj.los_id == 9999)
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
        self.assertTrue(len(Forest.mask_fields) == 6)
        self.assertTrue(Forest.mask_fields[0] == "flux")
        self.assertTrue(Forest.mask_fields[1] == "ivar")
        self.assertTrue(Forest.mask_fields[2] == "transmission_correction")
        self.assertTrue(Forest.mask_fields[3] == "log_lambda")
        self.assertTrue(Forest.mask_fields[4] == "exposures_diff")
        self.assertTrue(Forest.mask_fields[5] == "reso")
        self.assertTrue(np.allclose(test_obj.transmission_correction, np.ones(5)))
        self.assertTrue(np.allclose(test_obj.mean_snr, 8))
        self.assertTrue(Pk1dForest.lambda_abs_igm == 1215.67)
        self.assertTrue(np.allclose(test_obj.exposures_diff, np.ones(5)*2))
        self.assertTrue(np.allclose(test_obj.reso, np.ones(5)))
        self.assertTrue(np.isclose(test_obj.mean_z, 1.965425))

        # create a third Pk1dForest with different targetid
        kwargs_other = {
            "los_id": 9998,
            "ra": 0.15,
            "dec": 0.0,
            "z": 2.1,
            "flux": np.ones(10),
            "ivar": np.ones(10)*4,
            "log_lambda": np.array([3.5565, 3.55655, 3.5567, 3.55675, 3.5569,
                                    3.55695, 3.5571, 3.55715, 3.5573, 3.55735]),
            "exposures_diff": np.ones(10)*3,
            "reso": np.ones(10),
        }
        test_obj_other = Pk1dForest(**kwargs_other)

        # coadding them should raise an error
        with self.assertRaises(AstronomicalObjectError):
            test_obj.coadd(test_obj_other)

        # set class variables; case: linear wavelength solution
        reset_forest()
        setup_forest(wave_solution="lin")
        setup_pk1d_forest("LYA")

        # create a Forest
        kwargs = {
            "los_id": 100000000,
            "ra": 0.15,
            "dec": 0.0,
            "z": 2.1,
            "flux": np.ones(10),
            "ivar": np.ones(10)*4,
            "lambda": np.array([3610, 3610.4, 3650, 3650.4, 3670, 3670.4,
                                3680, 3680.4, 3700, 3700.4]),
            "exposures_diff": np.ones(10),
            "reso": np.ones(10),
        }
        test_obj = Pk1dForest(**kwargs)

        # create a second Forest
        kwargs_other = {
            "los_id": 100000000,
            "ra": 0.1,
            "dec": 0.01,
            "z": 2.2,
            "flux": np.ones(10)*3,
            "ivar": np.ones(10)*4,
            "lambda": np.array([3610, 3610.4, 3650, 3650.4, 3670, 3670.4,
                                3680, 3680.4, 3700, 3700.4]),
            "exposures_diff": np.ones(10)*3,
            "reso": np.ones(10),
        }
        test_obj_other = Pk1dForest(**kwargs_other)

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
        self.assertTrue(len(Forest.mask_fields) == 6)
        self.assertTrue(Forest.mask_fields[0] == "flux")
        self.assertTrue(Forest.mask_fields[1] == "ivar")
        self.assertTrue(Forest.mask_fields[2] == "transmission_correction")
        self.assertTrue(Forest.mask_fields[3] == "lambda_")
        self.assertTrue(Forest.mask_fields[4] == "exposures_diff")
        self.assertTrue(Forest.mask_fields[5] == "reso")
        self.assertTrue(np.allclose(test_obj.transmission_correction, np.ones(5)))
        self.assertTrue(np.allclose(test_obj.mean_snr, 8))
        self.assertTrue(np.allclose(test_obj.exposures_diff, np.ones(5)*2))
        self.assertTrue(np.allclose(test_obj.reso, np.ones(5)))
        self.assertTrue(np.isclose(test_obj.mean_z, 2.0065725073416303))

    def test_pk1d_forest_get_data(self):
        """Test method get_data for Pk1dForest."""
        # set class variables; case: logarithmic wavelength solution
        setup_forest(wave_solution="log")
        setup_pk1d_forest("LYA")

        # create a Pk1dForest
        kwargs = {
            "los_id": 9999,
            "ra": 0.15,
            "dec": 0.0,
            "z": 2.1,
            "flux": np.ones(15),
            "ivar": np.ones(15)*4,
            "log_lambda": np.ones(15),
            "exposures_diff": np.ones(15),
            "reso": np.ones(15),
        }
        test_obj = Pk1dForest(**kwargs)

        cols, names, units, comments = test_obj.get_data()
        self.assertTrue(len(names) == 5)
        self.assertTrue(len(cols) == 5)
        self.assertTrue(len(units) == 5)
        self.assertTrue(len(comments) == 5)
        self.assertTrue(names[0] == "LOGLAM")
        self.assertTrue(np.allclose(cols[0], np.ones(15)))
        self.assertTrue(units[0] == "log Angstrom")
        self.assertTrue(comments[0] == "Log lambda")
        self.assertTrue(names[1] == "CONT")
        self.assertTrue(np.allclose(cols[1], np.zeros(15)))
        self.assertTrue(units[1] == "Flux units")
        self.assertTrue(comments[1] == ("Quasar continuum if BAD_CONT is 'None'. "
                                        "Check input spectra for units"))
        self.assertTrue(names[2] == "DELTA")
        self.assertTrue(np.allclose(cols[2], np.zeros(15)))
        self.assertTrue(units[2] == "")
        self.assertTrue(comments[2] == "Delta field")
        self.assertTrue(names[3] == "WEIGHT")
        self.assertTrue(np.allclose(cols[3], np.zeros(15)))
        self.assertTrue(units[3] == "")
        self.assertTrue(comments[3] == "Pixel weights")
        self.assertTrue(names[4] == "DIFF")
        self.assertTrue(np.allclose(cols[4], np.ones(15)))
        self.assertTrue(units[4] == "Flux units")
        self.assertTrue(comments[4] == "Difference. Check input spectra for units")


        # set class variables; case: linear wavelength solution
        reset_forest()
        setup_forest(wave_solution="lin")
        setup_pk1d_forest("LYA")

        # create a Forest
        kwargs = {
            "los_id": 9999,
            "ra": 0.15,
            "dec": 0.0,
            "z": 2.1,
            "flux": np.ones(15),
            "ivar": np.ones(15)*4,
            "lambda": np.ones(15),
            "exposures_diff": np.ones(15),
            "reso": np.ones(15),
        }
        test_obj = Pk1dForest(**kwargs)

        cols, names, units, comments = test_obj.get_data()
        self.assertTrue(len(names) == 5)
        self.assertTrue(len(cols) == 5)
        self.assertTrue(len(units) == 5)
        self.assertTrue(len(comments) == 5)
        self.assertTrue(names[0] == "LAMBDA")
        self.assertTrue(np.allclose(cols[0], np.ones(15)))
        self.assertTrue(units[0] == "Angstrom")
        self.assertTrue(comments[0] == "Lambda")
        self.assertTrue(names[1] == "CONT")
        self.assertTrue(np.allclose(cols[1], np.zeros(15)))
        self.assertTrue(units[1] == "Flux units")
        self.assertTrue(comments[1] == ("Quasar continuum if BAD_CONT is 'None'. "
                                        "Check input spectra for units"))
        self.assertTrue(names[2] == "DELTA")
        self.assertTrue(np.allclose(cols[2], np.zeros(15)))
        self.assertTrue(units[2] == "")
        self.assertTrue(comments[2] == "Delta field")
        self.assertTrue(names[3] == "WEIGHT")
        self.assertTrue(np.allclose(cols[3], np.zeros(15)))
        self.assertTrue(units[3] == "")
        self.assertTrue(comments[3] == "Pixel weights")
        self.assertTrue(names[4] == "DIFF")
        self.assertTrue(np.allclose(cols[4], np.ones(15)))
        self.assertTrue(units[4] == "Flux units")
        self.assertTrue(comments[4] == "Difference. Check input spectra for units")

    def test_pk1d_forest_get_header(self):
        """Test method get_header for Pk1dForest."""
        # set class variables; case: logarithmic wavelength solution
        setup_forest(wave_solution="log")
        setup_pk1d_forest("LYA")

        # create a Pk1dForest
        kwargs = {
            "los_id": 9999,
            "ra": 0.15,
            "dec": 0.0,
            "z": 2.1,
            "flux": np.ones(10),
            "ivar": np.ones(10)*4,
            "log_lambda": np.ones(10),
            "exposures_diff": np.ones(10),
            "reso": np.ones(10),
        }
        test_obj = Pk1dForest(**kwargs)

        # get header and test
        header = test_obj.get_header()
        self.assertTrue(len(header) == 8)
        self.assertTrue(header[0].get("name") == "LOS_ID")
        self.assertTrue(header[0].get("value") == 9999)
        self.assertTrue(header[1].get("name") == "RA")
        self.assertTrue(header[1].get("value") == 0.15)
        self.assertTrue(header[2].get("name") == "DEC")
        self.assertTrue(header[2].get("value") == 0.0)
        self.assertTrue(header[3].get("name") == "Z")
        self.assertTrue(header[3].get("value") == 2.1)
        self.assertTrue(header[4].get("name") == "BAD_CONT")
        self.assertTrue(header[4].get("value") == "None")
        self.assertTrue(header[5].get("name") == "MEANSNR")
        self.assertTrue(header[5].get("value") == 2)
        self.assertTrue(header[6].get("name") == "MEANZ")
        self.assertTrue(header[6].get("value") == -0.9917740834272459)
        self.assertTrue(header[7].get("name") == "MEANRESO")
        self.assertTrue(header[7].get("value") == 1.0)

        # set class variables; case: linear wavelength solution
        setup_forest(wave_solution="lin")

        # create a Pk1dForest
        kwargs = {
            "los_id": 9999,
            "ra": 0.15,
            "dec": 0.0,
            "z": 2.1,
            "flux": np.ones(10),
            "ivar": np.ones(10)*4,
            "lambda": np.ones(10),
            "exposures_diff": np.ones(10),
            "reso": np.ones(10),
        }
        test_obj = Pk1dForest(**kwargs)

        # get header and test
        header = test_obj.get_header()
        self.assertTrue(len(header) == 8)
        self.assertTrue(header[0].get("name") == "LOS_ID")
        self.assertTrue(header[0].get("value") == 9999)
        self.assertTrue(header[1].get("name") == "RA")
        self.assertTrue(header[1].get("value") == 0.15)
        self.assertTrue(header[2].get("name") == "DEC")
        self.assertTrue(header[2].get("value") == 0.0)
        self.assertTrue(header[3].get("name") == "Z")
        self.assertTrue(header[3].get("value") == 2.1)
        self.assertTrue(header[4].get("name") == "BAD_CONT")
        self.assertTrue(header[4].get("value") == "None")
        self.assertTrue(header[5].get("name") == "MEANSNR")
        self.assertTrue(header[5].get("value") == 2)
        self.assertTrue(header[6].get("name") == "MEANZ")
        self.assertTrue(header[6].get("value") == -0.9991774083427246)
        self.assertTrue(header[7].get("name") == "MEANRESO")
        self.assertTrue(header[7].get("value") == 1.0)

    def test_sdss_forest(self):
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

        # set class variables
        setup_forest(wave_solution="log")

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
        self.assertTrue(len(Forest.mask_fields) == 4)
        self.assertTrue(Forest.mask_fields[0] == "flux")
        self.assertTrue(Forest.mask_fields[1] == "ivar")
        self.assertTrue(Forest.mask_fields[2] == "transmission_correction")
        self.assertTrue(Forest.mask_fields[3] == "log_lambda")
        self.assertTrue(np.allclose(test_obj.transmission_correction, np.ones(5)))
        self.assertTrue(np.allclose(test_obj.mean_snr, 2.8284271247461903))
        self.assertTrue(isinstance(test_obj.plate, list))
        self.assertTrue(test_obj.plate[0] == 0)
        self.assertTrue(isinstance(test_obj.fiberid, list))
        self.assertTrue(test_obj.fiberid[0] == 0)
        self.assertTrue(isinstance(test_obj.mjd, list))
        self.assertTrue(test_obj.mjd[0] == 0)
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
        self.assertTrue(isinstance(test_obj, Forest))
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
        self.assertTrue(len(Forest.mask_fields) == 4)
        self.assertTrue(Forest.mask_fields[0] == "flux")
        self.assertTrue(Forest.mask_fields[1] == "ivar")
        self.assertTrue(Forest.mask_fields[2] == "transmission_correction")
        self.assertTrue(Forest.mask_fields[3] == "log_lambda")
        self.assertTrue(np.allclose(test_obj.transmission_correction, np.ones(5)))
        self.assertTrue(np.allclose(test_obj.mean_snr, 2.8284271247461903))
        self.assertTrue(isinstance(test_obj.plate, list))
        self.assertTrue(test_obj.plate[0] == 0)
        self.assertTrue(isinstance(test_obj.fiberid, list))
        self.assertTrue(test_obj.fiberid[0] == 0)
        self.assertTrue(isinstance(test_obj.mjd, list))
        self.assertTrue(test_obj.mjd[0] == 0)
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

    def test_sdss_forest_coadd(self):
        """Test the coadd function in SdssForest"""
        # set class variables
        setup_forest(wave_solution="log")

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
            "thingid": 100000000,
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
        self.assertTrue(len(Forest.mask_fields) == 4)
        self.assertTrue(Forest.mask_fields[0] == "flux")
        self.assertTrue(Forest.mask_fields[1] == "ivar")
        self.assertTrue(Forest.mask_fields[2] == "transmission_correction")
        self.assertTrue(Forest.mask_fields[3] == "log_lambda")
        self.assertTrue(np.allclose(test_obj.transmission_correction, np.ones(5)))
        self.assertTrue(np.allclose(test_obj.mean_snr, 8))
        self.assertTrue(isinstance(test_obj.plate, list))
        self.assertTrue(test_obj.plate[0] == 0)
        self.assertTrue(test_obj.plate[1] == 1)
        self.assertTrue(isinstance(test_obj.fiberid, list))
        self.assertTrue(test_obj.fiberid[0] == 0)
        self.assertTrue(test_obj.fiberid[1] == 2)
        self.assertTrue(isinstance(test_obj.mjd, list))
        self.assertTrue(test_obj.mjd[0] == 0)
        self.assertTrue(test_obj.mjd[1] == 3)
        self.assertTrue(test_obj.thingid == 100000000)

        # create a third SdssForest with different targetid
        kwargs_other = {
            "ra": 0.1,
            "dec": 0.01,
            "z": 2.2,
            "flux": np.ones(10)*3,
            "ivar": np.ones(10)*4,
            "log_lambda": np.array([3.5565, 3.55655, 3.5567, 3.55675, 3.5569,
                                    3.55695, 3.5571, 3.55715, 3.5573, 3.55735]),
            "thingid": 100010000,
            "plate": 1,
            "fiberid": 2,
            "mjd": 3,
        }
        test_obj_other = SdssForest(**kwargs_other)

        # coadding them should raise an error
        with self.assertRaises(AstronomicalObjectError):
            test_obj.coadd(test_obj_other)

    def test_sdss_forest_get_data(self):
        """Test method get_data for SdssForest."""
        # set class variables
        setup_forest(wave_solution="log")

        # create an SdssForest
        kwargs = {
            "ra": 0.15,
            "dec": 0.0,
            "z": 2.1,
            "flux": np.ones(10),
            "ivar": np.ones(10)*4,
            "log_lambda": np.ones(10),
            "thingid": 100000000,
            "plate": 0,
            "fiberid": 0,
            "mjd": 0,
        }
        test_obj = SdssForest(**kwargs)

        cols, names, units, comments = test_obj.get_data()
        self.assertTrue(len(names) == 4)
        self.assertTrue(len(cols) == 4)
        self.assertTrue(len(units) == 4)
        self.assertTrue(len(comments) == 4)
        self.assertTrue(names[0] == "LOGLAM")
        self.assertTrue(np.allclose(cols[0], np.ones(10)))
        self.assertTrue(units[0] == "log Angstrom")
        self.assertTrue(comments[0] == "Log lambda")
        self.assertTrue(names[1] == "CONT")
        self.assertTrue(np.allclose(cols[1], np.zeros(10)))
        self.assertTrue(units[1] == "Flux units")
        self.assertTrue(comments[1] == ("Quasar continuum if BAD_CONT is 'None'. "
                                        "Check input spectra for units"))
        self.assertTrue(names[2] == "DELTA")
        self.assertTrue(np.allclose(cols[2], np.zeros(10)))
        self.assertTrue(units[2] == "")
        self.assertTrue(comments[2] == "Delta field")
        self.assertTrue(names[3] == "WEIGHT")
        self.assertTrue(np.allclose(cols[3], np.zeros(10)))
        self.assertTrue(units[3] == "")
        self.assertTrue(comments[3] == "Pixel weights")

    def test_sdss_forest_get_header(self):
        """Test method get_header for SdssForest."""
        # set class variables
        setup_forest(wave_solution="log")

        # create an SdssForest
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

        # get header and test
        header = test_obj.get_header()
        self.assertTrue(len(header) == 10)
        self.assertTrue(header[0].get("name") == "LOS_ID")
        self.assertTrue(header[0].get("value") == 100000000)
        self.assertTrue(header[1].get("name") == "RA")
        self.assertTrue(header[1].get("value") == 0.15)
        self.assertTrue(header[2].get("name") == "DEC")
        self.assertTrue(header[2].get("value") == 0.0)
        self.assertTrue(header[3].get("name") == "Z")
        self.assertTrue(header[3].get("value") == 2.1)
        self.assertTrue(header[4].get("name") == "BAD_CONT")
        self.assertTrue(header[4].get("value") == "None")
        self.assertTrue(header[5].get("name") == "MEANSNR")
        self.assertTrue(np.isclose(header[5].get("value"), 2.828427))
        self.assertTrue(header[6].get("name") == "THING_ID")
        self.assertTrue(header[6].get("value") == 100000000)
        self.assertTrue(header[7].get("name") == "PLATE")
        self.assertTrue(header[7].get("value") == "0000")
        self.assertTrue(header[8].get("name") == "MJD")
        self.assertTrue(header[8].get("value") == "00000")
        self.assertTrue(header[9].get("name") == "FIBERID")
        self.assertTrue(header[9].get("value") == "0000")

        # create a second SdssForest and coadd it to the first
        kwargs_other = {
            "ra": 0.1,
            "dec": 0.01,
            "z": 2.2,
            "flux": np.ones(10)*3,
            "ivar": np.ones(10)*4,
            "log_lambda": np.array([3.5565, 3.55655, 3.5567, 3.55675, 3.5569,
                                    3.55695, 3.5571, 3.55715, 3.5573, 3.55735]),
            "thingid": 100000000,
            "plate": 1,
            "fiberid": 2,
            "mjd": 3,
        }
        test_obj_other = SdssForest(**kwargs_other)
        test_obj.coadd(test_obj_other)

        # get header and test
        header = test_obj.get_header()
        self.assertTrue(len(header) == 10)
        self.assertTrue(header[0].get("name") == "LOS_ID")
        self.assertTrue(header[0].get("value") == 100000000)
        self.assertTrue(header[1].get("name") == "RA")
        self.assertTrue(header[1].get("value") == 0.15)
        self.assertTrue(header[2].get("name") == "DEC")
        self.assertTrue(header[2].get("value") == 0.0)
        self.assertTrue(header[3].get("name") == "Z")
        self.assertTrue(header[3].get("value") == 2.1)
        self.assertTrue(header[4].get("name") == "BAD_CONT")
        self.assertTrue(header[4].get("value") == "None")
        self.assertTrue(header[5].get("name") == "MEANSNR")
        self.assertTrue(header[5].get("value") == 8)
        self.assertTrue(header[6].get("name") == "THING_ID")
        self.assertTrue(header[6].get("value") == 100000000)
        self.assertTrue(header[7].get("name") == "PLATE")
        self.assertTrue(header[7].get("value") == "0000-0001")
        self.assertTrue(header[8].get("name") == "MJD")
        self.assertTrue(header[8].get("value") == "00000-00003")
        self.assertTrue(header[9].get("name") == "FIBERID")
        self.assertTrue(header[9].get("value") == "0000-0002")


    def test_sdss_pk1d_forest(self):
        """Test constructor for SdssPk1dForest.
        This includes a test of function rebin.
        """
        # create a SdssPk1dForest
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
            "exposures_diff": np.ones(10),
            "reso": np.ones(10),
        }
        # expected error as Pk1dForest class variables are not yet set
        with self.assertRaises(AstronomicalObjectError):
            SdssPk1dForest(**kwargs)

        # set class variables
        setup_pk1d_forest("LYA")

        # expected error as Forest class variables are not yet set
        with self.assertRaises(AstronomicalObjectError):
            SdssPk1dForest(**kwargs)

        # set class variables
        setup_forest(wave_solution="log")

        # create a SdssPk1dForest
        test_obj = SdssPk1dForest(**kwargs)

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
        self.assertTrue(len(Forest.mask_fields) == 6)
        self.assertTrue(Forest.mask_fields[0] == "flux")
        self.assertTrue(Forest.mask_fields[1] == "ivar")
        self.assertTrue(Forest.mask_fields[2] == "transmission_correction")
        self.assertTrue(Forest.mask_fields[3] == "log_lambda")
        self.assertTrue(Forest.mask_fields[4] == "exposures_diff")
        self.assertTrue(Forest.mask_fields[5] == "reso")
        self.assertTrue(np.allclose(test_obj.transmission_correction, np.ones(5)))
        self.assertTrue(np.allclose(test_obj.mean_snr, 2.8284271247461903))
        self.assertTrue(isinstance(test_obj.plate, list))
        self.assertTrue(test_obj.plate[0] == 0)
        self.assertTrue(isinstance(test_obj.fiberid, list))
        self.assertTrue(test_obj.fiberid[0] == 0)
        self.assertTrue(isinstance(test_obj.mjd, list))
        self.assertTrue(test_obj.mjd[0] == 0)
        self.assertTrue(test_obj.thingid == 100000000)

        # create SdssPk1dForest with extra variables
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
            "exposures_diff": np.ones(10),
            "reso": np.ones(10),
            "test variable": "test",
        }
        test_obj = SdssForest(**kwargs)
        self.assertTrue(isinstance(test_obj, Forest))
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
        self.assertTrue(len(Forest.mask_fields) == 6)
        self.assertTrue(Forest.mask_fields[0] == "flux")
        self.assertTrue(Forest.mask_fields[1] == "ivar")
        self.assertTrue(Forest.mask_fields[2] == "transmission_correction")
        self.assertTrue(Forest.mask_fields[3] == "log_lambda")
        self.assertTrue(Forest.mask_fields[4] == "exposures_diff")
        self.assertTrue(Forest.mask_fields[5] == "reso")
        self.assertTrue(np.allclose(test_obj.transmission_correction, np.ones(5)))
        self.assertTrue(np.allclose(test_obj.mean_snr, 2.8284271247461903))
        self.assertTrue(isinstance(test_obj.plate, list))
        self.assertTrue(test_obj.plate[0] == 0)
        self.assertTrue(isinstance(test_obj.fiberid, list))
        self.assertTrue(test_obj.fiberid[0] == 0)
        self.assertTrue(isinstance(test_obj.mjd, list))
        self.assertTrue(test_obj.mjd[0] == 0)
        self.assertTrue(test_obj.thingid == 100000000)

        # create a SdssPk1dForest with missing SdssForest variables
        kwargs = {
            "ra": 0.15,
            "dec": 0.0,
            "z": 2.1,
            "flux": np.ones(15),
            "ivar": np.ones(15)*4,
            "log_lambda": np.array([3.5565, 3.55655, 3.5567, 3.55675, 3.5569,
                                    3.55695, 3.5571, 3.55715, 3.5573, 3.55735]),
            "exposures_diff": np.ones(15),
            "reso": np.ones(15),
        }
        with self.assertRaises(AstronomicalObjectError):
            SdssPk1dForest(**kwargs)

        # create a SdssPk1dForest with missing Pk1dForest variables
        kwargs = {
            "ra": 0.15,
            "dec": 0.0,
            "z": 2.1,
            "flux": np.ones(15),
            "ivar": np.ones(15)*4,
            "thingid": 100000000,
            "plate": 0,
            "fiberid": 0,
            "mjd": 0,
        }
        with self.assertRaises(AstronomicalObjectError):
            SdssPk1dForest(**kwargs)

        # create SdssPk1dForest with missing Forest variables
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
            "exposures_diff": np.ones(15),
            "reso": np.ones(15),
        }
        with self.assertRaises(AstronomicalObjectError):
            SdssPk1dForest(**kwargs)

    def test_sdss_pk1d_forest_coadd(self):
        """Test the coadd function in SdssPk1dForest"""
        # set class variables
        setup_forest(wave_solution="log")
        setup_pk1d_forest("LYA")

        # create a SdssPk1dForest
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
            "exposures_diff": np.ones(10),
            "reso": np.ones(10),
        }
        test_obj = SdssPk1dForest(**kwargs)

        # create a second SdssPk1dForest
        kwargs_other = {
            "ra": 0.1,
            "dec": 0.01,
            "z": 2.2,
            "flux": np.ones(10)*3,
            "ivar": np.ones(10)*4,
            "log_lambda": np.array([3.5565, 3.55655, 3.5567, 3.55675, 3.5569,
                                    3.55695, 3.5571, 3.55715, 3.5573, 3.55735]),
            "thingid": 100000000,
            "plate": 1,
            "fiberid": 2,
            "mjd": 3,
            "exposures_diff": np.ones(10)*3,
            "reso": np.ones(10),
        }
        test_obj_other = SdssPk1dForest(**kwargs_other)

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
        self.assertTrue(len(Forest.mask_fields) == 6)
        self.assertTrue(Forest.mask_fields[0] == "flux")
        self.assertTrue(Forest.mask_fields[1] == "ivar")
        self.assertTrue(Forest.mask_fields[2] == "transmission_correction")
        self.assertTrue(Forest.mask_fields[3] == "log_lambda")
        self.assertTrue(Forest.mask_fields[4] == "exposures_diff")
        self.assertTrue(Forest.mask_fields[5] == "reso")
        self.assertTrue(np.allclose(test_obj.transmission_correction, np.ones(5)))
        self.assertTrue(np.allclose(test_obj.mean_snr, 8))
        self.assertTrue(isinstance(test_obj.plate, list))
        self.assertTrue(test_obj.plate[0] == 0)
        self.assertTrue(test_obj.plate[1] == 1)
        self.assertTrue(isinstance(test_obj.fiberid, list))
        self.assertTrue(test_obj.fiberid[0] == 0)
        self.assertTrue(test_obj.fiberid[1] == 2)
        self.assertTrue(isinstance(test_obj.mjd, list))
        self.assertTrue(test_obj.mjd[0] == 0)
        self.assertTrue(test_obj.mjd[1] == 3)
        self.assertTrue(test_obj.thingid == 100000000)
        self.assertTrue(Pk1dForest.lambda_abs_igm == 1215.67)
        self.assertTrue(np.allclose(test_obj.exposures_diff, np.ones(5)*2))
        self.assertTrue(np.allclose(test_obj.reso, np.ones(5)))
        self.assertTrue(np.isclose(test_obj.mean_z, 1.965425))

        # create a third SdssForest with different targetid
        kwargs_other = {
            "ra": 0.1,
            "dec": 0.01,
            "z": 2.2,
            "flux": np.ones(10)*3,
            "ivar": np.ones(10)*4,
            "log_lambda": np.array([3.5565, 3.55655, 3.5567, 3.55675, 3.5569,
                                    3.55695, 3.5571, 3.55715, 3.5573, 3.55735]),
            "thingid": 100010000,
            "plate": 1,
            "fiberid": 2,
            "mjd": 3,
            "exposures_diff": np.ones(10)*3,
            "reso": np.ones(10),
        }
        test_obj_other = SdssPk1dForest(**kwargs_other)

        # coadding them should raise an error
        with self.assertRaises(AstronomicalObjectError):
            test_obj.coadd(test_obj_other)

    def test_sdss_pk1d_forest_get_data(self):
        """Test method get_data for SdssPk1dForest."""
        # set class variables
        setup_forest(wave_solution="log")
        setup_pk1d_forest("LYA")

        # create an SdssPk1dForest
        kwargs = {
            "ra": 0.15,
            "dec": 0.0,
            "z": 2.1,
            "flux": np.ones(10),
            "ivar": np.ones(10)*4,
            "log_lambda": np.ones(10),
            "thingid": 100000000,
            "plate": 0,
            "fiberid": 0,
            "mjd": 0,
            "exposures_diff": np.ones(10),
            "reso": np.ones(10),
        }
        test_obj = SdssPk1dForest(**kwargs)

        cols, names, units, comments = test_obj.get_data()
        self.assertTrue(len(names) == 5)
        self.assertTrue(len(cols) == 5)
        self.assertTrue(len(units) == 5)
        self.assertTrue(len(comments) == 5)
        self.assertTrue(names[0] == "LOGLAM")
        self.assertTrue(np.allclose(cols[0], np.ones(10)))
        self.assertTrue(units[0] == "log Angstrom")
        self.assertTrue(comments[0] == "Log lambda")
        self.assertTrue(names[1] == "CONT")
        self.assertTrue(np.allclose(cols[1], np.zeros(10)))
        self.assertTrue(units[1] == "Flux units")
        self.assertTrue(comments[1] == ("Quasar continuum if BAD_CONT is 'None'. "
                                        "Check input spectra for units"))
        self.assertTrue(names[2] == "DELTA")
        self.assertTrue(np.allclose(cols[2], np.zeros(10)))
        self.assertTrue(units[2] == "")
        self.assertTrue(comments[2] == "Delta field")
        self.assertTrue(names[3] == "WEIGHT")
        self.assertTrue(np.allclose(cols[3], np.zeros(10)))
        self.assertTrue(units[3] == "")
        self.assertTrue(comments[3] == "Pixel weights")
        self.assertTrue(names[4] == "DIFF")
        self.assertTrue(np.allclose(cols[4], np.ones(10)))
        self.assertTrue(units[4] == "Flux units")
        self.assertTrue(comments[4] == "Difference. Check input spectra for units")

    def test_sdss_pk1d_forest_get_header(self):
        """Test method get_header for SdssPk1dForest."""
        # set class variables
        setup_forest(wave_solution="log")
        setup_pk1d_forest("LYA")

        # create an SdssPk1dForest
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
            "exposures_diff": np.ones(10),
            "reso": np.ones(10),
        }
        test_obj = SdssPk1dForest(**kwargs)

        # get header and test
        header = test_obj.get_header()
        self.assertTrue(len(header) == 12)
        self.assertTrue(header[0].get("name") == "LOS_ID")
        self.assertTrue(header[0].get("value") == 100000000)
        self.assertTrue(header[1].get("name") == "RA")
        self.assertTrue(header[1].get("value") == 0.15)
        self.assertTrue(header[2].get("name") == "DEC")
        self.assertTrue(header[2].get("value") == 0.0)
        self.assertTrue(header[3].get("name") == "Z")
        self.assertTrue(header[3].get("value") == 2.1)
        self.assertTrue(header[4].get("name") == "BAD_CONT")
        self.assertTrue(header[4].get("value") == "None")
        self.assertTrue(header[5].get("name") == "MEANSNR")
        self.assertTrue(np.isclose(header[5].get("value"), 2.828427))
        self.assertTrue(header[6].get("name") == "MEANZ")
        self.assertTrue(header[6].get("value") == 1.9654252799454879)
        self.assertTrue(header[7].get("name") == "MEANRESO")
        self.assertTrue(header[7].get("value") == 1.0)
        self.assertTrue(header[8].get("name") == "THING_ID")
        self.assertTrue(header[8].get("value") == 100000000)
        self.assertTrue(header[9].get("name") == "PLATE")
        self.assertTrue(header[9].get("value") == "0000")
        self.assertTrue(header[10].get("name") == "MJD")
        self.assertTrue(header[10].get("value") == "00000")
        self.assertTrue(header[11].get("name") == "FIBERID")
        self.assertTrue(header[11].get("value") == "0000")

        # create a second SdssPk1dForest and coadd it to the first
        kwargs_other = {
            "ra": 0.1,
            "dec": 0.01,
            "z": 2.2,
            "flux": np.ones(10)*3,
            "ivar": np.ones(10)*4,
            "log_lambda": np.array([3.5565, 3.55655, 3.5567, 3.55675, 3.5569,
                                    3.55695, 3.5571, 3.55715, 3.5573, 3.55735]),
            "thingid": 100000000,
            "plate": 1,
            "fiberid": 2,
            "mjd": 3,
            "exposures_diff": np.ones(10),
            "reso": np.ones(10),
        }
        test_obj_other = SdssPk1dForest(**kwargs_other)
        test_obj.coadd(test_obj_other)

        # get header and test
        header = test_obj.get_header()
        self.assertTrue(len(header) == 12)
        self.assertTrue(header[0].get("name") == "LOS_ID")
        self.assertTrue(header[0].get("value") == 100000000)
        self.assertTrue(header[1].get("name") == "RA")
        self.assertTrue(header[1].get("value") == 0.15)
        self.assertTrue(header[2].get("name") == "DEC")
        self.assertTrue(header[2].get("value") == 0.0)
        self.assertTrue(header[3].get("name") == "Z")
        self.assertTrue(header[3].get("value") == 2.1)
        self.assertTrue(header[4].get("name") == "BAD_CONT")
        self.assertTrue(header[4].get("value") == "None")
        self.assertTrue(header[5].get("name") == "MEANSNR")
        self.assertTrue(header[5].get("value") == 8)
        self.assertTrue(header[6].get("name") == "MEANZ")
        self.assertTrue(header[6].get("value") == 1.9654252799454879)
        self.assertTrue(header[7].get("name") == "MEANRESO")
        self.assertTrue(header[7].get("value") == 1.0)
        self.assertTrue(header[8].get("name") == "THING_ID")
        self.assertTrue(header[8].get("value") == 100000000)
        self.assertTrue(header[9].get("name") == "PLATE")
        self.assertTrue(header[9].get("value") == "0000-0001")
        self.assertTrue(header[10].get("name") == "MJD")
        self.assertTrue(header[10].get("value") == "00000-00003")
        self.assertTrue(header[11].get("name") == "FIBERID")
        self.assertTrue(header[11].get("value") == "0000-0002")

if __name__ == '__main__':
    unittest.main()
