"""This file contains tests related to AstronomicalObject and its childs"""
import unittest

import healpy
import numpy as np

from picca.delta_extraction.astronomical_object import AstronomicalObject
from picca.delta_extraction.astronomical_objects.desi_forest import DesiForest
from picca.delta_extraction.astronomical_objects.desi_pk1d_forest import DesiPk1dForest
from picca.delta_extraction.astronomical_objects.forest import Forest
from picca.delta_extraction.astronomical_objects.pk1d_forest import Pk1dForest
from picca.delta_extraction.astronomical_objects.sdss_forest import SdssForest
from picca.delta_extraction.astronomical_objects.sdss_pk1d_forest import SdssPk1dForest
from picca.delta_extraction.errors import AstronomicalObjectError
from picca.tests.delta_extraction.abstract_test import AbstractTest
from picca.tests.delta_extraction.test_utils import (reset_forest, setup_forest,
                                                     setup_pk1d_forest)

# define auxiliar variables
FLUX = np.ones(10)
FLUX2 = np.ones(10) * 3
FLUX_REBIN = np.ones(5)
FLUX_COADD = np.ones(5) * 2
IVAR = np.ones(10) * 4
IVAR_REBIN = np.ones(5) * 8
IVAR_COADD = np.ones(5) * 16
EXPOSURES_DIFF = np.ones(10)
EXPOSURES_DIFF2 = np.ones(10) * 3
EXPOSURES_DIFF_REBIN = np.ones(5)
EXPOSURES_DIFF_COADD = np.ones(5) * 2
RESO = np.ones(10)
RESO2 = np.ones(10) * 3
RESO_REBIN = np.ones(5)
RESO_COADD = np.ones(5) * 2
LOG_LAMBDA = np.array([
    3.5564725, 3.5564325, 3.5567725, 3.5567325, 3.5570725, 3.5570325,
    3.5573725, 3.5573325, 3.5576725, 3.5576325
])
LOG_LAMBDA_REBIN = np.array(
    [3.5564525, 3.5567525, 3.5570525, 3.5573525, 3.5576525])
LAMBDA_ = np.array(
    [3610.5, 3610.9, 3650.5, 3650.9, 3670.5, 3670.9, 3680.5, 3680.9, 3700.5, 3700.9])
LAMBDA_REBIN = np.array([3610.5, 3650.5, 3670.5, 3680.5, 3700.5])

THINGID = 100000000
TARGETID = 100000000

# define contructor for AstronomicalObject
kwargs_astronomical_object = {
    "los_id": 9999,
    "ra": 0.15,
    "dec": 0.0,
    "z": 2.1,
}

# define contructor for AstronomicalObject comparison objects
kwargs_astronomical_object_gt = {
    "healpix_ordering": {
        "los_id": 9999,
        "ra": 0.0,
        "dec": 0.0,
        "z": 2.1
    },
    "ra_ordering": {
        "los_id": 9999,
        "ra": 0.1,
        "dec": 0.0,
        "z": 2.1
    },
    "dec_ordering": {
        "los_id": 9999,
        "ra": 0.15,
        "dec": -0.01,
        "z": 2.1
    },
    "z_ordering": {
        "los_id": 9999,
        "ra": 0.15,
        "dec": 0.0,
        "z": 2.0
    },
}

# define contructors for Forest
kwargs_forest = kwargs_astronomical_object.copy()
kwargs_forest.update({
    "flux": FLUX,
    "ivar": IVAR,
})

kwargs_forest2 = kwargs_astronomical_object.copy()
kwargs_forest2.update({
    "flux": FLUX2,
    "ivar": IVAR,
})

kwargs_forest_log = kwargs_forest.copy()
kwargs_forest_log.update({
    "log_lambda": LOG_LAMBDA,
})

kwargs_forest_log2 = kwargs_forest2.copy()
kwargs_forest_log2.update({
    "log_lambda": LOG_LAMBDA,
})

kwargs_forest_lin = kwargs_forest.copy()
kwargs_forest_lin.update({
    "lambda": LAMBDA_,
})

kwargs_forest_lin2 = kwargs_forest2.copy()
kwargs_forest_lin2.update({
    "lambda": LAMBDA_,
})

kwargs_forest_rebin = kwargs_astronomical_object.copy()
kwargs_forest_rebin.update({
    "flux": FLUX_REBIN,
    "ivar": IVAR_REBIN,
})

kwargs_forest_log_rebin = kwargs_forest_rebin.copy()
kwargs_forest_log_rebin.update({
    "log_lambda": LOG_LAMBDA_REBIN,
})

kwargs_forest_lin_rebin = kwargs_forest_rebin.copy()
kwargs_forest_lin_rebin.update({
    "lambda": LAMBDA_REBIN,
})

kwargs_forest_coadd = kwargs_astronomical_object.copy()
kwargs_forest_coadd.update({
    "flux": FLUX_COADD,
    "ivar": IVAR_COADD,
})

kwargs_forest_log_coadd = kwargs_forest_coadd.copy()
kwargs_forest_log_coadd.update({
    "log_lambda": LOG_LAMBDA_REBIN,
})

kwargs_forest_lin_coadd = kwargs_forest_coadd.copy()
kwargs_forest_lin_coadd.update({
    "lambda": LAMBDA_REBIN,
})

# define contructors for Pk1dForest
kwargs_pk1d_forest = kwargs_forest.copy()
kwargs_pk1d_forest.update({
    "exposures_diff": EXPOSURES_DIFF,
    "reso": RESO,
})

kwargs_pk1d_forest2 = kwargs_forest2.copy()
kwargs_pk1d_forest2.update({
    "exposures_diff": EXPOSURES_DIFF2,
    "reso": RESO2,
})

kwargs_pk1d_forest_log = kwargs_pk1d_forest.copy()
kwargs_pk1d_forest_log.update({
    "log_lambda": LOG_LAMBDA,
})

kwargs_pk1d_forest_log2 = kwargs_pk1d_forest2.copy()
kwargs_pk1d_forest_log2.update({
    "log_lambda": LOG_LAMBDA,
})

kwargs_pk1d_forest_lin = kwargs_pk1d_forest.copy()
kwargs_pk1d_forest_lin.update({
    "lambda": LAMBDA_,
})

kwargs_pk1d_forest_lin2 = kwargs_pk1d_forest2.copy()
kwargs_pk1d_forest_lin2.update({
    "lambda": LAMBDA_,
})

kwargs_pk1d_forest_rebin = kwargs_forest_rebin.copy()
kwargs_pk1d_forest_rebin.update({
    "exposures_diff": EXPOSURES_DIFF_REBIN,
    "reso": RESO_REBIN,
})

kwargs_pk1d_forest_log_rebin = kwargs_pk1d_forest_rebin.copy()
kwargs_pk1d_forest_log_rebin.update({
    "log_lambda": LOG_LAMBDA_REBIN,
})

kwargs_pk1d_forest_lin_rebin = kwargs_pk1d_forest_rebin.copy()
kwargs_pk1d_forest_lin_rebin.update({
    "lambda": LAMBDA_REBIN,
})

kwargs_pk1d_forest_coadd = kwargs_forest_coadd.copy()
kwargs_pk1d_forest_coadd.update({
    "exposures_diff": EXPOSURES_DIFF_COADD,
    "reso": RESO_COADD,
})

kwargs_pk1d_forest_log_coadd = kwargs_pk1d_forest_coadd.copy()
kwargs_pk1d_forest_log_coadd.update({
    "log_lambda": LOG_LAMBDA_REBIN,
})

kwargs_pk1d_forest_lin_coadd = kwargs_pk1d_forest_coadd.copy()
kwargs_pk1d_forest_lin_coadd.update({
    "lambda": LAMBDA_REBIN,
})

# define contructors for DesiForest
kwargs_desi_forest = kwargs_forest_lin.copy()
del kwargs_desi_forest["los_id"]
kwargs_desi_forest.update({
    "targetid": TARGETID,
    "night": 0,
    "petal": 0,
    "tile": 0,
})

kwargs_desi_forest2 = kwargs_forest_lin2.copy()
del kwargs_desi_forest2["los_id"]
kwargs_desi_forest2.update({
    "targetid": TARGETID,
    "night": 1,
    "petal": 2,
    "tile": 3,
})

kwargs_desi_forest_rebin = kwargs_forest_lin_rebin.copy()
kwargs_desi_forest_rebin.update({
    "targetid": TARGETID,
    "night": [0],
    "petal": [0],
    "tile": [0],
})
kwargs_desi_forest_rebin["los_id"] = TARGETID

kwargs_desi_forest_coadd = kwargs_forest_lin_coadd.copy()
kwargs_desi_forest_coadd.update({
    "targetid": TARGETID,
    "night": [0, 1],
    "petal": [0, 2],
    "tile": [0, 3],
})
kwargs_desi_forest_coadd["los_id"] = TARGETID

# define contructors for DesiPk1dForest
kwargs_desi_pk1d_forest = kwargs_desi_forest.copy()
kwargs_desi_pk1d_forest.update(kwargs_pk1d_forest_lin)
del kwargs_desi_pk1d_forest["los_id"]

kwargs_desi_pk1d_forest2 = kwargs_desi_forest2.copy()
kwargs_desi_pk1d_forest2.update(kwargs_pk1d_forest_lin2)
del kwargs_desi_pk1d_forest2["los_id"]

kwargs_desi_pk1d_forest_rebin = kwargs_pk1d_forest_rebin.copy()
kwargs_desi_pk1d_forest_rebin.update(kwargs_desi_forest_rebin)

kwargs_desi_pk1d_forest_coadd = kwargs_pk1d_forest_coadd.copy()
kwargs_desi_pk1d_forest_coadd.update(kwargs_desi_forest_coadd)

# define contructors for SdssForest
kwargs_sdss_forest = kwargs_forest_log.copy()
del kwargs_sdss_forest["los_id"]
kwargs_sdss_forest.update({
    "thingid": THINGID,
    "plate": 0,
    "fiberid": 0,
    "mjd": 0,
})

kwargs_sdss_forest2 = kwargs_forest_log2.copy()
del kwargs_sdss_forest2["los_id"]
kwargs_sdss_forest2.update({
    "thingid": THINGID,
    "plate": 1,
    "fiberid": 2,
    "mjd": 3,
})

kwargs_sdss_forest_rebin = kwargs_forest_log_rebin.copy()
kwargs_sdss_forest_rebin.update({
    "thingid": THINGID,
    "plate": [0],
    "fiberid": [0],
    "mjd": [0],
})
kwargs_sdss_forest_rebin["los_id"] = THINGID

kwargs_sdss_forest_coadd = kwargs_forest_log_coadd.copy()
kwargs_sdss_forest_coadd.update({
    "thingid": THINGID,
    "plate": [0, 1],
    "fiberid": [0, 2],
    "mjd": [0, 3],
})
kwargs_sdss_forest_coadd["los_id"] = THINGID

# define contructors for SdssPk1dForest
kwargs_sdss_pk1d_forest = kwargs_sdss_forest.copy()
kwargs_sdss_pk1d_forest.update(kwargs_pk1d_forest_lin)
del kwargs_sdss_pk1d_forest["los_id"]

kwargs_sdss_pk1d_forest2 = kwargs_sdss_forest2.copy()
kwargs_sdss_pk1d_forest2.update(kwargs_pk1d_forest_lin2)
del kwargs_sdss_pk1d_forest2["los_id"]

kwargs_sdss_pk1d_forest_rebin = kwargs_pk1d_forest_rebin.copy()
kwargs_sdss_pk1d_forest_rebin.update(kwargs_sdss_forest_rebin)

kwargs_sdss_pk1d_forest_coadd = kwargs_pk1d_forest_coadd.copy()
kwargs_sdss_pk1d_forest_coadd.update(kwargs_sdss_forest_coadd)

# pylint: disable-msg=too-many-public-methods
# this is a test class
class AstronomicalObjectTest(AbstractTest):
    """Test AstronomicalObject and its childs.

    Methods
    -------
    compare_ascii (from AbstractTest)
    compare_fits (from AbstractTest)
    assert_astronomical_object
    assert_forest_object
    assert_get_data
    assert_get_header
    setUp
    tearDown
    test_astronomical_object
    test_astronomical_object_comparison
    test_astronomical_object_get_header
    test_desi_forest
    test_desi_forest_coadd
    test_desi_forest_get_data
    test_desi_forest_get_header
    test_desi_pk1d_forest
    test_desi_pk1d_forest_coadd
    test_desi_pk1d_forest_get_data
    test_desi_pk1d_forest_get_header
    test_forest
    test_forest_coadd
    test_forest_get_data
    test_forest_get_header
    test_pk1d_forest
    test_pk1d_forest_coadd
    test_pk1d_forest_get_data
    test_pk1d_forest_get_header
    test_sdss_forest
    test_sdss_forest_coadd
    test_sdss_forest_get_data
    test_sdss_forest_get_header
    test_sdss_pk1d_forest
    test_sdss_pk1d_forest_coadd
    test_sdss_pk1d_forest_get_data
    test_sdss_pk1d_forest_get_header
    """
    def assert_astronomical_object(self, test_obj, kwargs):
        """Assert the correct properties of a test AstronomicalObject

        Arguments
        ---------
        test_obj: AstronomicalObject
        The Forest instance to check

        kwargs: dictionary
        Dictionary used to initialize instance
        """
        self.assertTrue(isinstance(test_obj, AstronomicalObject))
        ra = kwargs.get("ra")
        dec = kwargs.get("dec")
        healpix = healpy.ang2pix(16, np.pi / 2 - dec, ra)
        self.assertTrue(test_obj.los_id == kwargs.get("los_id"))
        self.assertTrue(test_obj.healpix == healpix)
        self.assertTrue(test_obj.z == kwargs.get("z"))

    def assert_forest_object(self, test_obj, kwargs):
        """Assert the correct properties of a test Forest

        Arguments
        ---------
        test_obj: Forest
        The Forest instance to check

        kwargs: dictionary
        Dictionary used to initialize instance
        """
        self.assertTrue(isinstance(test_obj, Forest))

        self.assert_astronomical_object(test_obj, kwargs)

        self.assertTrue(test_obj.bad_continuum_reason is None)
        if "continuum" in kwargs:
            self.assertTrue(
                np.allclose(test_obj.continuum, kwargs.get("continuum")))
        else:
            self.assertTrue(test_obj.continuum is None)
        if "deltas" in kwargs:
            self.assertTrue(np.allclose(test_obj.deltas, kwargs.get("deltas")))
        else:
            self.assertTrue(test_obj.deltas is None)
        if Forest.wave_solution == "log":
            if (test_obj.log_lambda.size != kwargs.get("log_lambda").size):
                print("\nlog_lambda")
                print(test_obj.log_lambda)
                print("compared to")
                print(kwargs.get("log_lambda"))
            self.assertTrue(
                np.allclose(test_obj.log_lambda, kwargs.get("log_lambda")))
            self.assertTrue(test_obj.lambda_ is None)
        elif Forest.wave_solution == "lin":
            self.assertTrue(test_obj.log_lambda is None)
            self.assertTrue(np.allclose(test_obj.lambda_, kwargs.get("lambda")))
        flux = kwargs.get("flux")
        ivar = kwargs.get("ivar")
        self.assertTrue(np.allclose(test_obj.flux, flux))
        self.assertTrue(np.allclose(test_obj.ivar, ivar))

        if isinstance(test_obj, Pk1dForest):
            self.assertTrue(len(Forest.mask_fields) == 6)
        else:
            self.assertTrue(len(Forest.mask_fields) == 4)
        self.assertTrue(Forest.mask_fields[0] == "flux")
        self.assertTrue(Forest.mask_fields[1] == "ivar")
        self.assertTrue(Forest.mask_fields[2] == "transmission_correction")
        if Forest.wave_solution == "log":
            self.assertTrue(Forest.mask_fields[3] == "log_lambda")
        elif Forest.wave_solution == "lin":
            self.assertTrue(Forest.mask_fields[3] == "lambda_")
        self.assertTrue(
            np.allclose(test_obj.transmission_correction, np.ones_like(flux)))
        mean_snr = (flux * np.sqrt(ivar)).mean()
        self.assertTrue(np.allclose(test_obj.mean_snr, mean_snr))

        if isinstance(test_obj, SdssForest):
            self.assertTrue(isinstance(test_obj.plate, list))
            self.assertTrue(test_obj.plate == kwargs.get("plate"))
            self.assertTrue(isinstance(test_obj.fiberid, list))
            self.assertTrue(test_obj.fiberid == kwargs.get("fiberid"))
            self.assertTrue(isinstance(test_obj.mjd, list))
            self.assertTrue(test_obj.mjd == kwargs.get("mjd"))
            self.assertTrue(test_obj.thingid == kwargs.get("thingid"))

        if isinstance(test_obj, DesiForest):
            self.assertTrue(isinstance(test_obj.night, list))
            self.assertTrue(test_obj.night == kwargs.get("night"))
            self.assertTrue(isinstance(test_obj.petal, list))
            self.assertTrue(test_obj.petal == kwargs.get("petal"))
            self.assertTrue(isinstance(test_obj.tile, list))
            self.assertTrue(test_obj.tile == kwargs.get("tile"))
            self.assertTrue(test_obj.targetid == kwargs.get("targetid"))

        if isinstance(test_obj, Pk1dForest):
            self.assertTrue(Pk1dForest.lambda_abs_igm == 1215.67)
            self.assertTrue(
                np.allclose(test_obj.exposures_diff,
                            kwargs.get("exposures_diff")))
            self.assertTrue(np.allclose(test_obj.reso, kwargs.get("reso")))
            if Forest.wave_solution == "log":
                log_lambda = kwargs.get("log_lambda")
                mean_z = ((np.power(10., log_lambda[len(log_lambda) - 1]) +
                           np.power(10., log_lambda[0])) / 2. /
                          Pk1dForest.lambda_abs_igm - 1.0)
            elif Forest.wave_solution == "lin":
                lambda_ = kwargs.get("lambda")
                mean_z = ((lambda_[len(lambda_) - 1] + lambda_[0]) / 2. /
                          Pk1dForest.lambda_abs_igm - 1.0)
            if not np.isclose(test_obj.mean_z, mean_z):
                print(test_obj.mean_z, mean_z)
                print(log_lambda, test_obj.log_lambda,
                      log_lambda == test_obj.log_lambda)
            self.assertTrue(np.isclose(test_obj.mean_z, mean_z))
            self.assertTrue(
                np.isclose(test_obj.mean_reso,
                           kwargs.get("reso").mean()))

    def assert_get_data(self, test_obj):
        """Assert the correct properties of the return of method get_data

        Arguments
        ---------
        test_obj: Forest
        The Forest instance to check
        """
        self.assertTrue(isinstance(test_obj, Forest))
        cols, names, units, comments = test_obj.get_data()

        if Forest.wave_solution == "log":
            self.assertTrue(names[0] == "LOGLAM")
            self.assertTrue(np.allclose(cols[0], test_obj.log_lambda))
            self.assertTrue(units[0] == "log Angstrom")
            self.assertTrue(comments[0] == "Log lambda")
        elif Forest.wave_solution == "lin":
            self.assertTrue(names[0] == "LAMBDA")
            self.assertTrue(np.allclose(cols[0], test_obj.lambda_))
            self.assertTrue(units[0] == "Angstrom")
            self.assertTrue(comments[0] == "Lambda")

        if test_obj.deltas is None:
            deltas = np.zeros_like(test_obj.flux)
        else:
            deltas = test_obj.deltas
        self.assertTrue(names[1] == "DELTA")
        self.assertTrue(np.allclose(cols[1], deltas))
        self.assertTrue(units[1] == "")
        self.assertTrue(comments[1] == "Delta field")

        if test_obj.weights is None:
            weights = np.zeros_like(test_obj.flux)
        else:
            weights = test_obj.weights
        self.assertTrue(names[2] == "WEIGHT")
        self.assertTrue(np.allclose(cols[2], weights))
        self.assertTrue(units[2] == "")
        self.assertTrue(comments[2] == "Pixel weights")

        if test_obj.continuum is None:
            continuum = np.zeros_like(test_obj.flux)
        else:
            continuum = test_obj.continuum
        self.assertTrue(names[3] == "CONT")
        self.assertTrue(np.allclose(cols[3], continuum))
        self.assertTrue(units[3] == "Flux units")
        self.assertTrue(
            comments[3] == ("Quasar continuum. "
                            "Check input spectra for units"))

        if isinstance(test_obj, Pk1dForest):
            self.assertTrue(names[4] == "IVAR")
            self.assertTrue(np.allclose(cols[4], test_obj.ivar))
            self.assertTrue(units[4] == "Flux units")
            self.assertTrue(
                comments[4] == "Inverse variance. Check input spectra for units")

            self.assertTrue(names[5] == "DIFF")
            self.assertTrue(np.allclose(cols[5], test_obj.exposures_diff))
            self.assertTrue(units[5] == "Flux units")
            self.assertTrue(
                comments[5] == "Difference. Check input spectra for units")

    def assert_get_header(self, test_obj):
        """Assert the correct properties of the return of method get_data

        Arguments
        ---------
        test_obj: Forest
        The Forest instance to check
        """
        self.assertTrue(isinstance(test_obj, AstronomicalObject))
        header = test_obj.get_header()

        self.assertTrue(header[0].get("name") == "LOS_ID")
        self.assertTrue(header[0].get("value") == test_obj.los_id)
        self.assertTrue(header[1].get("name") == "RA")
        self.assertTrue(header[1].get("value") == test_obj.ra)
        self.assertTrue(header[2].get("name") == "DEC")
        self.assertTrue(header[2].get("value") == test_obj.dec)
        self.assertTrue(header[3].get("name") == "Z")
        self.assertTrue(header[3].get("value") == test_obj.z)

        index = 3
        if isinstance(test_obj, Forest):
            self.assertTrue(header[index + 1].get("name") == "MEANSNR")
            self.assertTrue(header[index + 1].get("value") == test_obj.mean_snr)
            self.assertTrue(header[index + 2].get("name") == "BLINDING")
            self.assertTrue(header[index + 2].get("value") == "none")
            index += 2
        if isinstance(test_obj, Pk1dForest):
            self.assertTrue(header[index + 1].get("name") == "MEANZ")
            self.assertTrue(header[index + 1].get("value") == test_obj.mean_z)
            self.assertTrue(header[index + 2].get("name") == "MEANRESO")
            self.assertTrue(header[index +
                                   2].get("value") == test_obj.mean_reso)
            index += 2
        if isinstance(test_obj, SdssForest):
            self.assertTrue(header[index + 1].get("name") == "THING_ID")
            self.assertTrue(header[index + 1].get("value") == test_obj.thingid)
            self.assertTrue(header[index + 2].get("name") == "PLATE")
            plate = "-".join([f"{plate:04d}" for plate in test_obj.plate])
            self.assertTrue(header[index + 2].get("value") == plate)
            self.assertTrue(header[index + 3].get("name") == "MJD")
            mjd = "-".join([f"{mjd:05d}" for mjd in test_obj.mjd])
            self.assertTrue(header[index + 3].get("value") == mjd)
            self.assertTrue(header[index + 4].get("name") == "FIBERID")
            fiberid = "-".join(
                [f"{fiberid:04d}" for fiberid in test_obj.fiberid])
            self.assertTrue(header[index + 4].get("value") == fiberid)
            index += 4
        if isinstance(test_obj, DesiForest):
            self.assertTrue(header[index + 1].get("name") == "TARGETID")
            self.assertTrue(header[index + 1].get("value") == test_obj.targetid)
            self.assertTrue(header[index + 2].get("name") == "NIGHT")
            night = "-".join([f"{night}" for night in test_obj.night])
            self.assertTrue(header[index + 2].get("value") == night)
            self.assertTrue(header[index + 3].get("name") == "PETAL")
            petal = "-".join([f"{petal}" for petal in test_obj.petal])
            self.assertTrue(header[index + 3].get("value") == petal)
            self.assertTrue(header[index + 4].get("name") == "TILE")
            tile = "-".join([f"{tile}" for tile in test_obj.tile])
            self.assertTrue(header[index + 4].get("value") == tile)
            index += 4

    def setUp(self):
        super().setUp()
        reset_forest()

    def tearDown(self):
        reset_forest()

    def test_astronomical_object(self):
        """Test constructor for AstronomicalObject."""
        test_obj = AstronomicalObject(**kwargs_astronomical_object)
        self.assert_astronomical_object(test_obj, kwargs_astronomical_object)

    def test_astronomical_object_comparison(self):
        """Test comparison between instances of AstronomicalObject."""
        test_obj = AstronomicalObject(**kwargs_astronomical_object)

        for kwargs in kwargs_astronomical_object_gt.values():
            other = AstronomicalObject(**kwargs)
            self.assertTrue(test_obj > other)
            self.assertTrue(test_obj != other)
            self.assertFalse(test_obj == other)
            self.assertFalse(test_obj < other)

        # equal objects (but with different los_id)
        # it makes sense that the los_id be different if we combine deltas
        # from different surveys
        kwargs = {
            "los_id": 1234,
            "ra": 0.15,
            "dec": 0.0,
            "z": 2.1,
        }
        other = AstronomicalObject(**kwargs)
        self.assertFalse(test_obj > other)
        self.assertTrue(test_obj == other)
        self.assertFalse(test_obj < other)

    def test_astronomical_object_get_header(self):
        """Test method get_header for AstronomicalObject."""
        test_obj = AstronomicalObject(**kwargs_astronomical_object)

        # get header and test
        self.assert_get_header(test_obj)

    def test_desi_forest(self):
        """Test constructor for DesiForest.
        This includes a test of function rebin.
        """
        # expected error as class variables are not yet set
        with self.assertRaises(AstronomicalObjectError):
            DesiForest(**kwargs_desi_forest)

        # set Forest class variables
        setup_forest(wave_solution="lin")

        # create a DesiForest
        test_obj = DesiForest(**kwargs_desi_forest)
        test_obj.rebin()
        self.assert_forest_object(test_obj, kwargs_desi_forest_rebin)

        # create forest with extra variables
        kwargs = kwargs_desi_forest.copy()
        kwargs.update({
            "test variable": "test",
        })
        test_obj = DesiForest(**kwargs)
        test_obj.rebin()
        self.assert_forest_object(test_obj, kwargs_desi_forest_rebin)

        # create a DesiForest with missing night, petal and tile
        kwargs = kwargs_desi_forest.copy()
        del kwargs["night"], kwargs["petal"], kwargs["tile"]
        test_obj = DesiForest(**kwargs)
        test_obj.rebin()

        kwargs = kwargs_desi_forest_rebin.copy()
        kwargs["night"] = []
        kwargs["petal"] = []
        kwargs["tile"] = []
        self.assert_forest_object(test_obj, kwargs)

        # create a DesiForest with missing DesiForest variables
        kwargs = {
            "ra":
                0.15,
            "dec":
                0.0,
            "z":
                2.1,
            "flux":
                np.ones(15),
            "ivar":
                np.ones(15) * 4,
            "lambda":
                np.array([
                    3610, 3610.4, 3650, 3650.4, 3670, 3670.4, 3680, 3680.4,
                    3700, 3700.4
                ]),
        }
        with self.assertRaises(AstronomicalObjectError):
            DesiForest(**kwargs)

        # create forest with missing Forest variables
        kwargs = {
            "los_id": 9999,
            "ra": 0.15,
            "dec": 0.0,
            "z": 2.1,
            "ivar": np.ones(15) * 4,
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
        test_obj = DesiForest(**kwargs_desi_forest)
        test_obj.rebin()

        # create a second DesiForest
        test_obj_other = DesiForest(**kwargs_desi_forest2)
        test_obj_other.rebin()

        # coadd them
        test_obj.coadd(test_obj_other)

        self.assert_forest_object(test_obj, kwargs_desi_forest_coadd)

        # create a third DesiForest with different targetid
        kwargs = kwargs_desi_forest2.copy()
        kwargs["targetid"] = 999
        test_obj_other = DesiForest(**kwargs)
        test_obj_other.rebin()

        # coadding them whould raise an error
        with self.assertRaises(AstronomicalObjectError):
            test_obj.coadd(test_obj_other)

    def test_desi_forest_get_data(self):
        """Test method get_data for DesiForest."""
        # set class variables
        setup_forest(wave_solution="lin")

        # create a DesiForest
        test_obj = DesiForest(**kwargs_desi_forest)
        test_obj.rebin()

        self.assert_get_data(test_obj)

    def test_desi_forest_get_header(self):
        """Test method get_header for DesiForest."""
        # set class variables
        setup_forest(wave_solution="lin")

        # create a DesiForest
        test_obj = DesiForest(**kwargs_desi_forest)
        test_obj.rebin()

        # get header and test
        self.assert_get_header(test_obj)

        # create a second DesiForest and coadd it to the first
        test_obj_other = DesiForest(**kwargs_desi_forest2)
        test_obj_other.rebin()
        test_obj.coadd(test_obj_other)

        # get header and test
        self.assert_get_header(test_obj)

    def test_desi_pk1d_forest(self):
        """Test constructor for DesiPk1dForest.
        This includes a test of function rebin.
        """
        # create a DesiPk1dForest class variables are not yet set
        with self.assertRaises(AstronomicalObjectError):
            DesiPk1dForest(**kwargs_desi_pk1d_forest)

        # set class variables
        setup_pk1d_forest("LYA")

        # expected error as Forest class variables are not yet set
        with self.assertRaises(AstronomicalObjectError):
            DesiPk1dForest(**kwargs_desi_pk1d_forest)

        # set class variables
        setup_forest(wave_solution="lin")

        test_obj = DesiPk1dForest(**kwargs_desi_pk1d_forest)
        test_obj.rebin()
        self.assertTrue(isinstance(test_obj, DesiPk1dForest))
        self.assertTrue(isinstance(test_obj, DesiForest))
        self.assertTrue(isinstance(test_obj, Pk1dForest))
        self.assert_forest_object(test_obj, kwargs_desi_pk1d_forest_rebin)

        # create forest with extra variables
        kwargs = kwargs_desi_pk1d_forest.copy()
        kwargs.update({
            "test_variable": "test",
        })
        self.assertTrue(isinstance(test_obj, DesiPk1dForest))
        self.assertTrue(isinstance(test_obj, DesiForest))
        self.assertTrue(isinstance(test_obj, Pk1dForest))
        test_obj = DesiPk1dForest(**kwargs)
        test_obj.rebin()

        # create a DesiPk1dForest with missing night, petal and tile
        # create a DesiForest with missing night, petal and tile
        kwargs = kwargs_desi_pk1d_forest.copy()
        del kwargs["night"], kwargs["petal"], kwargs["tile"]
        test_obj = DesiPk1dForest(**kwargs)
        test_obj.rebin()

        kwargs = kwargs_desi_pk1d_forest_rebin.copy()
        kwargs["night"] = []
        kwargs["petal"] = []
        kwargs["tile"] = []
        self.assert_forest_object(test_obj, kwargs)

        # create a DesiForest with missing DesiForest variables
        kwargs = {
            "ra":
                0.15,
            "dec":
                0.0,
            "z":
                2.1,
            "flux":
                np.ones(15),
            "ivar":
                np.ones(15) * 4,
            "lambda":
                np.array([
                    3610, 3610.4, 3650, 3650.4, 3670, 3670.4, 3680, 3680.4,
                    3700, 3700.4
                ]),
            "targetid":
                100000000,
            "exposures_diff":
                np.ones(10),
            "reso":
                np.ones(10),
        }
        with self.assertRaises(AstronomicalObjectError):
            DesiPk1dForest(**kwargs)

        # create a DesiForest with missing Pk1dForest variables
        kwargs = {
            "ra":
                0.15,
            "dec":
                0.0,
            "z":
                2.1,
            "flux":
                np.ones(15),
            "ivar":
                np.ones(15) * 4,
            "lambda":
                np.array([
                    3610, 3610.4, 3650, 3650.4, 3670, 3670.4, 3680, 3680.4,
                    3700, 3700.4
                ]),
            "targetid":
                100000000,
        }
        with self.assertRaises(AstronomicalObjectError):
            DesiPk1dForest(**kwargs)

        # create forest with missing Forest variables
        kwargs = {
            "los_id": 9999,
            "ra": 0.15,
            "dec": 0.0,
            "z": 2.1,
            "ivar": np.ones(15) * 4,
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
        test_obj = DesiPk1dForest(**kwargs_desi_pk1d_forest)
        test_obj.rebin()

        # create a second DesiPk1dForest
        test_obj_other = DesiPk1dForest(**kwargs_desi_pk1d_forest2)
        test_obj_other.rebin()

        # coadd them
        test_obj.coadd(test_obj_other)
        test_obj.rebin()
        self.assert_forest_object(test_obj, kwargs_desi_pk1d_forest_coadd)

        # create a third DesiPk1dForest with different targetid
        kwargs = kwargs_desi_pk1d_forest2.copy()
        kwargs["targetid"] = 999
        test_obj_other = DesiPk1dForest(**kwargs)

        # coadding them whould raise an error
        with self.assertRaises(AstronomicalObjectError):
            test_obj.coadd(test_obj_other)

    def test_desi_pk1d_forest_get_data(self):
        """Test method get_data for DesiPk1dForest."""
        # set class variables
        setup_forest(wave_solution="lin")
        setup_pk1d_forest("LYA")

        # create a DesiPk1dForest
        test_obj = DesiPk1dForest(**kwargs_desi_pk1d_forest)
        test_obj.rebin()
        self.assert_get_data(test_obj)

    def test_desi_pk1d_forest_get_header(self):
        """Test method get_header for DesiPk1dForest."""
        # set class variables
        setup_forest(wave_solution="lin")
        setup_pk1d_forest("LYA")

        # create a DesiPk1dForest
        test_obj = DesiPk1dForest(**kwargs_desi_pk1d_forest)
        test_obj.rebin()
        self.assert_get_header(test_obj)

        # create a second DesiPk1dForest and coadd it to the first
        test_obj_other = DesiPk1dForest(**kwargs_desi_pk1d_forest2)
        test_obj_other.rebin()
        test_obj.coadd(test_obj_other)

        # get header and test
        self.assert_get_header(test_obj)

    def test_forest(self):
        """Test constructor for Forest object."""
        # create a Forest with missing Forest class variables
        with self.assertRaises(AstronomicalObjectError):
            Forest(**kwargs_forest_log)

        # set class variables; case: logarithmic wavelength solution
        setup_forest(wave_solution="log", rebin=3)

        # create a Forest
        test_obj = Forest(**kwargs_forest_log)
        self.assertTrue(isinstance(test_obj, Forest))
        self.assertTrue(isinstance(test_obj, AstronomicalObject))
        self.assert_forest_object(test_obj, kwargs_forest_log)

        # create a Forest specifying all variables
        kwargs = kwargs_forest_log.copy()
        kwargs.update({
            "continuum": np.ones(15),
            "deltas": np.zeros(15),
        })
        test_obj = Forest(**kwargs)
        self.assert_forest_object(test_obj, kwargs)

        # create a Forest with extra variables
        kwargs = kwargs_forest_log.copy()
        kwargs.update({
            "test_variable": "test",
        })
        test_obj = Forest(**kwargs)
        self.assert_forest_object(test_obj, kwargs)

        # create a Forest with missing AstronomicalObject variables
        kwargs = {
            "flux": np.ones(15),
            "ivar": np.ones(15) * 4,
        }
        with self.assertRaises(AstronomicalObjectError):
            Forest(**kwargs)

        # set class variables; case: linear wavelength solution
        reset_forest()
        setup_forest(wave_solution="lin")

        # create a Forest
        test_obj = Forest(**kwargs_forest_lin)
        self.assert_forest_object(test_obj, kwargs_forest_lin)

    def test_forest_comparison(self):
        """Test comparison is properly inheried in Forest."""
        setup_forest(wave_solution="log", rebin=3)

        test_obj = Forest(**kwargs_forest_log)
        test_obj.rebin()

        kwargs_forest_gt = kwargs_astronomical_object_gt.copy()
        for kwargs in kwargs_forest_gt.values():
            kwargs.update({
                "flux": FLUX,
                "ivar": IVAR,
                "log_lambda": LOG_LAMBDA,
            })

        for kwargs in kwargs_forest_gt.values():
            other = Forest(**kwargs)
            self.assertTrue(test_obj > other)
            self.assertTrue(test_obj != other)
            self.assertFalse(test_obj == other)
            self.assertFalse(test_obj < other)

        # equal objects (but with different los_id)
        # it makes sense that the los_id be different if we combine deltas
        # from different surveys
        kwargs = {
            "los_id": 1234,
            "ra": 0.15,
            "dec": 0.0,
            "z": 2.1,
            "flux": FLUX,
            "ivar": IVAR,
            "log_lambda": LOG_LAMBDA,
        }
        other = Forest(**kwargs)
        self.assertFalse(test_obj > other)
        self.assertTrue(test_obj == other)
        self.assertFalse(test_obj < other)

    def test_forest_rebin(self):
        """Test the rebin function in Forest."""
        # set class variables; case: logarithmic wavelength solution
        setup_forest(wave_solution="log", rebin=3)

        # create a Forest
        test_obj = Forest(**kwargs_forest_log)

        # rebin and test results
        test_obj.rebin()
        self.assert_forest_object(test_obj, kwargs_forest_log_rebin)

        # set class variables; case: linear wavelength solution
        reset_forest()
        setup_forest(wave_solution="lin")

        # create a Forest
        test_obj = Forest(**kwargs_forest_lin)

        # rebin and test results
        test_obj.rebin()
        self.assert_forest_object(test_obj, kwargs_forest_lin_rebin)

    def test_forest_coadd(self):
        """Test the coadd function in Forest."""
        # set class variables; case: logarithmic wavelength solution
        setup_forest(wave_solution="log", rebin=3)

        # create a Forest
        test_obj = Forest(**kwargs_forest_log)
        test_obj.rebin()

        # create a second SdssForest
        test_obj_other = Forest(**kwargs_forest_log2)
        test_obj_other.rebin()

        # coadd them
        test_obj.coadd(test_obj_other)
        self.assert_forest_object(test_obj, kwargs_forest_log_coadd)

        # create a third Forest with different los_id
        kwargs = kwargs_forest_log2.copy()
        kwargs["los_id"] = 999
        test_obj_other = Forest(**kwargs)
        test_obj_other.rebin()

        # coadding them whould raise an error
        with self.assertRaises(AstronomicalObjectError):
            test_obj.coadd(test_obj_other)

        # set class variables; case: linear wavelength solution
        reset_forest()
        setup_forest(wave_solution="lin")

        # create a Forest
        test_obj = Forest(**kwargs_forest_lin)
        test_obj.rebin()

        # create a second Forest
        test_obj_other = Forest(**kwargs_forest_lin2)
        test_obj_other.rebin()

        # coadd them
        test_obj.coadd(test_obj_other)
        self.assert_forest_object(test_obj, kwargs_forest_lin_coadd)

    def test_forest_get_data(self):
        """Test method get_data for Forest."""
        # set class variables; case: logarithmic wavelength solution
        setup_forest(wave_solution="log", rebin=3)

        # create a Forest
        test_obj = Forest(**kwargs_forest_log)
        test_obj.rebin()
        self.assert_get_data(test_obj)

        # set class variables; case: linear wavelength solution
        reset_forest()
        setup_forest(wave_solution="lin")

        # create a Forest
        test_obj = Forest(**kwargs_forest_lin)
        test_obj.rebin()
        self.assert_get_data(test_obj)

    def test_forest_get_header(self):
        """Test method get_header for Forest."""
        # set class variables; case: logarithmic wavelength solution
        setup_forest(wave_solution="log", rebin=3)

        # create a Forest
        test_obj = Forest(**kwargs_forest_log)
        test_obj.rebin()

        # get header and test
        self.assert_get_header(test_obj)

        # set class variables; case: linear wavelength solution
        reset_forest()
        setup_forest(wave_solution="lin")

        # create a Forest
        test_obj = Forest(**kwargs_forest_lin)
        test_obj.rebin()

        # get header and test
        self.assert_get_header(test_obj)

    def test_pk1d_forest(self):
        """Test constructor for Pk1dForest object."""
        # create a Pk1dForest with missing Forest class variables
        with self.assertRaises(AstronomicalObjectError):
            Pk1dForest(**kwargs_pk1d_forest_log)

        # set class variables; case: logarithmic wavelength solution
        setup_forest(wave_solution="log", rebin=3)

        # create a Pk1dForest with missing Pk1dForest variables
        with self.assertRaises(AstronomicalObjectError):
            Pk1dForest(**kwargs_pk1d_forest_log)

        # set class variables
        setup_pk1d_forest("LYA")

        # create a Pk1dForest
        test_obj = Pk1dForest(**kwargs_pk1d_forest_log)
        self.assertTrue(isinstance(test_obj, Pk1dForest))
        self.assertTrue(isinstance(test_obj, Forest))
        self.assert_forest_object(test_obj, kwargs_pk1d_forest_log)

        # set class variables; case: linear wavelength solution
        reset_forest()
        setup_forest(wave_solution="lin")
        setup_pk1d_forest("LYA")

        # create a Forest
        test_obj = Pk1dForest(**kwargs_pk1d_forest_lin)
        self.assertTrue(isinstance(test_obj, Pk1dForest))
        self.assertTrue(isinstance(test_obj, Forest))
        self.assert_forest_object(test_obj, kwargs_pk1d_forest_lin)

    def test_pk1d_forest_coadd(self):
        """Test the coadd function in Pk1dForest"""
        # set class variables; case: logarithmic wavelength solution
        setup_forest(wave_solution="log", rebin=3)
        setup_pk1d_forest("LYA")

        # create a Pk1dForest
        test_obj = Pk1dForest(**kwargs_pk1d_forest_log)
        test_obj.rebin()

        # create a second Pk1dForest
        test_obj_other = Pk1dForest(**kwargs_pk1d_forest_log2)
        test_obj_other.rebin()

        # coadd them
        test_obj.coadd(test_obj_other)
        self.assert_forest_object(test_obj, kwargs_pk1d_forest_log_coadd)

        # create a third Pk1dForest with different targetid
        kwargs = kwargs_pk1d_forest_log2.copy()
        kwargs["los_id"] = 999
        test_obj_other = Pk1dForest(**kwargs)
        test_obj_other.rebin()

        # coadding them should raise an error
        with self.assertRaises(AstronomicalObjectError):
            test_obj.coadd(test_obj_other)

        # set class variables; case: linear wavelength solution
        reset_forest()
        setup_forest(wave_solution="lin")
        setup_pk1d_forest("LYA")

        # create a Forest
        test_obj = Pk1dForest(**kwargs_pk1d_forest_lin)
        test_obj.rebin()

        # create a second Forest
        test_obj_other = Pk1dForest(**kwargs_pk1d_forest_lin2)
        test_obj_other.rebin()

        # coadd them
        test_obj.coadd(test_obj_other)
        self.assert_forest_object(test_obj, kwargs_pk1d_forest_lin_coadd)

    def test_pk1d_forest_get_data(self):
        """Test method get_data for Pk1dForest."""
        # set class variables; case: logarithmic wavelength solution
        setup_forest(wave_solution="log", rebin=3)
        setup_pk1d_forest("LYA")

        # create a Pk1dForest
        test_obj = Pk1dForest(**kwargs_pk1d_forest_log)
        test_obj.rebin()
        self.assert_get_data(test_obj)

        # set class variables; case: linear wavelength solution
        reset_forest()
        setup_forest(wave_solution="lin")
        setup_pk1d_forest("LYA")

        # create a Forest
        test_obj = Pk1dForest(**kwargs_pk1d_forest_lin)
        test_obj.rebin()
        self.assert_get_data(test_obj)

    def test_pk1d_forest_get_header(self):
        """Test method get_header for Pk1dForest."""
        # set class variables; case: logarithmic wavelength solution
        setup_forest(wave_solution="log", rebin=3)
        setup_pk1d_forest("LYA")

        # create a Pk1dForest
        test_obj = Pk1dForest(**kwargs_pk1d_forest_log)
        test_obj.rebin()

        # get header and test
        self.assert_get_header(test_obj)

        # set class variables; case: linear wavelength solution
        reset_forest()
        setup_forest(wave_solution="lin")
        setup_pk1d_forest("LYA")

        # create a Pk1dForest
        test_obj = Pk1dForest(**kwargs_pk1d_forest_lin)
        test_obj.rebin()

        # get header and test
        self.assert_get_header(test_obj)

    def test_sdss_forest(self):
        """Test constructor for SdssForest.
        This includes a test of function rebin.
        """
        # expected error as class variables are not yet set
        with self.assertRaises(AstronomicalObjectError):
            SdssForest(**kwargs_sdss_forest)

        # set class variables
        setup_forest(wave_solution="log", rebin=3)

        # create a SdssForest
        test_obj = SdssForest(**kwargs_sdss_forest)
        test_obj.rebin()
        self.assertTrue(isinstance(test_obj, SdssForest))
        self.assertTrue(isinstance(test_obj, Forest))
        self.assert_forest_object(test_obj, kwargs_sdss_forest_rebin)

        # create forest with extra variables
        kwargs = kwargs_sdss_forest.copy()
        kwargs.update({
            "test_variable": "test",
        })
        test_obj = SdssForest(**kwargs)
        test_obj.rebin()
        self.assert_forest_object(test_obj, kwargs_sdss_forest_rebin)

        # create a SdssForest with missing SdssForest variables
        kwargs = {
            "ra": 0.15,
            "dec": 0.0,
            "z": 2.1,
            "flux": np.ones(15),
            "ivar": np.ones(15) * 4,
        }
        with self.assertRaises(AstronomicalObjectError):
            SdssForest(**kwargs)

        # create forest with missing Forest variables
        kwargs = {
            "los_id": 9999,
            "ra": 0.15,
            "dec": 0.0,
            "z": 2.1,
            "ivar": np.ones(15) * 4,
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
        setup_forest(wave_solution="log", rebin=3)

        # create a SdssForest
        test_obj = SdssForest(**kwargs_sdss_forest)
        test_obj.rebin()

        # create a second SdssForest
        test_obj_other = SdssForest(**kwargs_sdss_forest2)
        test_obj_other.rebin()

        # coadd them
        test_obj.coadd(test_obj_other)
        self.assert_forest_object(test_obj, kwargs_sdss_forest_coadd)

        # create a third SdssForest with different targetid
        kwargs = kwargs_sdss_forest2.copy()
        kwargs["thingid"] = 999
        test_obj_other = SdssForest(**kwargs)
        test_obj_other.rebin()

        # coadding them should raise an error
        with self.assertRaises(AstronomicalObjectError):
            test_obj.coadd(test_obj_other)

    def test_sdss_forest_get_data(self):
        """Test method get_data for SdssForest."""
        # set class variables
        setup_forest(wave_solution="log", rebin=3)

        # create an SdssForest
        test_obj = SdssForest(**kwargs_sdss_forest)
        test_obj.rebin()
        self.assert_get_data(test_obj)

    def test_sdss_forest_get_header(self):
        """Test method get_header for SdssForest."""
        # set class variables
        setup_forest(wave_solution="log", rebin=3)

        # create an SdssForest
        test_obj = SdssForest(**kwargs_sdss_forest)
        test_obj.rebin()

        # get header and test
        self.assert_get_header(test_obj)

        # create a second SdssForest and coadd it to the first
        test_obj_other = SdssForest(**kwargs_sdss_forest2)
        test_obj_other.rebin()
        test_obj.coadd(test_obj_other)

        # get header and test
        self.assert_get_header(test_obj)

    def test_sdss_pk1d_forest(self):
        """Test constructor for SdssPk1dForest.
        This includes a test of function rebin.
        """
        # expected error as Pk1dForest class variables are not yet set
        with self.assertRaises(AstronomicalObjectError):
            SdssPk1dForest(**kwargs_sdss_pk1d_forest)

        # set class variables
        setup_pk1d_forest("LYA")

        # expected error as Forest class variables are not yet set
        with self.assertRaises(AstronomicalObjectError):
            SdssPk1dForest(**kwargs_sdss_pk1d_forest)

        # set class variables
        setup_forest(wave_solution="log", rebin=3)

        # create a SdssPk1dForest
        test_obj = SdssPk1dForest(**kwargs_sdss_pk1d_forest)
        test_obj.rebin()
        self.assertTrue(isinstance(test_obj, SdssPk1dForest))
        self.assertTrue(isinstance(test_obj, SdssForest))
        self.assertTrue(isinstance(test_obj, Pk1dForest))
        self.assert_forest_object(test_obj, kwargs_sdss_pk1d_forest_rebin)

        # create SdssPk1dForest with extra variables
        kwargs = kwargs_sdss_pk1d_forest.copy()
        kwargs.update({
            "test_variable": "test",
        })
        test_obj = SdssPk1dForest(**kwargs)
        test_obj.rebin()
        self.assert_forest_object(test_obj, kwargs_sdss_pk1d_forest_rebin)

        # create a SdssPk1dForest with missing SdssForest variables
        kwargs = {
            "ra":
                0.15,
            "dec":
                0.0,
            "z":
                2.1,
            "flux":
                np.ones(15),
            "ivar":
                np.ones(15) * 4,
            "log_lambda":
                np.array([
                    3.5565, 3.55655, 3.5567, 3.55675, 3.5569, 3.55695, 3.5571,
                    3.55715, 3.5573, 3.55735
                ]),
            "exposures_diff":
                np.ones(15),
            "reso":
                np.ones(15),
        }
        with self.assertRaises(AstronomicalObjectError):
            SdssPk1dForest(**kwargs)

        # create a SdssPk1dForest with missing Pk1dForest variables
        kwargs = {
            "ra": 0.15,
            "dec": 0.0,
            "z": 2.1,
            "flux": np.ones(15),
            "ivar": np.ones(15) * 4,
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
            "ivar": np.ones(15) * 4,
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
        setup_forest(wave_solution="log", rebin=3)
        setup_pk1d_forest("LYA")

        # create a SdssPk1dForest
        test_obj = SdssPk1dForest(**kwargs_sdss_pk1d_forest)
        test_obj.rebin()

        # create a second SdssPk1dForest
        test_obj_other = SdssPk1dForest(**kwargs_sdss_pk1d_forest2)
        test_obj_other.rebin()

        # coadd them
        test_obj.coadd(test_obj_other)
        self.assert_forest_object(test_obj, kwargs_sdss_pk1d_forest_coadd)

        # create a third SdssForest with different targetid
        kwargs = kwargs_sdss_pk1d_forest2.copy()
        kwargs["thingid"] = 999
        test_obj_other = SdssPk1dForest(**kwargs)
        test_obj_other.rebin()

        # coadding them should raise an error
        with self.assertRaises(AstronomicalObjectError):
            test_obj.coadd(test_obj_other)

    def test_sdss_pk1d_forest_get_data(self):
        """Test method get_data for SdssPk1dForest."""
        # set class variables
        setup_forest(wave_solution="log", rebin=3)
        setup_pk1d_forest("LYA")

        # create an SdssPk1dForest
        test_obj = SdssPk1dForest(**kwargs_sdss_pk1d_forest)
        self.assert_get_data(test_obj)

    def test_sdss_pk1d_forest_get_header(self):
        """Test method get_header for SdssPk1dForest."""
        # set class variables
        setup_forest(wave_solution="log", rebin=3)
        setup_pk1d_forest("LYA")

        # create an SdssPk1dForest
        test_obj = SdssPk1dForest(**kwargs_sdss_pk1d_forest)
        test_obj.rebin()

        # get header and test
        self.assert_get_header(test_obj)

        # create a second SdssPk1dForest and coadd it to the first
        test_obj_other = SdssPk1dForest(**kwargs_sdss_pk1d_forest2)
        test_obj_other.rebin()
        test_obj.coadd(test_obj_other)

        # get header and test
        self.assert_get_header(test_obj)


if __name__ == '__main__':
    unittest.main()
