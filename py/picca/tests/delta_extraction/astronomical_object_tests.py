"""This file contains tests related to AstronomicalObject and its childs"""
import unittest

import healpy
import numpy as np

from picca.delta_extraction.astronomical_object import AstronomicalObject
from picca.delta_extraction.astronomical_objects.desi_forest import DesiForest
from picca.delta_extraction.astronomical_objects.desi_pk1d_forest import DesiPk1dForest
from picca.delta_extraction.astronomical_objects.forest import Forest
from picca.delta_extraction.astronomical_objects.forest import defaults as defaults_forest
from picca.delta_extraction.astronomical_objects.pk1d_forest import Pk1dForest
from picca.delta_extraction.astronomical_objects.sdss_forest import SdssForest
from picca.delta_extraction.astronomical_objects.sdss_pk1d_forest import SdssPk1dForest
from picca.delta_extraction.errors import AstronomicalObjectError
from picca.tests.delta_extraction.abstract_test import AbstractTest
from picca.tests.delta_extraction.test_utils import (reset_forest, setup_forest,
                                                     setup_pk1d_forest)

def _isin_float(elements, test_elements):
    """ Helper function to find indices where 'test_elements'
    are numerically close 'elements'.

    Arguments
    ---------
    elements: array
    Source array

    test_elements: array
    Test elements to find in elements.

    Returns:
    mask: array of bool
    masking array such that
    np.allclose(elements[mask], test_elements) == True

    """
    mask = np.zeros(len(elements), dtype=bool)
    for test_e in test_elements:
        mask |= np.isclose(elements, test_e)
    return mask

# define auxiliar variables
LOG_LAMBDA = np.array([
    3.5562825, 3.5563225, 3.5565825, 3.5566225, 3.5568825, 3.5569225,
    3.5571825, 3.5572225, 3.5574825, 3.5575225
])
FILLED_LAMBDA_LOG_POINTS = np.array([3.5563025, 3.5566025, 3.5569025, 3.5572025, 3.5575025])

LOG_LAMBDA_LIN = np.log10(np.array([
3610.0, 3610.4, 3650.0, 3650.4, 3670.0, 3670.4, 3680.0, 3680.4, 3700.0, 3700.4
]))
FILLED_LAMBDA_LIN_POINTS = np.log10(np.array([3610, 3650, 3670, 3680, 3700]))

# others
SIZE = 10
RESO_NDIAGS = 7
forest_dtype = np.dtype([('flux','f8'), ('ivar','f8'),
    ('exposures_diff','f8'), ('reso','f8'),
    ('reso_pix','f8'), ('resolution_matrix','f8', RESO_NDIAGS)
])
BASE_FOREST = np.ones(SIZE, dtype=forest_dtype)

# Values for three types of spectra
SPECTRA_VALUES_DICT = {
    '1': {
        'flux': 1,
        'ivar': 4,
        'ivar_rebin': 8,
        'exposures_diff': 1,
        'reso': 1,
        'reso_pix': 1,
        'resolution_matrix': 1
    },
    '2': {
        'flux': 3,
        'ivar': 4,
        'ivar_rebin': 8,
        'exposures_diff': 3,
        'reso': 3,
        'reso_pix': 3,
        'resolution_matrix': 3
    },
    'COADD': {
        'flux': 2,
        'ivar': 16,
        'ivar_rebin': 16,
        'exposures_diff': 2,
        'reso': 2,
        'reso_pix': 2,
        'resolution_matrix': 2
    }
}

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

def get_kwargs_input(wave_solution, which_spectrum, is_p1d=False, is_desi=False):
    """ This function creates the sparse input spectrum. BASE_FOREST values are
    scaled with SPECTRA_VALUES_DICT for a given which_spectrum.

    Arguments
    ---------
    wave_solution: str
    lin or log

    which_spectrum: str
    It should be "1", "2" or "COADD".

    is_p1d: bool
    Adds P1D expected fields

    is_desi: bool
    Adds "resolution_matrix" to dictionary.

    Returns
    ---------
    rebin_kwargs_forest: dict
    A copy of kwargs_astronomical_object as base and truth updated.

    """
    base_kwargs_forest = kwargs_astronomical_object.copy()

    if wave_solution == "lin":
        in_wave = LOG_LAMBDA_LIN
    else:
        in_wave = LOG_LAMBDA

    spectrum_vals = SPECTRA_VALUES_DICT[which_spectrum]
    update_dict = { "log_lambda": in_wave }

    keys_to_update = ["flux", "ivar"]
    if is_p1d:
        keys_to_update += ["exposures_diff", "reso", "reso_pix"]
        if is_desi:
            keys_to_update += ["resolution_matrix"]

    for key in keys_to_update:
        update_dict[key] = BASE_FOREST[key]*spectrum_vals[key]
    if "resolution_matrix" in keys_to_update:
        update_dict["resolution_matrix"] = update_dict["resolution_matrix"].T

    base_kwargs_forest.update(update_dict)

    return base_kwargs_forest

def get_kwargs_rebin(wave_solution, which_spectrum, rebin=1, is_p1d=False, is_desi=False):
    """ This function creates the truth for rebinned spectrum. Masked pixels are present
    with flux, ivar and exposures_diff set to 0, whereas resolution related values are not.

    Arguments
    ---------
    wave_solution: str
    lin or log

    which_spectrum: str
    It should be "1", "2" or "COADD".

    rebin: int
    Rebinning factor

    is_p1d: bool
    Adds P1D expected fields

    is_desi: bool
    Adds "resolution_matrix" to dictionary.

    Returns
    ---------
    rebin_kwargs_forest: dict
    A copy of kwargs_astronomical_object as base and truth updated.

    """
    rebin_kwargs_forest = kwargs_astronomical_object.copy()

    if wave_solution == "lin":
        step = 1. * rebin
        rebin_wave = np.log10(np.arange(3610, 3700+step/2., step=step))
        filled_points = FILLED_LAMBDA_LIN_POINTS
    else:
        step = 1e-4 * rebin
        rebin_wave = np.arange(3.5563025, 3.5575025+step/2., step=step)
        filled_points = FILLED_LAMBDA_LOG_POINTS

    w_filled_points = _isin_float(rebin_wave, filled_points)
    rebin_forest_array = np.ones(rebin_wave.size, dtype=forest_dtype)
    rebin_forest_array['flux'][~w_filled_points] = 0
    rebin_forest_array['ivar'][~w_filled_points] = 0
    rebin_forest_array['exposures_diff'][~w_filled_points] = 0

    spectrum_vals = SPECTRA_VALUES_DICT[which_spectrum]
    update_dict = {
        "log_lambda": rebin_wave,
        "ivar":rebin_forest_array["ivar"]*spectrum_vals["ivar_rebin"]
    }

    keys_to_update = ["flux"]
    if is_p1d:
        keys_to_update += ["exposures_diff", "reso", "reso_pix"]
        if is_desi:
            keys_to_update += ["resolution_matrix"]

    for key in keys_to_update:
        update_dict[key] = rebin_forest_array[key]*spectrum_vals[key]

    # Fix the shape for picca expected
    if "resolution_matrix" in keys_to_update:
        update_dict["resolution_matrix"] = update_dict["resolution_matrix"].T
    rebin_kwargs_forest.update(update_dict)

    return rebin_kwargs_forest

# define contructors for DesiForest
def get_desi_kwargs_input(wave_solution, which_spectrum, is_p1d=False):
    """ This function creates the sparse input spectrum for DESI.
    Also includes targetid, night, petal and tile information.

    Arguments
    ---------
    wave_solution: str
    lin or log

    which_spectrum: str
    It should be "1" or "2". Otherwise returns None

    is_p1d: bool
    Adds P1D expected fields

    Returns
    ---------
    kwargs_desi_forest: dict
    A copy of kwargs_astronomical_object as base and truth updated.

    """
    kwargs_desi_forest = get_kwargs_input(wave_solution, which_spectrum,
        is_p1d=is_p1d, is_desi=True)
    del kwargs_desi_forest["los_id"]
    if which_spectrum == "1":
        kwargs_desi_forest.update({
            "targetid": TARGETID,
            "night": 0,
            "petal": 0,
            "tile": 0,
        })
    elif which_spectrum == "2":
        kwargs_desi_forest.update({
            "targetid": TARGETID,
            "night": 1,
            "petal": 2,
            "tile": 3,
        })
    else:
        return None

    return kwargs_desi_forest

def get_desi_kwargs_rebin(wave_solution, which_spectrum, rebin=1, is_p1d=False):
    """ This function creates the rebinned spectrum for DESI.
    Also includes targetid, night, petal and tile information.

    Arguments
    ---------
    wave_solution: str
    lin or log

    which_spectrum: str
    Pass "COADD" for coadded tests. Otherwise returns rebin values
    for both "1" and "2" spectra.

    rebin: int
    Rebinning factor.

    is_p1d: bool
    Adds P1D expected fields

    Returns
    ---------
    kwargs_desi_forest: dict
    A copy of kwargs_astronomical_object as base and truth updated.

    """
    kwargs_desi_forest = get_kwargs_rebin(wave_solution, which_spectrum,
        rebin=rebin, is_p1d=is_p1d, is_desi=True)
    del kwargs_desi_forest["los_id"]
    if which_spectrum == "COADD":
        kwargs_desi_forest.update({
            "targetid": TARGETID,
            "night": [0, 1],
            "petal": [0, 2],
            "tile": [0, 3],
        })
    else:
        kwargs_desi_forest.update({
            "targetid": TARGETID,
            "night": [0],
            "petal": [0],
            "tile": [0],
        })

    kwargs_desi_forest["los_id"] = TARGETID

    return kwargs_desi_forest

# define contructors for SdssForest
def get_sdss_kwargs_input(wave_solution, which_spectrum, is_p1d=False):
    """ This function creates the sparse input spectrum for SDSS.
    Also includes thingid, plate, fiberid and mjd information.

    Arguments
    ---------
    wave_solution: str
    lin or log

    which_spectrum: str
    It should be "1" or "2". Otherwise returns None

    is_p1d: bool
    Adds P1D expected fields

    Returns
    ---------
    kwargs_sdss_forest: dict
    A copy of kwargs_astronomical_object as base and truth updated.

    """
    kwargs_sdss_forest = get_kwargs_input(wave_solution, which_spectrum, is_p1d=is_p1d)
    del kwargs_sdss_forest["los_id"]

    if which_spectrum == "1":
        kwargs_sdss_forest.update({
        "thingid": THINGID,
        "plate": 0,
        "fiberid": 0,
        "mjd": 0,
    })
    elif which_spectrum == "2":
        kwargs_sdss_forest.update({
        "thingid": THINGID,
        "plate": 1,
        "fiberid": 2,
        "mjd": 3,
    })
    else:
        return None

    return kwargs_sdss_forest

def get_sdss_kwargs_rebin(wave_solution, which_spectrum, rebin=3, is_p1d=False):
    """ This function creates the rebinned spectrum for SDSS.
    Also includes thingid, plate, fiberid and mjd information.

    Arguments
    ---------
    wave_solution: str
    lin or log

    which_spectrum: str
    Pass "COADD" for coadded tests. Otherwise returns rebin values
    for both "1" and "2" spectra.

    rebin: int
    Rebinning factor.

    is_p1d: bool
    Adds P1D expected fields

    Returns
    ---------
    kwargs_sdss_forest_rebin: dict
    A copy of kwargs_astronomical_object as base and truth updated.

    """
    kwargs_sdss_forest_rebin = get_kwargs_rebin(wave_solution, which_spectrum,
        rebin=rebin, is_p1d=is_p1d, is_desi=False)

    if which_spectrum == "COADD":
        kwargs_sdss_forest_rebin.update({
            "thingid": THINGID,
            "plate": [0, 1],
            "fiberid": [0, 2],
            "mjd": [0, 3]
        })
    else:
        kwargs_sdss_forest_rebin.update({
            "thingid": THINGID,
            "plate": [0],
            "fiberid": [0],
            "mjd": [0],
        })

    kwargs_sdss_forest_rebin["los_id"] = THINGID

    return kwargs_sdss_forest_rebin

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
        if len(test_obj.log_lambda) != len(kwargs.get("log_lambda")):
            print()
            print(test_obj.log_lambda)
            print(kwargs.get("log_lambda"))
        self.assertTrue(
            np.allclose(test_obj.log_lambda, kwargs.get("log_lambda")))
        flux = kwargs.get("flux")
        ivar = kwargs.get("ivar")
        self.assertTrue(np.allclose(test_obj.flux, flux))
        self.assertTrue(np.allclose(test_obj.ivar, ivar))

        if isinstance(test_obj, DesiPk1dForest):
            self.assertTrue(len(Forest.mask_fields) == 8)
        elif isinstance(test_obj, Pk1dForest):
            self.assertTrue(len(Forest.mask_fields) == 7)
        else:
            self.assertTrue(len(Forest.mask_fields) == 4)
        self.assertTrue(Forest.mask_fields[0] == "flux")
        self.assertTrue(Forest.mask_fields[1] == "ivar")
        self.assertTrue(Forest.mask_fields[2] == "transmission_correction")
        self.assertTrue(Forest.mask_fields[3] == "log_lambda")
        true_transmission_correction = np.where(ivar>0, 1, 0)
        self.assertTrue(np.allclose(test_obj.transmission_correction,
            true_transmission_correction))
        mean_snr = np.sum(flux * np.sqrt(ivar))/np.sum(ivar>0)
        self.assertTrue(np.isclose(test_obj.mean_snr, mean_snr))

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
            self.assertTrue(np.allclose(test_obj.reso_pix, kwargs.get("reso_pix")))

            log_lambda = kwargs.get("log_lambda")
            mean_z = ((np.power(10., log_lambda[len(log_lambda) - 1]) +
                       np.power(10., log_lambda[0])) / 2. /
                      Pk1dForest.lambda_abs_igm - 1.0)
            if not np.isclose(test_obj.mean_z, mean_z):
                print(test_obj.mean_z, mean_z)
                print(log_lambda, test_obj.log_lambda,
                      log_lambda == test_obj.log_lambda)
            self.assertTrue(np.isclose(test_obj.mean_z, mean_z))
            self.assertTrue(
                np.isclose(test_obj.mean_reso,
                           kwargs.get("reso").mean()))

    def assert_get_data(self, test_obj, blinding=False):
        """Assert the correct properties of the return of method get_data

        Arguments
        ---------
        test_obj: Forest
        The Forest instance to check

        blinding: bool - Default: False
        If True, get the data format assuming a blinding strategy is engaged
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
            self.assertTrue(np.allclose(cols[0], 10**test_obj.log_lambda))
            self.assertTrue(units[0] == "Angstrom")
            self.assertTrue(comments[0] == "Lambda")

        if test_obj.deltas is None:
            deltas = np.zeros_like(test_obj.flux)
        else:
            deltas = test_obj.deltas
        if blinding:
            self.assertTrue(names[1] == "DELTA_BLIND")
        else:
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
            self.assertTrue(header[index + 3].get("name") == "WAVE_SOLUTION")
            if Forest.wave_solution == "log":
                self.assertTrue(header[index + 3].get("value") == "log")
                self.assertTrue(header[index + 4].get("name") == "DELTA_LOG_LAMBDA")
                self.assertTrue(np.isclose(header[index + 4].get("value"), 3e-4))
            elif Forest.wave_solution == "lin":
                self.assertTrue(header[index + 3].get("value") == "lin")
                self.assertTrue(header[index + 4].get("name") == "DELTA_LAMBDA")
                self.assertTrue(np.isclose(header[index + 4].get("value"), 1.0))
            else:
                print(f"Forest.wave_solution={Forest.wave_solution}, expected "
                      "'log' or 'lin'")
                self.assertTrue(False)
            index += 4
        if isinstance(test_obj, Pk1dForest):
            self.assertTrue(header[index + 1].get("name") == "MEANZ")
            self.assertTrue(header[index + 1].get("value") == test_obj.mean_z)
            self.assertTrue(header[index + 2].get("name") == "MEANRESO")
            self.assertTrue(header[index +
                                   2].get("value") == test_obj.mean_reso)
            self.assertTrue(header[index + 3].get("name") == "MEANRESO_PIX")
            self.assertTrue(header[index +
                                   3].get("value") == test_obj.mean_reso_pix)

            index += 3
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

    def test_astronomical_object(self):
        """Test constructor for AstronomicalObject."""
        test_obj = AstronomicalObject(**kwargs_astronomical_object)
        self.assert_astronomical_object(test_obj, kwargs_astronomical_object)

    def test_astronomical_object_missing_variables(self):
        """Test constructor errors for AstronomicalObject."""
        kwargs = {}

        # missing dec
        expected_message = (
            "Error constructing AstronomicalObject. Missing variable 'dec'"
        )
        with self.assertRaises(AstronomicalObjectError) as context_manager:
            test_obj = AstronomicalObject(**kwargs)
        self.compare_error_message(context_manager, expected_message)

        # missing los_id
        kwargs["dec"] = 0.0
        expected_message = (
            "Error constructing AstronomicalObject. Missing variable 'los_id'"
        )
        with self.assertRaises(AstronomicalObjectError) as context_manager:
            test_obj = AstronomicalObject(**kwargs)
        self.compare_error_message(context_manager, expected_message)

        # missing ra
        kwargs["los_id"] = 1234
        expected_message = (
            "Error constructing AstronomicalObject. Missing variable 'ra'"
        )
        with self.assertRaises(AstronomicalObjectError) as context_manager:
            test_obj = AstronomicalObject(**kwargs)
        self.compare_error_message(context_manager, expected_message)

        # missing z
        kwargs["ra"] = 0.0
        expected_message = (
            "Error constructing AstronomicalObject. Missing variable 'z'"
        )
        with self.assertRaises(AstronomicalObjectError) as context_manager:
            test_obj = AstronomicalObject(**kwargs)
        self.compare_error_message(context_manager, expected_message)

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
        kwargs_desi_forest = get_desi_kwargs_input("lin", "1")

        # expected error as class variables are not yet set
        expected_message = (
            "Error constructing Forest. Class variable 'log_lambda_grid' must "
            "be set prior to initialize instances of this type. This probably "
            "means you did not run Forest.set_class_variables"
        )
        with self.assertRaises(AstronomicalObjectError) as context_manager:
            DesiForest(**kwargs_desi_forest)
        self.compare_error_message(context_manager, expected_message)

        # set Forest class variables
        setup_forest(wave_solution="lin")
        kwargs_desi_forest_rebin = get_desi_kwargs_rebin("lin", "1")

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
        expected_message = (
            "Error constructing DesiForest. Missing variable 'targetid'"
        )
        with self.assertRaises(AstronomicalObjectError) as context_manager:
            DesiForest(**kwargs)
        self.compare_error_message(context_manager, expected_message)

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
        expected_message = (
            "Error constructing Forest. Missing variable 'log_lambda'"
        )
        with self.assertRaises(AstronomicalObjectError) as context_manager:
            DesiForest(**kwargs)
        self.compare_error_message(context_manager, expected_message)

    def test_desi_forest_coadd(self):
        """Test the coadd function in DesiForest"""
        # set class variables
        setup_forest(wave_solution="lin")
        kwargs_desi_forest  = get_desi_kwargs_input("lin", "1")
        kwargs_desi_forest2 = get_desi_kwargs_input("lin", "2")
        kwargs_desi_forest_coadd = get_desi_kwargs_rebin("lin", "COADD")

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
        expected_message = (
            "Attempting to coadd two Forests "
            "with different los_id. This should "
            f"not happen. this.los_id={test_obj.los_id}, "
            f"other.los_id={test_obj_other.los_id}."
        )
        with self.assertRaises(AstronomicalObjectError) as context_manager:
            test_obj.coadd(test_obj_other)
        self.compare_error_message(context_manager, expected_message)

        # create a Forest object
        kwargs = kwargs_desi_forest2.copy()
        kwargs["los_id"] = 999
        test_obj_other = Forest(**kwargs)
        test_obj_other.rebin()

        # coadding them should raise an error
        expected_message = (
            "Error coadding DesiForest. Expected DesiForest instance in other. "
            "Found: Forest"
        )
        with self.assertRaises(AstronomicalObjectError) as context_manager:
            test_obj.coadd(test_obj_other)
        self.compare_error_message(context_manager, expected_message)

    def test_desi_forest_get_data(self):
        """Test method get_data for DesiForest."""
        # set class variables
        setup_forest(wave_solution="lin")
        kwargs_desi_forest  = get_desi_kwargs_input("lin", "1")

        # create a DesiForest
        test_obj = DesiForest(**kwargs_desi_forest)
        test_obj.rebin()

        self.assert_get_data(test_obj)

    def test_desi_forest_get_header(self):
        """Test method get_header for DesiForest."""
        # set class variables
        setup_forest(wave_solution="lin")
        kwargs_desi_forest  = get_desi_kwargs_input("lin", "1")
        kwargs_desi_forest2 = get_desi_kwargs_input("lin", "2")

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
        kwargs_desi_pk1d_forest = get_desi_kwargs_input("lin", "1", is_p1d=True)
        kwargs_desi_pk1d_forest_rebin = get_desi_kwargs_rebin("lin", "1", is_p1d=True)
        # create a DesiPk1dForest class variables are not yet set
        expected_message = (
            "Error constructing Pk1dForest. Class variable 'lambda_abs_igm' "
            "must be set prior to initialize instances of this type"
        )
        with self.assertRaises(AstronomicalObjectError) as context_manager:
            DesiPk1dForest(**kwargs_desi_pk1d_forest)
        self.compare_error_message(context_manager, expected_message)

        # set class variables
        setup_pk1d_forest("LYA")

        # expected error as Forest class variables are not yet set
        expected_message = (
            "Error constructing Forest. Class variable 'log_lambda_grid' must "
            "be set prior to initialize instances of this type. This probably "
            "means you did not run Forest.set_class_variables"
        )
        with self.assertRaises(AstronomicalObjectError) as context_manager:
            DesiPk1dForest(**kwargs_desi_pk1d_forest)
        self.compare_error_message(context_manager, expected_message)

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
        kwargs = kwargs_desi_pk1d_forest.copy()
        del kwargs["night"], kwargs["petal"], kwargs["tile"]
        test_obj = DesiPk1dForest(**kwargs)
        test_obj.rebin()

        kwargs = kwargs_desi_pk1d_forest_rebin.copy()
        kwargs["night"] = []
        kwargs["petal"] = []
        kwargs["tile"] = []
        self.assert_forest_object(test_obj, kwargs)

        # create a DesiForest with missing DesiPk1dForest variables
        kwargs = {
            "ra": 0.15,
            "dec": 0.0,
            "z": 2.1,
            "flux": np.ones(15),
            "ivar": np.ones(15) * 4,
            "lambda":np.array([
                    3610, 3610.4, 3650, 3650.4, 3670, 3670.4, 3680, 3680.4,
                    3700, 3700.4 ]),
            "targetid": 100000000,
            "reso": np.ones(10),
            "reso_pix": np.ones(10),
        }
        expected_message = (
            "Error constructing DesiPk1dForest. Missing variable "
            "'resolution_matrix'"
        )
        with self.assertRaises(AstronomicalObjectError) as context_manager:
            DesiPk1dForest(**kwargs)
        self.compare_error_message(context_manager, expected_message)

        # create a DesiPk1dForest with missing DesiForest variables
        kwargs = {
            "ra": 0.15,
            "dec": 0.0,
            "z": 2.1,
            "flux": np.ones(15),
            "ivar": np.ones(15) * 4,
            "lambda": np.array([
                    3610, 3610.4, 3650, 3650.4, 3670, 3670.4, 3680, 3680.4,
                    3700, 3700.4]),
            "exposures_diff": np.ones(10),
            "reso": np.ones(10),
            "reso_pix": np.ones(10),
            "resolution_matrix": np.ones([7, 10])
        }
        expected_message = (
            "Error constructing DesiForest. Missing variable 'targetid'"
        )
        with self.assertRaises(AstronomicalObjectError) as context_manager:
            DesiPk1dForest(**kwargs)
        self.compare_error_message(context_manager, expected_message)

        # create forest with missing Pk1dForest variables
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
            "reso": np.ones(10),
            "reso_pix": np.ones(15),
            "resolution_matrix": np.ones([7, 10])
        }
        expected_message = (
            "Error constructing Pk1dForest. Missing variable 'exposures_diff'"
        )
        with self.assertRaises(AstronomicalObjectError) as context_manager:
            DesiPk1dForest(**kwargs)
        self.compare_error_message(context_manager, expected_message)

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
            "reso_pix": np.ones(15),
            "resolution_matrix": np.ones([7, 10])
        }
        expected_message = (
            "Error constructing Forest. Missing variable 'log_lambda'"
        )
        with self.assertRaises(AstronomicalObjectError) as context_manager:
            DesiPk1dForest(**kwargs)
        self.compare_error_message(context_manager, expected_message)

    def test_desi_pk1d_forest_coadd(self):
        """Test the coadd function in DesiPk1d_Forest"""
        # set class variables
        setup_forest(wave_solution="lin")
        setup_pk1d_forest("LYA")

        kwargs_desi_pk1d_forest = get_desi_kwargs_input("lin", "1", is_p1d=True)
        kwargs_desi_pk1d_forest2 = get_desi_kwargs_input("lin", "2", is_p1d=True)
        kwargs_desi_pk1d_forest_coadd = get_desi_kwargs_rebin("lin", "COADD", is_p1d=True)

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
        expected_message = (
            "Attempting to coadd two Forests "
            "with different los_id. This should "
            f"not happen. this.los_id={test_obj.los_id}, "
            f"other.los_id={test_obj_other.los_id}."
        )
        with self.assertRaises(AstronomicalObjectError) as context_manager:
            test_obj.coadd(test_obj_other)
        self.compare_error_message(context_manager, expected_message)

        # create a Forest object
        kwargs = kwargs_desi_pk1d_forest.copy()
        kwargs["los_id"] = 999
        test_obj_other = Forest(**kwargs)
        test_obj_other.rebin()

        # coadding them should raise an error
        expected_message = (
            "Error coadding DesiPk1dForest. Expected DesiPk1dForest instance in other. "
            "Found: Forest"
        )
        with self.assertRaises(AstronomicalObjectError) as context_manager:
            test_obj.coadd(test_obj_other)
        self.compare_error_message(context_manager, expected_message)

    def test_desi_pk1d_forest_consistency_check(self):
        """Test method consistency_check from DesiPk1dForest"""
        setup_forest("log")
        setup_pk1d_forest("LYA")
        kwargs_desi_pk1d_forest = get_desi_kwargs_input("lin", "1", is_p1d=True)

        # create a DesiPk1dForest with flux and resolution_matrix with
        # incompatible sizes
        kwargs = kwargs_desi_pk1d_forest.copy()
        kwargs["resolution_matrix"] = kwargs["resolution_matrix"][:,::2]
        expected_message = (
            "Error constructing DesiPk1dForest. 'resolution_matrix' and 'flux' "
            "don't have the same size"
        )
        with self.assertRaises(AstronomicalObjectError) as context_manager:
            DesiPk1dForest(**kwargs)
        self.compare_error_message(context_manager, expected_message)

    def test_desi_pk1d_forest_get_data(self):
        """Test method get_data for DesiPk1dForest."""
        # set class variables
        setup_forest(wave_solution="lin")
        setup_pk1d_forest("LYA")
        kwargs_desi_pk1d_forest = get_desi_kwargs_input("lin", "1", is_p1d=True)

        # create a DesiPk1dForest
        test_obj = DesiPk1dForest(**kwargs_desi_pk1d_forest)
        test_obj.rebin()
        self.assert_get_data(test_obj)

    def test_desi_pk1d_forest_get_header(self):
        """Test method get_header for DesiPk1dForest."""
        # set class variables
        setup_forest(wave_solution="lin")
        setup_pk1d_forest("LYA")
        kwargs_desi_pk1d_forest = get_desi_kwargs_input("lin", "1", is_p1d=True)
        kwargs_desi_pk1d_forest2 = get_desi_kwargs_input("lin", "2", is_p1d=True)

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
        kwargs_forest_log = get_kwargs_input("log", "1")
        kwargs_forest_lin = get_kwargs_input("lin", "1")

        # create a Forest with missing Forest class variables
        expected_message = (
            "Error constructing Forest. Class variable 'log_lambda_grid' must "
            "be set prior to initialize instances of this type. This probably "
            "means you did not run Forest.set_class_variables"
        )
        with self.assertRaises(AstronomicalObjectError) as context_manager:
            Forest(**kwargs_forest_log)
        self.compare_error_message(context_manager, expected_message)

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
        kwargs.update({ "test_variable": "test" })
        test_obj = Forest(**kwargs)
        self.assert_forest_object(test_obj, kwargs)

        # create a Forest with missing log_lambda
        kwargs = {}
        expected_message = (
            "Error constructing Forest. Missing variable 'log_lambda'"
        )
        with self.assertRaises(AstronomicalObjectError) as context_manager:
            Forest(**kwargs)
        self.compare_error_message(context_manager, expected_message)

        # create a Forest with missing flux
        kwargs = { "log_lambda": np.ones(15) }
        expected_message = (
            "Error constructing Forest. Missing variable 'flux'"
        )
        with self.assertRaises(AstronomicalObjectError) as context_manager:
            Forest(**kwargs)
        self.compare_error_message(context_manager, expected_message)

        # create a Forest with missing ivar
        kwargs = {
            "log_lambda": np.ones(15),
            "flux": np.ones(15),
        }
        expected_message = (
            "Error constructing Forest. Missing variable 'ivar'"
        )
        with self.assertRaises(AstronomicalObjectError) as context_manager:
            Forest(**kwargs)
        self.compare_error_message(context_manager, expected_message)

        # create a Forest with missing AstronomicalObject variables
        kwargs = {
            "log_lambda": np.ones(15),
            "flux": np.ones(15),
            "ivar": np.ones(15) * 4,
        }
        expected_message = (
            "Error constructing AstronomicalObject. Missing variable 'dec'"
        )
        with self.assertRaises(AstronomicalObjectError) as context_manager:
            Forest(**kwargs)
        self.compare_error_message(context_manager, expected_message)

        # create a Forest with weights defined
        kwargs = kwargs_forest_log.copy()
        kwargs.update({
            "log_lambda": np.ones(15),
            "flux": np.ones(15),
            "ivar": np.ones(15) * 4,
            "weights": np.ones(15) * 4,
        })
        Forest(**kwargs)

        # set class variables; case: linear wavelength solution
        reset_forest()
        setup_forest(wave_solution="lin")

        # create a Forest
        test_obj = Forest(**kwargs_forest_lin)
        self.assert_forest_object(test_obj, kwargs_forest_lin)

    def test_forest_class_variable_check(self):
        """Test class method class_variable_check from Forest"""
        kwargs_forest_log = get_kwargs_input("log", "1")
        # create a Forest with missing Forest.log_lambda_grid
        expected_message = (
            "Error constructing Forest. Class variable 'log_lambda_grid' must "
            "be set prior to initialize instances of this type. This probably "
            "means you did not run Forest.set_class_variables"
        )
        with self.assertRaises(AstronomicalObjectError) as context_manager:
            Forest(**kwargs_forest_log)
        self.compare_error_message(context_manager, expected_message)

        # create a Forest with missing Forest.log_lambda_rest_frame_grid
        Forest.log_lambda_grid = np.ones(15)
        expected_message = (
            "Error constructing Forest. Class variable "
            "'log_lambda_rest_frame_grid' must be set prior to initialize "
            "instances of this type. This probably means you did not run "
            "Forest.set_class_variables"
        )
        with self.assertRaises(AstronomicalObjectError) as context_manager:
            Forest(**kwargs_forest_log)
        self.compare_error_message(context_manager, expected_message)

        # create a Forest with missing Forest.mask_fields
        Forest.log_lambda_rest_frame_grid = np.ones(15)
        expected_message = (
            "Error constructing Forest. Class variable "
            "'mask_fields' must be set prior to initialize "
            "instances of this type. This probably means you did not run "
            "Forest.set_class_variables"
        )
        with self.assertRaises(AstronomicalObjectError) as context_manager:
            Forest(**kwargs_forest_log)
        self.compare_error_message(context_manager, expected_message)

        # create a Forest with incorrect Forest.mask_fields
        Forest.mask_fields = "flux"
        expected_message = (
            "Error constructing Forest. "
            "Expected list in class variable 'mask fields'. "
            "Found 'flux'."
        )
        with self.assertRaises(AstronomicalObjectError) as context_manager:
            Forest(**kwargs_forest_log)
        self.compare_error_message(context_manager, expected_message)

        # create a Forest with missing Forest.wave_solution
        Forest.mask_fields = ["flux"]
        expected_message = (
            "Error constructing Forest. Class variable "
            "'wave_solution' must be set prior to initialize "
            "instances of this type. This probably means you did not run "
            "Forest.set_class_variables"
        )
        with self.assertRaises(AstronomicalObjectError) as context_manager:
            Forest(**kwargs_forest_log)
        self.compare_error_message(context_manager, expected_message)

    def test_forest_comparison(self):
        """Test comparison is properly inheried in Forest."""
        setup_forest(wave_solution="log", rebin=3)
        kwargs_forest_log = get_kwargs_input("log", "1")

        test_obj = Forest(**kwargs_forest_log)
        test_obj.rebin()

        kwargs_forest_gt = kwargs_astronomical_object_gt.copy()
        for kwargs in kwargs_forest_gt.values():
            kwargs.update({
                "flux": BASE_FOREST['flux']*SPECTRA_VALUES_DICT["1"]['flux'],
                "ivar": BASE_FOREST['ivar']*SPECTRA_VALUES_DICT["1"]['ivar'],
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
            "flux": BASE_FOREST['flux']*SPECTRA_VALUES_DICT["1"]['flux'],
            "ivar": BASE_FOREST['ivar']*SPECTRA_VALUES_DICT["1"]['ivar'],
            "log_lambda": LOG_LAMBDA,
        }
        other = Forest(**kwargs)
        self.assertFalse(test_obj > other)
        self.assertTrue(test_obj == other)
        self.assertFalse(test_obj < other)

    def test_forest_consistency_check(self):
        """Test method consistency_check from Forest"""
        setup_forest("log")
        kwargs_forest_log = get_kwargs_input("log", "1")

        # create a Forest with flux and ivar of different sizes
        kwargs = kwargs_forest_log.copy()
        kwargs["ivar"] = kwargs["ivar"][::2]
        expected_message = (
            "Error constructing Forest. 'flux' and 'ivar' don't have the "
            "same size"
        )
        with self.assertRaises(AstronomicalObjectError) as context_manager:
            Forest(**kwargs)
        self.compare_error_message(context_manager, expected_message)

        # create a Forest with log_lambda and flux of different sizes
        kwargs = kwargs_forest_log.copy()
        kwargs["flux"] = kwargs["flux"][::2]
        kwargs["ivar"] = kwargs["ivar"][::2]
        expected_message = (
            "Error constructing Forest. 'flux' and 'log_lambda' don't have the "
            "same size"
        )
        with self.assertRaises(AstronomicalObjectError) as context_manager:
            Forest(**kwargs)
        self.compare_error_message(context_manager, expected_message)

    def test_forest_rebin(self):
        """Test the rebin function in Forest."""
        # set class variables; case: logarithmic wavelength solution
        setup_forest(wave_solution="log", rebin=3)
        kwargs_forest_log = get_kwargs_input("log", "1")
        kwargs_forest_log_rebin = get_kwargs_rebin("log", "1", rebin=3)

        # create a Forest, rebin and test results
        test_obj = Forest(**kwargs_forest_log)
        test_obj.rebin()
        self.assert_forest_object(test_obj, kwargs_forest_log_rebin)

        # set class variables; case: linear wavelength solution
        reset_forest()
        setup_forest(wave_solution="lin")

        kwargs_forest_lin = get_kwargs_input("lin", "1")
        kwargs_forest_lin_rebin = get_kwargs_rebin("lin", "1", rebin=1)

        # create a Forest
        test_obj = Forest(**kwargs_forest_lin)

        # rebin and test results
        test_obj.rebin()
        self.assert_forest_object(test_obj, kwargs_forest_lin_rebin)

        # if Forest.wave_solution is reset this should cause an error
        reset_forest()
        expected_message = (
            "Error in Forest.rebin(). Class variable 'wave_solution' "
            "must be either 'lin' or 'log'."
        )
        with self.assertRaises(AstronomicalObjectError) as context_manager:
            test_obj.rebin()
        self.compare_error_message(context_manager, expected_message)

    def test_forest_coadd(self):
        """Test the coadd function in Forest."""
        # set class variables; case: logarithmic wavelength solution
        setup_forest(wave_solution="log", rebin=3)
        kwargs_forest_log  = get_kwargs_input("log", "1")
        kwargs_forest_log2 = get_kwargs_input("log", "2")
        kwargs_forest_log_coadd = get_kwargs_rebin("log", "COADD", rebin=3)

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
        expected_message = (
            "Attempting to coadd two Forests "
            "with different los_id. This should "
            f"not happen. this.los_id={test_obj.los_id}, "
            f"other.los_id={test_obj_other.los_id}."
        )
        with self.assertRaises(AstronomicalObjectError) as context_manager:
            test_obj.coadd(test_obj_other)
        self.compare_error_message(context_manager, expected_message)

        # create an AstronomicalObject
        kwargs = kwargs_forest_log2.copy()
        test_obj_other = AstronomicalObject(**kwargs)

        # coadding them whould raise an error
        expected_message = (
            "Error coadding Forest. Expected Forest instance in other. Found: "
            "AstronomicalObject"
        )
        with self.assertRaises(AstronomicalObjectError) as context_manager:
            test_obj.coadd(test_obj_other)
        self.compare_error_message(context_manager, expected_message)

        # set class variables; case: linear wavelength solution
        reset_forest()
        setup_forest(wave_solution="lin")

        kwargs_forest_lin  = get_kwargs_input("lin", "1")
        kwargs_forest_lin2 = get_kwargs_input("lin", "2")
        kwargs_forest_lin_coadd = get_kwargs_rebin("lin", "COADD", rebin=1)

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
        kwargs_forest_log  = get_kwargs_input("log", "1")

        # create a Forest
        test_obj = Forest(**kwargs_forest_log)
        test_obj.rebin()
        self.assert_get_data(test_obj)

        # set class variables; case: linear wavelength solution
        reset_forest()
        setup_forest(wave_solution="lin")
        kwargs_forest_lin  = get_kwargs_input("lin", "1")

        # create a Forest
        test_obj = Forest(**kwargs_forest_lin)
        test_obj.rebin()
        self.assert_get_data(test_obj)

        # check data format when blinding is engaged
        Forest.blinding = "corr_yshift"
        self.assert_get_data(test_obj, blinding=True)
        # restore blinding to the default value for future runs
        Forest.blinding = "none"

        # if Forest.wave_solution is reset this should cause an error
        reset_forest()
        expected_message = (
            "Error in Forest.get_data(). Class variable 'wave_solution' "
            "must be either 'lin' or 'log'. Found: 'None'"
        )
        with self.assertRaises(AstronomicalObjectError) as context_manager:
            test_obj.get_data()
        self.compare_error_message(context_manager, expected_message)

    def test_forest_get_header(self):
        """Test method get_header for Forest."""
        # set class variables; case: logarithmic wavelength solution
        setup_forest(wave_solution="log", rebin=3)
        kwargs_forest_log  = get_kwargs_input("log", "1")

        # create a Forest
        test_obj = Forest(**kwargs_forest_log)
        test_obj.rebin()

        # get header and test
        self.assert_get_header(test_obj)

        # set class variables; case: linear wavelength solution
        reset_forest()
        setup_forest(wave_solution="lin")
        kwargs_forest_lin  = get_kwargs_input("lin", "1")

        # create a Forest
        test_obj = Forest(**kwargs_forest_lin)
        test_obj.rebin()

        # get header and test
        self.assert_get_header(test_obj)

        # if Forest.wave_solution is reset this should cause an error
        reset_forest()
        expected_message = (
            "Error in Forest.get_header(). Class variable 'wave_solution' "
            "must be either 'lin' or 'log'. Found: 'None'"
        )
        with self.assertRaises(AstronomicalObjectError) as context_manager:
            test_obj.get_header()
        self.compare_error_message(context_manager, expected_message)

    def test_forest_set_class_variables(self):
        """Test class method set_class_variables from Forest"""
        # logarithmic binning
        Forest.set_class_variables(3600.0, 5500.0, 1040.0, 1200.0, 50e-4, 50e-4,
                                   "log")

        log_lambda_grid = np.array([
            3.5563025, 3.5613025, 3.5663025, 3.5713025, 3.5763025, 3.5813025,
            3.5863025, 3.5913025, 3.5963025, 3.6013025, 3.6063025, 3.6113025,
            3.6163025, 3.6213025, 3.6263025, 3.6313025, 3.6363025, 3.6413025,
            3.6463025, 3.6513025, 3.6563025, 3.6613025, 3.6663025, 3.6713025,
            3.6763025, 3.6813025, 3.6863025, 3.6913025, 3.6963025, 3.7013025,
            3.7063025, 3.7113025, 3.7163025, 3.7213025, 3.7263025, 3.7313025,
            3.7363025, 3.7413025
        ])
        self.assertTrue(np.allclose(Forest.log_lambda_grid, log_lambda_grid))

        log_lambda_rest_frame_grid = np.array([
            3.01953334, 3.02453334, 3.02953334, 3.03453334, 3.03953334,
            3.04453334, 3.04953334, 3.05453334, 3.05953334, 3.06453334,
            3.06953334, 3.07453334
        ])
        self.assertTrue(np.allclose(Forest.log_lambda_rest_frame_grid,
                                    log_lambda_rest_frame_grid))

        self.assertTrue(Forest.mask_fields, defaults_forest.get("mask fields"))
        self.assertTrue(Forest.wave_solution == "log")

        # linear binning
        Forest.set_class_variables(3600.0, 5500.0, 1040.0, 1200.0, 100, 100,
                                   "lin")

        log_lambda_grid = np.array([
            3.5563025 , 3.56820172, 3.5797836 , 3.59106461, 3.60205999,
            3.61278386, 3.62324929, 3.63346846, 3.64345268, 3.65321251,
            3.66275783, 3.67209786, 3.68124124, 3.69019608, 3.69897   ,
            3.70757018, 3.71600334, 3.72427587, 3.73239376, 3.74036269
        ])
        self.assertTrue(np.allclose(Forest.log_lambda_grid, log_lambda_grid))

        log_lambda_rest_frame_grid = np.array([
            3.0374265 , 3.07554696
        ])
        self.assertTrue(np.allclose(Forest.log_lambda_rest_frame_grid,
                                    log_lambda_rest_frame_grid))

        self.assertTrue(Forest.mask_fields, defaults_forest.get("mask fields"))
        self.assertTrue(Forest.wave_solution == "lin")

        # specifying wrong bining should raise and error
        expected_message = (
            "Error in setting Forest class variables. 'wave_solution' "
            "must be either 'lin' or 'log'. Found: wrong"
        )
        with self.assertRaises(AstronomicalObjectError) as context_manager:
            Forest.set_class_variables(3600.0, 5500.0, 1040.0, 1200.0, 100, 100,
                                       "wrong")
        self.compare_error_message(context_manager, expected_message)

    def test_pk1d_forest(self):
        """Test constructor for Pk1dForest object."""
        kwargs_pk1d_forest_log = get_kwargs_input("log", "1", is_p1d=True)

        # create a Pk1dForest with missing Pk1dForest class variables
        expected_message = (
            "Error constructing Pk1dForest. Class variable 'lambda_abs_igm' "
            "must be set prior to initialize instances of this type"
        )
        with self.assertRaises(AstronomicalObjectError) as context_manager:
            Pk1dForest(**kwargs_pk1d_forest_log)
        self.compare_error_message(context_manager, expected_message)

        # set class variables
        setup_pk1d_forest("LYA")

        # create a Pk1dForest with missing Forest variables
        expected_message = (
            "Error constructing Forest. Class variable 'log_lambda_grid' "
            "must be set prior to initialize instances of this type. This "
            "probably means you did not run Forest.set_class_variables"
        )
        with self.assertRaises(AstronomicalObjectError) as context_manager:
            Pk1dForest(**kwargs_pk1d_forest_log)
        self.compare_error_message(context_manager, expected_message)

        # set class variables; case: logarithmic wavelength solution
        setup_forest(wave_solution="log", rebin=3)

        # create a Pk1dForest
        test_obj = Pk1dForest(**kwargs_pk1d_forest_log)
        self.assertTrue(isinstance(test_obj, Pk1dForest))
        self.assertTrue(isinstance(test_obj, Forest))
        self.assert_forest_object(test_obj, kwargs_pk1d_forest_log)

        # set class variables; case: linear wavelength solution
        reset_forest()
        setup_forest(wave_solution="lin")
        setup_pk1d_forest("LYA")

        kwargs_pk1d_forest_lin = get_kwargs_input("lin", "1", is_p1d=True)

        # create a Pk1dForest
        test_obj = Pk1dForest(**kwargs_pk1d_forest_lin)
        self.assertTrue(isinstance(test_obj, Pk1dForest))
        self.assertTrue(isinstance(test_obj, Forest))
        self.assert_forest_object(test_obj, kwargs_pk1d_forest_lin)

        # create a Pk1dForest with missing exposures_diff
        kwargs = {
            "ra": 0.15,
            "dec": 0.0,
            "z": 2.1,
            "flux": np.ones(15),
            "ivar": np.ones(15) * 4,
        }
        expected_message = (
            "Error constructing Pk1dForest. Missing variable 'exposures_diff'"
        )
        with self.assertRaises(AstronomicalObjectError) as context_manager:
            Pk1dForest(**kwargs)
        self.compare_error_message(context_manager, expected_message)

        # create a Pk1dForest with missing reso
        kwargs = {
            "ra": 0.15,
            "dec": 0.0,
            "z": 2.1,
            "flux": np.ones(15),
            "ivar": np.ones(15) * 4,
            "exposures_diff":
                np.ones(15),
        }
        expected_message = (
            "Error constructing Pk1dForest. Missing variable 'reso'"
        )
        with self.assertRaises(AstronomicalObjectError) as context_manager:
            Pk1dForest(**kwargs)
        self.compare_error_message(context_manager, expected_message)

        # create a Pk1dForest with missing reso_pix
        kwargs = {
            "ra": 0.15,
            "dec": 0.0,
            "z": 2.1,
            "flux": np.ones(15),
            "ivar": np.ones(15) * 4,
            "exposures_diff":
                np.ones(15),
            "reso":
                np.ones(15),
        }
        expected_message = (
            "Error constructing Pk1dForest. Missing variable 'reso_pix'"
        )
        with self.assertRaises(AstronomicalObjectError) as context_manager:
            Pk1dForest(**kwargs)
        self.compare_error_message(context_manager, expected_message)

    def test_pk1d_forest_coadd(self):
        """Test the coadd function in Pk1dForest"""
        # set class variables; case: logarithmic wavelength solution
        setup_forest(wave_solution="log", rebin=3)
        setup_pk1d_forest("LYA")
        kwargs_pk1d_forest_log  = get_kwargs_input("log", "1", is_p1d=True)
        kwargs_pk1d_forest_log2 = get_kwargs_input("log", "2", is_p1d=True)
        kwargs_pk1d_forest_log_coadd = get_kwargs_rebin("log", "COADD", rebin=3, is_p1d=True)

        # create a Pk1dForest
        test_obj = Pk1dForest(**kwargs_pk1d_forest_log)
        test_obj.rebin()

        # create a second Pk1dForest
        test_obj_other = Pk1dForest(**kwargs_pk1d_forest_log2)
        test_obj_other.rebin()

        # coadd them
        test_obj.coadd(test_obj_other)
        self.assert_forest_object(test_obj, kwargs_pk1d_forest_log_coadd)

        # create a third Pk1dForest with different los_id
        kwargs = kwargs_pk1d_forest_log2.copy()
        kwargs["los_id"] = 999
        test_obj_other = Pk1dForest(**kwargs)
        test_obj_other.rebin()

        # coadding them should raise an error
        expected_message = (
            "Attempting to coadd two Forests "
            "with different los_id. This should "
            f"not happen. this.los_id={test_obj.los_id}, "
            f"other.los_id={test_obj_other.los_id}."
        )
        with self.assertRaises(AstronomicalObjectError) as context_manager:
            test_obj.coadd(test_obj_other)
        self.compare_error_message(context_manager, expected_message)

        # set class variables; case: linear wavelength solution
        reset_forest()
        setup_forest(wave_solution="lin")
        setup_pk1d_forest("LYA")

        kwargs_pk1d_forest_lin  = get_kwargs_input("lin", "1", is_p1d=True)
        kwargs_pk1d_forest_lin2 = get_kwargs_input("lin", "2", is_p1d=True)
        kwargs_pk1d_forest_lin_coadd = get_kwargs_rebin("lin", "COADD", rebin=1, is_p1d=True)

        # create a Forest
        test_obj = Pk1dForest(**kwargs_pk1d_forest_lin)
        test_obj.rebin()

        # create a second Forest
        test_obj_other = Pk1dForest(**kwargs_pk1d_forest_lin2)
        test_obj_other.rebin()

        # coadd them
        test_obj.coadd(test_obj_other)
        self.assert_forest_object(test_obj, kwargs_pk1d_forest_lin_coadd)

        # create a Forest object
        kwargs = kwargs_pk1d_forest_log2.copy()
        kwargs["los_id"] = 999
        test_obj_other = Forest(**kwargs)
        test_obj_other.rebin()

        # coadding them should raise an error
        expected_message = (
            "Error coadding Pk1dForest. Expected Pk1dForest instance in other. "
            "Found: Forest"
        )
        with self.assertRaises(AstronomicalObjectError) as context_manager:
            test_obj.coadd(test_obj_other)
        self.compare_error_message(context_manager, expected_message)

    def test_pk1d_forest_consistency_check(self):
        """Test method consistency_check from Pk1dForest"""
        setup_forest("log")
        setup_pk1d_forest("LYA")
        kwargs_pk1d_forest_log  = get_kwargs_input("log", "1", is_p1d=True)

        # create a Pk1dForest with flux and ivar of different sizes
        kwargs = kwargs_pk1d_forest_log.copy()
        kwargs["ivar"] = kwargs["ivar"][::2]
        expected_message = (
            "Error constructing Forest. 'flux' and 'ivar' don't have the "
            "same size"
        )
        with self.assertRaises(AstronomicalObjectError) as context_manager:
            Pk1dForest(**kwargs)
        self.compare_error_message(context_manager, expected_message)

        # create a Pk1dForest with flux and exposures_diff of different sizes
        kwargs = kwargs_pk1d_forest_log.copy()
        kwargs["exposures_diff"] = kwargs["exposures_diff"][::2]
        expected_message = (
            "Error constructing Pk1dForest. 'flux' and 'exposures_diff' don't "
            "have the same size"
        )
        with self.assertRaises(AstronomicalObjectError) as context_manager:
            Pk1dForest(**kwargs)
        self.compare_error_message(context_manager, expected_message)

    def test_pk1d_forest_get_data(self):
        """Test method get_data for Pk1dForest."""
        # set class variables; case: logarithmic wavelength solution
        setup_forest(wave_solution="log", rebin=3)
        setup_pk1d_forest("LYA")
        kwargs_pk1d_forest_log = get_kwargs_input("log", "1", is_p1d=True)
        kwargs_pk1d_forest_lin = get_kwargs_input("lin", "1", is_p1d=True)

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
        kwargs_pk1d_forest_log  = get_kwargs_input("log", "1", is_p1d=True)
        kwargs_pk1d_forest_lin = get_kwargs_input("lin", "1", is_p1d=True)

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
        kwargs_sdss_forest = get_sdss_kwargs_input("log", "1")
        kwargs_sdss_forest_rebin = get_sdss_kwargs_rebin("log", "1")

        # expected error as class variables are not yet set
        expected_message = (
            "Error constructing Forest. Class variable 'log_lambda_grid' "
            "must be set prior to initialize instances of this type. This "
            "probably means you did not run Forest.set_class_variables"
        )
        with self.assertRaises(AstronomicalObjectError) as context_manager:
            SdssForest(**kwargs_sdss_forest)
        self.compare_error_message(context_manager, expected_message)

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

        # create a SdssForest with missing fiberid
        kwargs = {
            "ra": 0.15,
            "dec": 0.0,
            "z": 2.1,
            "flux": np.ones(15),
            "ivar": np.ones(15) * 4,
        }
        expected_message = (
            "Error constructing SdssForest. Missing variable 'fiberid'"
        )
        with self.assertRaises(AstronomicalObjectError) as context_manager:
            SdssForest(**kwargs)
        self.compare_error_message(context_manager, expected_message)

        # create a SdssForest with missing mjd
        kwargs = {
            "ra": 0.15,
            "dec": 0.0,
            "z": 2.1,
            "flux": np.ones(15),
            "ivar": np.ones(15) * 4,
            "fiberid": 0,
        }
        expected_message = (
            "Error constructing SdssForest. Missing variable 'mjd'"
        )
        with self.assertRaises(AstronomicalObjectError) as context_manager:
            SdssForest(**kwargs)
        self.compare_error_message(context_manager, expected_message)

        # create a SdssForest with missing plate
        kwargs = {
            "ra": 0.15,
            "dec": 0.0,
            "z": 2.1,
            "flux": np.ones(15),
            "ivar": np.ones(15) * 4,
            "fiberid": 0,
            "mjd": 0,
        }
        expected_message = (
            "Error constructing SdssForest. Missing variable 'plate'"
        )
        with self.assertRaises(AstronomicalObjectError) as context_manager:
            SdssForest(**kwargs)
        self.compare_error_message(context_manager, expected_message)

        # create a SdssForest with missing thingid
        kwargs = {
            "ra": 0.15,
            "dec": 0.0,
            "z": 2.1,
            "flux": np.ones(15),
            "ivar": np.ones(15) * 4,
            "fiberid": 0,
            "mjd": 0,
            "plate": 0,
        }
        expected_message = (
            "Error constructing SdssForest. Missing variable 'thingid'"
        )
        with self.assertRaises(AstronomicalObjectError) as context_manager:
            SdssForest(**kwargs)
        self.compare_error_message(context_manager, expected_message)

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
        expected_message = (
            "Error constructing Forest. Missing variable 'log_lambda'"
        )
        with self.assertRaises(AstronomicalObjectError) as context_manager:
            SdssForest(**kwargs)
        self.compare_error_message(context_manager, expected_message)

    def test_sdss_forest_coadd(self):
        """Test the coadd function in SdssForest"""
        # set class variables
        setup_forest(wave_solution="log", rebin=3)
        kwargs_sdss_forest = get_sdss_kwargs_input("log", "1")
        kwargs_sdss_forest2 = get_sdss_kwargs_input("log", "2")
        kwargs_sdss_forest_coadd = get_sdss_kwargs_rebin("log", "COADD")

        # create a SdssForest
        test_obj = SdssForest(**kwargs_sdss_forest)
        test_obj.rebin()

        # create a second SdssForest
        test_obj_other = SdssForest(**kwargs_sdss_forest2)
        test_obj_other.rebin()

        # coadd them
        test_obj.coadd(test_obj_other)
        self.assert_forest_object(test_obj, kwargs_sdss_forest_coadd)

        # create a third SdssForest with different thingid
        kwargs = kwargs_sdss_forest2.copy()
        kwargs["thingid"] = 999
        test_obj_other = SdssForest(**kwargs)
        test_obj_other.rebin()

        # coadding them should raise an error
        expected_message = (
            "Attempting to coadd two Forests "
            "with different los_id. This should "
            f"not happen. this.los_id={test_obj.los_id}, "
            f"other.los_id={test_obj_other.los_id}."
        )
        with self.assertRaises(AstronomicalObjectError) as context_manager:
            test_obj.coadd(test_obj_other)
        self.compare_error_message(context_manager, expected_message)

        # create a Forest object
        kwargs = kwargs_sdss_forest2.copy()
        kwargs["los_id"] = 999
        test_obj_other = Forest(**kwargs)
        test_obj_other.rebin()

        # coadding them should raise an error
        expected_message = (
            "Error coadding SdssForest. Expected SdssForest instance in other. "
            "Found: Forest"
        )
        with self.assertRaises(AstronomicalObjectError) as context_manager:
            test_obj.coadd(test_obj_other)
        self.compare_error_message(context_manager, expected_message)

    def test_sdss_forest_get_data(self):
        """Test method get_data for SdssForest."""
        # set class variables
        setup_forest(wave_solution="log", rebin=3)
        kwargs_sdss_forest = get_sdss_kwargs_input("log", "1")

        # create an SdssForest
        test_obj = SdssForest(**kwargs_sdss_forest)
        test_obj.rebin()
        self.assert_get_data(test_obj)

    def test_sdss_forest_get_header(self):
        """Test method get_header for SdssForest."""
        # set class variables
        setup_forest(wave_solution="log", rebin=3)
        kwargs_sdss_forest = get_sdss_kwargs_input("log", "1")
        kwargs_sdss_forest2 = get_sdss_kwargs_input("log", "2")

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
        kwargs_sdss_pk1d_forest = get_sdss_kwargs_input("log", "1", is_p1d=True)
        kwargs_sdss_pk1d_forest_rebin = get_sdss_kwargs_rebin("log", "1", is_p1d=True)

        # expected error as Pk1dForest class variables are not yet set
        expected_message = (
            "Error constructing Pk1dForest. Class variable 'lambda_abs_igm' "
            "must be set prior to initialize instances of this type"
        )
        with self.assertRaises(AstronomicalObjectError) as context_manager:
            SdssPk1dForest(**kwargs_sdss_pk1d_forest)
        self.compare_error_message(context_manager, expected_message)

        # set class variables
        setup_pk1d_forest("LYA")

        # expected error as Forest class variables are not yet set
        expected_message = (
            "Error constructing Forest. Class variable 'log_lambda_grid' "
            "must be set prior to initialize instances of this type. This "
            "probably means you did not run Forest.set_class_variables"
        )
        with self.assertRaises(AstronomicalObjectError) as context_manager:
            SdssPk1dForest(**kwargs_sdss_pk1d_forest)
        self.compare_error_message(context_manager, expected_message)

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
            "ra": 0.15,
            "dec": 0.0,
            "z": 2.1,
            "flux": np.ones(15),
            "ivar": np.ones(15) * 4,
            "log_lambda": np.array([
                    3.5565, 3.55655, 3.5567, 3.55675, 3.5569, 3.55695, 3.5571,
                    3.55715, 3.5573, 3.55735 ]),
            "exposures_diff": np.ones(15),
            "reso": np.ones(15),
            "reso_pix": np.ones(15),
        }
        expected_message = (
            "Error constructing SdssForest. Missing variable 'fiberid'"
        )
        with self.assertRaises(AstronomicalObjectError) as context_manager:
            SdssPk1dForest(**kwargs)
        self.compare_error_message(context_manager, expected_message)

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
        expected_message = (
            "Error constructing Pk1dForest. Missing variable 'exposures_diff'"
        )
        with self.assertRaises(AstronomicalObjectError) as context_manager:
            SdssPk1dForest(**kwargs)
        self.compare_error_message(context_manager, expected_message)

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
            "reso_pix": np.ones(15),
        }
        expected_message = (
            "Error constructing Forest. Missing variable 'log_lambda'"
        )
        with self.assertRaises(AstronomicalObjectError) as context_manager:
            SdssPk1dForest(**kwargs)
        self.compare_error_message(context_manager, expected_message)

    def test_sdss_pk1d_forest_coadd(self):
        """Test the coadd function in SdssPk1dForest"""
        # set class variables
        setup_forest(wave_solution="log", rebin=3)
        setup_pk1d_forest("LYA")

        kwargs_sdss_pk1d_forest = get_sdss_kwargs_input("log", "1", is_p1d=True)
        kwargs_sdss_pk1d_forest2 = get_sdss_kwargs_input("log", "2", is_p1d=True)
        kwargs_sdss_pk1d_forest_coadd = get_sdss_kwargs_rebin("log", "COADD", is_p1d=True)

        # create a SdssPk1dForest
        test_obj = SdssPk1dForest(**kwargs_sdss_pk1d_forest)
        test_obj.rebin()

        # create a second SdssPk1dForest
        test_obj_other = SdssPk1dForest(**kwargs_sdss_pk1d_forest2)
        test_obj_other.rebin()

        # coadd them
        test_obj.coadd(test_obj_other)
        self.assert_forest_object(test_obj, kwargs_sdss_pk1d_forest_coadd)

        # create a third SdssForest with different thingid
        kwargs = kwargs_sdss_pk1d_forest2.copy()
        kwargs["thingid"] = 999
        test_obj_other = SdssPk1dForest(**kwargs)
        test_obj_other.rebin()

        # coadding them should raise an error
        expected_message = (
            "Attempting to coadd two Forests "
            "with different los_id. This should "
            f"not happen. this.los_id={test_obj.los_id}, "
            f"other.los_id={test_obj_other.los_id}."
        )
        with self.assertRaises(AstronomicalObjectError) as context_manager:
            test_obj.coadd(test_obj_other)
        self.compare_error_message(context_manager, expected_message)

    def test_sdss_pk1d_forest_get_data(self):
        """Test method get_data for SdssPk1dForest."""
        # set class variables
        setup_forest(wave_solution="log", rebin=3)
        setup_pk1d_forest("LYA")
        kwargs_sdss_pk1d_forest = get_sdss_kwargs_input("log", "1", is_p1d=True)

        # create an SdssPk1dForest
        test_obj = SdssPk1dForest(**kwargs_sdss_pk1d_forest)
        self.assert_get_data(test_obj)

    def test_sdss_pk1d_forest_get_header(self):
        """Test method get_header for SdssPk1dForest."""
        # set class variables
        setup_forest(wave_solution="log", rebin=3)
        setup_pk1d_forest("LYA")
        kwargs_sdss_pk1d_forest = get_sdss_kwargs_input("log", "1", is_p1d=True)
        kwargs_sdss_pk1d_forest2 = get_sdss_kwargs_input("log", "2", is_p1d=True)

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
