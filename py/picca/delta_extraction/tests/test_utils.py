"""This file contains functions and variables used in different tests"""
import logging
import os

import numpy as np

from picca.delta_extraction.astronomical_objects.forest import Forest
from picca.delta_extraction.astronomical_objects.pk1d_forest import Pk1dForest
from picca.delta_extraction.astronomical_objects.sdss_forest import SdssForest
from picca.delta_extraction.utils import ABSORBER_IGM

# reset Forest and Pk1dForest class variables
def reset_forest():
    """Reset the class variables of Forest and Pk1dForest"""
    Forest.wave_solution = None
    Forest.delta_log_lambda = None
    Forest.delta_log_lambda = None
    Forest.lambda_max = None
    Forest.lambda_max_rest_frame = None
    Forest.lambda_min = None
    Forest.lambda_min_rest_frame = None
    Forest.log_lambda_max = None
    Forest.log_lambda_max_rest_frame = None
    Forest.log_lambda_min = None
    Forest.log_lambda_min_rest_frame = None
    Forest.mask_fields = []
    Pk1dForest.lambda_abs_igm = None


# setup Forest class variables
def setup_forest(wave_solution):
    """Set Forest class variables

    Arguments
    ---------
    wave_solution: "log" or "lin"
    Determines whether the wavelength solution has linear spacing ("lin") or
    logarithmic spacing ("log").
    """
    assert wave_solution in ["log", "lin"]

    if wave_solution == "log":
        Forest.wave_solution = "log"
        Forest.delta_log_lambda = 1e-4
        Forest.log_lambda_max = np.log10(5500.0)
        Forest.log_lambda_max_rest_frame = np.log10(1200.0)
        Forest.log_lambda_min = np.log10(3600.0)
        Forest.log_lambda_min_rest_frame = np.log10(1040.0)
    elif wave_solution == "lin":
        Forest.wave_solution = "lin"
        Forest.delta_lambda = 1.
        Forest.lambda_max = 5500.0
        Forest.lambda_max_rest_frame = 1200.0
        Forest.lambda_min = 3600.0
        Forest.lambda_min_rest_frame = 1040.0

setup_forest("log")

# setup Pk1dForest class variables
def setup_pk1d_forest(absorber):
    """Set Pk1dForest class variables

    Arguments
    ---------
    absorber: "str"
    Key of ABSORBER_IGM selecting the absorber
    """
    Pk1dForest.lambda_abs_igm = ABSORBER_IGM.get(absorber)

setup_pk1d_forest("LYA")

# create SdssForest instance forest1
# has:
# * 1 DLA in dummy_absorbers_cat.fits.gz
# * 1 absorber in dummy_absorbers_cat.fits.gz
forest1_log_lambda = np.arange(3.562, 3.62, 1e-3)
kwargs1 = {
    "los_id": 9999,
    "ra": 0.15,
    "dec": 0.0,
    "z": 2.5,
    "flux": np.ones_like(forest1_log_lambda),
    "ivar": np.ones_like(forest1_log_lambda)*4,
    "log_lambda": forest1_log_lambda,
    "thingid": 10000,
    "plate": 0,
    "fiberid": 0,
    "mjd": 0,
}
forest1 = SdssForest(**kwargs1)
# forest 1 properties
assert np.allclose(forest1.flux, np.ones_like(forest1_log_lambda))
assert np.allclose(forest1.log_lambda, forest1_log_lambda)
assert np.allclose(forest1.ivar, np.ones_like(forest1_log_lambda)*4)
assert np.allclose(forest1.transmission_correction,
                   np.ones_like(forest1_log_lambda))

# create SdssForest instance forest2
# has:
# * 1 DLA in dummy_absorbers_cat.fits.gz
# * 1 absorber in dummy_absorbers_cat.fits.gz
forest2_log_lambda = np.arange(3.562, 3.62, 1e-3)
kwargs2 = {
    "los_id": 9999,
    "ra": 0.15,
    "dec": 0.0,
    "z": 2.5,
    "flux": np.ones_like(forest2_log_lambda),
    "ivar": np.ones_like(forest2_log_lambda)*4,
    "log_lambda": forest2_log_lambda,
    "thingid": 10001,
    "plate": 0,
    "fiberid": 0,
    "mjd": 0,
}
forest2 = SdssForest(**kwargs2)
# forest 2 properties
assert np.allclose(forest2.flux, np.ones_like(forest2_log_lambda))
assert np.allclose(forest2.log_lambda, forest2_log_lambda)
assert np.allclose(forest2.ivar, np.ones_like(forest2_log_lambda)*4)
assert np.allclose(forest2.transmission_correction,
                   np.ones_like(forest2_log_lambda))

# create SdssForest instance forest3
# has:
# * 0 DLA in dummy_absorbers_cat.fits.gz
# * 0 absorber in dummy_absorbers_cat.fits.gz
forest3_log_lambda = np.arange(3.562, 3.62, 1e-3)
kwargs3 = {
    "los_id": 9999,
    "ra": 0.15,
    "dec": 0.0,
    "z": 2.5,
    "flux": np.ones_like(forest3_log_lambda),
    "ivar": np.ones_like(forest3_log_lambda)*4,
    "log_lambda": forest3_log_lambda,
    "thingid": 10002,
    "plate": 0,
    "fiberid": 0,
    "mjd": 0,
}
forest3 = SdssForest(**kwargs3)
# forest 2 properties
assert np.allclose(forest3.flux, np.ones_like(forest3_log_lambda))
assert np.allclose(forest3.log_lambda, forest3_log_lambda)
assert np.allclose(forest3.ivar, np.ones_like(forest3_log_lambda)*4)
assert np.allclose(forest3.transmission_correction,
                   np.ones_like(forest3_log_lambda))


# Dictionary to load SdssData
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sdss_data_kwargs = {
    "input directory":
        f"{THIS_DIR}/data",
    "output directory":
        f"{THIS_DIR}/results",
    "drq catalogue":
        f"{THIS_DIR}/data/cat_for_clustering_plate3655.fits.gz",
    "z max": 3.5,
    "z min": 2.1,
}
sdss_data_kwargs_filter_forest = {
    "input directory":
        f"{THIS_DIR}/data",
    "output directory":
        f"{THIS_DIR}/results",
    "drq catalogue":
        f"{THIS_DIR}/data/cat_for_clustering_plate3655.fits.gz",
    "mode": "spec",
    "lambda min": 3600.0,
    "lambda max": 7235.0,
    "lambda min rest frame": 2900.0,
    "lambda max rest frame": 3120.0,
}



def reset_logger():
    """This function reset the logger picca.delta_extraction by closing
    and removing its handlers.
    """
    logger = logging.getLogger("picca.delta_extraction")
    handlers = logger.handlers
    for handler in handlers:
        handler.close()
        logger.removeHandler(handler)
    logger.addHandler(logging.NullHandler())


if __name__ == '__main__':
    pass
