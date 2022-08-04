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
    Forest.log_lambda_grid = np.array([])
    Forest.log_lambda_rest_frame_grid = np.array([])
    Forest.mask_fields = []
    Pk1dForest.lambda_abs_igm = None

# setup Forest class variables
def setup_forest(wave_solution, rebin=1, pixel_step=None):
    """Set Forest class variables

    Arguments
    ---------
    wave_solution: "log" or "lin"
    Determines whether the wavelength solution has linear spacing ("lin") or
    logarithmic spacing ("log").

    rebin: int
    Number of pixels that will be rebinned

    pixel_step: float
    Pixel size to be used

    """
    assert wave_solution in ["log", "lin"]

    if pixel_step is None:
        if wave_solution == "log":
            pixel_step = 1e-4
        elif wave_solution == "lin":
            pixel_step = 1

    pixel_step *= rebin
    pixel_step_rf = pixel_step

    Forest.set_class_variables(3600.0, 5500.0, 1040.0, 1200.0, pixel_step, pixel_step_rf,
                               wave_solution)

setup_forest("log", rebin=3)

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
forest1_log_lambda = np.arange(3.5617025, 3.6200525, 3e-4)
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
forest1.rebin()
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
forest2_log_lambda = np.arange(3.5617025, 3.6200525, 3e-4)
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
forest2.rebin()
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
forest3_log_lambda = np.arange(3.5617025, 3.6200525, 3e-4)
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
forest3.rebin()
# forest 2 properties
assert np.allclose(forest3.flux, np.ones_like(forest3_log_lambda))
assert np.allclose(forest3.log_lambda, forest3_log_lambda)
assert np.allclose(forest3.ivar, np.ones_like(forest3_log_lambda)*4)
assert np.allclose(forest3.transmission_correction,
                   np.ones_like(forest3_log_lambda))



# Dictionary to load data
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
kwargs_data = {
    "input directory":
        f"{THIS_DIR}/data",
    "out dir":
        f"{THIS_DIR}/results",
    "rejection log file": "rejection_log.fits.gz",
    "z max": 3.5,
    "z min": 2.1,
}

# Dictionary to load DesiHealpix
desi_healpix_kwargs = kwargs_data.copy()
desi_healpix_kwargs.update({
    "catalogue":
        f"{THIS_DIR}/data/QSO_cat_fuji_dark_healpix.fits.gz",
    "num processors": 1,
})

# Dictionary to load SdssData
sdss_data_kwargs = kwargs_data.copy()
sdss_data_kwargs.update({
    "drq catalogue":
        f"{THIS_DIR}/data/cat_for_clustering_plate3655.fits.gz",
    "num processors": 1,
})
sdss_data_kwargs_filter_forest = {
    "input directory":
        f"{THIS_DIR}/data",
    "out dir":
        f"{THIS_DIR}/results",
    "rejection log file": "sdss_data_rejection_log.fits.gz",
    "drq catalogue":
        f"{THIS_DIR}/data/cat_for_clustering_plate3655.fits.gz",
    "mode": "spec",
    "lambda min": 3600.0,
    "lambda max": 7235.0,
    "lambda min rest frame": 2900.0,
    "lambda max rest frame": 3120.0,
    "num processors": 1,
}

desi_mock_data_kwargs = {
    "input directory":
        f"{THIS_DIR}/data",
    "out dir":
        f"{THIS_DIR}/results",
    "rejection log file": "rejection_log.fits.gz",
    "catalogue":
        f"{THIS_DIR}/data/desi_mock_test_catalogue.fits",
    "z max": 3.5,
    "z min": 2.1,
    "wave solution": "lin",
    "delta lambda": 2.4,
    "type": "DesisimMocks",
    "num processors": 1,
}

reset_forest()

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

class WarningsHandler(logging.Handler):
    """Handler to raise Errors when logging warnings"""
    def __init__(self, ErrorType):
        """Initialize instance

        Arguments
        ---------
        ErrorType: Exception or child of Exception
        Error type
        """
        super().__init__()
        self.ErrorType = ErrorType

    def handle(self, record):
        """Raise error instead of logging a warning

        Arguments
        ---------
        record: logging.LogRecord
        The logger record

        Raise
        -----
        ErrorType if the record has warning level or higher
        """
        if record.levelno >= logging.WARN:
            raise self.ErrorType(record.getMessage())
        return record

def setup_test_logger(logger_name, ErrorType, reset=False):
    """This function set up the logger for the package
    picca.delta_extraction so that warnings raise errors
    of type ErrorType instead

    Arguments
    ---------
    logger_name: str
    Name of the logger

    ErrorType: Exception or child of Exception
    Error type

    reset: bool - Default: False
    If True, reset the logger
    """
    logger = logging.getLogger(logger_name)

    if reset:
        handlers = logger.handlers
        for handler in handlers:
            handler.close()
            logger.removeHandler(handler)
        logger.addHandler(logging.NullHandler())
    else:
        # add handler
        warnings_handler = WarningsHandler(ErrorType)
        warnings_handler.setLevel(logging.WARN)
        logger.addHandler(warnings_handler)


if __name__ == '__main__':
    pass
