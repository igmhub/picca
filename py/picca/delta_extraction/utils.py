"""This module define several functions and variables used throughout the
package"""
import importlib
import logging
import os
import sys

from numba import njit
import numpy as np
from scipy.constants import speed_of_light as speed_light

from picca.delta_extraction.errors import DeltaExtractionError

module_logger = logging.getLogger(__name__)

SPEED_LIGHT = speed_light / 1000.  # [km/s]

ABSORBER_IGM = {
    "Halpha": 6562.8,
    "Hbeta": 4862.68,
    "MgI(2853)": 2852.96,
    "MgII(2804)": 2803.5324,
    "MgII(2796)": 2796.3511,
    "FeII(2600)": 2600.1724835,
    "FeII(2587)": 2586.6495659,
    "MnII(2577)": 2576.877,
    "FeII(2383)": 2382.7641781,
    "FeII(2374)": 2374.4603294,
    "FeII(2344)": 2344.2129601,
    "AlIII(1863)": 1862.79113,
    "AlIII(1855)": 1854.71829,
    "AlII(1671)": 1670.7886,
    "FeII(1608)": 1608.4511,
    "CIV(1551)": 1550.77845,
    "CIV(eff)": 1549.06,
    "CIV(1548)": 1548.2049,
    "SiII(1527)": 1526.70698,
    "NiII(1455)": 1454.842,
    "SiIV(1403)": 1402.77291,
    "SiIV(1394)": 1393.76018,
    "NiII(1370)": 1370.132,
    "CII(1335)": 1334.5323,
    "NiII(1317)": 1317.217,
    "SiII(1304)": 1304.3702,
    "OI(1302)": 1302.1685,
    "SiII(1260)": 1260.4221,
    "SII(1254)": 1253.811,
    "SII(1251)": 1250.584,
    "NV(1243)": 1242.804,
    "NV(1239)": 1238.821,
    "LYA": 1215.67,
    "SiIII(1207)": 1206.500,
    "NI(1200)": 1200.,
    "SiII(1193)": 1193.2897,
    "SiII(1190)": 1190.4158,
    "PII(1153)": 1152.818,
    "FeII(1145)": 1144.9379,
    "FeII(1143)": 1143.2260,
    "NI(1134)": 1134.4149,
    "FeII(1125)": 1125.4477,
    "FeIII(1123)": 1122.526,
    "FeII(1097)": 1096.8769,
    "NII(1084)": 1083.990,
    "FeII(1082)": 1081.8748,
    "FeII(1063)": 1063.002,
    "OI(1039)": 1039.230,
    "OVI(1038)": 1037.613,
    "CII(1037)": 1036.7909,
    "OVI(1032)": 1031.912,
    "LYB": 1025.72,
    "SiII(1021)": 1020.6989,
    "SIII(1013)": 1012.502,
    "SiII(990)": 989.8731,
    "OI(989)": 988.7,
    "CIII(977)": 977.020,
    "LY3": 972.537,
    "LY4": 949.7431,
    "LY5": 937.8035,
    "LY6": 930.7483,
    "LY7": 926.2257,
    "LY8": 923.1504,
    "LY9": 920.9631,
    "LY10": 919.3514,
}

ACCEPTED_BLINDING_STRATEGIES = [
    "none", "desi_m2", "desi_y1", "desi_y3"]
# TODO: add tags here when we are allowed to unblind them
UNBLINDABLE_STRATEGIES = []

def class_from_string(class_name, module_name):
    """Return a class from a string. The class must be saved in a module
    under picca.delta_extraction with the same name as the class but
    lowercase and with and underscore. For example class 'MyClass' should
    be in module picca.delta_extraction.my_class

    Arguments
    ---------
    class_name: str
    Name of the class to load

    module_name: str
    Name of the module containing the class

    Return
    ------
    class_object: Class
    The loaded class

    deafult_args: dict
    A dictionary with the default options (empty for no default options)

    accepted_options: list
    A list with the names of the accepted options

    Raise
    -----
    ImportError if module cannot be loaded
    AttributeError if class cannot be found
    """
    # load module
    module_object = importlib.import_module(module_name)
    # get the class
    class_object = getattr(module_object, class_name)
    # get the dictionary with the default arguments
    try:
        default_args = getattr(module_object, "defaults")
    except AttributeError:
        default_args = {}
    # get the list with the valid options
    try:
        accepted_options = getattr(module_object, "accepted_options")
    except AttributeError:
        accepted_options = []
    return class_object, default_args, accepted_options


@njit()
def find_bins(original_array, grid_array, wave_solution):
    """For each element in original_array, find the corresponding bin in grid_array

    Arguments
    ---------
    original_array: array of float
    Read array, e.g. forest.log_lambda

    grid_array: array of float
    Common array, e.g. Forest.log_lambda_grid

    wave_solution: "log" or "lin"
    Specifies whether we want to construct a wavelength grid that is evenly
    spaced on wavelength (lin) or on the logarithm of the wavelength (log)

    Return
    ------
    found_bin: array of int
    An array of size original_array.size filled with values smaller than
    grid_array.size with the bins correspondance
    """
    if wave_solution == "log":
        pass
    elif wave_solution == "lin":
        original_array = 10**original_array
        grid_array = 10**grid_array
    else:  # pragma: no cover
        raise DeltaExtractionError(
            "Error in function find_bins from py/picca/delta_extraction/utils.py"
            "expected wavelength solution to be either 'log' or 'lin'. ")
    original_array_size = original_array.size
    grid_array_size = grid_array.size
    found_bin = np.zeros(original_array_size, dtype=np.int64)
    for index1 in range(original_array_size):
        min_dist = np.finfo(np.float64).max
        for index2 in range(grid_array_size):
            dist = np.abs(grid_array[index2] - original_array[index1])
            if dist < min_dist:
                min_dist = dist
                found_bin[index1] = index2
            else:
                break
    return found_bin

PROGRESS_LEVEL_NUM = 15
logging.addLevelName(PROGRESS_LEVEL_NUM, "PROGRESS")


def progress(self, message, *args, **kws):
    """Function to log with level PROGRESS"""
    if self.isEnabledFor(PROGRESS_LEVEL_NUM):  # pragma: no branch
        # pylint: disable-msg=protected-access
        # this method will be attached to logging.Logger
        self._log(PROGRESS_LEVEL_NUM, message, args, **kws)


logging.Logger.progress = progress

OK_WARNING_LEVEL_NUM = 31
logging.addLevelName(OK_WARNING_LEVEL_NUM, "WARNING OK")


def ok_warning(self, message, *args, **kws):
    """Function to log with level WARNING OK"""
    if self.isEnabledFor(OK_WARNING_LEVEL_NUM):  # pragma: no branch
        # pylint: disable-msg=protected-access
        # this method will be attached to logging.Logger
        self._log(OK_WARNING_LEVEL_NUM, message, args, **kws)


logging.Logger.ok_warning = ok_warning


def setup_logger(logging_level_console=logging.DEBUG,
                 log_file=None,
                 logging_level_file=logging.DEBUG):
    """This function set up the logger for the package
    picca.delta_extraction

    Arguments
    ---------
    logging_level_console: int or str - Default: logging.DEBUG
    Logging level for the console handler. If str, it should be a Level from
    the logging module (i.e. CRITICAL, ERROR, WARNING, INFO, DEBUG, NOTSET).
    Additionally, the user-defined levels PROGRESS and WARNING_OK are allowed.

    log_file: str or None
    Log file for logging

    logging_level_file: int or str - Default: logging.DEBUG
    Logging level for the file handler. If str, it should be a Level from
    the logging module (i.e. CRITICAL, ERROR, WARNING, INFO, DEBUG, NOTSET).
    Additionally, the user-defined level PROGRESS and WARNING_OK are allowed.
    Ignored if log_file is None.
    """
    if isinstance(logging_level_console, str):
        if logging_level_console.upper() == "PROGRESS":
            logging_level_console = PROGRESS_LEVEL_NUM
        else:
            logging_level_console = getattr(logging,
                                            logging_level_console.upper())

    if isinstance(logging_level_file, str):
        if logging_level_file.upper() == "PROGRESS":
            logging_level_file = PROGRESS_LEVEL_NUM
        else:
            logging_level_file = getattr(logging, logging_level_file.upper())

    logger = logging.getLogger("picca.delta_extraction")
    logger.setLevel(logging.DEBUG)

    # logging formatter
    formatter = logging.Formatter('[%(levelname)s]: %(message)s')

    # create console handler to logs messages
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging_level_console)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # create file handler which logs messages to file
    if log_file is not None:
        if os.path.exists(log_file):
            newfilename = f'{log_file}.{os.path.getmtime(log_file)}'
            os.rename(log_file, newfilename)
        file_handler = logging.FileHandler(log_file, mode="w")
        file_handler.setLevel(logging_level_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # sets up numba logger
    #logging.getLogger('numba').setLevel(logging.WARNING)

def update_accepted_options(accepted_options, new_options, remove=False):
    """Update the content of the list of accepted options

    Arguments
    ---------
    accepted_options: list of string
    The current accepted options

    new_options: list of string
    The new options

    remove: bool - Default: False
    If True, then remove the elements of new_options from accepted_options.
    If False, then add new_options to accepted_options

    Return
    ------
    accepted_options: list of string
    The updated accepted options
    """
    if remove:
        accepted_options = accepted_options.copy()
        for item in new_options:
            if item in accepted_options:
                accepted_options.remove(item)
    else:
        accepted_options = sorted(list(set(accepted_options + new_options)))

    return accepted_options

def update_default_options(default_options, new_options):
    """Update the content of the list of accepted options

    Arguments
    ---------
    default_options: dict
    The current default options

    new_options: dict
    The new options

    Return
    ------
    default_options: dict
    The updated default options
    """
    default_options = default_options.copy()
    for key, value in new_options.items():
        if key in default_options:
            default_value = default_options.get(key)
            if type(default_value) is not type(value):
                raise DeltaExtractionError(
                    f"Incompatible defaults are being added. Key {key} "
                    "found to have values with different type: "
                    f"{type(default_value)} and {type(value)}. "
                    "Revise your recent changes or contact picca developpers.")
            if default_value != value:
                raise DeltaExtractionError(
                    f"Incompatible defaults are being added. Key {key} "
                    f"found to have two default values: '{value}' and '{default_value}' "
                    "Revise your recent changes or contact picca developpers.")
        else:
            default_options[key] = value

    return default_options
