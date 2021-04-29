"""This module define several functions and variables used throughout the
package"""
import importlib
import logging

from scipy.constants import speed_of_light as speed_light

module_logger = logging.getLogger(__name__)

SPEED_LIGHT = speed_light/1000. # [km/s]

ABSORBER_IGM = {
    "Halpha"      : 6562.8,
    "Hbeta"       : 4862.68,
    "MgI(2853)"   : 2852.96,
    "MgII(2804)"  : 2803.5324,
    "MgII(2796)"  : 2796.3511,
    "FeII(2600)"  : 2600.1724835,
    "FeII(2587)"  : 2586.6495659,
    "MnII(2577)"  : 2576.877,
    "FeII(2383)"  : 2382.7641781,
    "FeII(2374)"  : 2374.4603294,
    "FeII(2344)"  : 2344.2129601,
    "AlIII(1863)" : 1862.79113,
    "AlIII(1855)" : 1854.71829,
    "AlII(1671)"  : 1670.7886,
    "FeII(1608)"  : 1608.4511,
    "CIV(1551)"   : 1550.77845,
    "CIV(eff)"    : 1549.06,
    "CIV(1548)"   : 1548.2049,
    "SiII(1527)"  : 1526.70698,
    "NiII(1455)"  : 1454.842,
    "SiIV(1403)"  : 1402.77291,
    "SiIV(1394)"  : 1393.76018,
    "NiII(1370)"  : 1370.132,
    "CII(1335)"   : 1334.5323,
    "NiII(1317)"  : 1317.217,
    "SiII(1304)"  : 1304.3702,
    "OI(1302)"    : 1302.1685,
    "SiII(1260)"  : 1260.4221,
    "SII(1254)"   : 1253.811,
    "SII(1251)"   : 1250.584,
    "NV(1243)"    : 1242.804,
    "NV(1239)"    : 1238.821,
    "LYA"         : 1215.67,
    "SiIII(1207)" : 1206.500,
    "NI(1200)"    : 1200.,
    "SiII(1193)"  : 1193.2897,
    "SiII(1190)"  : 1190.4158,
    "PII(1153)"   : 1152.818,
    "FeII(1145)"  : 1144.9379,
    "FeII(1143)"  : 1143.2260,
    "NI(1134)"    : 1134.4149,
    "FeII(1125)"  : 1125.4477,
    "FeIII(1123)" : 1122.526,
    "FeII(1097)"  : 1096.8769,
    "NII(1084)"   : 1083.990,
    "FeII(1082)"  : 1081.8748,
    "FeII(1063)"  : 1063.002,
    "OI(1039)"    : 1039.230,
    "OVI(1038)"   : 1037.613,
    "CII(1037)"   : 1036.7909,
    "OVI(1032)"   : 1031.912,
    "LYB"         : 1025.72,
    "SiII(1021)"  : 1020.6989,
    "SIII(1013)"  : 1012.502,
    "SiII(990)"   : 989.8731,
    "OI(989)"     : 988.7,
    "CIII(977)"   : 977.020,
    "LY3"         : 972.537,
    "LY4"         : 949.7431,
    "LY5"         : 937.8035,
    "LY6"         : 930.7483,
    "LY7"         : 926.2257,
    "LY8"         : 923.1504,
    "LY9"         : 920.9631,
    "LY10"        : 919.3514,
}

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
    The loaded class

    Raise
    -----
    ImportError if module cannot be loaded
    AttributeError if class cannot be found
    """
    # load module
    module_object = importlib.import_module(module_name)
    # get the class, will raise
    class_object = getattr(module_object, class_name)
    return class_object


PROGRESS_LEVEL_NUM = 15
logging.addLevelName(PROGRESS_LEVEL_NUM, "PROGRESS")
def progress(self, message, *args, **kws):
    """Function to log with level PROGRESS"""
    if self.isEnabledFor(PROGRESS_LEVEL_NUM):
        # pylint: disable-msg=protected-access
        # this method will be attached to logging.Logger
        self._log(PROGRESS_LEVEL_NUM, message, args, **kws)
logging.Logger.progress = progress

OK_WARNING_LEVEL_NUM = 31
logging.addLevelName(OK_WARNING_LEVEL_NUM, "WARNING OK")
def ok_warning(self, message, *args, **kws):
    """Function to log with level WARNING OK"""
    if self.isEnabledFor(OK_WARNING_LEVEL_NUM):
        # pylint: disable-msg=protected-access
        # this method will be attached to logging.Logger
        self._log(OK_WARNING_LEVEL_NUM, message, args, **kws)
logging.Logger.ok_warning = ok_warning

def setup_logger(logging_level_console=logging.DEBUG, log_file=None,
                 logging_level_file=logging.DEBUG):
    """This function set up the logger for the package
    picca.delta_extraction

    Arguments
    ---------
    logging_level_console: int or str - Default: logging.DEBUG
    Logging level for the console handler. If str, it should be a Level from
    the logging module (i.e. CRITICAL, ERROR, WARNING, INFO, DEBU, NOTSET).
    Additionally, the user-defined level PROGRESS is allowed.

    log_file: str or None
    Log file for logging

    logging_level_file: int or str - Default: logging.DEBUG
    Logging level for the file handler. If str, it should be a Level from
    the logging module (i.e. CRITICAL, ERROR, WARNING, INFO, DEBU, NOTSET).
    Additionally, the user-defined level PROGRESS is allowed. Ignored if
    log_file is None.
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
            logging_level_file = getattr(logging,
                                         logging_level_file.upper())

    logger = logging.getLogger("picca.delta_extraction")
    logger.setLevel(logging.DEBUG)

    # logging formatter
    formatter = logging.Formatter('[%(levelname)s]: %(message)s')

    # create console handler to logs messages
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging_level_console)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # create file handler which logs messages to file
    if log_file is not None:
        file_handler = logging.FileHandler(log_file, mode="w")
        file_handler.setLevel(logging_level_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # sets up numba logger
    #logging.getLogger('numba').setLevel(logging.WARNING)
