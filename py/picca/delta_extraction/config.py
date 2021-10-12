"""This module defines the Config class.
This class is responsible for managing the options selected for the user and
contains the default configuration.
"""
from configparser import ConfigParser
import logging
import os
import re
from datetime import datetime
import git

from picca.delta_extraction.errors import ConfigError
from picca.delta_extraction.utils import class_from_string, setup_logger

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PICCA_BASE = THIS_DIR.split("py/picca")[0]

default_config = {
    "general": {
        "overwrite": False,
        # New logging level defined in setup_logger.
        # Numeric value is PROGRESS_LEVEL_NUM defined in utils.py
        "logging level console": "PROGRESS",
        "logging level file": "PROGRESS",
        "log": "run.log",
    },
    "run specs": {
        "git hash": git.Repo(PICCA_BASE).head.object.hexsha,
        "timestamp": str(datetime.now()),
    }
}

class Config:
    """Class to manage the configuration file

    Methods
    -------
    __init__
    __format_correction_section
    __format_data_section
    __format_expected_flux_section
    __format_general_section
    __format_masks_section
    __parse_environ_variables
    write_config

    Attributes
    ---------
    config: ConfigParser
    A ConfigParser instance with the user configuration

    corrections: list of (class, dict)
    A list of (class, dict). For each element, class should be a child of
    Correction and dict a dictionary with the keyword arguments necesssary
    to initialize it

    data: (class, dict)
    Class should be a child of Data and dict a dictionary with the keyword
    arguments necesssary to initialize it

    expected_flux: (class, dict)
    Class should be a child of ExpectedFlux and dict a dictionary with the keyword
    arguments necesssary to initialize it

    log: str or None
    Name of the log file. None for no log file

    logger: logging.Logger
    Logger object

    logging_level_console: str
    Level of console logging. Messages with lower priorities will not be logged.
    Accepted values are (in order of priority) NOTSET, DEBUG, PROGRESS, INFO,
    WARNING, WARNING_OK, ERROR, CRITICAL.

    logging_level_file: str
    Level of file logging. Messages with lower priorities will not be logged.
    Accepted values are (in order of priority) NOTSET, DEBUG, PROGRESS, INFO,
    WARNING, WARNING_OK, ERROR, CRITICAL.

    masks: list of (class, dict)
    A list of (class, dict). For each element, class should be a child of
    Mask and dict a dictionary with the keyword arguments necesssary
    to initialize it

    num_corrections: int
    Number of elements in self.corrections

    num_masks: int
    Number of elements in self.masks

    out_dir: str
    Name of the directory where the deltas will be saved

    overwrite: bool
    If True, overwrite a previous run in the saved in the same output
    directory. Does not have any effect if the folder `out_dir` does not
    exist.
    """
    def __init__(self, filename):
        """Initializes class instance

        Arguments
        ---------
        filename: str
        Name of the config file
        """
        self.logger = logging.getLogger(__name__)

        self.config = ConfigParser()
        # load default configuration
        self.config.read_dict(default_config)
        # now read the configuration file
        self.config.read(filename)

        # parse the environ variables
        self.__parse_environ_variables()

        # format the sections
        self.overwrite = None
        self.log = None
        self.logging_level_console = None
        self.logging_level_file = None
        self.__format_general_section()
        self.corrections = None
        self.num_corrections = None
        self.__format_corrections_section()
        self.masks = None
        self.num_masks = None
        self.__format_masks_section()
        self.data = None
        self.__format_data_section()
        self.expected_flux = None
        self.__format_expected_flux_section()

        # initialize folders where data will be saved
        self.initialize_folders()

        # setup logger
        setup_logger(logging_level_console=self.logging_level_console,
                     log_file=self.log,
                     logging_level_file=self.logging_level_file)

    def __format_corrections_section(self):
        """Format the corrections section of the parser into usable data

        Raise
        -----
        ConfigError if the config file is not correct
        """
        self.corrections = []
        if "corrections" not in self.config:
            self.logger.warning("Missing section [corrections]. No Corrections will"
                                "be applied to data")
        section = self.config["corrections"]
        self.num_corrections = section.getint("num corrections")
        if self.num_corrections is None:
            raise ConfigError("In section 'corrections', variable 'num corrections' "
                              "is required")
        if self.num_corrections < 0:
            raise ConfigError("In section 'corrections', variable 'num corrections' "
                              "must be a non-negative integer")
        for correction_index in range(self.num_corrections):
            # first load the correction class
            correction_name = section.get(f"type {correction_index}")
            if correction_name is None:
                raise ConfigError("In section [corrections], missing variable "
                                  f"[type {correction_index}]")
            module_name = section.get(f"module name {correction_index}")
            if module_name is None:
                module_name = re.sub('(?<!^)(?=[A-Z])', '_', correction_name).lower()
                module_name = f"picca.delta_extraction.corrections.{module_name.lower()}"
            try:
                CorrectionType, default_args = class_from_string(correction_name,
                                                                 module_name)
            except ImportError:
                raise ConfigError(f"Error loading class {correction_name}, "
                                  f"module {module_name} could not be loaded")
            except AttributeError:
                raise ConfigError(f"Error loading class {correction_name}, "
                                  f"module {module_name} did not contain class")

            # now load the arguments with which to initialize this class
            if f"correction arguments {correction_index}" not in self.config:
                self.logger.warning(f"Missing section [correction arguments {correction_index}]. "
                                    f"Correction {correction_name} will be called without "
                                    "arguments")
                self.config.read_dict({f"correction arguments {correction_index}":{}})
            correction_args = self.config[f"correction arguments {correction_index}"]

            # update the section adding the default choices when necessary
            for key, value in default_args.items():
                if key not in correction_args:
                    correction_args[key] = str(value)

            # finally add the correction to self.corrections
            self.corrections.append((CorrectionType, correction_args))

    def __format_data_section(self):
        """Format the data section of the parser into usable data

        Raise
        -----
        ConfigError if the config file is not correct
        """
        if "data" not in self.config:
            raise ConfigError("Missing section [data]")
        section = self.config["data"]

        # first load the data class
        data_name = section.get("type")
        module_name = section.get("module name")
        if module_name is None:
            module_name = re.sub('(?<!^)(?=[A-Z])', '_', data_name).lower()
            module_name = f"picca.delta_extraction.data_catalogues.{module_name.lower()}"
        try:
            DataType, default_args = class_from_string(data_name, module_name)
        except ImportError:
            raise ConfigError(f"Error loading class {data_name}, "
                              f"module {module_name} could not be loaded")
        except AttributeError:
            raise ConfigError(f"Error loading class {data_name}, "
                              f"module {module_name} did not contain class")

        # update the section adding the default choices when necessary
        for key, value in default_args.items():
            if key not in section:
                section[key] = str(value)

        # finally add the information to self.data
        self.data = (DataType, section)

    def __format_expected_flux_section(self):
        """Format the expected flux section of the parser into usable data

        Raise
        -----
        ConfigError if the config file is not correct
        """
        if "expected flux" not in self.config:
            raise ConfigError("Missing section [expected flux]")
        section = self.config["expected flux"]

        # first load the data class
        expected_flux_name = section.get("type")
        module_name = section.get("module name")
        if module_name is None:
            module_name = re.sub('(?<!^)(?=[A-Z])', '_', expected_flux_name).lower()
            module_name = f"picca.delta_extraction.expected_fluxes.{module_name.lower()}"
        try:
            ExpectedFluxType, default_args = class_from_string(expected_flux_name,
                                                               module_name)
        except ImportError:
            raise ConfigError(f"Error loading class {expected_flux_name}, "
                              f"module {module_name} could not be loaded")
        except AttributeError:
            raise ConfigError(f"Error loading class {expected_flux_name}, "
                              f"module {module_name} did not contain class")

        # update the section adding the default choices when necessary
        for key, value in default_args.items():
            if key not in section:
                section[key] = str(value)

        # finally add the information to self.continua
        self.expected_flux = (ExpectedFluxType, section)

    def __format_general_section(self):
        """Format the general section of the parser into usable data

        Raise
        -----
        ConfigError if the config file is not correct
        """
        if "general" not in self.config:
            raise ConfigError("Missing section [general]")
        section = self.config["general"]
        self.out_dir = section.get("out dir")
        if self.out_dir is None:
            raise ConfigError("In section 'general', variable 'out dir' is required")
        self.overwrite = section.getboolean("overwrite")

        self.log = section.get("log")
        if self.log is not None and not (self.log.startswith(".") or self.log.startswith("/")):
            self.log = self.out_dir + "Log/" + self.log
            section["log"] = self.log

        self.logging_level_console = section.get("logging level console")
        if self.logging_level_console is None:
            raise ConfigError("In section 'general', variable 'logging level console' is required")
        self.logging_level_console = self.logging_level_console.upper()

        self.logging_level_file = section.get("logging level file")
        if self.logging_level_file is None:
            raise ConfigError("In section 'general', variable 'logging level file' is required")
        self.logging_level_file = self.logging_level_file.upper()

    def __format_masks_section(self):
        """Format the masks section of the parser into usable data

        Raise
        -----
        ConfigError if the config file is not correct
        """
        self.masks = []
        if "masks" not in self.config:
            self.logger.warning("Missing section [masks]. No Masks will"
                                "be applied to data")
            return
        section = self.config["masks"]
        self.num_masks = section.getint("num masks")
        for mask_index in range(self.num_masks):
            # first load the mask class
            mask_name = section.get(f"type {mask_index}")
            if mask_name is None:
                raise ConfigError("In section [masks], missing variable [type "
                                  f"{mask_index}]")
            module_name = section.get(f"module name {mask_index}")
            if module_name is None:
                module_name = re.sub('(?<!^)(?=[A-Z])', '_', mask_name).lower()
                module_name = f"picca.delta_extraction.masks.{module_name.lower()}"
            try:
                MaskType, default_args = class_from_string(mask_name, module_name)
            except ImportError:
                raise ConfigError(f"Error loading class {mask_name}, "
                                  f"module {module_name} could not be loaded")
            except AttributeError:
                raise ConfigError(f"Error loading class {mask_name}, "
                                  f"module {module_name} did not contain class")

            # now load the arguments with which to initialize this class
            if f"mask arguments {mask_index}" not in self.config:
                self.logger.warning(f"Missing section [mask arguments {mask_index}]. "
                                    f"Correction {mask_name} will be called without "
                                    "arguments")
                self.config.read_dict({f"mask arguments {mask_index}":{}})
            mask_args = self.config[f"mask arguments {mask_index}"]

            # update the section adding the default choices when necessary
            for key, value in default_args.items():
                if key not in mask_args:
                    mask_args[key] = str(value)

            # finally add the correction to self.masks
            self.masks.append((MaskType, mask_args))

    def __parse_environ_variables(self):
        """Read all variables and replaces the enviroment variables for their
        actual values. This assumes that enviroment variables are only used
        at the beggining of the paths.

        Raise
        -----
        ConfigError if an environ variable was not defined
        """
        for section in self.config:
            for key, value in self.config[section].items():
                if value.startswith("$"):
                    pos = value.find("/")
                    if os.getenv(value[1:pos]) is None:
                        raise ConfigError(f"In section [{section}], undefined "
                                          f"environment variable {value[:pos]} "
                                          "was found")
                    self.config[section][key] = value.replace(value[:pos],
                                                              os.getenv(value[1:pos]))

    def initialize_folders(self):
        """Initialize output folders

        Raise
        -----
        ConfigError if the output path was already used and the
        overwrite is not selected
        """
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)
            os.makedirs(self.out_dir+"Delta/")
            os.makedirs(self.out_dir+"Log/")
            self.write_config()
        elif self.overwrite:
            self.write_config()
        else:
            raise ConfigError("Specified folder contains a previous run."
                              "Pass overwrite option in configuration file"
                              "in order to ignore the previous run or"
                              "change the output path variable to point "
                              "elsewhere")

    def write_config(self):
        """This function writes the configuration options for later
        usages. The file is saved under the name .config.ini and in
        the self.out_dir folder
        """
        config_file = open(f"{self.out_dir}/.config.ini", 'w')
        self.config.write(config_file)
        config_file.close()
