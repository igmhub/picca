"""This module defines the Config class.
This class is responsible for managing the options selected for the user and
contains the default configuration.
"""
import os
import re
import warnings
from configparser import ConfigParser

from picca.delta_extraction.errors import ConfigError, ConfigWarning
from picca.delta_extraction.utils import class_from_string

default_config = {
    "general": {
        "overwrite": False,
        "quiet": False,
    },
    "data": {
    },
    "corrections":{
    },
    "masks": {
    },
    "expected flux":{
    },
    "empty":{
    },
}

class Config:
    """Class to manage the configuration file

    Methods
    -------
    __init__
    __format_continua_section
    __format_correction_section
    __format_expected_flux
    __format_data_section
    __format_general_section
    __format_masks_section
    __parse_environ_variables
    write_config

    Attributes
    ---------
    config: ConfigParser
    A ConfigParser instance with the user configuration

    continua: (class, dict)
    Class should be a child of Continua and dict a dictionary with the keyword
    arguments necesssary to initialize it

    corrections: list of (class, dict)
    A list of (class, dict). For each element, class should be a child of
    Correction and dict a dictionary with the keyword arguments necesssary
    to initialize it

    data: (class, dict)
    Class should be a child of Forest and dict a dictionary with the keyword
    arguments necesssary to initialize it

    log: str or None
    Name of the log file. None for no log file

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

    quiet: bool
    Printing flag. If True, no information will be printed

    """
    def __init__(self, filename):
        """Initializes class instance

        Arguments
        ---------
        filename: str
        Name of the config file
        """
        self.config = ConfigParser()
        # load default configuration
        self.config.read_dict(default_config)
        # now read the configuration file
        self.config.read(filename)

        # parse the environ variables
        self.__parse_environ_variables()

        # format the sections
        self.overwrite = None
        self.quiet = None
        self.log = None
        self.__format_general_section()
        self.corrections = None
        self.num_corrections = None
        self.__format_corrections_section()
        self.masks = None
        self.num_masks = None
        self.__format_masks_section()
        self.data = None
        self.__format_data_section()
        self.mean_expected_flux = None
        self.__format_expected_flux()

    def __format_corrections_section(self):
        """Formats the corrections section of the parser into usable data

        Raises
        ------
        ConfigError if the config file is not correct

        Warnings
        --------
        ConfigWarning if no arguments were found to pass to CorrectionType
        """
        self.corrections = []
        if "corrections" not in self.config:
            warnings.warn("Missing section [corrections]. No Corrections will"
                          "be applied to data", ConfigWarning)
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
                CorrectionType = class_from_string(correction_name, module_name)
            except ImportError:
                raise ConfigError(f"Error loading class {correction_name}, "
                                  f"module {module_name} could not be loaded")
            except AttributeError:
                raise ConfigError(f"Error loading class {correction_name}, "
                                  f"module {module_name} did not contain class")

            # now load the arguments with which to initialize this class
            if f"correction arguments {correction_index}" not in self.config:
                warnings.warn(f"Missing section [correction arguments {correction_index}]."
                              f"Correction {correction_name} will be called without "
                              "arguments", ConfigWarning)
                correction_args = self.config["empty"]
            else:
                correction_args = self.config[f"correction arguments {correction_index}"]

            # finally add the correction to self.corrections
            self.corrections.append((CorrectionType, correction_args))

    def __format_data_section(self):
        """Formats the data section of the parser into usable data

        Raises
        ------
        ConfigError if the config file is not correct

        Warnings
        --------
        ConfigWarning if no arguments were found to pass to DataType
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
            DataType = class_from_string(data_name, module_name)
        except ImportError:
            raise ConfigError(f"Error loading class {data_name}, "
                              f"module {module_name} could not be loaded")
        except AttributeError:
            raise ConfigError(f"Error loading class {data_name}, "
                              f"module {module_name} did not contain class")

        # finally add the information to self.data
        self.data = (DataType, section)

    def __format_expected_flux(self):
        """Formats the expected flux section of the parser into usable data

        Raises
        ------
        ConfigError if the config file is not correct

        Warnings
        --------
        ConfigWarning if no arguments were found to pass to ContinuaType
        """
        if "expected flux" not in self.config:
            raise ConfigError("Missing section [expected flux]")
        section = self.config["expected flux"]
        section["out dir"] = self.out_dir

        # first load the data class
        expected_flux_name = section.get("type")
        module_name = section.get("module name")
        if module_name is None:
            module_name = re.sub('(?<!^)(?=[A-Z])', '_', expected_flux_name).lower()
            module_name = f"picca.delta_extraction.expected_fluxes.{module_name.lower()}"
        try:
            ExpectedFluxType = class_from_string(expected_flux_name, module_name)
        except ImportError:
            raise ConfigError(f"Error loading class {expected_flux_name}, "
                              f"module {module_name} could not be loaded")
        except AttributeError:
            raise ConfigError(f"Error loading class {expected_flux_name}, "
                              f"module {module_name} did not contain class")

        # finally add the information to self.continua
        self.expected_flux = (ExpectedFluxType, section)

    def __format_general_section(self):
        """Formats the general section of the parser into usable data

        Raises
        ------
        ConfigError if the config file is not correct

        Warnings
        --------
        ConfigWarning if no arguments were found to pass to MaskType
        """
        if "general" not in self.config:
            raise ConfigError("Missing section [general]")
        section = self.config["general"]
        self.out_dir = section.get("out dir", None)
        if self.out_dir is None:
            raise ConfigError("In section 'general', variable 'out dir' is required")
        self.overwrite = section.getboolean("overwrite")
        self.quiet = section.getboolean("quiet")
        self.log = section.get("log")

    def __format_masks_section(self):
        """Formats the masks section of the parser into usable data

        Raises
        ------
        ConfigError if the config file is not correct
        """
        self.masks = []
        if "masks" not in self.config:
            warnings.warn("Missing section [masks]. No Masks will"
                          "be applied to data", ConfigWarning)
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
                MaskType = class_from_string(mask_name, module_name)
            except ImportError:
                raise ConfigError(f"Error loading class {mask_name}, "
                                  f"module {module_name} could not be loaded")
            except AttributeError:
                raise ConfigError(f"Error loading class {mask_name}, "
                                  f"module {module_name} did not contain class")

            # now load the arguments with which to initialize this class
            if f"mask arguments {mask_index}" not in self.config:
                warnings.warn(f"Missing section [mask arguments {mask_index}]."
                              f"Correction {mask_name} will be called without "
                              "arguments", ConfigWarning)
                mask_args = self.config["empty"]
            else:
                mask_args = self.config[f"mask arguments {mask_index}"]

            # finally add the correction to self.corrections
            #mask = MaskType(**mask_args)
            self.masks.append((MaskType, mask_args))

    def __parse_environ_variables(self):
        """Reads all variables and replaces the enviroment variables for their
        actual values. This assumes that enviroment variables are only used
        at the beggining of the paths.

        Raises
        ------
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

    def write_config(self):
        """This function writes the configuration options for later
        usages. The file is saved under the name .config.ini and in
        the self.out_dir folder
        """
        config_file = open(f"{self.out_dir}/.config.ini", 'w')
        self.config.write(config_file)
        config_file.close()
