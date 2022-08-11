"""This file contains configuration tests"""
import os
import unittest
from configparser import ConfigParser

import numpy as np

from picca.delta_extraction.config import Config
from picca.delta_extraction.config import accepted_corrections_options
from picca.delta_extraction.config import accepted_masks_options
from picca.delta_extraction.corrections.dust_correction import defaults as defaults_dust_correction
from picca.delta_extraction.errors import ConfigError
from picca.delta_extraction.masks.dla_mask import defaults as defaults_dla_mask
from picca.tests.delta_extraction.abstract_test import AbstractTest
from picca.tests.delta_extraction.test_utils import reset_logger

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ["THIS_DIR"] = THIS_DIR

class ConfigTest(AbstractTest):
    """Test the configuration.

    Methods
    -------
    compare_ascii (from AbstractTest)
    compare_fits (from AbstractTest)
    setUp (from AbstractTest)
    test_config
    """
    def check_error(self, in_file, expected_message, startswith=False):
        """Load a Configuration instance expecting an error
        Check the error message

        Arguments
        ---------
        in_file: str
        Input configuration file to construct the Configuration instance

        expected_message: str
        Expected error message

        startswith: bool - Default: False
        If True, check that expected_message is the beginning of the actual error
        message. Otherwise check that expected_message is the entire message
        """
        with self.assertRaises(ConfigError) as context_manager:
            config = Config(in_file)

        self.compare_error_message(context_manager, expected_message,
                                   startswith=startswith)

    def compare_config(self, orig_file, new_file):
        """Compares two configuration files to check that they are equal

        Arguments
        ---------
        orig_file: str
        Control file

        new_file: str
        New file
        """
        orig_config = ConfigParser()
        orig_config.read(orig_file)
        new_config = ConfigParser()
        new_config.read(new_file)

        for section in orig_config.sections():
            if not section in new_config.sections():
                print(f"Section [{section}] missing on {new_file}")
            self.assertTrue(section in new_config.sections())
            orig_section = orig_config[section]
            new_section = new_config[section]
            if section == "run specs":
                for key in orig_section.keys():
                    self.assertTrue(key in new_section.keys())
            else:
                for key, orig_value in orig_section.items():
                    if key not in new_section.keys():
                        print(f"key '{key}' in section [{new_section}] missing in "
                              "new file")
                    self.assertTrue(key in new_section.keys())
                    new_value = new_section.get(key)
                    # this is necessary to remove the system dependent bits of
                    # the paths
                    base_path = "py/picca/tests/delta_extraction"
                    if base_path in new_value:
                        new_value = new_value.split(base_path)[-1]
                        orig_value = orig_value.split(base_path)[-1]

                    if not orig_value == new_value:
                        print(f"In section [{section}], for key '{key}'' found "
                              f"orig value = {orig_value} but new value = "
                              f"{new_value}")
                    self.assertTrue(orig_value == new_value)
            for key in new_section.keys():
                if key not in orig_section.keys():
                    print(f"key '{key}' in section [{section}] missing in original "
                          "file")
                    for key2, value in new_section.items():
                        print(key2, value)
                self.assertTrue(key in orig_section.keys())

        for section in new_config.sections():
            if not section in orig_config.sections():
                print(f"Section [{section}] missing on '{orig_file}'")

            self.assertTrue(section in orig_config.sections())

    def test_config(self):
        """Basic test for config.

        Load a config file and then print it
        """
        in_file = f"{THIS_DIR}/data/config_overwrite.ini"
        out_file = f"{THIS_DIR}/results/config_tests/.config.ini"
        test_file = f"{THIS_DIR}/data/.config.ini"
        out_warning_file = f"{THIS_DIR}/results/config_tests/Log/run.log"
        test_warning_file = f"{THIS_DIR}/data/config_test.txt"

        config = Config(in_file)
        config.write_config()
        self.compare_config(test_file, out_file)

        reset_logger()
        self.compare_ascii(test_warning_file, out_warning_file)

        # this should raise an error as folder exists and overwrite is False
        in_file = f"{THIS_DIR}/data/config.ini"
        expected_message = (
            "Specified folder contains a previous run. Pass overwrite "
            "option in configuration file in order to ignore the "
            "previous run or change the output path variable to point "
            f"elsewhere. Folder: {THIS_DIR}/results/config_tests/"
        )
        self.check_error(in_file, expected_message)

        # this should not raise an error as folder exists and overwrite is True
        in_file = f"{THIS_DIR}/data/config_overwrite.ini"
        config = Config(in_file)

    def test_config_check_defaults_overwrite(self):
        """ Test that passing default values are not overwriting choices"""
        folder = f"{THIS_DIR}/data/config_extra/"
        # check that default values do not overwrite chosen options
        # corrections section
        in_file = f"{folder}/config_check_defaults_overwrite.ini"
        config = Config(in_file)

        # check the corrections section
        correction_args = config.corrections[2][1]
        self.assertTrue(np.isclose(
            correction_args.getfloat("extinction_conversion_r"),
            2.0
        ))

        # check the mask section
        mask_args = config.masks[0][1]
        self.assertTrue(np.isclose(
            mask_args.getfloat("dla mask limit"),
            1.0
        ))

        # check that out dir has an ending /
        self.assertTrue(config.out_dir.endswith("/"))

    def test_config_missing_arguments_section(self):
        """ Test that not passing sections [arguments correction 0] and
        [arguments mask 0] behaves as expected"""
        folder = f"{THIS_DIR}/data/config_extra/"
        # check that default values do not overwrite chosen options
        # corrections section
        in_file = f"{folder}/config_missing_arguments_section.ini"
        config = Config(in_file)

        # check corrections dictionary
        correction_args0 = config.corrections[0][1]
        self.assertTrue(len(correction_args0) == 0)
        correction_args1 = config.corrections[1][1]
        self.assertTrue(len(correction_args1) == 1)
        self.assertTrue(np.isclose(
            correction_args1.getfloat("extinction_conversion_r"),
            defaults_dust_correction.get("extinction_conversion_r")
        ))

        # check masks dictionary
        mask_args0 = config.masks[0][1]
        self.assertTrue(len(mask_args0) == 3)
        self.assertTrue(np.isclose(
            mask_args0.getfloat("dla mask limit"),
            defaults_dla_mask.get("dla mask limit")
        ))

    def test_config_invalid_correction_options(self):
        """ Test that passing invalid options to the correction classes
        raise errors """
        prefix = f"{THIS_DIR}/data/config_wrong_options/config_wrong_options"

        # check corrections section
        in_file = f"{prefix}_corrections.ini"
        expected_message = (
            "Unrecognised option in section [corrections]. "
            f"Found: 'name 0'. Accepted options are "
            f"{accepted_corrections_options}"
        )
        self.check_error(in_file, expected_message)

        # check bad correction
        in_file = f"{prefix}_corrections_bad_inheritance.ini"
        expected_message = (
            "Error loading class Mask. "
            "This class should inherit from Correction but "
            "it does not. Please check for correct inheritance "
            "pattern."
        )
        self.check_error(in_file, expected_message)

        # check case num_corrections is missing
        in_file = f"{prefix}_corrections_no_num_corrections.ini"
        expected_message = (
            "In section [corrections], variable 'num corrections' is required"
        )
        self.check_error(in_file, expected_message)

        # check case num_corrections is not positive
        in_file = f"{prefix}_corrections_num_corrections.ini"
        expected_message = (
            "In section [corrections], variable 'num corrections' "
            "must be a non-negative integer"
        )
        self.check_error(in_file, expected_message)

        # check case missing type
        in_file = f"{prefix}_corrections_no_type.ini"
        expected_message = (
            "In section [corrections], missing variable [type 0]"
        )
        self.check_error(in_file, expected_message)

        # check case type is not correct
        in_file = f"{prefix}_corrections_bad_type.ini"
        expected_message = (
            "Unrecognised option in section [corrections]. "
            f"Found: 'type a'. Accepted options are "
            f"{accepted_corrections_options}"
        )
        self.check_error(in_file, expected_message)

        # check case number in type is too large
        in_file = f"{prefix}_corrections_bad_type2.ini"
        expected_message = (
            "In section [corrections] found option 'type 3', but "
            "'num corrections' is '1' (keep in mind python zero indexing)"
        )
        self.check_error(in_file, expected_message)

        # check case module name is not correct
        in_file = f"{prefix}_corrections_bad_module_name.ini"
        expected_message = (
            "Unrecognised option in section [corrections]. "
            f"Found: 'module name a'. Accepted options are "
            f"{accepted_corrections_options}"
        )
        self.check_error(in_file, expected_message)

        # check case number in module name is too large
        in_file = f"{prefix}_corrections_bad_module_name2.ini"
        expected_message = (
            "In section [corrections] found option 'module name 3', but "
            "'num corrections' is '1' (keep in mind python zero indexing)"
        )
        self.check_error(in_file, expected_message)

        # check case module does not exist
        in_file = f"{prefix}_corrections_inexistent_module_name.ini"
        expected_message = (
            f"Error loading class Correction, "
            f"module picca.delta_extraction.fake_correction could not be loaded"
        )
        self.check_error(in_file, expected_message)

        # check case module does not contain the class
        in_file = f"{prefix}_corrections_module_no_class.ini"
        expected_message = (
            "Error loading class DustCorrection, "
            "module picca.delta_extraction.corrections.calibration_correction "
            "did not contain requested class"
        )
        self.check_error(in_file, expected_message)

        # now check arguments of the different Correction child classes
        expected_message = (
            "Unrecognised option in section [correction arguments 0]"
        )

        # check arguments of CalibrationCorrection
        in_file = f"{prefix}_calibration_correction.ini"
        self.check_error(in_file, expected_message, startswith=True)

        # check arguments of DustCorrection
        in_file = f"{prefix}_dust_correction.ini"
        self.check_error(in_file, expected_message, startswith=True)

        # check arguments of IvarCorrection
        in_file = f"{prefix}_ivar_correction.ini"
        self.check_error(in_file, expected_message, startswith=True)

        # check arguments of OpticalDepthCorrection
        in_file = f"{prefix}_optical_depth_correction.ini"
        self.check_error(in_file, expected_message, startswith=True)

    def test_config_invalid_data_options(self):
        """ Test that passing invalid options to the data classes raise errors"""
        prefix = f"{THIS_DIR}/data/config_wrong_options/config_wrong_options"

        # missing section
        in_file = f"{prefix}_no_data.ini"
        expected_message = "Missing section [data]"
        self.check_error(in_file, expected_message)

        # check bad data
        in_file = f"{prefix}_data_bad_inheritance.ini"
        expected_message = (
            "Error loading class Mask. "
            "This class should inherit from Data but "
            "it does not. Please check for correct inheritance "
            "pattern."
        )
        self.check_error(in_file, expected_message)

        # missing type
        in_file = f"{prefix}_data_no_type.ini"
        expected_message = "In section [data], variable 'type' is required"
        self.check_error(in_file, expected_message)

        # check case module does not exist
        in_file = f"{prefix}_data_no_module.ini"
        expected_message = (
            f"Error loading class Data, "
            f"module picca.fake_data could not be loaded"
        )
        self.check_error(in_file, expected_message)

        # check case module does not contain the class
        in_file = f"{prefix}_data_module_no_class.ini"
        expected_message = (
            "Error loading class SdssData, "
            "module picca.delta_extraction.data_catalogues.desi_data "
            "did not contain requested class"
        )
        self.check_error(in_file, expected_message)

        # now check arguments of the different Data child classes
        expected_message = "Unrecognised option in section [data]"

        # check arguments of DesiHealpix
        in_file = f"{prefix}_desi_healpix.ini"
        self.check_error(in_file, expected_message, startswith=True)

        # check arguments of DesiTile
        in_file = f"{prefix}_desi_tile.ini"
        self.check_error(in_file, expected_message, startswith=True)

        # check arguments of DesisimMocks
        in_file = f"{prefix}_desisim_mocks.ini"
        self.check_error(in_file, expected_message, startswith=True)

        # check arguments of SdssData
        in_file = f"{prefix}_sdss_data.ini"
        self.check_error(in_file, expected_message, startswith=True)

    def test_config_invalid_expected_flux_options(self):
        """ Test that passing invalid options to the expected flux classes
        raise errors """
        prefix = f"{THIS_DIR}/data/config_wrong_options/config_wrong_options"

        # missing section
        in_file = f"{prefix}_no_expected_flux.ini"
        expected_message = "Missing section [expected flux]"
        self.check_error(in_file, expected_message)

        # check bad expected_flux
        in_file = f"{prefix}_expected_flux_bad_inheritance.ini"
        expected_message = (
            "Error loading class Mask. "
            "This class should inherit from ExpectedFlux but "
            "it does not. Please check for correct inheritance "
            "pattern."
        )
        self.check_error(in_file, expected_message)

        # missing type
        in_file = f"{prefix}_expected_flux_no_type.ini"
        expected_message = "In section [expected flux], variable 'type' is required"
        self.check_error(in_file, expected_message)

        # check case module does not exist
        in_file = f"{prefix}_expected_flux_no_module.ini"
        expected_message = (
            f"Error loading class Dr16ExpectedFlux, "
            f"module picca.fake_expected_flux could not be loaded"
        )
        self.check_error(in_file, expected_message)

        # check case module does not contain the class
        in_file = f"{prefix}_expected_flux_module_no_class.ini"
        expected_message = (
            "Error loading class Dr16ExpectedFlux, "
            "module picca.delta_extraction.expected_fluxes.true_continuum "
            "did not contain requested class"
        )
        self.check_error(in_file, expected_message)

        # now check arguments of the different Data child classes
        expected_message = "Unrecognised option in section [expected flux]"

        # check arguments of Dr16ExpectedFlux
        in_file = f"{prefix}_dr16_expected_flux.ini"
        self.check_error(in_file, expected_message, startswith=True)

        # check arguments of TrueContinuum
        in_file = f"{prefix}_true_continuum.ini"
        self.check_error(in_file, expected_message, startswith=True)

    def test_config_invalid_general_options(self):
        """ Test that passing invalid options to the general section raise errors """
        prefix = f"{THIS_DIR}/data/config_wrong_options/config_wrong_options"

        # missing out dir
        in_file = f"{prefix}_general_no_out_dir.ini"
        expected_message = "Missing variable 'out dir' in section [general]"
        self.check_error(in_file, expected_message)

        # invalid log
        in_file = f"{prefix}_general_invalid_log.ini"
        expected_message = (
            "Variable 'log' in section [general] should not incude folders. "
            "Found: log/my_log.log"
        )
        self.check_error(in_file, expected_message)

        # now check arguments of the general section
        expected_message = "Unrecognised option in section [general]"

        in_file = f"{prefix}_general.ini"
        self.check_error(in_file, expected_message, startswith=True)

    def test_config_invalid_mask_options(self):
        """ Test that passing invalid options to the mask classes raise errors """
        prefix = f"{THIS_DIR}/data/config_wrong_options/config_wrong_options"

        # firt check masks section
        in_file = f"{prefix}_masks.ini"
        expected_message = (
            "Unrecognised option in section [masks]. Found: 'name 0'. "
            "Accepted options are "
            f"{accepted_masks_options}"
        )
        self.check_error(in_file, expected_message)

        # check bad mask
        in_file = f"{prefix}_mask_bad_inheritance.ini"
        expected_message = (
            "Error loading class Correction. "
            "This class should inherit from Mask but "
            "it does not. Please check for correct inheritance "
            "pattern."
        )
        self.check_error(in_file, expected_message)

        # check case num_masks is missing
        in_file = f"{prefix}_masks_no_num_masks.ini"
        expected_message = (
            "In section [masks], variable 'num masks' is required"
        )
        self.check_error(in_file, expected_message)

        # check case num_masks is not positive
        in_file = f"{prefix}_masks_num_masks.ini"
        expected_message = (
            "In section [masks], variable 'num masks' "
            "must be a non-negative integer"
        )
        self.check_error(in_file, expected_message)

        # check case missing type
        in_file = f"{prefix}_masks_no_type.ini"
        expected_message = (
            "In section [masks], missing variable [type 0]"
        )
        self.check_error(in_file, expected_message)

        # check case type is not correct
        in_file = f"{prefix}_masks_bad_type.ini"
        expected_message = (
            "Unrecognised option in section [masks]. "
            f"Found: 'type a'. Accepted options are "
            f"{accepted_masks_options}"
        )
        self.check_error(in_file, expected_message)

        # check case number in type is too large
        in_file = f"{prefix}_masks_bad_type2.ini"
        expected_message = (
            "In section [masks] found option 'type 3', but "
            "'num masks' is '1' (keep in mind python zero indexing)"
        )
        self.check_error(in_file, expected_message)

        # check case module name is not correct
        in_file = f"{prefix}_masks_bad_module_name.ini"
        expected_message = (
            "Unrecognised option in section [masks]. "
            f"Found: 'module name a'. Accepted options are "
            f"{accepted_masks_options}"
        )
        self.check_error(in_file, expected_message)

        # check case number in module name is too large
        in_file = f"{prefix}_masks_bad_module_name2.ini"
        expected_message = (
            "In section [masks] found option 'module name 3', but "
            "'num masks' is '1' (keep in mind python zero indexing)"
        )
        self.check_error(in_file, expected_message)

        # check case module does not exist
        in_file = f"{prefix}_masks_inexistent_module_name.ini"
        expected_message = (
            f"Error loading class Mask, "
            f"module picca.delta_extraction.fake_mask could not be loaded"
        )
        self.check_error(in_file, expected_message)

        # check case module does not contain the class
        in_file = f"{prefix}_masks_module_no_class.ini"
        expected_message = (
            "Error loading class BalMask, "
            "module picca.delta_extraction.masks.dla_mask "
            "did not contain requested class"
        )
        self.check_error(in_file, expected_message)

        # now check arguments of the different Correction child classes
        expected_message = (
            "Unrecognised option in section [mask arguments 0]"
        )

        # check arguments of LinesMask
        in_file = f"{prefix}_lines_mask.ini"
        self.check_error(in_file, expected_message, startswith=True)

        # check arguments of AbsorberMask
        in_file = f"{prefix}_absorber_mask.ini"
        self.check_error(in_file, expected_message, startswith=True)

        # check arguments of BalMask
        in_file = f"{prefix}_bal_mask.ini"
        self.check_error(in_file, expected_message, startswith=True)

        # check arguments of DlaMask
        in_file = f"{prefix}_dla_mask.ini"
        self.check_error(in_file, expected_message, startswith=True)

    def test_config_no_file(self):
        """Check behaviour of config when the file is not valid"""
        in_file = f"{THIS_DIR}/data/non_existent_config_overwrite.ini"
        expected_message = f"Config file not found: {in_file}"
        self.check_error(in_file, expected_message)

    def test_config_undefined_environment_variable(self):
        """Check the behaviour for undefined environment variables"""
        prefix = f"{THIS_DIR}/data/config_wrong_options/config_wrong_options"
        in_file = f"{prefix}_undefined_environment.ini"
        expected_message = (
            "In section [general], undefined environment variable UNDEFINED "
            "was found"
        )
        self.check_error(in_file, expected_message)

if __name__ == '__main__':
    unittest.main()
