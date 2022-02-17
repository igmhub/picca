"""This file contains configuration tests"""
import os
import unittest
from configparser import ConfigParser

from picca.delta_extraction.config import Config
from picca.delta_extraction.errors import ConfigError
from picca.tests.delta_extraction.abstract_test import AbstractTest
from picca.tests.delta_extraction.test_utils import reset_logger
from picca.delta_extraction.utils import setup_logger

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
                print(f"Section {section} missing on {new_file}")
            self.assertTrue(section in new_config.sections())
            orig_section = orig_config[section]
            new_section = new_config[section]
            if section == "run specs":
                for key in orig_section.keys():
                    self.assertTrue(key in new_section.keys())
            else:
                for key, orig_value in orig_section.items():
                    if key not in new_section.keys():
                        print(f"key {key} in section {new_section} missing in new file")
                        self.assertTrue(key in new_section.keys())
                    new_value = new_section.get(key)
                    # this is necessary to remove the system dependent bits of
                    # the paths
                    if "py/picca/delta_extraction/tests" in new_value:
                        new_value = new_value.split("py/picca/delta_extraction/tests")[-1]
                        orig_value = orig_value.split("py/picca/delta_extraction/tests")[-1]

                    if not orig_value == new_value:
                        print(f"For key {key} found orig value = {orig_value} but new value = {new_value}")
                    self.assertTrue(orig_value == new_value)
            for key in new_section.keys():
                if key not in orig_section.keys():
                    print(f"key {key} in section {new_section} missing in original file")
                    self.assertTrue(key in orig_section.keys())

        for section in new_config.sections():
            if not section in orig_config.sections():
                print(f"Section {section} missing on {orig_file}")

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

        in_file = f"{THIS_DIR}/data/config.ini"
        with self.assertRaises(ConfigError):
            config = Config(in_file)

    def test_config_invalid_correction_options(self):
        """ Test that passing invalid options to the correction classes raise errors """

        # firt check corrections section
        in_file = f"{THIS_DIR}/data/config_wrong_options/config_wrong_options_corrections.ini"
        with self.assertRaises(ConfigError) as context_manager:
            config = Config(in_file)
        if not str(context_manager.exception).startswith("Unrecognised option in section [corrections]"):
            print(context_manager.exception)
            self.assertTrue(str(context_manager.exception).startswith("Unrecognised option in section [corrections]"))

        # check arguments of CalibrationCorrection
        in_file = f"{THIS_DIR}/data/config_wrong_options/config_wrong_options_calibration_correction.ini"
        with self.assertRaises(ConfigError) as context_manager:
            config = Config(in_file)
        if not str(context_manager.exception).startswith("Unrecognised option in section [correction arguments 0]"):
            print(context_manager.exception)
            self.assertTrue(str(context_manager.exception).startswith("Unrecognised option in section [correction arguments 0]"))

        # check arguments of DustCorrection
        in_file = f"{THIS_DIR}/data/config_wrong_options/config_wrong_options_dust_correction.ini"
        with self.assertRaises(ConfigError) as context_manager:
            config = Config(in_file)
        if not str(context_manager.exception).startswith("Unrecognised option in section [correction arguments 0]"):
            print(context_manager.exception)
            self.assertTrue(str(context_manager.exception).startswith("Unrecognised option in section [correction arguments 0]"))

        # check arguments of IvarCorrection
        in_file = f"{THIS_DIR}/data/config_wrong_options/config_wrong_options_ivar_correction.ini"
        with self.assertRaises(ConfigError) as context_manager:
            config = Config(in_file)
        if not str(context_manager.exception).startswith("Unrecognised option in section [correction arguments 0]"):
            print(context_manager.exception)
            self.assertTrue(str(context_manager.exception).startswith("Unrecognised option in section [correction arguments 0]"))

        # check arguments of OpticalDepthCorrection
        in_file = f"{THIS_DIR}/data/config_wrong_options/config_wrong_options_optical_depth_correction.ini"
        with self.assertRaises(ConfigError) as context_manager:
            config = Config(in_file)
        if not str(context_manager.exception).startswith("Unrecognised option in section [correction arguments 0]"):
            print(context_manager.exception)
            self.assertTrue(str(context_manager.exception).startswith("Unrecognised option in section [correction arguments 0]"))

    def test_config_invalid_data_options(self):
        """ Test that passing invalid options to the data classes raise errors """

        # check arguments of DesiHealpix
        in_file = f"{THIS_DIR}/data/config_wrong_options/config_wrong_options_desi_healpix.ini"
        with self.assertRaises(ConfigError) as context_manager:
            config = Config(in_file)
        if not str(context_manager.exception).startswith("Unrecognised option in section [data]"):
            print(context_manager.exception)
            self.assertTrue(str(context_manager.exception).startswith("Unrecognised option in section [data]"))

        # check arguments of DesiTile
        in_file = f"{THIS_DIR}/data/config_wrong_options/config_wrong_options_desi_tile.ini"
        with self.assertRaises(ConfigError) as context_manager:
            config = Config(in_file)
        if not str(context_manager.exception).startswith("Unrecognised option in section [data]"):
            print(context_manager.exception)
            self.assertTrue(str(context_manager.exception).startswith("Unrecognised option in section [data]"))

        # check arguments of DesisimMocks
        in_file = f"{THIS_DIR}/data/config_wrong_options/config_wrong_options_desisim_mocks.ini"
        with self.assertRaises(ConfigError) as context_manager:
            config = Config(in_file)
        if not str(context_manager.exception).startswith("Unrecognised option in section [data]"):
            print(context_manager.exception)
            self.assertTrue(str(context_manager.exception).startswith("Unrecognised option in section [data]"))

        # check arguments of SdssData
        in_file = f"{THIS_DIR}/data/config_wrong_options/config_wrong_options_sdss_data.ini"
        with self.assertRaises(ConfigError) as context_manager:
            config = Config(in_file)
        if not str(context_manager.exception).startswith("Unrecognised option in section [data]"):
            print(context_manager.exception)
            self.assertTrue(str(context_manager.exception).startswith("Unrecognised option in section [data]"))

    def test_config_invalid_expected_flux_options(self):
        """ Test that passing invalid options to the expected flux classes raise errors """
        # check arguments of Dr16ExpectedFlux
        in_file = f"{THIS_DIR}/data/config_wrong_options/config_wrong_options_dr16_expected_flux.ini"
        with self.assertRaises(ConfigError) as context_manager:
            config = Config(in_file)
        if not str(context_manager.exception).startswith("Unrecognised option in section [expected flux]"):
            print(context_manager.exception)
            self.assertTrue(str(context_manager.exception).startswith("Unrecognised option in section [expected flux]"))

    def test_config_invalid_general_options(self):
        """ Test that passing invalid options to the general section raise errors """
        in_file = f"{THIS_DIR}/data/config_wrong_options/config_wrong_options_general.ini"

        with self.assertRaises(ConfigError) as context_manager:
            config = Config(in_file)
        if not str(context_manager.exception).startswith("Unrecognised option in section [general]"):
            print(context_manager.exception)
            self.assertTrue(str(context_manager.exception).startswith("Unrecognised option in section [general]"))

    def test_config_invalid_mask_options(self):
        """ Test that passing invalid options to the mask classes raise errors """

        # firt check masks section
        in_file = f"{THIS_DIR}/data/config_wrong_options/config_wrong_options_masks.ini"
        with self.assertRaises(ConfigError) as context_manager:
            config = Config(in_file)
        if not str(context_manager.exception).startswith("Unrecognised option in section [masks]"):
            print(context_manager.exception)
            self.assertTrue(str(context_manager.exception).startswith("Unrecognised option in section [masks]"))

        # check arguments of LinesMask
        in_file = f"{THIS_DIR}/data/config_wrong_options/config_wrong_options_lines_mask.ini"
        with self.assertRaises(ConfigError) as context_manager:
            config = Config(in_file)
        if not str(context_manager.exception).startswith("Unrecognised option in section [mask arguments 0]"):
            print(context_manager.exception)
            self.assertTrue(str(context_manager.exception).startswith("Unrecognised option in section [mask arguments 0]"))

        # check arguments of AbsorberMask
        in_file = f"{THIS_DIR}/data/config_wrong_options/config_wrong_options_absorber_mask.ini"
        with self.assertRaises(ConfigError) as context_manager:
            config = Config(in_file)
        if not str(context_manager.exception).startswith("Unrecognised option in section [mask arguments 0]"):
            print(context_manager.exception)
            self.assertTrue(str(context_manager.exception).startswith("Unrecognised option in section [mask arguments 0]"))

        # check arguments of BalMask
        in_file = f"{THIS_DIR}/data/config_wrong_options/config_wrong_options_bal_mask.ini"
        with self.assertRaises(ConfigError) as context_manager:
            config = Config(in_file)
        if not str(context_manager.exception).startswith("Unrecognised option in section [mask arguments 0]"):
            print(context_manager.exception)
            self.assertTrue(str(context_manager.exception).startswith("Unrecognised option in section [mask arguments 0]"))

        # check arguments of DlaMask
        in_file = f"{THIS_DIR}/data/config_wrong_options/config_wrong_options_dla_mask.ini"
        with self.assertRaises(ConfigError) as context_manager:
            config = Config(in_file)
        if not str(context_manager.exception).startswith("Unrecognised option in section [mask arguments 0]"):
            print(context_manager.exception)
            self.assertTrue(str(context_manager.exception).startswith("Unrecognised option in section [mask arguments 0]"))


if __name__ == '__main__':
    unittest.main()
