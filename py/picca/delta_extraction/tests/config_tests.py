"""This file contains configuration tests"""
import os
import unittest
from configparser import ConfigParser

from picca.delta_extraction.config import Config
from picca.delta_extraction.errors import ConfigError
from picca.delta_extraction.tests.abstract_test import AbstractTest
from picca.delta_extraction.tests.test_utils import reset_logger
from picca.delta_extraction.utils import setup_logger

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ["THIS_DIR"] = THIS_DIR

class ConfigurationTest(AbstractTest):
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
                if not key in orig_section.keys():
                    print(f"key {key} in section {section} missing on {new_file}")

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

        #setup_logger(log_file=out_warning_file)

        config = Config(in_file)
        config.write_config()
        self.compare_config(test_file, out_file)

        reset_logger()
        self.compare_ascii(test_warning_file, out_warning_file)

        in_file = f"{THIS_DIR}/data/config.ini"
        with self.assertRaises(ConfigError):
            config = Config(in_file)


if __name__ == '__main__':
    unittest.main()
