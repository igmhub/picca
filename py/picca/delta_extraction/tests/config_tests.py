"""This file contains configuration tests"""
import os
import unittest

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

    def test_config(self):
        """Basic test for config.

        Load a config file and then print it
        """
        in_file = f"{THIS_DIR}/data/config_overwrite.ini"
        out_file = f"{THIS_DIR}/results/.config.ini"
        test_file = f"{THIS_DIR}/data/config_overwrite.ini"
        out_warning_file = f"{THIS_DIR}/results/config_test.txt"
        test_warning_file = f"{THIS_DIR}/data/config_test.txt"

        setup_logger(log_file=out_warning_file)

        config = Config(in_file)
        config.write_config()
        self.compare_ascii(test_file, out_file, expand_dir=True)

        reset_logger()
        self.compare_ascii(test_warning_file, out_warning_file, expand_dir=True)

        in_file = f"{THIS_DIR}/data/config.ini"
        with self.assertRaises(ConfigError):
            config = Config(in_file)


if __name__ == '__main__':
    unittest.main()
