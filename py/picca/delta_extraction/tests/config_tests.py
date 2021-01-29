"""This file contains configuration tests"""
import unittest
import os

from picca.delta_extraction.config import Config
from picca.delta_extraction.errors import ConfigWarning

from picca.delta_extraction.tests.abstract_test import AbstractTest

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ["THIS_DIR"] = THIS_DIR

class TestConfiguration(AbstractTest):
    """Test the configuration."""

    def test_config(self):
        """Basic test for config.

        Load a config file and then print it
        """
        in_file = f"{THIS_DIR}/data/config.ini"
        out_file = f"{THIS_DIR}/results/.config.ini"
        test_file = f"{THIS_DIR}/data/config.ini"

        with self.assertWarns(ConfigWarning):
            config = Config(in_file)
        config.write_config()
        self.compare_ascii(test_file, out_file, expand_dir=True)

if __name__ == '__main__':
    unittest.main()
