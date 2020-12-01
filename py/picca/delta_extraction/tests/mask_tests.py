"""This file contains configuration tests"""
import os
import unittest

from picca.delta_extraction.errors import MaskError
from picca.delta_extraction.mask import Mask

from picca.delta_extraction.masks.sdss_dla_mask import SdssDlaMask
from picca.delta_extraction.masks.sdss_absorber_mask import SdssAbsorberMask

from picca.delta_extraction.tests.abstract_test import AbstractTest

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class TestConfiguration(AbstractTest):
    """Test the configuration."""

    def test_mask(self):
        """Tests Abstract class Mask

        Load a Mask instace and check that it cannot be
        initialized.

        #TODO:Check that the function apply_correction
        is correctly implemented in a dummy Forest instance

        """
        with self.assertRaises(MaskError):
            mask = Mask()
            mask.apply_mask("fake data")

    def test_dla_mask(self):
        """Tests correct initialisation and inheritance for class
        SdssDlaMask

        Load a SdssDlaMask instace and check that it is
        correctly initialized.

        #TODO:Check that the function apply_mask
        is correctly implemented in a dummy Forest instance

        """
        in_file = f"{THIS_DIR}/data/delta_attributes.fits.gz"

        mask = SdssDlaMask({"dla catalogue": in_file})
        self.assertTrue(isinstance(mask, Mask))

    def test_absorber_mask(self):
        """Tests correct initialisation and inheritance for class
        SdssAbsorberMask

        Load a SdssAbsorberMask instace and check that it is
        correctly initialized.

        #TODO:Check that the function apply_mask
        is correctly implemented in a dummy Forest instance

        """
        in_file = f"{THIS_DIR}/data/delta_attributes.fits.gz"

        mask = SdssAbsorberMask({"absorbers catalogue": in_file})
        self.assertTrue(isinstance(mask, Mask))

if __name__ == '__main__':
    unittest.main()
