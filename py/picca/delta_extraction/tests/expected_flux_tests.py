"""This file contains tests related to Data and its childs"""
import os
import unittest
import copy
import numpy as np

from picca.delta_extraction.errors import ExpectedFluxError
from picca.delta_extraction.expected_flux import ExpectedFlux
from picca.delta_extraction.tests.abstract_test import AbstractTest
from picca.delta_extraction.tests.test_utils import forest1

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class ExpectedFluxTest(AbstractTest):
    """Test class ExpectedFlux and its childs."""

    def test_expected_flux(self):
        """Test Abstract class ExpectedFlux

        Load an ExpectedFlux instace.
        """
        expected_flux = ExpectedFlux()

        with self.assertRaises(ExpectedFluxError):
            expected_flux.compute_expected_flux([])

        forest = copy.deepcopy(forest1)
        self.assertTrue(forest.deltas is None)

        # los_id not in dictionary: extract deltas does nothing
        expected_flux.extract_deltas(forest)
        self.assertTrue(forest.deltas is None)

        # los_id in dictionary: extract deltas
        expected_flux.los_ids = {forest1.los_id: {"mean expected flux": np.ones_like(forest1.flux),
                                                  "weights": np.ones_like(forest1.flux)}}
        expected_flux.extract_deltas(forest)
        self.assertTrue(all(forest.deltas == np.zeros_like(forest1.flux)))


if __name__ == '__main__':
    unittest.main()
