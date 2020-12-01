"""This file contains configuration tests"""
import unittest
import os

from picca.delta_extraction.astronomical_object import AstronomicalObject

from picca.delta_extraction.astronomical_objects.drq_object import DrqObject

from picca.delta_extraction.tests.abstract_test import AbstractTest

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ["THIS_DIR"] = THIS_DIR

class TestConfiguration(AbstractTest):
    """Test the configuration."""

    def test_astronomical_object(self):
        """Test constructor for astronomical object."""
        kwargs = {
            "los_id": 9999,
            "ra": 0.15,
            "dec": 0.0,
            "z": 2.1,
        }
        test_obj = AstronomicalObject(**kwargs)
        self.assertTrue(test_obj.los_id == 9999)
        self.assertTrue(test_obj.healpix == 1505)
        self.assertTrue(test_obj.z == 2.1)

    def test_astronomical_object_comparison(self):
        """Test comparison between astronomical object."""
        kwargs = {
            "los_id": 9999,
            "ra": 0.15,
            "dec": 0.0,
            "z": 2.1,
        }
        test_obj = AstronomicalObject(**kwargs)

        kwargs_gt = {
            "healpix_ordering": {"los_id": 9999, "ra": 0.0, "dec": 0.0, "z": 2.1},
            "ra_ordering": {"los_id": 9999, "ra": 0.1, "dec": 0.0, "z": 2.1},
            "dec_ordering": {"los_id": 9999, "ra": 0.15, "dec": -0.01, "z": 2.1},
            "z_ordering": {"los_id": 9999, "ra": 0.15, "dec": 0.0, "z": 2.0},
        }
        for kwargs_other in kwargs_gt.values():
            other = AstronomicalObject(**kwargs_other)
            self.assertTrue(test_obj > other)
            self.assertTrue(test_obj != other)
            self.assertFalse(test_obj == other)
            self.assertFalse(test_obj < other)

        # equal objects
        kwargs_other = {
            "los_id": 1234,
            "ra": 0.15,
            "dec": 0.0,
            "z": 2.1,
        }
        other = AstronomicalObject(**kwargs_other)
        self.assertFalse(test_obj > other)
        self.assertTrue(test_obj == other)
        self.assertFalse(test_obj < other)

    def test_drq_object(self):
        """Tests the DrqObject class

        Create a DrqObject and checks that the inheritance is correct
        and that los_id=thing_id
        """
        kwargs = {
            "plate": 1234,
            "thingid": 9999,
            "fiberid": 444,
            "mjd": 55999,
            "ra": 0.0,
            "dec": 1.0,
            "z": 2.1,
        }
        test_obj = DrqObject(**kwargs)
        self.assertTrue(test_obj.los_id == 9999)
        self.assertTrue(test_obj.healpix == 264)
        self.assertTrue(test_obj.z == 2.1)

if __name__ == '__main__':
    unittest.main()
