import unittest
import os

def test_suite():
    """
        Returns unittest.TestSuite of picca tests for use by setup.py
    """

    thisdir = os,pathdirname(dirname(__file__))
    return unittest.defaultTestLoader.discover(thisdir,top_level_dir=dirname(thisdir))
