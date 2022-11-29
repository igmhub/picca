"""This file contains tests related to the script
picca_delta_extraction.py under $PICCA_HOME/bin"""
import glob
import os
import unittest
import subprocess
from subprocess import CalledProcessError

from picca.tests.delta_extraction.abstract_test import AbstractTest

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

PICCA_BIN = THIS_DIR.split("py/picca")[0]+"bin/"

class ScriptsTest(AbstractTest):
    """Test script
    picca_delta_extraction.py under $PICCA_HOME/bin

    Methods
    -------
    compare_ascii (from AbstractTest)
    compare_fits (from AbstractTest)
    setUp (from AbstractTest)
    """
    def run_delta_extraction(self, config_file, out_dir, test_dir):
        """ Run delta_extraction.py with the specified configuration
        and test its results

        Parameters
        ----------
        command : list
        A list of items with the script to run and its options
        """
        command = ["python",
                   "{}/picca_delta_extraction.py".format(PICCA_BIN),
                   config_file,
                   ]
        print("Running command: ", " ".join(command))

        try:
            subprocess.run(command, check=True, capture_output=True,
                           env=dict(os.environ, THIS_DIR=THIS_DIR))
        except CalledProcessError as e:
            print(e.stderr)
            raise e

        # compare attributes
        test_files = sorted(glob.glob(f"{test_dir}/Log/delta_attributes*.fits.gz"))
        out_files = sorted(glob.glob(f"{out_dir}/Log/delta_attributes*.fits.gz"))
        for test_file, out_file in zip(test_files, out_files):
            self.assertTrue(test_file.split("/")[-1] == out_file.split("/")[-1])
            self.compare_fits(test_file, out_file)

        # compare deltas
        test_files = sorted(glob.glob(f"{test_dir}/Delta/delta-*.fits.gz"))
        out_files = sorted(glob.glob(f"{out_dir}/Delta/delta-*.fits.gz"))
        self.assertTrue(len(out_files) == len(test_files))
        for test_file, out_file in zip(test_files, out_files):
            self.assertTrue(test_file.split("/")[-1] == out_file.split("/")[-1])
            self.compare_fits(test_file, out_file)

    def test_delta_calib(self):
        """End-to-end test using 'calib' setup including the sky mask"""
        config_file = "{}/data/delta_calib.ini".format(THIS_DIR)
        out_dir = "{}/results/delta_extraction_calib".format(THIS_DIR)
        test_dir = "{}/data/delta_extraction_calib".format(THIS_DIR)

        self.run_delta_extraction(config_file, out_dir, test_dir)

    def test_delta_calib_nomask(self):
        """End-to-end test using 'calib' setup"""
        config_file = "{}/data/delta_calib_nomask.ini".format(THIS_DIR)
        out_dir = "{}/results/delta_extraction_calib_nomask".format(THIS_DIR)
        test_dir = "{}/data/delta_extraction_calib_nomask".format(THIS_DIR)

        self.run_delta_extraction(config_file, out_dir, test_dir)

    def test_delta_calib2_nomask(self):
        """End-to-end test using 'calib2' setup without sky masking"""
        config_file = "{}/data/delta_calib2_nomask.ini".format(THIS_DIR)
        out_dir = "{}/results/delta_extraction_calib2_nomask".format(THIS_DIR)
        test_dir = "{}/data/delta_extraction_calib2_nomask".format(THIS_DIR)

        self.run_delta_extraction(config_file, out_dir, test_dir)

    def test_delta_calib2(self):
        """End-to-end test using 'calib2' setup"""
        config_file = "{}/data/delta_calib2.ini".format(THIS_DIR)
        out_dir = "{}/results/delta_extraction_calib2".format(THIS_DIR)
        test_dir = "{}/data/delta_extraction_calib2".format(THIS_DIR)

        self.run_delta_extraction(config_file, out_dir, test_dir)

    def test_delta_lin(self):
        """End-to-end test using 'LYA' setup with  a linear wavelenth solution"""
        config_file = "{}/data/delta_lin.ini".format(THIS_DIR)
        out_dir = "{}/results/delta_extraction_lin".format(THIS_DIR)
        test_dir = "{}/data/delta_extraction_lin".format(THIS_DIR)

        self.run_delta_extraction(config_file, out_dir, test_dir)

    def test_delta_lin_image(self):
        """End-to-end test using 'LYA' linear setup storing data as image"""
        config_file = "{}/data/delta_lin_image.ini".format(THIS_DIR)
        out_dir = "{}/results/delta_extraction_lin_image".format(THIS_DIR)
        test_dir = "{}/data/delta_extraction_lin_image".format(THIS_DIR)

        self.run_delta_extraction(config_file, out_dir, test_dir)

    def test_delta_lin_pk1d(self):
        """End-to-end test using 'LYA' setup with a linear wavelenth solution and Pk1D Forests.
        """
        config_file = "{}/data/delta_lin_pk1d.ini".format(THIS_DIR)
        out_dir = "{}/results/delta_extraction_lin_pk1d".format(THIS_DIR)
        test_dir = "{}/data/delta_extraction_lin_pk1d".format(THIS_DIR)

        self.run_delta_extraction(config_file, out_dir, test_dir)

    def test_delta_lya_nodla(self):
        """End-to-end test using 'LYA' setup wihtout masking DLAs"""
        config_file = "{}/data/delta_lya_nodla.ini".format(THIS_DIR)
        out_dir = "{}/results/delta_extraction_lya_nodla".format(THIS_DIR)
        test_dir = "{}/data/delta_extraction_lya_nodla".format(THIS_DIR)

        self.run_delta_extraction(config_file, out_dir, test_dir)

    def test_delta_lya_nomask_nodla(self):
        """End-to-end test using 'LYA' setup wihtout masking sky lines nor DLAs"""
        config_file = "{}/data/delta_lya_nomask_nodla.ini".format(THIS_DIR)
        out_dir = "{}/results/delta_extraction_lya_nomask_nodla".format(THIS_DIR)
        test_dir = "{}/data/delta_extraction_lya_nomask_nodla".format(THIS_DIR)

        self.run_delta_extraction(config_file, out_dir, test_dir)

    def test_delta_lya_desi_mocks(self):
        """End-to-end test using 'LYA' setup without corrections or masking,
        for desi mocks"""
        config_file = "{}/data/delta_lya_desi_mocks.ini".format(THIS_DIR)
        out_dir = "{}/results/delta_extraction_lya_desi_mocks".format(THIS_DIR)
        test_dir = "{}/data/delta_extraction_lya_desi_mocks".format(THIS_DIR)

        self.run_delta_extraction(config_file, out_dir, test_dir)


if __name__ == '__main__':
    unittest.main()
