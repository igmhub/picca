import unittest
import numpy as np
import fitsio
import healpy
import subprocess
import os
import tempfile
import shutil
from pkg_resources import resource_filename
import sys
import picca.bin.picca_Pk1D as picca_Pk1D

from picca.utils import userprint

from .test_helpers import update_system_status_values, compare_fits, compare_h5py, send_requirements, load_requirements


class TestPk1d(unittest.TestCase):
    #TODO: bad style, using it for the moment while transitioning, remove later
    compare_fits = compare_fits
    compare_h5py = compare_h5py

    @classmethod
    def setUpClass(cls):
        cls._branchFiles = tempfile.mkdtemp() + "/"
        cls.produce_folder(cls)
        cls.picca_base = resource_filename('picca',
                                           './').replace('py/picca/./', '')
        send_requirements(load_requirements(cls.picca_base))
        np.random.seed(42)
        cls._masterFiles = cls.picca_base + '/py/picca/test/data/'

        userprint("\n")

    @classmethod
    def tearDownClass(cls):
        if os.path.isdir(cls._branchFiles):
            shutil.rmtree(cls._branchFiles, ignore_errors=True)

    def produce_folder(self):
        """
            Create the necessary folders
        """

        userprint("\n")
        lst_fold = ["/Products/", "/Products/Pk1D/"]

        for fold in lst_fold:
            if not os.path.isdir(self._branchFiles + fold):
                os.mkdir(self._branchFiles + fold)

        return

    def test_Pk1D(self):
        self._test = True
        userprint("\n")
        ### Send
        #cmd = " picca_Pk1D.py"
        cmd = "--in-dir " + self._masterFiles + "/test_delta/Delta_Pk1D/"
        cmd += " --out-dir " + self._branchFiles + "/Products/Pk1D/"
        picca_Pk1D.main(cmd.split(" "))
        #subprocess.call(cmd, shell=True)

        ### Test
        if self._test:
            path1 = self._masterFiles + "/test_Pk1D/Pk1D.fits.gz"
            path2 = self._branchFiles + "/Products/Pk1D/Pk1D-0.fits.gz"
            self.compare_fits(path1, path2, "picca_Pk1D.py")

        return


if __name__ == '__main__':
    unittest.main()
