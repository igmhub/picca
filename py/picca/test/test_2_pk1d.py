
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

from picca.utils import userprint

from .test_helpers import update_system_status_values, compare_fits, compare_h5py

class TestPk1d(unittest.TestCase):
    #TODO: bad style, using it for the moment while transitioning, remove later
    compare_fits = compare_fits
    compare_h5py = compare_h5py

    @classmethod
    def setUpClass(cls):
        cls._branchFiles = tempfile.mkdtemp() + "/"

    @classmethod
    def tearDownClass(cls):
        if os.path.isdir(cls._branchFiles):
            shutil.rmtree(cls._branchFiles, ignore_errors=True)

    def test_pk1d(self):

        self.picca_base = resource_filename('picca',
                                            './').replace('py/picca/./', '')
        self.send_requirements()
        np.random.seed(42)

        userprint("\n")
        self._test = True
        self._masterFiles = self.picca_base + '/py/picca/test/data/'
        self.produce_folder()
        self.send_Pk1D()

        if self._test:
            self.remove_folder()

        return


    def load_requirements(self):

        req = {}

        if sys.version_info > (3, 0):
            path = self.picca_base + '/requirements.txt'
        else:
            path = self.picca_base + '/requirements-python2.txt'
        with open(path, 'r') as f:
            for l in f:
                l = l.replace('\n', '').replace('==',
                                                ' ').replace('>=',
                                                             ' ').split()
                self.assertTrue(
                    len(l) == 2,
                    "requirements.txt attribute is not valid: {}".format(
                        str(l)))
                req[l[0]] = l[1]

        return req

    def send_requirements(self):

        userprint("\n")
        req = self.load_requirements()
        for req_lib, req_ver in req.items():
            try:
                local_ver = __import__(req_lib).__version__
                if local_ver != req_ver:
                    userprint(
                        "WARNING: The local version of {}: {} is different from the required version: {}"
                        .format(req_lib, local_ver, req_ver))
            except ImportError:
                userprint("WARNING: Module {} can't be found".format(req_lib))

        return

    def produce_folder(self):
        """
            Create the necessary folders
        """

        userprint("\n")
        lst_fold = [
            "/Products/", 
            "/Products/Pk1D/"
        ]

        for fold in lst_fold:
            if not os.path.isdir(self._branchFiles + fold):
                os.mkdir(self._branchFiles + fold)

        return

    def remove_folder(self):
        """
            Remove the produced folders
        """

        userprint("\n")
        shutil.rmtree(self._branchFiles, ignore_errors=True)

        return

    def send_Pk1D(self):

        userprint("\n")
        ### Send
        cmd = " picca_Pk1D.py"
        cmd += " --in-dir " + self._masterFiles + "/test_Pk1D/delta_Pk1D/"
        cmd += " --out-dir " + self._branchFiles + "/Products/Pk1D/"
        subprocess.call(cmd, shell=True)

        ### Test
        if self._test:
            path1 = self._masterFiles + "/test_Pk1D/Pk1D.fits.gz"
            path2 = self._branchFiles + "/Products/Pk1D/Pk1D-0.fits.gz"
            self.compare_fits(path1, path2, "picca_Pk1D.py")

        return


if __name__ == '__main__':
    unittest.main()
