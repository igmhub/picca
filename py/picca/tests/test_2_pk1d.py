import unittest
import os
import glob
import numpy as np
import fitsio
import healpy

from picca.utils import userprint

from picca.tests.test_helpers import AbstractTest


class TestPk1d(AbstractTest):
    """
        Test the Pk1d routines
    """

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
        """
            Runs a simple test of Pk1d routines
        """
        import picca.bin.picca_Pk1D as picca_Pk1D

        self._test = True
        userprint("\n")
        ### Send
        cmd = "picca_Pk1D.py "
        cmd += "--in-dir " + self._masterFiles + "/test_delta/Delta_Pk1D/"
        cmd += " --out-dir " + self._branchFiles + "/Products/Pk1D/"
        picca_Pk1D.main(cmd.split()[1:])

        ### Test
        if self._test:
            path1 = self._masterFiles + "/test_Pk1D/Pk1D.fits.gz"
            path2 = self._branchFiles + "/Products/Pk1D/Pk1D-0.fits.gz"
            self.compare_fits(path1, path2, "picca_Pk1D.py")

        return

    def test_Pk1D_raw(self):
        """
            Runs a simple test of Pk1d routines
        """
        import picca.bin.picca_Pk1D as picca_Pk1D

        self._test = True
        userprint("\n")
        ### Send
        cmd = "picca_Pk1D.py "
        cmd += "--in-dir " + self._masterFiles + "/test_delta/Delta_LYA/"
        cmd += " --out-dir " + self._branchFiles + "/Products/Pk1D_raw/"
        picca_Pk1D.main(cmd.split()[1:])

        ### Test
        if self._test:
            path1 = self._masterFiles + "/test_Pk1D/Pk1D_raw.fits.gz"
            path2 = self._branchFiles + "/Products/Pk1D_raw/Pk1D-0.fits.gz"
            self.compare_fits(path1, path2, "picca_Pk1D.py")

        return


if __name__ == '__main__':
    unittest.main()
