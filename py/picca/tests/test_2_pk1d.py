import unittest
import os
import glob
import shutil
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
        lst_fold = ["/Products/", "/Products/Pk1D/", "/Products/meanPk1D/"]

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
            path2 = self._branchFiles + "/Products/Pk1D_raw/Pk1D-272.fits.gz"
            self.compare_fits(path1, path2, "picca_Pk1D.py")

        return

    def test_meanPk1D(self):
        """
            Runs a simple test of Pk1d postprocessing
        """
        import picca.bin.picca_Pk1D_postprocess as picca_Pk1D_postprocess

        self._test = True
        userprint("\n")
        ### Send
        #- picca_Pk1D_postprocess.py takes all Pk1D*.fits.gz files in in-dir
        #  => copy a single file
        shutil.copy(self._masterFiles + "/test_Pk1D/Pk1D.fits.gz",
                    self._branchFiles + "/Products/meanPk1D")
        print(os.listdir(self._branchFiles + "/Products/meanPk1D"))
        cmd = "picca_Pk1D_postprocess.py "
        cmd += " --in-dir " + self._branchFiles + "/Products/meanPk1D"
        cmd += " --output-file " + self._branchFiles + "/Products/meanPk1D/meanPk1D.fits.gz"
        #- small sample => k,z-bins changed wrt default ones
        cmd += " --zedge-min 2.1 --zedge-max 3.1 --zedge-bin 0.2"
        cmd += " --kedge-min 0.015 --kedge-max 0.035 --kedge-bin 0.005"
        picca_Pk1D_postprocess.main(cmd.split()[1:])

        ### Test
        if self._test:
            path1 = self._masterFiles + "/test_Pk1D/meanPk1D.fits.gz"
            path2 = self._branchFiles + "/Products/meanPk1D/meanPk1D.fits.gz"
            self.compare_fits(path1, path2, "picca_Pk1D_postprocess.py")


    def test_meanPk1D_covariance(self):
        """
            Runs a covariance test of Pk1d postprocessing
        """
        import picca.bin.picca_Pk1D_postprocess as picca_Pk1D_postprocess

        self._test = True
        userprint("\n")
        ### Send
        #- picca_Pk1D_postprocess.py takes all Pk1D*.fits.gz files in in-dir
        #  => copy a single file
        shutil.copy(self._masterFiles + "/test_Pk1D/Pk1D.fits.gz",
                    self._branchFiles + "/Products/meanPk1D")
        print(os.listdir(self._branchFiles + "/Products/meanPk1D"))
        cmd = "picca_Pk1D_postprocess.py "
        cmd += " --in-dir " + self._branchFiles + "/Products/meanPk1D"
        cmd += " --output-file " + self._branchFiles + "/Products/meanPk1D/meanPk1D_covariance.fits.gz"
        #- small sample => k,z-bins changed wrt default ones
        cmd += " --zedge-min 2.1 --zedge-max 3.1 --zedge-bin 0.2"
        cmd += " --kedge-min 0.015 --kedge-max 0.035 --kedge-bin 0.005"
        cmd += " --covariance"
        picca_Pk1D_postprocess.main(cmd.split()[1:])

        ### Test
        if self._test:
            path1 = self._masterFiles + "/test_Pk1D/meanPk1D_covariance.fits.gz"
            path2 = self._branchFiles + "/Products/meanPk1D/meanPk1D_covariance.fits.gz"
            self.compare_fits(path1, path2, "picca_Pk1D_postprocess.py")



if __name__ == '__main__':
    unittest.main()
