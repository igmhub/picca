'''
Test module
'''
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

from .test_helpers import update_system_status_values, compare_fits, compare_h5py, send_requirements, load_requirements


class TestCor(unittest.TestCase):
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
        cls._test=True
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
        lst_fold = [
            "/Products/",
            "/Products/Spectra/",
            "/Products/Correlations/",
            "/Products/Correlations/Co_Random/",
            "/Products/Correlations/Fit/",
        ]

        for fold in lst_fold:
            if not os.path.isdir(self._branchFiles + fold):
                os.mkdir(self._branchFiles + fold)

        return

    def test_cf1d(self):

        userprint("\n")
        ### Send
        cmd = "picca_cf1d.py"
        cmd += " --in-dir " + self._masterFiles + "/test_delta/Delta_LYA/"
        cmd += " --out " + self._branchFiles + "/Products/Correlations/cf1d.fits.gz"
        cmd += " --nproc 1"
        subprocess.call(cmd, shell=True)

        ### Test
        if self._test:
            path1 = self._masterFiles + "/test_cor/cf1d.fits.gz"
            path2 = self._branchFiles + "/Products/Correlations/cf1d.fits.gz"
            self.compare_fits(path1, path2, "picca_cf1d.py")

        return

    def test_cf1d_cross(self):

        userprint("\n")
        ### Send
        cmd = "picca_cf1d.py"
        cmd += " --in-dir " + self._masterFiles + "/test_delta/Delta_LYA/"
        cmd += " --in-dir2 " + self._masterFiles + "/test_delta/Delta_LYA/"
        cmd += " --out " + self._branchFiles + "/Products/Correlations/cf1d_cross.fits.gz"
        cmd += " --nproc 1"
        subprocess.call(cmd, shell=True)

        ### Test
        if self._test:
            path1 = self._masterFiles + "/test_cor/cf1d_cross.fits.gz"
            path2 = self._branchFiles + "/Products/Correlations/cf1d_cross.fits.gz"
            self.compare_fits(path1, path2, "picca_cf1d.py")

        return

    def test_cf_angl(self):

        userprint("\n")
        ### Send
        cmd = "picca_cf_angl.py"
        cmd += " --in-dir " + self._masterFiles + "/test_delta/Delta_LYA/"
        cmd += " --out " + self._branchFiles + "/Products/Correlations/cf_angl.fits.gz"
        cmd += " --nproc 1"
        subprocess.call(cmd, shell=True)

        ### Test
        if self._test:
            path1 = self._masterFiles + "/test_cor/cf_angl.fits.gz"
            path2 = self._branchFiles + "/Products/Correlations/cf_angl.fits.gz"
            self.compare_fits(path1, path2, "picca_cf_angl.py")

        return

    def test_cf(self):

        userprint("\n")
        ### Send
        cmd = "picca_cf.py"
        cmd += " --in-dir " + self._masterFiles + "/test_delta/Delta_LYA/"
        cmd += " --out " + self._branchFiles + "/Products/Correlations/cf.fits.gz"
        cmd += " --rp-min +0.0"
        cmd += " --rp-max +60.0"
        cmd += " --rt-max +60.0"
        cmd += " --np 15"
        cmd += " --nt 15"
        cmd += " --nproc 1"
        cmd += ' --remove-same-half-plate-close-pairs'
        subprocess.call(cmd, shell=True)

        ### Test
        if self._test:
            path1 = self._masterFiles + "/test_cor/cf.fits.gz"
            path2 = self._branchFiles + "/Products/Correlations/cf.fits.gz"
            self.compare_fits(path1, path2, "picca_cf.py")

        return

    def test_dmat(self):

        userprint("\n")
        ### Send
        cmd = "picca_dmat.py"
        cmd += " --in-dir " + self._masterFiles + "/test_delta/Delta_LYA/"
        cmd += " --out " + self._branchFiles + "/Products/Correlations/dmat.fits.gz"
        cmd += " --rp-min +0.0"
        cmd += " --rp-max +60.0"
        cmd += " --rt-max +60.0"
        cmd += " --np 15"
        cmd += " --nt 15"
        cmd += " --rej 0.99 "
        cmd += " --nproc 1"
        cmd += ' --remove-same-half-plate-close-pairs'
        subprocess.call(cmd, shell=True)

        ### Test
        if self._test:
            path1 = self._masterFiles + "/test_cor/dmat.fits.gz"
            path2 = self._branchFiles + "/Products/Correlations/dmat.fits.gz"
            self.compare_fits(path1, path2, "picca_dmat.py")

        return

    def test_metal_dmat(self):

        userprint("\n")
        ### Send
        cmd = "picca_metal_dmat.py"
        cmd += " --in-dir " + self._masterFiles + "/test_delta/Delta_LYA/"
        cmd += " --out " + self._branchFiles + "/Products/Correlations/metal_dmat.fits.gz"
        cmd += r" --abs-igm SiIII\(1207\)"
        cmd += " --rp-min +0.0"
        cmd += " --rp-max +60.0"
        cmd += " --rt-max +60.0"
        cmd += " --np 15"
        cmd += " --nt 15"
        cmd += " --rej 0.99 "
        cmd += " --nproc 1"
        cmd += ' --remove-same-half-plate-close-pairs'
        subprocess.call(cmd, shell=True)

        ### Test
        if self._test:
            path1 = self._masterFiles + "/test_cor/metal_dmat.fits.gz"
            path2 = self._branchFiles + "/Products/Correlations/metal_dmat.fits.gz"
            self.compare_fits(path1, path2, "picca_metal_dmat.py")

        return

    def test_wick(self):

        userprint("\n")
        ### Send
        cmd = "picca_wick.py"
        cmd += " --in-dir " + self._masterFiles + "/test_delta/Delta_LYA/"
        cmd += " --out " + self._branchFiles + "/Products/Correlations/wick.fits.gz"
        cmd += " --cf1d " + self._masterFiles + "/test_cor/cf1d.fits.gz"
        cmd += " --rp-min +0.0"
        cmd += " --rp-max +60.0"
        cmd += " --rt-max +60.0"
        cmd += " --np 15"
        cmd += " --nt 15"
        cmd += " --rej 0.99 "
        cmd += " --nproc 1"
        subprocess.call(cmd, shell=True)

        ### Test
        if self._test:
            path1 = self._masterFiles + "/test_cor/wick.fits.gz"
            path2 = self._branchFiles + "/Products/Correlations/wick.fits.gz"
            self.compare_fits(path1, path2, "picca_wick.py")

        return

    def test_export_cf(self):

        userprint("\n")
        ### Send
        cmd = "picca_export.py"
        cmd += " --data " + self._masterFiles + "/test_cor/cf.fits.gz"
        cmd += " --dmat " + self._masterFiles + "/test_cor/dmat.fits.gz"
        cmd += " --out " + self._branchFiles + "/Products/Correlations/exported_cf.fits.gz"
        returncode = subprocess.call(cmd, shell=True)
        self.assertEqual(returncode, 0, "export_cf did not finish")
        return

    def test_cf_cross(self):

        userprint("\n")
        ### Send
        cmd = "picca_cf.py"
        cmd += " --in-dir  " + self._masterFiles + "/test_delta/Delta_LYA/"
        cmd += " --in-dir2 " + self._masterFiles + "/test_delta/Delta_LYA/"
        cmd += " --out " + self._branchFiles + "/Products/Correlations/cf_cross.fits.gz"
        cmd += " --rp-min -60.0"
        cmd += " --rp-max +60.0"
        cmd += " --rt-max +60.0"
        cmd += " --np 30"
        cmd += " --nt 15"
        cmd += " --nproc 1"
        cmd += ' --remove-same-half-plate-close-pairs'
        cmd += " --unfold-cf"
        subprocess.call(cmd, shell=True)

        ### Test
        if self._test:
            path1 = self._masterFiles + "/test_cor/cf_cross.fits.gz"
            path2 = self._branchFiles + "/Products/Correlations/cf_cross.fits.gz"
            self.compare_fits(path1, path2, "picca_cf.py")

        return

    def test_dmat_cross(self):

        userprint("\n")
        ### Send
        cmd = "picca_dmat.py"
        cmd += " --in-dir  " + self._masterFiles + "/test_delta/Delta_LYA/"
        cmd += " --in-dir2 " + self._masterFiles + "/test_delta/Delta_LYA/"
        cmd += " --out " + self._branchFiles + "/Products/Correlations/dmat_cross.fits.gz"
        cmd += " --rp-min -60.0"
        cmd += " --rp-max +60.0"
        cmd += " --rt-max +60.0"
        cmd += " --np 30"
        cmd += " --nt 15"
        cmd += " --rej 0.99 "
        cmd += " --nproc 1"
        cmd += ' --remove-same-half-plate-close-pairs'
        cmd += " --unfold-cf"
        subprocess.call(cmd, shell=True)

        ### Test
        if self._test:
            path1 = self._masterFiles + "/test_cor/dmat_cross.fits.gz"
            path2 = self._branchFiles + "/Products/Correlations/dmat_cross.fits.gz"
            self.compare_fits(path1, path2, "picca_dmat.py")

        return

    def test_metal_dmat_cross(self):

        userprint("\n")
        ### Send
        cmd = "picca_metal_dmat.py"
        cmd += " --in-dir " + self._masterFiles + "/test_delta/Delta_LYA/"
        cmd += " --in-dir2 " + self._masterFiles + "/test_delta/Delta_LYA/"
        cmd += " --out " + self._branchFiles + \
            "/Products/Correlations/metal_dmat_cross.fits.gz"
        cmd += r" --abs-igm SiIII\(1207\)"
        cmd += r" --abs-igm2 SiIII\(1207\)"
        cmd += " --rp-min -60.0"
        cmd += " --rp-max +60.0"
        cmd += " --rt-max +60.0"
        cmd += " --np 30"
        cmd += " --nt 15"
        cmd += " --rej 0.99 "
        cmd += " --nproc 1"
        cmd += ' --remove-same-half-plate-close-pairs'
        cmd += " --unfold-cf"
        subprocess.call(cmd, shell=True)

        ### Test
        if self._test:
            path1 = self._masterFiles + "/test_cor/metal_dmat_cross.fits.gz"
            path2 = self._branchFiles + "/Products/Correlations/metal_dmat_cross.fits.gz"
            self.compare_fits(path1, path2, "picca_metal_dmat.py")

        return

    def test_export_cf_cross(self):

        userprint("\n")
        ### Send
        cmd = "picca_export.py"
        cmd += " --data " + self._masterFiles + "/test_cor/cf_cross.fits.gz"
        cmd += " --dmat " + self._masterFiles + "/test_cor/dmat_cross.fits.gz"
        cmd += " --out " + self._branchFiles + \
            "/Products/Correlations/exported_cf_cross.fits.gz"
        returncode = subprocess.call(cmd, shell=True)
        self.assertEqual(returncode, 0, "export_cf_cross did not finish")

        return

    def test_xcf_angl(self):

        userprint("\n")
        ### Send
        cmd = "picca_xcf_angl.py"
        cmd += " --in-dir " + self._masterFiles + "/test_delta/Delta_LYA/"
        cmd += " --drq " + self._masterFiles + "/test_delta/cat.fits"
        cmd += " --out " + self._branchFiles + "/Products/Correlations/xcf_angl.fits.gz"
        cmd += " --nproc 1"
        subprocess.call(cmd, shell=True)

        ### Test
        if self._test:
            path1 = self._masterFiles + "/test_cor/xcf_angl.fits.gz"
            path2 = self._branchFiles + "/Products/Correlations/xcf_angl.fits.gz"
            self.compare_fits(path1, path2, "picca_xcf_angl.py")

        return

    def test_xcf(self):

        userprint("\n")
        ### Send
        cmd = "picca_xcf.py"
        cmd += " --in-dir " + self._masterFiles + "/test_delta/Delta_LYA/"
        cmd += " --drq " + self._masterFiles + "/test_delta/cat.fits"
        cmd += " --out " + self._branchFiles + "/Products/Correlations/xcf.fits.gz"
        cmd += " --rp-min -60.0"
        cmd += " --rp-max +60.0"
        cmd += " --rt-max +60.0"
        cmd += " --np 30"
        cmd += " --nt 15"
        cmd += " --nproc 1"
        subprocess.call(cmd, shell=True)

        ### Test
        if self._test:
            path1 = self._masterFiles + "/test_cor/xcf.fits.gz"
            path2 = self._branchFiles + "/Products/Correlations/xcf.fits.gz"
            self.compare_fits(path1, path2, "picca_xcf.py")

        return

    def test_xdmat(self):

        userprint("\n")
        ### Send
        cmd = "picca_xdmat.py"
        cmd += " --in-dir  " + self._masterFiles + "/test_delta/Delta_LYA/"
        cmd += " --drq " + self._masterFiles + "/test_delta/cat.fits"
        cmd += " --out " + self._branchFiles + "/Products/Correlations/xdmat.fits.gz"
        cmd += " --rp-min -60.0"
        cmd += " --rp-max +60.0"
        cmd += " --rt-max +60.0"
        cmd += " --np 30"
        cmd += " --nt 15"
        cmd += " --rej 0.99 "
        cmd += " --nproc 1"
        subprocess.call(cmd, shell=True)

        ### Test
        if self._test:
            path1 = self._masterFiles + "/test_cor/xdmat.fits.gz"
            path2 = self._branchFiles + "/Products/Correlations/xdmat.fits.gz"
            self.compare_fits(path1, path2, "picca_xdmat.py")

        return

    def test_metal_xdmat(self):

        userprint("\n")
        ### Send
        cmd = "picca_metal_xdmat.py"
        cmd += " --in-dir " + self._masterFiles + "/test_delta/Delta_LYA/"
        cmd += " --drq " + self._masterFiles + "/test_delta/cat.fits"
        cmd += " --out " + self._branchFiles + "/Products/Correlations/metal_xdmat.fits.gz"
        cmd += r" --abs-igm SiIII\(1207\)"
        cmd += " --rp-min -60.0"
        cmd += " --rp-max +60.0"
        cmd += " --rt-max +60.0"
        cmd += " --np 30"
        cmd += " --nt 15"
        cmd += " --rej 0.99 "
        cmd += " --nproc 1"
        subprocess.call(cmd, shell=True)

        ### Test
        if self._test:
            path1 = self._masterFiles + "/test_cor/metal_xdmat.fits.gz"
            path2 = self._branchFiles + "/Products/Correlations/metal_xdmat.fits.gz"
            self.compare_fits(path1, path2, "picca_metal_xdmat.py")

        return

    def test_xwick(self):

        userprint("\n")
        ### Send
        cmd = "picca_xwick.py"
        cmd += " --in-dir " + self._masterFiles + "/test_delta/Delta_LYA/"
        cmd += " --drq " + self._masterFiles + "/test_delta/cat.fits"
        cmd += " --out " + self._branchFiles + "/Products/Correlations/xwick.fits.gz"
        cmd += " --cf1d " + self._masterFiles + "/test_cor/cf1d.fits.gz"
        cmd += " --rp-min -60.0"
        cmd += " --rp-max +60.0"
        cmd += " --rt-max +60.0"
        cmd += " --np 30"
        cmd += " --nt 15"
        cmd += " --rej 0.99 "
        cmd += " --nproc 1"
        subprocess.call(cmd, shell=True)

        ### Test
        if self._test:
            path1 = self._masterFiles + "/test_cor/xwick.fits.gz"
            path2 = self._branchFiles + "/Products/Correlations/xwick.fits.gz"
            self.compare_fits(path1, path2, "picca_xwick.py")

        return

    def test_export_xcf(self):

        userprint("\n")
        ### Send
        cmd = "picca_export.py"
        cmd += " --data " + self._masterFiles + "/test_cor/xcf.fits.gz"
        cmd += " --dmat " + self._masterFiles + "/test_cor/xdmat.fits.gz"
        cmd += " --out " + self._branchFiles + \
            "/Products/Correlations/exported_xcf.fits.gz"
        returncode = subprocess.call(cmd, shell=True)
        self.assertEqual(returncode, 0, "export_xcf did not finish")

        return

    def test_export_cross_covariance_cf_xcf(self):

        userprint("\n")
        ### Send
        cmd = "picca_export_cross_covariance.py"
        cmd += " --data1 " + self._masterFiles + "/test_cor/cf.fits.gz"
        cmd += " --data2 " + self._masterFiles + "/test_cor/xcf.fits.gz"
        cmd += " --out " + self._branchFiles + \
            "/Products/Correlations/exported_cross_covariance_cf_xcf.fits.gz"
        returncode = subprocess.call(cmd, shell=True)
        self.assertEqual(returncode, 0,
                         "export_cross_covariance_cf_xcf did not finish")

        return

    def test_co(self):

        userprint("\n")
        ### Send
        cmd = "picca_co.py"
        cmd += " --drq " + self._masterFiles + "/test_delta/cat.fits"
        cmd += " --out " + self._branchFiles + "/Products/Correlations/co_DD.fits.gz"
        cmd += " --rp-min 0."
        cmd += " --rp-max +60.0"
        cmd += " --rt-max +60.0"
        cmd += " --np 15"
        cmd += " --nt 15"
        cmd += " --nproc 1"
        cmd += " --type-corr DD"
        returncode = subprocess.call(cmd, shell=True)
        self.assertEqual(returncode, 0, "picca_co did not finish on DD")
        ### Send
        cmd = "picca_co.py"
        cmd += " --drq " + self._masterFiles + "/test_delta/random.fits"
        cmd += " --out " + self._branchFiles + \
            "/Products/Correlations/Co_Random/co_RR.fits.gz"
        cmd += " --rp-min 0."
        cmd += " --rp-max +60.0"
        cmd += " --rt-max +60.0"
        cmd += " --np 15"
        cmd += " --nt 15"
        cmd += " --nproc 1"
        cmd += " --type-corr RR"
        returncode = subprocess.call(cmd, shell=True)
        self.assertEqual(returncode, 0, "picca_co did not finish on RR")
        ### Send
        cmd = "picca_co.py"
        cmd += " --drq " + self._masterFiles + "/test_delta/cat.fits"
        cmd += " --drq2 " + self._masterFiles + "/test_delta/random.fits"
        cmd += " --out " + self._branchFiles + \
            "/Products/Correlations/Co_Random/co_DR.fits.gz"
        cmd += " --rp-min 0."
        cmd += " --rp-max +60.0"
        cmd += " --rt-max +60.0"
        cmd += " --np 15"
        cmd += " --nt 15"
        cmd += " --nproc 1"
        cmd += " --type-corr DR"
        returncode = subprocess.call(cmd, shell=True)
        self.assertEqual(returncode, 0, "picca_co did not finish on DR")
        ### Send
        cmd = "picca_co.py"
        cmd += " --drq " + self._masterFiles + "/test_delta/random.fits"
        cmd += " --drq2 " + self._masterFiles + "/test_delta/cat.fits"
        cmd += " --out " + self._branchFiles + \
            "/Products/Correlations/Co_Random/co_RD.fits.gz"
        cmd += " --rp-min 0."
        cmd += " --rp-max +60.0"
        cmd += " --rt-max +60.0"
        cmd += " --np 15"
        cmd += " --nt 15"
        cmd += " --nproc 1"
        cmd += " --type-corr RD"
        returncode = subprocess.call(cmd, shell=True)
        self.assertEqual(returncode, 0, "picca_co did not finish on RD")

        ### Test
        if self._test:
            path1 = self._masterFiles + "/test_cor/co_DD.fits.gz"
            path2 = self._branchFiles + "/Products/Correlations/co_DD.fits.gz"
            self.compare_fits(path1, path2, "picca_co.py DD")

            path1 = self._masterFiles + "/test_cor/co_RR.fits.gz"
            path2 = self._branchFiles + "/Products/Correlations/Co_Random/co_RR.fits.gz"
            self.compare_fits(path1, path2, "picca_co.py RR")

            path1 = self._masterFiles + "/test_cor/co_DR.fits.gz"
            path2 = self._branchFiles + "/Products/Correlations/Co_Random/co_DR.fits.gz"
            self.compare_fits(path1, path2, "picca_co.py DR")

            path1 = self._masterFiles + "/test_cor/co_RD.fits.gz"
            path2 = self._branchFiles + "/Products/Correlations/Co_Random/co_RD.fits.gz"
            self.compare_fits(path1, path2, "picca_co.py RD")

        return

    def test_export_co(self):

        userprint("\n")
        ### Send
        cmd = "picca_export_co.py"
        cmd += " --DD-file " + self._masterFiles + "/test_cor/co_DD.fits.gz"
        cmd += " --RR-file " + self._masterFiles + \
            "/test_cor/co_RR.fits.gz"
        cmd += " --DR-file " + self._masterFiles + \
            "/test_cor/co_DR.fits.gz"
        cmd += " --RD-file " + self._masterFiles + \
            "/test_cor/co_RD.fits.gz"
        cmd += " --out " + self._branchFiles + "/Products/Correlations/exported_co.fits.gz"
        cmd += " --get-cov-from-poisson"
        returncode = subprocess.call(cmd, shell=True)
        self.assertEqual(returncode, 0, "picca_export_co did not finish")

        return


if __name__ == '__main__':
    unittest.main()
