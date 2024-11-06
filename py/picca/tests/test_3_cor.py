'''
Test module
'''

import unittest
import os
import importlib
import numpy as np

from picca.utils import userprint

### need to load those modules here and later reload
### as global variables inside will be overwritten
import picca.cf
import picca.xcf


import picca.bin.picca_co
import picca.bin.picca_cf1d
import picca.bin.picca_cf_angl
import picca.bin.picca_cf
import picca.bin.picca_wick
import picca.bin.picca_xwick
import picca.bin.picca_xcf
import picca.bin.picca_xcf_angl
import picca.bin.picca_dmat
import picca.bin.picca_xdmat

import picca.bin.picca_metal_dmat
import picca.bin.picca_metal_xdmat
import picca.bin.picca_fast_metal_dmat
import picca.bin.picca_fast_metal_xdmat
import picca.bin.picca_export
import picca.bin.picca_export_cross_covariance
import picca.bin.picca_export_co

from picca.tests.test_helpers import AbstractTest

def reset_cf():
    """Resets the parameters of picca_cf"""
    picca.cf.num_bins_r_par = None
    picca.cf.num_bins_r_trans = None
    picca.cf.num_model_bins_r_trans = None
    picca.cf.num_model_bins_r_par = None
    picca.cf.r_par_max = None
    picca.cf.r_par_min = None
    picca.cf.z_cut_max = None
    picca.cf.z_cut_min = None
    picca.cf.r_trans_max = None
    picca.cf.ang_max = None
    picca.cf.nside = None

    picca.cf.counter = None
    picca.cf.num_data = None
    picca.cf.num_data2 = None

    picca.cf.z_ref = None
    picca.cf.alpha = None
    picca.cf.alpha2 = None
    picca.cf.alpha_abs = None
    picca.cf.lambda_abs = None
    picca.cf.lambda_abs2 = None

    picca.cf.data = None
    picca.cf.data2 = None

    picca.cf.cosmo = None

    picca.cf.reject = None
    picca.cf.lock = None
    picca.cf.x_correlation = False
    picca.cf.ang_correlation = False
    picca.cf.remove_same_half_plate_close_pairs = False

    # variables used in the 1D correlation function analysis
    picca.cf.num_pixels = None
    picca.cf.log_lambda_min = None
    picca.cf.log_lambda_max = None
    picca.cf.delta_log_lambda = None

    # variables used in the wick covariance matrix computation
    picca.cf.et_variance_1d = {}
    picca.cf.xi_1d = {}
    picca.cf.max_diagram = None
    picca.cf.xi_wick = {}

def reset_xcf():
    """Resets the parameters of picca_xcf"""
    picca.xcf.num_bins_r_par = None
    picca.xcf.num_bins_r_trans = None
    picca.xcf.num_model_bins_r_par = None
    picca.xcf.num_model_bins_r_trans = None
    picca.xcf.r_par_max = None
    picca.xcf.r_par_min = None
    picca.xcf.r_trans_max = None
    picca.xcf.z_cut_max = None
    picca.xcf.z_cut_min = None
    picca.xcf.ang_max = None
    picca.xcf.nside = None

    picca.xcf.counter = None
    picca.xcf.num_data = None

    picca.xcf.z_ref = None
    picca.xcf.alpha = None
    picca.xcf.alpha_obj = None
    picca.xcf.lambda_abs = None
    picca.xcf.alpha_abs = None

    picca.xcf.data = None
    picca.xcf.objs = None

    picca.xcf.reject = None
    picca.xcf.lock = None

    picca.xcf.cosmo = None
    picca.xcf.ang_correlation = False

    # variables used in the wick covariance matrix computation
    picca.xcf.get_variance_1d = {}
    picca.xcf.xi_1d = {}
    picca.xcf.max_diagram = None
    picca.xcf.xi_wick = None

# TODO: add test for xcf1d
class TestCor(AbstractTest):
    """
        Tests the Correlation Function Computations
    """

    def tearDown(self):
        """ Actions done at test end
        Make sure that Forest and Pk1dForest class variables are reset
        """
        reset_cf()
        reset_xcf()


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
        """
            Test 1d correlation function
        """

        userprint("\n")
        ### Send
        cmd = "picca_cf1d.py"
        cmd += " --in-dir " + self._masterFiles + "/test_delta/Delta_LYA/"
        cmd += " --out " + self._branchFiles + "/Products/Correlations/cf1d.fits.gz"
        cmd += " --nproc 1"
        print(repr(cmd))
        picca.bin.picca_cf1d.main(cmd.split()[1:])

        ### Test
        if self._test:
            path1 = self._masterFiles + "/test_cor/cf1d.fits.gz"
            path2 = self._branchFiles + "/Products/Correlations/cf1d.fits.gz"
            self.compare_fits(path1, path2, "picca_cf1d.py")

        return

    def test_cf1d_cross(self):
        """
            Test 1d cross-correlation function
        """

        userprint("\n")
        ### Send
        cmd = "picca_cf1d.py"
        cmd += " --in-dir " + self._masterFiles + "/test_delta/Delta_LYA/"
        cmd += " --in-dir2 " + self._masterFiles + "/test_delta/Delta_LYA/"
        cmd += " --out " + self._branchFiles + "/Products/Correlations/cf1d_cross.fits.gz"
        cmd += " --nproc 1"
        print(repr(cmd))
        picca.bin.picca_cf1d.main(cmd.split()[1:])

        ### Test
        if self._test:
            path1 = self._masterFiles + "/test_cor/cf1d_cross.fits.gz"
            path2 = self._branchFiles + "/Products/Correlations/cf1d_cross.fits.gz"
            self.compare_fits(path1, path2, "picca_cf1d.py")

        return

    def test_cf_angl(self):
        """
            Test angular correlation function
        """
        importlib.reload(picca.cf)
        userprint("\n")
        ### Send
        cmd = "picca_cf_angl.py"
        cmd += " --in-dir " + self._masterFiles + "/test_delta/Delta_LYA/"
        cmd += " --out " + self._branchFiles + "/Products/Correlations/cf_angl.fits.gz"
        cmd += " --nproc 1"
        print(repr(cmd))
        picca.bin.picca_cf_angl.main(cmd.split()[1:])

        ### Test
        if self._test:
            path1 = self._masterFiles + "/test_cor/cf_angl.fits.gz"
            path2 = self._branchFiles + "/Products/Correlations/cf_angl.fits.gz"
            self.compare_fits(path1, path2, "picca_cf_angl.py")

        return

    def test_cf(self):
        """
            Test correlation function
        """
        importlib.reload(picca.cf)

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
        print(repr(cmd))
        picca.bin.picca_cf.main(cmd.split()[1:])

        ### Test
        if self._test:
            path1 = self._masterFiles + "/test_cor/cf.fits.gz"
            path2 = self._branchFiles + "/Products/Correlations/cf.fits.gz"
            self.compare_fits(path1, path2, "picca_cf.py")

        return

    def test_cf_image_data(self):
        """
            Test correlation function reading image data
        """
        importlib.reload(picca.cf)

        userprint("\n")
        ### Send
        cmd = "picca_cf.py"
        cmd += " --in-dir " + self._masterFiles + "/test_delta/Delta_LYA_image/"
        cmd += " --out " + self._branchFiles + "/Products/Correlations/cf.fits.gz"
        cmd += " --rp-min +0.0"
        cmd += " --rp-max +60.0"
        cmd += " --rt-max +60.0"
        cmd += " --np 15"
        cmd += " --nt 15"
        cmd += " --nproc 1"
        print(repr(cmd))
        picca.bin.picca_cf.main(cmd.split()[1:])

        ### Test
        if self._test:
            path1 = self._masterFiles + "/test_cor/cf_image.fits.gz"
            path2 = self._branchFiles + "/Products/Correlations/cf.fits.gz"
            self.compare_fits(path1, path2, "picca_cf.py")

        return

    def test_cf_image_data_rebin(self):
        """
            Test correlation function reading image data
        """
        importlib.reload(picca.cf)

        userprint("\n")
        ### Send
        cmd = "picca_cf.py"
        cmd += " --in-dir " + self._masterFiles + "/test_delta/Delta_LYA_image/"
        cmd += " --out " + self._branchFiles + "/Products/Correlations/cf.fits.gz"
        cmd += " --rp-min +0.0"
        cmd += " --rp-max +60.0"
        cmd += " --rt-max +60.0"
        cmd += " --np 15"
        cmd += " --nt 15"
        cmd += " --nproc 1"
        cmd += " --rebin-factor 3"
        print(repr(cmd))
        picca.bin.picca_cf.main(cmd.split()[1:])

        ### Test
        if self._test:
            path1 = self._masterFiles + "/test_cor/cf_image_rebinned.fits.gz"
            path2 = self._branchFiles + "/Products/Correlations/cf.fits.gz"
            self.compare_fits(path1, path2, "picca_cf.py")

        return

    def test_dmat(self):
        """
            Test distortion matrix
        """
        importlib.reload(picca.cf)

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
        cmd += " --rej 0.99"
        cmd += " --nproc 1"
        cmd += ' --remove-same-half-plate-close-pairs'
        cmd += ' --no-redshift-evolution'

        print(repr(cmd))
        picca.bin.picca_dmat.main(cmd.split()[1:])

        ### Test
        if self._test:
            path1 = self._masterFiles + "/test_cor/dmat.fits.gz"
            path2 = self._branchFiles + "/Products/Correlations/dmat.fits.gz"
            self.compare_fits(path1, path2, "picca_dmat.py")

        return

    def test_metal_dmat(self):
        """
            Test metal distortion matrix
        """
        importlib.reload(picca.cf)

        userprint("\n")
        ### Send
        cmd = "picca_metal_dmat.py"
        cmd += " --in-dir " + self._masterFiles + "/test_delta/Delta_LYA/"
        cmd += " --out " + self._branchFiles + "/Products/Correlations/metal_dmat.fits.gz"
        cmd += r" --abs-igm SiIII(1207)"
        cmd += " --rp-min +0.0"
        cmd += " --rp-max +60.0"
        cmd += " --rt-max +60.0"
        cmd += " --np 15"
        cmd += " --nt 15"
        cmd += " --rej 0.99"
        cmd += " --nproc 1"
        cmd += ' --remove-same-half-plate-close-pairs'
        print(repr(cmd))
        picca.bin.picca_metal_dmat.main(cmd.split()[1:])

        ### Test
        if self._test:
            path1 = self._masterFiles + "/test_cor/metal_dmat.fits.gz"
            path2 = self._branchFiles + "/Products/Correlations/metal_dmat.fits.gz"
            self.compare_fits(path1, path2, "picca_metal_dmat.py")

        return


    def test_fast_metal_dmat(self):
        """
            Test metal distortion matrix
        """
        importlib.reload(picca.cf)

        userprint("\n")
        ### Send
        cmd = "picca_fast_metal_dmat.py"
        cmd += " --in-attributes " + self._masterFiles + "/test_cor/input_from_delta_extraction_lya_nodla/Log/delta_attributes.fits.gz"
        cmd += " --out " + self._branchFiles + "/Products/Correlations/fast_metal_dmat.fits.gz"
        cmd += r" --abs-igm SiIII(1207)"
        cmd += " --rp-min +0.0"
        cmd += " --rp-max +60.0"
        cmd += " --rt-max +60.0"
        cmd += " --np 15"
        cmd += " --nt 15"
        print(repr(cmd))
        picca.bin.picca_fast_metal_dmat.main(cmd.split()[1:])

        ### Test
        if self._test:
            path1 = self._masterFiles + "/test_cor/fast_metal_dmat.fits.gz"
            path2 = self._branchFiles + "/Products/Correlations/fast_metal_dmat.fits.gz"
            self.compare_fits(path1, path2, "picca_fast_metal_dmat.py")

        return

    def test_wick(self):
        """
            Test wick covariances
        """
        importlib.reload(picca.cf)

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
        cmd += " --rej 0.99"
        cmd += " --nproc 1"
        print(repr(cmd))
        picca.bin.picca_wick.main(cmd.split()[1:])

        ### Test
        if self._test:
            path1 = self._masterFiles + "/test_cor/wick.fits.gz"
            path2 = self._branchFiles + "/Products/Correlations/wick.fits.gz"
            self.compare_fits(path1, path2, "picca_wick.py")

        return

    def test_export_cf(self):
        """
            Test export of correlation function
        """

        userprint("\n")
        ### Send
        cmd = "picca_export.py"
        cmd += " --data " + self._masterFiles + "/test_cor/cf.fits.gz"
        cmd += " --dmat " + self._masterFiles + "/test_cor/dmat.fits.gz"
        cmd += " --out " + self._branchFiles + "/Products/Correlations/exported_cf.fits.gz"
        print(repr(cmd))
        picca.bin.picca_export.main(cmd.split()[1:])
        return

    def test_cf_cross(self):
        """
            Test export of cross correlation function
        """
        importlib.reload(picca.cf)

        userprint("\n")
        ### Send
        cmd = "picca_cf.py"
        cmd += " --in-dir " + self._masterFiles + "/test_delta/Delta_LYA/"
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
        print(repr(cmd))
        picca.bin.picca_cf.main(cmd.split()[1:])

        ### Test
        if self._test:
            path1 = self._masterFiles + "/test_cor/cf_cross.fits.gz"
            path2 = self._branchFiles + "/Products/Correlations/cf_cross.fits.gz"
            self.compare_fits(path1, path2, "picca_cf.py")

        return

    def test_dmat_cross(self):
        """
            Test cross distortion matrix
        """
        importlib.reload(picca.cf)


        userprint("\n")
        ### Send
        cmd = "picca_dmat.py"
        cmd += " --in-dir " + self._masterFiles + "/test_delta/Delta_LYA/"
        cmd += " --in-dir2 " + self._masterFiles + "/test_delta/Delta_LYA/"
        cmd += " --out " + self._branchFiles + "/Products/Correlations/dmat_cross.fits.gz"
        cmd += " --rp-min -60.0"
        cmd += " --rp-max +60.0"
        cmd += " --rt-max +60.0"
        cmd += " --np 30"
        cmd += " --nt 15"
        cmd += " --rej 0.99"
        cmd += " --nproc 1"
        cmd += ' --remove-same-half-plate-close-pairs'
        cmd += " --unfold-cf"
        cmd += ' --no-redshift-evolution'

        print(repr(cmd))
        picca.bin.picca_dmat.main(cmd.split()[1:])

        ### Test
        if self._test:
            path1 = self._masterFiles + "/test_cor/dmat_cross.fits.gz"
            path2 = self._branchFiles + "/Products/Correlations/dmat_cross.fits.gz"
            self.compare_fits(path1, path2, "picca_dmat.py")

        return

    def test_metal_dmat_cross(self):
        """
            Test metal cross distortion matrix
        """
        importlib.reload(picca.cf)

        userprint("\n")
        ### Send
        cmd = "picca_metal_dmat.py"
        cmd += " --in-dir " + self._masterFiles + "/test_delta/Delta_LYA/"
        cmd += " --in-dir2 " + self._masterFiles + "/test_delta/Delta_LYA/"
        cmd += " --out " + self._branchFiles + \
            "/Products/Correlations/metal_dmat_cross.fits.gz"
        cmd += r" --abs-igm SiIII(1207)"
        cmd += r" --abs-igm2 SiIII(1207)"
        cmd += " --rp-min -60.0"
        cmd += " --rp-max +60.0"
        cmd += " --rt-max +60.0"
        cmd += " --np 30"
        cmd += " --nt 15"
        cmd += " --rej 0.99"
        cmd += " --nproc 1"
        cmd += ' --remove-same-half-plate-close-pairs'
        cmd += " --unfold-cf"
        print(repr(cmd))
        picca.bin.picca_metal_dmat.main(cmd.split()[1:])

        ### Test
        if self._test:
            path1 = self._masterFiles + "/test_cor/metal_dmat_cross.fits.gz"
            path2 = self._branchFiles + "/Products/Correlations/metal_dmat_cross.fits.gz"
            self.compare_fits(path1, path2, "picca_metal_dmat.py")

        return

    def test_fast_metal_dmat_cross(self):
        """
            Test metal cross distortion matrix
        """
        importlib.reload(picca.cf)

        userprint("\n")
        ### Send
        cmd = "picca_fast_metal_dmat.py"
        cmd += " --in-attributes " + self._masterFiles + "/test_cor/input_from_delta_extraction_lya_nodla/Log/delta_attributes.fits.gz"
        cmd += " --in-attributes " + self._masterFiles + "/test_cor/input_from_delta_extraction_lya_nodla/Log/delta_attributes.fits.gz"
        cmd += " --out " + self._branchFiles + \
            "/Products/Correlations/fast_metal_dmat_cross.fits.gz"
        cmd += r" --abs-igm SiIII(1207)"
        cmd += r" --abs-igm2 SiIII(1207)"
        cmd += " --rp-min -60.0"
        cmd += " --rp-max +60.0"
        cmd += " --rt-max +60.0"
        cmd += " --np 30"
        cmd += " --nt 15"
        cmd += " --unfold-cf"
        print(repr(cmd))
        picca.bin.picca_fast_metal_dmat.main(cmd.split()[1:])

        ### Test
        if self._test:
            path1 = self._masterFiles + "/test_cor/fast_metal_dmat_cross.fits.gz"
            path2 = self._branchFiles + "/Products/Correlations/fast_metal_dmat_cross.fits.gz"
            self.compare_fits(path1, path2, "picca_fast_metal_dmat.py")

        return

    def test_export_cf_cross(self):
        """
            Test export of cross correlation function
        """

        userprint("\n")
        ### Send
        cmd = "picca_export.py"
        cmd += " --data " + self._masterFiles + "/test_cor/cf_cross.fits.gz"
        cmd += " --dmat " + self._masterFiles + "/test_cor/dmat_cross.fits.gz"
        cmd += " --out " + self._branchFiles + \
            "/Products/Correlations/exported_cf_cross.fits.gz"
        print(repr(cmd))
        picca.bin.picca_export.main(cmd.split()[1:])

        return

    def test_xcf_angl(self):
        """
            Test angular cross correlation function
        """

        userprint("\n")
        ### Send
        cmd = "picca_xcf_angl.py"
        cmd += " --in-dir " + self._masterFiles + "/test_delta/Delta_LYA/"
        cmd += " --drq " + self._masterFiles + "/test_delta/cat.fits"
        cmd += " --out " + self._branchFiles + "/Products/Correlations/xcf_angl.fits.gz"
        cmd += " --nproc 1"
        cmd += " --z-evol-obj 1."
        print(repr(cmd))
        picca.bin.picca_xcf_angl.main(cmd.split()[1:])

        ### Test
        if self._test:
            path1 = self._masterFiles + "/test_cor/xcf_angl.fits.gz"
            path2 = self._branchFiles + "/Products/Correlations/xcf_angl.fits.gz"
            self.compare_fits(path1, path2, "picca_xcf_angl.py")

        return

    def test_xcf(self):
        """
            Test cross correlation function
        """

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
        cmd += " --z-evol-obj 1."
        print(repr(cmd))
        picca.bin.picca_xcf.main(cmd.split()[1:])

        ### Test
        if self._test:
            path1 = self._masterFiles + "/test_cor/xcf.fits.gz"
            path2 = self._branchFiles + "/Products/Correlations/xcf.fits.gz"
            self.compare_fits(path1, path2, "picca_xcf.py")

        return

    def test_xdmat(self):
        """
            Test cross distortion matrix
        """
        importlib.reload(picca.xcf)


        userprint("\n")
        ### Send
        cmd = "picca_xdmat.py"
        cmd += " --in-dir " + self._masterFiles + "/test_delta/Delta_LYA/"
        cmd += " --drq " + self._masterFiles + "/test_delta/cat.fits"
        cmd += " --out " + self._branchFiles + "/Products/Correlations/xdmat.fits.gz"
        cmd += " --rp-min -60.0"
        cmd += " --rp-max +60.0"
        cmd += " --rt-max +60.0"
        cmd += " --np 30"
        cmd += " --nt 15"
        cmd += " --rej 0.99"
        cmd += " --nproc 1"
        cmd += " --z-evol-obj 1."
        cmd += ' --no-redshift-evolution'
        print(repr(cmd))
        picca.bin.picca_xdmat.main(cmd.split()[1:])

        ### Test
        if self._test:
            path1 = self._masterFiles + "/test_cor/xdmat.fits.gz"
            path2 = self._branchFiles + "/Products/Correlations/xdmat.fits.gz"
            self.compare_fits(path1, path2, "picca_xdmat.py")

        return

    def test_metal_xdmat(self):
        """
            Test metal cross distortion matrix
        """
        userprint("\n")
        ### Send
        cmd = "picca_metal_xdmat.py"
        cmd += " --in-dir " + self._masterFiles + "/test_delta/Delta_LYA/"
        cmd += " --drq " + self._masterFiles + "/test_delta/cat.fits"
        cmd += " --out " + self._branchFiles + "/Products/Correlations/metal_xdmat.fits.gz"
        cmd += r" --abs-igm SiIII(1207)"
        cmd += " --rp-min -60.0"
        cmd += " --rp-max +60.0"
        cmd += " --rt-max +60.0"
        cmd += " --np 30"
        cmd += " --nt 15"
        cmd += " --rej 0.99"
        cmd += " --nproc 1"
        cmd += " --z-evol-obj 1."
        print(repr(cmd))
        picca.bin.picca_metal_xdmat.main(cmd.split()[1:])

        ### Test
        if self._test:
            path1 = self._masterFiles + "/test_cor/metal_xdmat.fits.gz"
            path2 = self._branchFiles + "/Products/Correlations/metal_xdmat.fits.gz"
            self.compare_fits(path1, path2, "picca_metal_xdmat.py")

        return

    def test_fast_metal_xdmat(self):
        """
            Test metal cross distortion matrix
        """
        userprint("\n")
        ### Send
        cmd = "picca_fast_metal_xdmat.py"
        cmd += " --in-attributes " + self._masterFiles + "/test_cor/input_from_delta_extraction_lya_nodla/Log/delta_attributes.fits.gz"
        cmd += " --drq " + self._masterFiles + "/test_delta/cat.fits"
        cmd += " --out " + self._branchFiles + "/Products/Correlations/fast_metal_xdmat.fits.gz"
        cmd += r" --abs-igm SiIII(1207)"
        cmd += " --rp-min -60.0"
        cmd += " --rp-max +60.0"
        cmd += " --rt-max +60.0"
        cmd += " --np 30"
        cmd += " --nt 15"
        cmd += " --z-evol-obj 1."
        print(repr(cmd))
        picca.bin.picca_fast_metal_xdmat.main(cmd.split()[1:])

        ### Test
        if self._test:
            path1 = self._masterFiles + "/test_cor/fast_metal_xdmat.fits.gz"
            path2 = self._branchFiles + "/Products/Correlations/fast_metal_xdmat.fits.gz"
            self.compare_fits(path1, path2, "picca_fast_metal_xdmat.py")

        return

    def test_xwick(self):
        """
            Test wick covariances for cross
        """

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
        cmd += " --rej 0.99"
        cmd += " --nproc 1"
        cmd += " --z-evol-obj 1."
        print(repr(cmd))
        picca.bin.picca_xwick.main(cmd.split()[1:])

        ### Test
        if self._test:
            path1 = self._masterFiles + "/test_cor/xwick.fits.gz"
            path2 = self._branchFiles + "/Products/Correlations/xwick.fits.gz"
            self.compare_fits(path1, path2, "picca_xwick.py")

        return

    def test_cf_angl_zcuts(self):
        """
            Test angular correlation function
        """
        importlib.reload(picca.cf)
        userprint("\n")
        ### Send
        cmd = "picca_cf_angl.py"
        cmd += " --in-dir " + self._masterFiles + "/test_delta/Delta_LYA/"
        cmd += " --out " + self._branchFiles + "/Products/Correlations/cf_angl_zcuts.fits.gz"
        cmd += " --nproc 1"
        cmd += " --z-cut-min 2.25"
        cmd += " --z-cut-max 2.3"
        cmd += " --z-min-sources 2.3"
        cmd += " --z-max-sources 2.5"
        print(repr(cmd))
        picca.bin.picca_cf_angl.main(cmd.split()[1:])

        ### Test
        if self._test:
            path1 = self._masterFiles + "/test_cor/cf_angl_zcuts.fits.gz"
            path2 = self._branchFiles + "/Products/Correlations/cf_angl_zcuts.fits.gz"
            self.compare_fits(path1, path2, "picca_cf_angl.py")

        return

    def test_cf_zcuts(self):
        """
            Test correlation function
        """
        importlib.reload(picca.cf)

        userprint("\n")
        ### Send
        cmd = "picca_cf.py"
        cmd += " --in-dir " + self._masterFiles + "/test_delta/Delta_LYA/"
        cmd += " --out " + self._branchFiles + "/Products/Correlations/cf_zcuts.fits.gz"
        cmd += " --rp-min +0.0"
        cmd += " --rp-max +60.0"
        cmd += " --rt-max +60.0"
        cmd += " --np 15"
        cmd += " --nt 15"
        cmd += " --nproc 1"
        cmd += ' --remove-same-half-plate-close-pairs'
        cmd += " --z-cut-min 2.25"
        cmd += " --z-cut-max 2.3"
        cmd += " --z-min-sources 2.3"
        cmd += " --z-max-sources 2.5"
        print(repr(cmd))
        picca.bin.picca_cf.main(cmd.split()[1:])

        ### Test
        if self._test:
            path1 = self._masterFiles + "/test_cor/cf_zcuts.fits.gz"
            path2 = self._branchFiles + "/Products/Correlations/cf_zcuts.fits.gz"
            self.compare_fits(path1, path2, "picca_cf.py")

        return

    def test_cf_image_data_zcuts(self):
        """
            Test correlation function reading image data
        """
        importlib.reload(picca.cf)

        userprint("\n")
        ### Send
        cmd = "picca_cf.py"
        cmd += " --in-dir " + self._masterFiles + "/test_delta/Delta_LYA_image/"
        cmd += " --out " + self._branchFiles + "/Products/Correlations/cf_zcuts.fits.gz"
        cmd += " --rp-min +0.0"
        cmd += " --rp-max +60.0"
        cmd += " --rt-max +60.0"
        cmd += " --np 15"
        cmd += " --nt 15"
        cmd += " --nproc 1"
        cmd += " --z-cut-min 2.25"
        cmd += " --z-cut-max 2.3"
        cmd += " --z-min-sources 2.3"
        cmd += " --z-max-sources 2.5"
        print(repr(cmd))
        picca.bin.picca_cf.main(cmd.split()[1:])

        ### Test
        if self._test:
            path1 = self._masterFiles + "/test_cor/cf_image_zcuts.fits.gz"
            path2 = self._branchFiles + "/Products/Correlations/cf_zcuts.fits.gz"
            self.compare_fits(path1, path2, "picca_cf.py")

        return

    def test_dmat_zcuts(self):
        """
            Test distortion matrix
        """
        importlib.reload(picca.cf)

        userprint("\n")
        ### Send
        cmd = "picca_dmat.py"
        cmd += " --in-dir " + self._masterFiles + "/test_delta/Delta_LYA/"
        cmd += " --out " + self._branchFiles + "/Products/Correlations/dmat_zcuts.fits.gz"
        cmd += " --rp-min +0.0"
        cmd += " --rp-max +60.0"
        cmd += " --rt-max +60.0"
        cmd += " --np 15"
        cmd += " --nt 15"
        cmd += " --rej 0.99"
        cmd += " --nproc 1"
        cmd += ' --remove-same-half-plate-close-pairs'
        cmd += " --z-cut-min 2.25"
        cmd += " --z-cut-max 2.3"
        cmd += " --z-min-sources 2.3"
        cmd += " --z-max-sources 2.5"
        cmd += ' --no-redshift-evolution'

        print(repr(cmd))
        picca.bin.picca_dmat.main(cmd.split()[1:])

        ### Test
        if self._test:
            path1 = self._masterFiles + "/test_cor/dmat_zcuts.fits.gz"
            path2 = self._branchFiles + "/Products/Correlations/dmat_zcuts.fits.gz"
            self.compare_fits(path1, path2, "picca_dmat.py")

        return

    def test_metal_dmat_zcuts(self):
        """
            Test metal distortion matrix
        """
        importlib.reload(picca.cf)

        userprint("\n")
        ### Send
        cmd = "picca_metal_dmat.py"
        cmd += " --in-dir " + self._masterFiles + "/test_delta/Delta_LYA/"
        cmd += " --out " + self._branchFiles + "/Products/Correlations/metal_dmat_zcuts.fits.gz"
        cmd += r" --abs-igm SiIII(1207)"
        cmd += " --rp-min +0.0"
        cmd += " --rp-max +60.0"
        cmd += " --rt-max +60.0"
        cmd += " --np 15"
        cmd += " --nt 15"
        cmd += " --rej 0.99"
        cmd += " --nproc 1"
        cmd += ' --remove-same-half-plate-close-pairs'
        cmd += " --z-cut-min 2.25"
        cmd += " --z-cut-max 2.3"
        cmd += " --z-min-sources 2.3"
        cmd += " --z-max-sources 2.5"
        print(repr(cmd))
        picca.bin.picca_metal_dmat.main(cmd.split()[1:])

        ### Test
        if self._test:
            path1 = self._masterFiles + "/test_cor/metal_dmat_zcuts.fits.gz"
            path2 = self._branchFiles + "/Products/Correlations/metal_dmat_zcuts.fits.gz"
            self.compare_fits(path1, path2, "picca_metal_dmat.py")

        return

    def test_fast_metal_dmat_zcuts(self):
        """
            Test metal distortion matrix
        """
        importlib.reload(picca.cf)

        userprint("\n")
        ### Send
        cmd = "picca_metal_dmat.py"
        cmd += " --in-attributes " + self._masterFiles + "/test_cor/input_from_delta_extraction_lya_nodla/Log/delta_attributes.fits.gz"
        cmd += " --out " + self._branchFiles + "/Products/Correlations/fast_metal_dmat_zcuts.fits.gz"
        cmd += r" --abs-igm SiIII(1207)"
        cmd += " --rp-min +0.0"
        cmd += " --rp-max +60.0"
        cmd += " --rt-max +60.0"
        cmd += " --np 15"
        cmd += " --nt 15"
        cmd += " --z-cut-min 2.25"
        cmd += " --z-cut-max 2.3"
        cmd += " --z-min-sources 2.3"
        cmd += " --z-max-sources 2.5"
        print(repr(cmd))
        picca.bin.picca_fast_metal_dmat.main(cmd.split()[1:])

        ### Test
        if self._test:
            path1 = self._masterFiles + "/test_cor/fast_metal_dmat_zcuts.fits.gz"
            path2 = self._branchFiles + "/Products/Correlations/fast_metal_dmat_zcuts.fits.gz"
            self.compare_fits(path1, path2, "picca_fast_metal_dmat.py")

        return

    def test_wick_zcuts(self):
        """
            Test wick covariances
        """
        importlib.reload(picca.cf)

        userprint("\n")
        ### Send
        cmd = "picca_wick.py"
        cmd += " --in-dir " + self._masterFiles + "/test_delta/Delta_LYA/"
        cmd += " --out " + self._branchFiles + "/Products/Correlations/wick_zcuts.fits.gz"
        cmd += " --cf1d " + self._masterFiles + "/test_cor/cf1d.fits.gz"
        cmd += " --rp-min +0.0"
        cmd += " --rp-max +60.0"
        cmd += " --rt-max +60.0"
        cmd += " --np 15"
        cmd += " --nt 15"
        cmd += " --rej 0.99"
        cmd += " --nproc 1"
        cmd += " --z-cut-min 2.25"
        cmd += " --z-cut-max 2.3"
        cmd += " --z-min-sources 2.3"
        cmd += " --z-max-sources 2.5"
        print(repr(cmd))
        picca.bin.picca_wick.main(cmd.split()[1:])

        ### Test
        if self._test:
            path1 = self._masterFiles + "/test_cor/wick_zcuts.fits.gz"
            path2 = self._branchFiles + "/Products/Correlations/wick_zcuts.fits.gz"
            self.compare_fits(path1, path2, "picca_wick.py")

        return

    def test_cf_cross_zcuts(self):
        """
            Test export of cross correlation function
        """
        importlib.reload(picca.cf)

        userprint("\n")
        ### Send
        cmd = "picca_cf.py"
        cmd += " --in-dir " + self._masterFiles + "/test_delta/Delta_LYA/"
        cmd += " --in-dir2 " + self._masterFiles + "/test_delta/Delta_LYA/"
        cmd += " --out " + self._branchFiles + "/Products/Correlations/cf_cross_zcuts.fits.gz"
        cmd += " --rp-min -60.0"
        cmd += " --rp-max +60.0"
        cmd += " --rt-max +60.0"
        cmd += " --np 30"
        cmd += " --nt 15"
        cmd += " --nproc 1"
        cmd += ' --remove-same-half-plate-close-pairs'
        cmd += " --unfold-cf"
        cmd += " --z-cut-min 2.25"
        cmd += " --z-cut-max 2.3"
        cmd += " --z-min-sources 2.3"
        cmd += " --z-max-sources 2.5"
        print(repr(cmd))
        picca.bin.picca_cf.main(cmd.split()[1:])

        ### Test
        if self._test:
            path1 = self._masterFiles + "/test_cor/cf_cross_zcuts.fits.gz"
            path2 = self._branchFiles + "/Products/Correlations/cf_cross_zcuts.fits.gz"
            self.compare_fits(path1, path2, "picca_cf.py")

        return

    def test_dmat_cross_zcuts(self):
        """
            Test cross distortion matrix
        """
        importlib.reload(picca.cf)


        userprint("\n")
        ### Send
        cmd = "picca_dmat.py"
        cmd += " --in-dir " + self._masterFiles + "/test_delta/Delta_LYA/"
        cmd += " --in-dir2 " + self._masterFiles + "/test_delta/Delta_LYA/"
        cmd += " --out " + self._branchFiles + "/Products/Correlations/dmat_cross_zcuts.fits.gz"
        cmd += " --rp-min -60.0"
        cmd += " --rp-max +60.0"
        cmd += " --rt-max +60.0"
        cmd += " --np 30"
        cmd += " --nt 15"
        cmd += " --rej 0.99"
        cmd += " --nproc 1"
        cmd += ' --remove-same-half-plate-close-pairs'
        cmd += " --unfold-cf"
        cmd += " --z-cut-min 2.25"
        cmd += " --z-cut-max 2.3"
        cmd += " --z-min-sources 2.3"
        cmd += " --z-max-sources 2.5"
        cmd += ' --no-redshift-evolution'

        print(repr(cmd))
        picca.bin.picca_dmat.main(cmd.split()[1:])

        ### Test
        if self._test:
            path1 = self._masterFiles + "/test_cor/dmat_cross_zcuts.fits.gz"
            path2 = self._branchFiles + "/Products/Correlations/dmat_cross_zcuts.fits.gz"
            self.compare_fits(path1, path2, "picca_dmat.py")

        return

    def test_metal_dmat_cross_zcuts(self):
        """
            Test metal cross distortion matrix
        """
        importlib.reload(picca.cf)

        userprint("\n")
        ### Send
        cmd = "picca_metal_dmat.py"
        cmd += " --in-dir " + self._masterFiles + "/test_delta/Delta_LYA/"
        cmd += " --in-dir2 " + self._masterFiles + "/test_delta/Delta_LYA/"
        cmd += " --out " + self._branchFiles + \
            "/Products/Correlations/metal_dmat_cross_zcuts.fits.gz"
        cmd += r" --abs-igm SiIII(1207)"
        cmd += r" --abs-igm2 SiIII(1207)"
        cmd += " --rp-min -60.0"
        cmd += " --rp-max +60.0"
        cmd += " --rt-max +60.0"
        cmd += " --np 30"
        cmd += " --nt 15"
        cmd += " --rej 0.99"
        cmd += " --nproc 1"
        cmd += ' --remove-same-half-plate-close-pairs'
        cmd += " --unfold-cf"
        cmd += " --z-cut-min 2.25"
        cmd += " --z-cut-max 2.3"
        cmd += " --z-min-sources 2.3"
        cmd += " --z-max-sources 2.5"
        print(repr(cmd))
        picca.bin.picca_metal_dmat.main(cmd.split()[1:])

        ### Test
        if self._test:
            path1 = self._masterFiles + "/test_cor/metal_dmat_cross_zcuts.fits.gz"
            path2 = self._branchFiles + "/Products/Correlations/metal_dmat_cross_zcuts.fits.gz"
            self.compare_fits(path1, path2, "picca_metal_dmat.py")

        return

    def test_fast_metal_dmat_cross_zcuts(self):
        """
            Test metal cross distortion matrix
        """
        importlib.reload(picca.cf)

        userprint("\n")
        ### Send
        cmd = "picca_fast_metal_dmat.py"
        cmd += " --in-attributes " + self._masterFiles + "/test_cor/input_from_delta_extraction_lya_nodla/Log/delta_attributes.fits.gz"
        cmd += " --in-attributes2 " + self._masterFiles + "/test_cor/input_from_delta_extraction_lya_nodla/Log/delta_attributes.fits.gz"
        cmd += " --out " + self._branchFiles + \
            "/Products/Correlations/fast_metal_dmat_cross_zcuts.fits.gz"
        cmd += r" --abs-igm SiIII(1207)"
        cmd += r" --abs-igm2 SiIII(1207)"
        cmd += " --rp-min -60.0"
        cmd += " --rp-max +60.0"
        cmd += " --rt-max +60.0"
        cmd += " --np 30"
        cmd += " --nt 15"
        cmd += " --unfold-cf"
        cmd += " --z-cut-min 2.25"
        cmd += " --z-cut-max 2.3"
        cmd += " --z-min-sources 2.3"
        cmd += " --z-max-sources 2.5"
        print(repr(cmd))
        picca.bin.picca_fast_metal_dmat.main(cmd.split()[1:])

        ### Test
        if self._test:
            path1 = self._masterFiles + "/test_cor/fast_metal_dmat_cross_zcuts.fits.gz"
            path2 = self._branchFiles + "/Products/Correlations/fast_metal_dmat_cross_zcuts.fits.gz"
            self.compare_fits(path1, path2, "picca_fast_metal_dmat.py")

        return

    def test_xcf_angl_zcuts(self):
        """
            Test angular cross correlation function
        """

        userprint("\n")
        ### Send
        cmd = "picca_xcf_angl.py"
        cmd += " --in-dir " + self._masterFiles + "/test_delta/Delta_LYA/"
        cmd += " --drq " + self._masterFiles + "/test_delta/cat.fits"
        cmd += " --out " + self._branchFiles + "/Products/Correlations/xcf_angl_zcuts.fits.gz"
        cmd += " --nproc 1"
        cmd += " --z-evol-obj 1."
        cmd += " --z-cut-min 2.25"
        cmd += " --z-cut-max 2.3"
        cmd += " --z-min-sources 2.3"
        cmd += " --z-max-sources 2.5"
        print(repr(cmd))
        picca.bin.picca_xcf_angl.main(cmd.split()[1:])

        ### Test
        if self._test:
            path1 = self._masterFiles + "/test_cor/xcf_angl_zcuts.fits.gz"
            path2 = self._branchFiles + "/Products/Correlations/xcf_angl_zcuts.fits.gz"
            self.compare_fits(path1, path2, "picca_xcf_angl.py")

        return

    def test_xcf_zcuts(self):
        """
            Test cross correlation function
        """

        userprint("\n")
        ### Send
        cmd = "picca_xcf.py"
        cmd += " --in-dir " + self._masterFiles + "/test_delta/Delta_LYA/"
        cmd += " --drq " + self._masterFiles + "/test_delta/cat.fits"
        cmd += " --out " + self._branchFiles + "/Products/Correlations/xcf_zcuts.fits.gz"
        cmd += " --rp-min -60.0"
        cmd += " --rp-max +60.0"
        cmd += " --rt-max +60.0"
        cmd += " --np 30"
        cmd += " --nt 15"
        cmd += " --nproc 1"
        cmd += " --z-evol-obj 1."
        cmd += " --z-cut-min 2.25"
        cmd += " --z-cut-max 2.3"
        cmd += " --z-min-sources 2.3"
        cmd += " --z-max-sources 2.5"
        print(repr(cmd))
        picca.bin.picca_xcf.main(cmd.split()[1:])

        ### Test
        if self._test:
            path1 = self._masterFiles + "/test_cor/xcf_zcuts.fits.gz"
            path2 = self._branchFiles + "/Products/Correlations/xcf_zcuts.fits.gz"
            self.compare_fits(path1, path2, "picca_xcf.py")

        return

    def test_xdmat_zcuts(self):
        """
            Test cross distortion matrix
        """
        importlib.reload(picca.xcf)


        userprint("\n")
        ### Send
        cmd = "picca_xdmat.py"
        cmd += " --in-dir " + self._masterFiles + "/test_delta/Delta_LYA/"
        cmd += " --drq " + self._masterFiles + "/test_delta/cat.fits"
        cmd += " --out " + self._branchFiles + "/Products/Correlations/xdmat_zcuts.fits.gz"
        cmd += " --rp-min -60.0"
        cmd += " --rp-max +60.0"
        cmd += " --rt-max +60.0"
        cmd += " --np 30"
        cmd += " --nt 15"
        cmd += " --rej 0.99"
        cmd += " --nproc 1"
        cmd += " --z-evol-obj 1."
        cmd += " --z-cut-min 2.25"
        cmd += " --z-cut-max 2.3"
        cmd += " --z-min-sources 2.3"
        cmd += " --z-max-sources 2.5"
        cmd += ' --no-redshift-evolution'
        print(repr(cmd))
        picca.bin.picca_xdmat.main(cmd.split()[1:])

        ### Test
        if self._test:
            path1 = self._masterFiles + "/test_cor/xdmat_zcuts.fits.gz"
            path2 = self._branchFiles + "/Products/Correlations/xdmat_zcuts.fits.gz"
            self.compare_fits(path1, path2, "picca_xdmat.py")

        return

    def test_metal_xdmat_zcuts(self):
        """
            Test metal cross distortion matrix
        """
        userprint("\n")
        ### Send
        cmd = "picca_metal_xdmat.py"
        cmd += " --in-dir " + self._masterFiles + "/test_delta/Delta_LYA/"
        cmd += " --drq " + self._masterFiles + "/test_delta/cat.fits"
        cmd += " --out " + self._branchFiles + "/Products/Correlations/metal_xdmat_zcuts.fits.gz"
        cmd += r" --abs-igm SiIII(1207)"
        cmd += " --rp-min -60.0"
        cmd += " --rp-max +60.0"
        cmd += " --rt-max +60.0"
        cmd += " --np 30"
        cmd += " --nt 15"
        cmd += " --rej 0.99"
        cmd += " --nproc 1"
        cmd += " --z-evol-obj 1."
        cmd += " --z-cut-min 2.25"
        cmd += " --z-cut-max 2.3"
        cmd += " --z-min-sources 2.3"
        cmd += " --z-max-sources 2.5"
        print(repr(cmd))
        picca.bin.picca_metal_xdmat.main(cmd.split()[1:])

        ### Test
        if self._test:
            path1 = self._masterFiles + "/test_cor/metal_xdmat_zcuts.fits.gz"
            path2 = self._branchFiles + "/Products/Correlations/metal_xdmat_zcuts.fits.gz"
            self.compare_fits(path1, path2, "picca_metal_xdmat.py")

        return

    def test_xwick_zcuts(self):
        """
            Test wick covariances for cross
        """

        userprint("\n")
        ### Send
        cmd = "picca_xwick.py"
        cmd += " --in-dir " + self._masterFiles + "/test_delta/Delta_LYA/"
        cmd += " --drq " + self._masterFiles + "/test_delta/cat.fits"
        cmd += " --out " + self._branchFiles + "/Products/Correlations/xwick_zcuts.fits.gz"
        cmd += " --cf1d " + self._masterFiles + "/test_cor/cf1d.fits.gz"
        cmd += " --rp-min -60.0"
        cmd += " --rp-max +60.0"
        cmd += " --rt-max +60.0"
        cmd += " --np 30"
        cmd += " --nt 15"
        cmd += " --rej 0.99"
        cmd += " --nproc 1"
        cmd += " --z-evol-obj 1."
        cmd += " --z-cut-min 2.25"
        cmd += " --z-cut-max 2.3"
        cmd += " --z-min-sources 2.3"
        cmd += " --z-max-sources 2.5"
        print(repr(cmd))
        picca.bin.picca_xwick.main(cmd.split()[1:])

        ### Test
        if self._test:
            path1 = self._masterFiles + "/test_cor/xwick_zcuts.fits.gz"
            path2 = self._branchFiles + "/Products/Correlations/xwick_zcuts.fits.gz"
            self.compare_fits(path1, path2, "picca_xwick.py")

        return

    def test_export_xcf(self):
        """
            Test the export of the cross correlation function
        """


        userprint("\n")
        ### Send
        cmd = "picca_export.py"
        cmd += " --data " + self._masterFiles + "/test_cor/xcf.fits.gz"
        cmd += " --dmat " + self._masterFiles + "/test_cor/xdmat.fits.gz"
        cmd += " --out " + self._branchFiles + \
            "/Products/Correlations/exported_xcf.fits.gz"
        print(repr(cmd))
        picca.bin.picca_export.main(cmd.split()[1:])

        return

    def test_export_cross_covariance_cf_xcf(self):
        """
            Test the export of cross_covariances between correlation function and cross correlation function
        """


        userprint("\n")
        ### Send
        cmd = "picca_export_cross_covariance.py"
        cmd += " --data1 " + self._masterFiles + "/test_cor/cf.fits.gz"
        cmd += " --data2 " + self._masterFiles + "/test_cor/xcf.fits.gz"
        cmd += " --out " + self._branchFiles + \
            "/Products/Correlations/exported_cross_covariance_cf_xcf.fits.gz"
        print(repr(cmd))
        picca.bin.picca_export_cross_covariance.main(cmd.split()[1:])

        return

    def test_co(self):
        """
            Test the covariances
        """

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
        print(repr(cmd))
        picca.bin.picca_co.main(cmd.split()[1:])
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
        print(repr(cmd))
        picca.bin.picca_co.main(cmd.split()[1:])
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
        print(repr(cmd))
        picca.bin.picca_co.main(cmd.split()[1:])
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
        print(repr(cmd))
        picca.bin.picca_co.main(cmd.split()[1:])

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
        """
            Test the export of covariances
        """

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
        print(repr(cmd))
        picca.bin.picca_export_co.main(cmd.split()[1:])

        return


if __name__ == '__main__':
    unittest.main()
