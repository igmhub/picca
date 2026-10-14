'''
Tests of the GPU (--gpu) versions of the correlation scripts.

Each case runs a script on the CPU and again with --gpu on the small test data of test_3_cor.py and checks that
  * the two outputs agree to a relative tolerance of 1e-10 (the GPU sums in a different order; at full scale the
    observed difference is about 1e-13),
  * the numbers of pairs (NB) are identical, and
  * the GPU output agrees with the stored reference output of the CPU tests.
All tests are skipped when no CUDA device is available.
'''

import importlib
import os
import unittest

import warnings

import fitsio
import numpy as np

import picca.cf
import picca.xcf
import picca.bin.picca_cf
import picca.bin.picca_xcf
import picca.bin.picca_dmat
import picca.bin.picca_xdmat
import picca.bin.picca_metal_dmat

from picca.tests.test_helpers import AbstractTest
from picca.tests.test_3_cor import reset_cf, reset_xcf


def cuda_available():
    """True if numba can use a CUDA GPU"""
    try:
        from numba import cuda  # pylint: disable=import-outside-toplevel
        return bool(cuda.is_available())
    except Exception:  # pylint: disable=broad-except
        return False


# GPU against CPU: float64 sums in a different order
GPU_RTOL = 1e-10
GPU_ATOL = 1e-12


@unittest.skipUnless(cuda_available(), "needs a CUDA GPU")
class TestGpu(AbstractTest):
    """Tests the --gpu option of the correlation function and distortion matrix scripts"""

    def produce_folder(self):
        """Creates the output folders"""
        for fold in ["/Products/", "/Products/Correlations/"]:
            if not os.path.isdir(self._branchFiles + fold):
                os.mkdir(self._branchFiles + fold)

    def setUp(self):
        """The tests use tiny data sets, so numba warns that the grids under-utilise the GPU"""
        warnings.filterwarnings("ignore", message=".*under-utilization.*")

    def tearDown(self):
        """Resets the module variables the scripts set"""
        reset_cf()
        reset_xcf()

    def delta_args(self):
        """Arguments with the test deltas"""
        return [
            "--in-dir", self._masterFiles + "/test_delta/Delta_LYA/",
            "--in-attributes", self._masterFiles + "/test_delta/delta_attributes.fits.gz",
        ]

    def cross_args(self):
        """Arguments with the test deltas and the quasar catalogue"""
        return self.delta_args() + ["--drq", self._masterFiles + "/test_delta/cat.fits"]

    def run_script(self, script, args, out, gpu):
        """Runs picca_<script>.py with a fresh state of the picca.cf and picca.xcf modules"""
        importlib.reload(picca.cf)
        importlib.reload(picca.xcf)
        module = getattr(picca.bin, "picca_" + script)
        cmd = list(args) + ["--out", out]
        if gpu:
            cmd.append("--gpu")
        print(repr(" ".join(["picca_%s.py" % script] + cmd)))
        module.main(cmd)
        reset_cf()
        reset_xcf()

    def check(self, script, name, args, reference=None, pair_counts=False):
        """Runs the script on the CPU and the GPU and compares the outputs

        Args:
            script: name of the script without the picca_ prefix and the .py
            name: name of the output file (without the extension)
            args: arguments of the script, without --out and --gpu
            reference: name of the stored reference file in test_cor (without the extension),
                if there is one with the same arguments
            pair_counts: also compare the number of pairs (NB) exactly
        """
        folder = self._branchFiles + "/Products/Correlations/"
        cpu = folder + name + "_cpu.fits.gz"
        gpu = folder + name + "_gpu.fits.gz"
        self.run_script(script, args, cpu, gpu=False)
        self.run_script(script, args, gpu, gpu=True)

        self.compare_fits(cpu, gpu, "picca_%s.py --gpu against the CPU" % script,
                          rel_tolerance=GPU_RTOL, abs_tolerance=GPU_ATOL)
        if pair_counts:
            with fitsio.FITS(cpu) as fits_cpu, fitsio.FITS(gpu) as fits_gpu:
                np.testing.assert_array_equal(fits_cpu["ATTRI"]["NB"][:], fits_gpu["ATTRI"]["NB"][:])
        if reference is not None and self._test:
            self.compare_fits(self._masterFiles + "/test_cor/" + reference + ".fits.gz", gpu,
                              "picca_%s.py --gpu against the reference" % script)

    # ---- auto-correlations of the forests

    def test_gpu_cf(self):
        """cf with the removal of close pairs on the same half plate (the slow neighbour search)"""
        self.check("cf", "cf", self.delta_args() + [
            "--rp-min", "+0.0", "--rp-max", "+60.0", "--rt-max", "+60.0", "--np", "15", "--nt", "15",
            "--nproc", "1", "--remove-same-half-plate-close-pairs"],
            reference="cf", pair_counts=True)

    def test_gpu_cf_fast_neighbours(self):
        """cf without that option (the vectorised neighbour search)"""
        self.check("cf", "cf_fast", self.delta_args() + [
            "--rp-min", "+0.0", "--rp-max", "+60.0", "--rt-max", "+60.0", "--np", "15", "--nt", "15",
            "--nproc", "1"], pair_counts=True)

    def test_gpu_cf_two_runs_different_data(self):
        """two GPU runs in the same process with different sets of forests (cached healpix ranges must not leak)"""
        common = ["--rp-min", "+0.0", "--rp-max", "+60.0", "--rt-max", "+60.0", "--np", "15", "--nt", "15",
                  "--nproc", "1"]
        self.check("cf", "cf_run1", self.delta_args() + common, pair_counts=True)
        self.check("cf", "cf_run2", self.delta_args() + common + [
            "--z-min-sources", "2.3", "--z-max-sources", "2.5"], pair_counts=True)

    def test_gpu_cf_cross(self):
        """cf of two sets of deltas"""
        self.check("cf", "cf_cross", self.delta_args() + [
            "--in-dir2", self._masterFiles + "/test_delta/Delta_LYA/",
            "--rp-min", "-60.0", "--rp-max", "+60.0", "--rt-max", "+60.0", "--np", "30", "--nt", "15",
            "--nproc", "1", "--remove-same-half-plate-close-pairs", "--unfold-cf"],
            reference="cf_cross", pair_counts=True)

    def test_gpu_cf_zcuts(self):
        """cf with cuts in the redshift of the pairs and of the sources"""
        self.check("cf", "cf_zcuts", self.delta_args() + [
            "--rp-min", "+0.0", "--rp-max", "+60.0", "--rt-max", "+60.0", "--np", "15", "--nt", "15",
            "--nproc", "1", "--remove-same-half-plate-close-pairs",
            "--z-min-pairs", "2.25", "--z-max-pairs", "2.3", "--z-min-sources", "2.3", "--z-max-sources", "2.5"],
            reference="cf_zcuts", pair_counts=True)

    def test_gpu_cf_rmu(self):
        """cf binned in r and mu"""
        self.check("cf", "cf_rmu", self.delta_args() + [
            "--rp-min", "+0.0", "--rp-max", "+60.0", "--rt-max", "+60.0", "--np", "15", "--nt", "15",
            "--nproc", "1", "--rmu-binning"], pair_counts=True)

    # ---- cross-correlation of the forests with quasars

    def test_gpu_xcf(self):
        """xcf"""
        self.check("xcf", "xcf", self.cross_args() + [
            "--rp-min", "-60.0", "--rp-max", "+60.0", "--rt-max", "+60.0", "--np", "30", "--nt", "15",
            "--nproc", "1", "--z-evol-obj", "1."], reference="xcf", pair_counts=True)

    def test_gpu_xcf_zcuts(self):
        """xcf with cuts in redshift"""
        self.check("xcf", "xcf_zcuts", self.cross_args() + [
            "--rp-min", "-60.0", "--rp-max", "+60.0", "--rt-max", "+60.0", "--np", "30", "--nt", "15",
            "--nproc", "1", "--z-evol-obj", "1.",
            "--z-min-pairs", "2.25", "--z-max-pairs", "2.3", "--z-min-sources", "2.3", "--z-max-sources", "2.5"],
            reference="xcf_zcuts", pair_counts=True)

    def test_gpu_xcf_rmu(self):
        """xcf binned in r and mu"""
        self.check("xcf", "xcf_rmu", self.cross_args() + [
            "--rp-min", "-60.0", "--rp-max", "+60.0", "--rt-max", "+60.0", "--np", "30", "--nt", "15",
            "--nproc", "1", "--z-evol-obj", "1.", "--rmu-binning"], pair_counts=True)

    def test_gpu_xcf_zerr_cut(self):
        """xcf with the cut on pairs of close quasars"""
        self.check("xcf", "xcf_zerr", self.cross_args() + [
            "--rp-min", "-60.0", "--rp-max", "+60.0", "--rt-max", "+60.0", "--np", "30", "--nt", "15",
            "--nproc", "1", "--z-evol-obj", "1.", "--zerr-cut-deg", "2.0", "--zerr-cut-kms", "20000.0"],
            pair_counts=True)

    # ---- distortion matrices

    def test_gpu_dmat(self):
        """dmat"""
        self.check("dmat", "dmat", self.delta_args() + [
            "--rp-min", "+0.0", "--rp-max", "+60.0", "--rt-max", "+60.0", "--np", "15", "--nt", "15",
            "--rej", "0.99", "--nproc", "1", "--remove-same-half-plate-close-pairs",
            "--no-redshift-evolution"], reference="dmat")

    def test_gpu_dmat_cross(self):
        """dmat of two sets of deltas"""
        self.check("dmat", "dmat_cross", self.delta_args() + [
            "--in-dir2", self._masterFiles + "/test_delta/Delta_LYA/",
            "--rp-min", "-60.0", "--rp-max", "+60.0", "--rt-max", "+60.0", "--np", "30", "--nt", "15",
            "--rej", "0.99", "--nproc", "1", "--remove-same-half-plate-close-pairs", "--unfold-cf",
            "--no-redshift-evolution"], reference="dmat_cross")

    def test_gpu_xdmat(self):
        """xdmat"""
        self.check("xdmat", "xdmat", self.cross_args() + [
            "--rp-min", "-60.0", "--rp-max", "+60.0", "--rt-max", "+60.0", "--np", "30", "--nt", "15",
            "--rej", "0.99", "--nproc", "1", "--z-evol-obj", "1.", "--no-redshift-evolution"],
            reference="xdmat")

    def test_gpu_metal_dmat(self):
        """metal_dmat"""
        self.check("metal_dmat", "metal_dmat", self.delta_args() + [
            "--abs-igm", "SiIII(1207)",
            "--rp-min", "+0.0", "--rp-max", "+60.0", "--rt-max", "+60.0", "--np", "15", "--nt", "15",
            "--rej", "0.99", "--nproc", "1"], reference="metal_dmat")

    # ---- the result of the distortion matrices must not depend on the number of processes with --seed-per-healpix

    def seed_check(self, script, name, args):
        """CPU with one process, GPU with two, both seeded per healpix: the same distortion matrix"""
        folder = self._branchFiles + "/Products/Correlations/"
        cpu = folder + name + "_cpu1.fits.gz"
        gpu = folder + name + "_gpu2.fits.gz"
        self.run_script(script, list(args) + ["--nproc", "1", "--seed-per-healpix"], cpu, gpu=False)
        self.run_script(script, list(args) + ["--nproc", "2", "--seed-per-healpix"], gpu, gpu=True)
        self.compare_fits(cpu, gpu, "picca_%s.py --seed-per-healpix, CPU on 1 process against GPU on 2" % script,
                          rel_tolerance=GPU_RTOL, abs_tolerance=GPU_ATOL)

    def test_gpu_dmat_seed_per_healpix(self):
        """dmat with --seed-per-healpix"""
        self.seed_check("dmat", "dmat_seed", self.delta_args() + [
            "--rp-min", "+0.0", "--rp-max", "+60.0", "--rt-max", "+60.0", "--np", "15", "--nt", "15",
            "--rej", "0.99", "--remove-same-half-plate-close-pairs", "--no-redshift-evolution"])

    def test_gpu_xdmat_seed_per_healpix(self):
        """xdmat with --seed-per-healpix"""
        self.seed_check("xdmat", "xdmat_seed", self.cross_args() + [
            "--rp-min", "-60.0", "--rp-max", "+60.0", "--rt-max", "+60.0", "--np", "30", "--nt", "15",
            "--rej", "0.99", "--z-evol-obj", "1.", "--no-redshift-evolution"])


if __name__ == '__main__':
    unittest.main()
