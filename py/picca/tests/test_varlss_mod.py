"""Tests of the recomputation of the delta weights with another var lss mod

(option varlss_mod_factor of picca.io.read_deltas, --varlss-mod-factor of
the correlation scripts)
"""
import glob
import os
import shutil
import tempfile
import unittest

import fitsio
import numpy as np
from scipy.interpolate import interp1d

from picca import io
import picca.bin.picca_cf

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# output of the delta extraction tests, with VARLSSMD = 1.0
DELTA_DIR = os.path.join(
    THIS_DIR, "delta_extraction", "data", "delta_extraction_lin_image")
IN_DIR = os.path.join(DELTA_DIR, "Delta")
ATTRIBUTES = os.path.join(DELTA_DIR, "Log", "delta_attributes.fits.gz")
NEW_VAR_LSS_MOD = 7.5


def sorted_deltas(data):
    """Deltas of a read_deltas output, in a reproducible order"""
    deltas = [delta for deltas in data.values() for delta in deltas]
    return sorted(deltas, key=lambda delta: (delta.los_id, delta.log_lambda[0]))


class TestVarLssMod(unittest.TestCase):
    """Tests of the weights recomputed with another var lss mod"""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def read_deltas(self, **kwargs):
        """Reads the test deltas, without projection nor redshift evolution
        of the weights"""
        data = io.read_deltas(
            IN_DIR, nside=16, lambda_abs=1215.67, alpha=1., z_ref=2.25,
            cosmo=None, no_project=True, nproc=1,
            **dict({"delta_attributes": ATTRIBUTES}, **kwargs))[0]
        return sorted_deltas(data)

    def test_read_var_lss(self):
        """read_var_lss returns the var lss grid and VARLSSMD"""
        log_lambda, var_lss, var_lss_mod = io.read_var_lss(ATTRIBUTES)
        self.assertEqual(var_lss_mod, 1.0)
        self.assertEqual(log_lambda.shape, var_lss.shape)
        self.assertTrue(np.all(np.diff(log_lambda) > 0))

    def test_same_var_lss_mod(self):
        """The weights are unchanged with the var lss mod of the extraction"""
        reference = self.read_deltas()
        same = self.read_deltas(varlss_mod_factor=1.0)
        self.assertEqual(len(reference), len(same))
        for delta, delta_same in zip(reference, same):
            np.testing.assert_array_equal(delta.weights, delta_same.weights)

    def test_new_var_lss_mod(self):
        """1/w' = 1/w + (new var lss mod - VARLSSMD) * var_lss"""
        reference = self.read_deltas()
        new = self.read_deltas(varlss_mod_factor=NEW_VAR_LSS_MOD)
        log_lambda, var_lss, var_lss_mod = io.read_var_lss(ATTRIBUTES)
        get_var_lss = interp1d(
            log_lambda, var_lss, kind="cubic", fill_value="extrapolate")
        for delta, delta_new in zip(reference, new):
            w = delta.weights > 0
            np.testing.assert_array_equal(delta_new.weights[~w], 0.)
            np.testing.assert_allclose(
                1. / delta_new.weights[w],
                1. / delta.weights[w]
                + (NEW_VAR_LSS_MOD - var_lss_mod) * get_var_lss(delta.log_lambda[w]),
                rtol=1e-12)
            # the weights decrease, as var lss mod increases
            self.assertTrue(np.all(delta_new.weights[w] < delta.weights[w]))
            # the deltas themselves are unchanged
            np.testing.assert_array_equal(delta_new.delta, delta.delta)

    def test_update_before_rebinning(self):
        """The weights are updated before rebinning"""
        log_lambda, var_lss, var_lss_mod = io.read_var_lss(ATTRIBUTES)
        var_lss_update = (log_lambda, var_lss, NEW_VAR_LSS_MOD - var_lss_mod)
        filename = sorted(glob.glob(IN_DIR + "/delta-*.fits*"))[0]
        rebinned = io.read_delta_file(
            filename, rebin_factor=3, var_lss_update=var_lss_update)
        manual = io.read_delta_file(filename, var_lss_update=var_lss_update)
        with fitsio.FITS(filename) as hdul:
            delta_lambda = hdul["LAMBDA"].read_header()["DELTA_LAMBDA"]
        for delta in manual:
            delta.rebin(3, dwave=delta_lambda)
        self.assertEqual(len(rebinned), len(manual))
        for delta, delta_manual in zip(rebinned, manual):
            np.testing.assert_allclose(delta.weights, delta_manual.weights, rtol=1e-14)
            np.testing.assert_allclose(delta.delta, delta_manual.delta, rtol=1e-14)

    def test_missing_varlssmd(self):
        """An attributes file without VARLSSMD raises a KeyError"""
        old_attributes = os.path.join(self.tmp_dir, "delta_attributes.fits.gz")
        with fitsio.FITS(ATTRIBUTES) as hdul, \
                fitsio.FITS(old_attributes, "rw", clobber=True) as old_hdul:
            for hdu in hdul[1:]:
                header = hdu.read_header()
                if hdu.get_extname() == "VAR_FUNC":
                    header.delete("VARLSSMD")
                old_hdul.write(hdu.read(), header=header, extname=hdu.get_extname())
        with self.assertRaises(KeyError):
            io.read_var_lss(old_attributes)
        with self.assertRaises(KeyError):
            self.read_deltas(varlss_mod_factor=NEW_VAR_LSS_MOD,
                             delta_attributes=old_attributes)

    def test_negative_var_lss_mod(self):
        """A negative var lss mod raises a ValueError"""
        with self.assertRaises(ValueError):
            self.read_deltas(varlss_mod_factor=-1.)

    def test_second_delta_attributes(self):
        """Choice of the attributes file of the second set of deltas"""
        # given explicitly
        self.assertEqual(
            io.second_delta_attributes("att1", "att2", "dir2", NEW_VAR_LSS_MOD), "att2")
        # same deltas, or weights unchanged: --in-attributes, as before
        self.assertEqual(
            io.second_delta_attributes("att1", None, None, NEW_VAR_LSS_MOD), "att1")
        self.assertEqual(io.second_delta_attributes("att1", None, "dir2", None), "att1")
        # standard location relative to the second set of deltas
        self.assertIsNone(
            io.second_delta_attributes(None, None, "dir2", NEW_VAR_LSS_MOD))
        # the var lss of the first set would be used for the second one
        with self.assertRaises(ValueError):
            io.second_delta_attributes("att1", None, "dir2", NEW_VAR_LSS_MOD)

    def test_picca_cf(self):
        """picca_cf: --varlss-mod-factor equal to VARLSSMD changes nothing,
        another value changes the weights"""
        results = {}
        for name, option in [("none", []), ("same", ["--varlss-mod-factor", "1.0"]),
                             ("new", ["--varlss-mod-factor", str(NEW_VAR_LSS_MOD)])]:
            out = os.path.join(self.tmp_dir, f"cf_{name}.fits.gz")
            picca.bin.picca_cf.main([
                "--in-dir", IN_DIR, "--out", out, "--rp-max", "60",
                "--rt-max", "60", "--np", "15", "--nt", "15", "--nproc", "1",
                "--no-project"] + option)
            results[name] = fitsio.read(out, ext=2)
        for column in ["DA", "WE"]:
            np.testing.assert_array_equal(results["none"][column], results["same"][column])
        self.assertFalse(np.allclose(results["none"]["WE"], results["new"]["WE"]))


if __name__ == "__main__":
    unittest.main()
