"""Tests for picca.cov_denoise and the covariance-denoising scripts.

Builds two tiny synthetic mock types with different correlation lengths and
runs the full fit + pipeline chain on them.
"""
import os
import shutil
import tempfile
import unittest

import fitsio
import numpy as np

from picca.cov_denoise import DenoiseModel, load_sample_paths, split_train_test
from picca.cov_denoise.io import WORKFLOW_HELP
from picca.bin import (picca_denoise_cov, picca_denoise_fit_classifier,
                       picca_denoise_pipeline)

N_BINS = 30
MOCK_TYPES = {"typea": 3.0, "typeb": 6.0}  # name: correlation length


def _write_cov(path, cov):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with fitsio.FITS(path, "rw", clobber=True) as fits_file:
        fits_file.write([cov], names=["COV"], extname="COVMAT")


class TestCovDenoise(unittest.TestCase):
    """Smoke tests on synthetic data."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        rng = np.random.default_rng(42)
        bins = np.arange(N_BINS)
        clean = {}
        config = ["[TOY]", f"mock_types = {' '.join(MOCK_TYPES)}",
                  r"train_only_regex = stack\d+x"]
        for name, length in MOCK_TYPES.items():
            corr = np.exp(-np.abs(bins[:, None] - bins[None, :]) / length)
            var = (1 + 0.3 * np.sin(bins / 4.0)) ** 2
            clean[name] = corr * np.sqrt(var * var[:, None])
            clean_path = os.path.join(self.tmp, name, "clean.fits")
            _write_cov(clean_path, clean[name])
            chol = np.linalg.cholesky(clean[name])
            for j in range(12):
                suffix = "x" if j >= 10 else ""  # two train-only files
                samples = chol @ rng.standard_normal((N_BINS, 200))
                _write_cov(os.path.join(self.tmp, name, "noisy",
                                        f"cov_stack{j}{suffix}.fits"),
                           np.cov(samples))
            config += [f"[TOY.{name}]",
                       f"noisy_glob = {self.tmp}/{name}/noisy/cov_stack*.fits",
                       f"clean_cov = {clean_path}"]
        self.ref_path = os.path.join(self.tmp, "ref.fits")
        _write_cov(self.ref_path, sum(clean.values()) / len(clean))
        self.config_path = os.path.join(self.tmp, "paths.ini")
        with open(self.config_path, "w", encoding="utf-8") as file:
            file.write("\n".join(config) + "\n")

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_split(self):
        """Train-only files never land in the test set; split is seeded."""
        sample_paths = load_sample_paths("toy", self.config_path)
        paths = {k: sorted(os.path.join(self.tmp, k, "noisy", f)
                           for f in os.listdir(os.path.join(self.tmp, k, "noisy")))
                 for k in sample_paths["mock_types"]}
        train, test = split_train_test(paths, sample_paths["train_only_regex"])
        self.assertEqual(len(train) + len(test), 24)
        self.assertEqual(len(test), 2 * round(0.2 * 12))
        self.assertFalse(any(p.endswith("x.fits") for p, _ in test))
        self.assertEqual((train, test),
                         split_train_test(paths, sample_paths["train_only_regex"]))

    def test_help_example_config(self):
        """The example in --help parses as intended (end-of-line comments)."""
        example = WORKFLOW_HELP.split("Minimal\n  example for a new sample:\n")[1]
        example = "\n".join(line.strip() for line in
                            example.split("\n  Run with")[0].splitlines())
        config_path = os.path.join(self.tmp, "example.ini")
        with open(config_path, "w", encoding="utf-8") as file:
            file.write(example)
        paths = load_sample_paths("DR2", config_path)
        self.assertEqual(paths["mock_types"], ["london", "saclay"])
        self.assertEqual(paths["train_only_regex"], "")
        self.assertEqual(paths["model"], "/path/to/model.npz")

    def test_show_config(self):
        """--show-config exits cleanly without requiring other arguments."""
        with self.assertRaises(SystemExit) as context:
            picca_denoise_fit_classifier.main(["--show-config"])
        self.assertEqual(context.exception.code, 0)

    def test_unknown_sample(self):
        """A sample missing from the config gives a clear error."""
        with self.assertRaises(ValueError):
            load_sample_paths("DR99", self.config_path)

    def test_fit_and_pipeline(self):
        """Full chain runs, classifies correctly and reduces the error."""
        model_path = os.path.join(self.tmp, "model.npz")
        out_prefix = os.path.join(self.tmp, "run")
        picca_denoise_fit_classifier.main([
            "--sample", "TOY", "--paths-config", self.config_path,
            "--ref-corr", self.ref_path, "--out", model_path, "--nproc", "2"])
        picca_denoise_pipeline.main([
            "--model", model_path, "--paths-config", self.config_path,
            "--out", out_prefix, "--nproc", "2",
            "--test-cov-output", os.path.join(self.tmp, "denoised")])

        stats = np.load(out_prefix + "_stats.npz", allow_pickle=True)
        for prefix in ("tr", "te"):
            self.assertTrue(np.all(stats[f"{prefix}_hard_pred"]
                                   == stats[f"{prefix}_true_keys"]))
            self.assertTrue(np.all(stats[f"{prefix}_mse_final"]
                                   < stats[f"{prefix}_mse_noisy"]))
        for out_path in stats["te_out_paths"]:
            self.assertTrue(os.path.isfile(out_path))

        # Notebook interface agrees with the pipeline on a test matrix
        model = DenoiseModel.load(model_path)
        result = model.denoise(str(stats["te_paths"][0]))
        self.assertEqual(result["final_cov"].shape, (N_BINS, N_BINS))
        np.testing.assert_allclose(result["pev"], stats["te_pev"][0])

    def test_load_sample(self):
        """Models are found through the config; a missing entry is explained."""
        with self.assertRaises(ValueError):  # no model listed yet
            DenoiseModel.load_sample("TOY", self.config_path)

        model_path = os.path.join(self.tmp, "model.npz")
        with open(self.config_path, encoding="utf-8") as file:
            config = file.read()
        with open(self.config_path, "w", encoding="utf-8") as file:
            file.write(config.replace("[TOY]\n", f"[TOY]\nmodel = {model_path}\n", 1))
        with self.assertRaises(FileNotFoundError):  # listed but not made yet
            DenoiseModel.load_sample("TOY", self.config_path)

        # Fitting works whether or not the model entry exists
        picca_denoise_fit_classifier.main([
            "--sample", "TOY", "--paths-config", self.config_path,
            "--ref-corr", self.ref_path, "--out", model_path, "--nproc", "2"])
        model = DenoiseModel.load_sample("TOY", self.config_path)
        self.assertEqual(model.clf.unique_keys, sorted(MOCK_TYPES))

    def test_denoise_cov_script(self):
        """The single-file script matches DenoiseModel and records metadata."""
        model_path = os.path.join(self.tmp, "model.npz")
        picca_denoise_fit_classifier.main([
            "--sample", "TOY", "--paths-config", self.config_path,
            "--ref-corr", self.ref_path, "--out", model_path, "--nproc", "2"])
        in_path = os.path.join(self.tmp, "typeb", "noisy", "cov_stack3.fits")
        out_path = os.path.join(self.tmp, "denoised.fits")
        picca_denoise_cov.main(["--model", model_path, "--in", in_path,
                                "--out", out_path])

        expected = DenoiseModel.load(model_path).denoise(in_path)
        np.testing.assert_array_equal(fitsio.read(out_path, ext=1)["COV"],
                                      expected["final_cov"])
        np.testing.assert_array_equal(fitsio.read(out_path, ext=2)["COV-INIT"],
                                      expected["initial_recon_cov"])
        header = fitsio.read_header(out_path, ext=1)
        self.assertEqual(header["DNSGROUP"].strip(), "typeb")
        self.assertAlmostEqual(sum(header[f"DNSWGT{i}"] for i in range(2)), 1.0)

        with self.assertRaises(SystemExit):  # refuses to overwrite its input
            picca_denoise_cov.main(["--model", model_path, "--in", in_path,
                                    "--out", in_path])

    def test_pipeline_rejects_changed_inputs(self):
        """The pipeline refuses to run if the training set changed."""
        model_path = os.path.join(self.tmp, "model.npz")
        picca_denoise_fit_classifier.main([
            "--sample", "TOY", "--paths-config", self.config_path,
            "--ref-corr", self.ref_path, "--out", model_path, "--nproc", "2"])
        os.remove(os.path.join(self.tmp, "typea", "noisy", "cov_stack10x.fits"))
        with self.assertRaises(ValueError):
            picca_denoise_pipeline.main([
                "--model", model_path, "--paths-config", self.config_path,
                "--out", os.path.join(self.tmp, "run"), "--nproc", "2"])


if __name__ == "__main__":
    unittest.main()
