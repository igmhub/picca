"""Input/output for covariance denoising: FITS files, sample paths, splits."""
import argparse
import configparser
import glob
import os
import re

import fitsio
import numpy as np

DEFAULT_PATHS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "default_paths.ini")


WORKFLOW_HELP = f"""\
workflow:
  1. picca_denoise_fit_classifier.py --sample DR1 --out model_dr1.npz
       Fits the classifier on the training mocks and saves a model.
  2. picca_denoise_pipeline.py --model model_dr1.npz --out run_dr1
       Denoises all train + test mocks and saves diagnostics.
  3. picca_denoise_cov.py --sample DR1 --in data_cov.fits --out denoised.fits
       Denoises one covariance matrix (e.g. the data) with a fitted model.
     Or in Python:
       from picca.cov_denoise import DenoiseModel
       model = DenoiseModel.load_sample("DR1")   # or DenoiseModel.load(path)
       result = model.denoise(noisy_cov)

paths config:
  Input paths for each sample are read from
    {DEFAULT_PATHS_FILE}
  To add a sample or change paths, write an .ini with only the sections and
  options you want to add/change and pass it with --paths-config. Minimal
  example for a new sample:

    [DR2]
    mock_types = london saclay      # one [DR2.<type>] section per type
    train_only_regex =              # optional, see --show-config
    ref_corr = /path/to/ref.fits    # optional, or use --ref-corr
    model = /path/to/model.npz      # optional, set after fitting

    [DR2.london]
    noisy_glob = /path/to/london/noisy/*.fits
    clean_cov = /path/to/london/clean.fits

    [DR2.saclay]
    noisy_glob = /path/to/saclay/noisy/*.fits
    clean_cov = /path/to/saclay/clean.fits

  Run with --show-config to print the default config with full
  documentation and the samples currently available.
"""


class HelpFormatter(argparse.ArgumentDefaultsHelpFormatter,
                    argparse.RawDescriptionHelpFormatter):
    """Show argument defaults and keep the epilog's line breaks."""


class _ShowConfigAction(argparse.Action):
    """Print the default paths config and available samples, then exit."""

    def __init__(self, option_strings, dest, **kwargs):
        super().__init__(option_strings, dest, nargs=0, default=argparse.SUPPRESS,
                         help="print the default paths config (with its "
                              "documentation) and the available samples, "
                              "then exit")

    def __call__(self, parser, namespace, values, option_string=None):
        print(f"# Default paths config: {DEFAULT_PATHS_FILE}\n")
        with open(DEFAULT_PATHS_FILE, encoding="utf-8") as file:
            print(file.read())
        paths_config = getattr(namespace, "paths_config", None)
        print(f"# Available samples: {available_samples(paths_config)}")
        parser.exit()


def add_paths_config_args(parser):
    """Add --paths-config and --show-config to a script's parser.

    Put these before other options so that --show-config sees a
    --paths-config given before it.
    """
    parser.add_argument("--paths-config", type=str, default=None,
                        help="optional .ini layered on top of the default "
                             "paths config (see --show-config)")
    parser.add_argument("--show-config", action=_ShowConfigAction)


def read_cov(path, ext=1):
    """Read a covariance (or correlation) matrix from a FITS file.

    Accepts a binary table with a 'COV' or 'COVMAT' column (one matrix row per
    table row), a table with a single column, or an image HDU.
    """
    data = fitsio.read(path, ext=ext)
    names = data.dtype.names
    if names is None:
        return np.asarray(data, dtype=np.float64)
    for name in ("COV", "COVMAT"):
        if name in names:
            return np.asarray(data[name], dtype=np.float64)
    if len(names) == 1:
        return np.asarray(data[names[0]], dtype=np.float64)
    raise ValueError(f"Could not find a covariance matrix in {path} "
                     f"(ext {ext}, columns {names})")


def read_reference_corr(path):
    """Read the reference correlation matrix used to build the eigenbasis.

    Uses a 'CORR' column if present; otherwise converts the covariance.
    """
    data = fitsio.read(path, ext=1)
    names = data.dtype.names
    if names is not None and "CORR" in names:
        return np.asarray(data["CORR"], dtype=np.float64)
    cov = read_cov(path)
    if names is None:
        return cov
    var = np.diagonal(cov)
    return cov / np.sqrt(var * var[:, None])


def write_denoised_cov(path, final_cov, initial_cov, header=None):
    """Write the denoised covariance and the initial reconstruction.

    HDU 1 ('COVMAT', column 'COV') holds the final covariance and HDU 2
    ('COVMAT_INIT', column 'COV-INIT') the initial reconstruction.

    Args:
        header: optional list of {'name', 'value', 'comment'} dicts added to
            the header of HDU 1
    """
    with fitsio.FITS(path, "rw", clobber=True) as fits_out:
        fits_out.write([final_cov], names=["COV"], units=[""], extname="COVMAT",
                       header=header)
        fits_out.write([initial_cov], names=["COV-INIT"], units=[""],
                       extname="COVMAT_INIT")


def _expand(path):
    return os.path.expanduser(os.path.expandvars(path))


def available_samples(paths_config=None):
    """Samples defined in the default (and optional user) paths config."""
    cfg = _read_paths_config(paths_config)
    return [s for s in cfg.sections() if "." not in s]


def _read_paths_config(paths_config=None):
    # '#' after whitespace starts an end-of-line comment
    cfg = configparser.ConfigParser(interpolation=None,
                                    inline_comment_prefixes=("#",))
    files = [DEFAULT_PATHS_FILE]
    if paths_config is not None:
        if not os.path.isfile(paths_config):
            raise FileNotFoundError(f"Paths config not found: {paths_config}")
        files.append(paths_config)
    cfg.read(files)
    return cfg


def load_sample_paths(sample, paths_config=None):
    """Input paths for a sample (e.g. 'DR1').

    Paths are read from default_paths.ini in this directory. If
    `paths_config` is given, it is read on top of the defaults, so it only
    needs to contain the sections/options that change. Environment variables
    and ~ are expanded.

    Returns:
        dict with keys
            'sample': normalized sample name
            'mock_types': sorted list of mock types (classifier groups)
            'noisy_glob': {mock_type: glob pattern for noisy covariances}
            'clean_cov': {mock_type: path to clean covariance}
            'ref_corr': path to reference correlation matrix, or None
            'model': path to the fitted model .npz for this sample, or None
                if no model has been made yet
            'train_only_regex': regex on file basename; matching files are
                always put in the training set ('' disables this)
    """
    cfg = _read_paths_config(paths_config)
    sample = sample.upper()
    if not cfg.has_section(sample):
        raise ValueError(
            f"Sample '{sample}' is not defined. Available samples: "
            f"{available_samples(paths_config)}. Add it to "
            f"{DEFAULT_PATHS_FILE} or pass a file with --paths-config.")

    section = cfg[sample]
    mock_types = sorted(section.get("mock_types", "").split())
    if not mock_types:
        raise ValueError(f"No mock_types listed in section [{sample}]")

    paths = {
        "sample": sample,
        "mock_types": mock_types,
        "noisy_glob": {},
        "clean_cov": {},
        "ref_corr": None,
        "model": None,
        "train_only_regex": section.get("train_only_regex", ""),
    }
    for option in ("ref_corr", "model"):
        if section.get(option):
            paths[option] = _expand(section[option])

    for mock_type in mock_types:
        name = f"{sample}.{mock_type}"
        if not cfg.has_section(name):
            raise ValueError(f"Missing section [{name}] for mock type "
                             f"'{mock_type}' of sample {sample}")
        for option in ("noisy_glob", "clean_cov"):
            if not cfg[name].get(option):
                raise ValueError(f"Missing option '{option}' in [{name}]")
            paths[option][mock_type] = _expand(cfg[name][option])
    return paths


def glob_noisy_paths(sample_paths):
    """Sorted noisy covariance paths per mock type."""
    paths_per_key = {}
    for key in sample_paths["mock_types"]:
        pattern = sample_paths["noisy_glob"][key]
        found = sorted(glob.glob(pattern))
        if not found:
            raise FileNotFoundError(f"No files match {pattern} ({key})")
        paths_per_key[key] = np.array(found)
    return paths_per_key


def split_train_test(paths_per_key, train_only_regex="", test_fraction=0.2,
                     seed=1216):
    """Per-group train/test split of noisy covariance paths.

    For each group, n_test = max(1, round(test_fraction * N_group)) files are
    drawn for testing, where N_group counts *all* files in the group. Test
    files are drawn only from files whose basename does not match
    `train_only_regex`; matching files always go to training.

    Groups are processed in sorted order with one random stream, so the split
    is fully determined by (file lists, train_only_regex, test_fraction, seed).

    Returns:
        (train, test): lists of (path, key) tuples
    """
    rng = np.random.RandomState(seed)
    pattern = re.compile(train_only_regex) if train_only_regex else None
    train, test = [], []
    for key in sorted(paths_per_key):
        paths = np.asarray(paths_per_key[key])
        is_train_only = np.array(
            [pattern is not None and pattern.search(os.path.basename(p)) is not None
             for p in paths], dtype=bool)
        splittable = paths[~is_train_only]
        idx = rng.permutation(len(splittable))
        n_test = max(1, int(round(test_fraction * len(paths))))
        test.extend((p, key) for p in splittable[idx[:n_test]])
        train.extend((p, key) for p in splittable[idx[n_test:]])
        train.extend((p, key) for p in paths[is_train_only])
    return train, test
