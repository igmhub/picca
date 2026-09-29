#!/usr/bin/env python3
"""Phase 1 of covariance denoising: fit the group classifier.

- Builds the reference eigenbasis V from a reference correlation matrix.
- Computes the clean residual of each mock type's clean matrix.
- Projects the noisy *training* matrices onto V and fits the classifier on
  their pseudo-eigenvalues. Test matrices are never read here.
- Saves V, clean residuals, classifier and run metadata (sample, split,
  smoothing) to a model .npz for picca_denoise_pipeline.py.

Example:
    picca_denoise_fit_classifier.py --sample DR1 --ref-corr ref.fits \\
        --out denoise_model_dr1.npz --nproc 128
"""
import argparse
import multiprocessing as mp
import os
import time

import numpy as np

from picca.cov_denoise.core import (GroupClassifier, compute_clean_residual,
                                    compute_reference_eigenbasis, corr_matrix,
                                    project, save_model, smooth_pev_adaptive)
from picca.cov_denoise.io import (WORKFLOW_HELP, HelpFormatter,
                                  add_paths_config_args,
                                  glob_noisy_paths, load_sample_paths,
                                  read_cov, read_reference_corr,
                                  split_train_test)
from picca.utils import userprint

# Set before the pool is created; workers inherit them through fork
_BASIS = None
_SMOOTH_PEV = False


def _project_worker(args):
    """Load one noisy covariance, return its (smoothed) pseudo-eigenvalues."""
    index, path = args
    pev = np.diag(project(corr_matrix(read_cov(path)), _BASIS)).copy()
    if _SMOOTH_PEV:
        pev = smooth_pev_adaptive(pev)
    return index, pev


def main(cmdargs=None):
    """Fit the denoising classifier."""
    parser = argparse.ArgumentParser(
        formatter_class=HelpFormatter, epilog=WORKFLOW_HELP,
        description="Fit the covariance-denoising classifier from noisy "
                    "mock covariance matrices.")
    add_paths_config_args(parser)
    parser.add_argument("--sample", type=str, required=True,
                        help="Sample to use (e.g. DR1), as defined in the "
                             "paths config")
    parser.add_argument("--ref-corr", type=str, default=None,
                        help="Reference correlation matrix FITS file. "
                             "Defaults to ref_corr in the paths config")
    parser.add_argument("--out", type=str, required=True,
                        help="Output model .npz")
    parser.add_argument("--nproc", type=int, default=128,
                        help="Number of worker processes")
    parser.add_argument("--test-fraction", type=float, default=0.2,
                        help="Fraction of each mock type held out for testing")
    parser.add_argument("--seed", type=int, default=1216,
                        help="Random seed for the train/test split")
    parser.add_argument("--smooth-pev", action="store_true",
                        help="Smooth pseudo-eigenvalues in log-k "
                             "(sigma=0.05) before fitting. Stored in the model "
                             "and used by picca_denoise_pipeline.py")
    parser.add_argument("--print-every", type=int, default=26,
                        help="Progress print interval")
    args = parser.parse_args(cmdargs)

    t_total = time.time()
    sample_paths = load_sample_paths(args.sample, args.paths_config)
    ref_corr_path = args.ref_corr or sample_paths["ref_corr"]
    if ref_corr_path is None:
        parser.error("no reference correlation matrix: pass --ref-corr or set "
                     f"ref_corr in the [{sample_paths['sample']}] section")
    n_workers = args.nproc or os.cpu_count()

    # Reference eigenbasis
    userprint(f"Sample: {sample_paths['sample']}")
    userprint(f"Loading reference correlation matrix:\n  {ref_corr_path}")
    basis = compute_reference_eigenbasis(read_reference_corr(ref_corr_path))
    n = basis.shape[0]
    userprint(f"  V shape: {basis.shape}\n")

    global _BASIS, _SMOOTH_PEV
    _BASIS = basis
    _SMOOTH_PEV = args.smooth_pev
    if args.smooth_pev:
        userprint("Pseudo-eigenvalue smoothing enabled (log-k, sigma=0.05)\n")

    # Clean residuals
    userprint("Computing clean residuals...")
    clean_resid = {}
    for key in sample_paths["mock_types"]:
        clean_corr = corr_matrix(read_cov(sample_paths["clean_cov"][key]))
        clean_resid[key] = compute_clean_residual(clean_corr, basis)
        userprint(f"  {key}: {sample_paths['clean_cov'][key]}")
    userprint()

    # Train/test split (test paths are not used here)
    paths_per_key = glob_noisy_paths(sample_paths)
    for key, paths in paths_per_key.items():
        userprint(f"  {key}: {len(paths)} noisy files")
    train, test = split_train_test(paths_per_key,
                                   sample_paths["train_only_regex"],
                                   args.test_fraction, args.seed)
    n_train = len(train)
    userprint(f"\nTrain: {n_train}  |  Test (held out): {len(test)}\n")

    # Project training matrices
    userprint(f"Projecting {n_train} training matrices ({n_workers} workers)...")
    train_pev = np.empty((n_train, n), dtype=np.float64)
    t_start = time.time()
    with mp.get_context("fork").Pool(processes=n_workers) as pool:
        work = [(i, path) for i, (path, _) in enumerate(train)]
        for ndone, (i, pev) in enumerate(
                pool.imap_unordered(_project_worker, work), start=1):
            train_pev[i] = pev
            if args.print_every > 0 and (ndone % args.print_every == 0
                                         or ndone == n_train):
                elapsed = time.time() - t_start
                rate = ndone / elapsed if elapsed > 0 else 0
                eta = (n_train - ndone) / rate if rate > 0 else float("inf")
                userprint(f"  [{ndone:4d}/{n_train}]  {elapsed:.1f}s  "
                          f"{rate:.2f} mat/s  ETA={eta:.0f}s")
    userprint(f"  Done in {time.time() - t_start:.1f}s\n")

    # Fit classifier
    userprint("Fitting classifier...")
    train_keys = [key for _, key in train]
    clf = GroupClassifier().fit(train_pev, train_keys)
    userprint(f"  Groups      : {clf.unique_keys}")
    userprint(f"  Weight range: [{clf.weights.min():.4f}, "
              f"{clf.weights.max():.4f}]")
    hard = clf.predict_hard(train_pev)
    n_correct = sum(h == t for h, t in zip(hard, train_keys))
    userprint(f"  Training accuracy: {n_correct}/{n_train} "
              f"({100 * n_correct / n_train:.1f}%)")
    for key in clf.unique_keys:
        idx = [j for j, t in enumerate(train_keys) if t == key]
        correct = sum(hard[j] == key for j in idx)
        userprint(f"    {key}: {correct}/{len(idx)} "
                  f"({100 * correct / len(idx):.1f}%)")
    userprint()

    metadata = {
        "sample": sample_paths["sample"],
        "smooth_pev": args.smooth_pev,
        "seed": args.seed,
        "test_fraction": args.test_fraction,
        "train_only_regex": sample_paths["train_only_regex"],
        "train_files": sorted(f"{key}/{os.path.basename(p)}" for p, key in train),
        "ref_corr": ref_corr_path,
    }
    out = save_model(args.out, basis, clf, clean_resid, metadata)
    userprint(f"Model saved to {out}")
    userprint(f"\nTotal wall time: {time.time() - t_total:.1f}s")
    userprint(f"\nNext step:\n  picca_denoise_pipeline.py --model {out} "
              "--out <prefix> --nproc <N>")
    userprint(f"\nTo make this the default model for {sample_paths['sample']}, "
              f"set 'model = {os.path.abspath(out)}' in the "
              f"[{sample_paths['sample']}] section of the paths config.")


if __name__ == "__main__":
    main()
