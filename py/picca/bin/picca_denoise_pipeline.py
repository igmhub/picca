#!/usr/bin/env python3
"""Phase 2 of covariance denoising: run the pipeline on all noisy mocks.

Loads a model from picca_denoise_fit_classifier.py, rebuilds the same
train/test split (checked against the split stored in the model), and
denoises every noisy matrix, comparing to the clean matrix of its mock type.

For each matrix:
  1. Correlation matrix -> project onto V -> pseudo-eigenvalues.
  2. Initial reconstruction; KL divergence vs clean.
  3. Classify, weight the clean residuals, eigendecompose.
  4. Project and reconstruct the noisy residual; add it back.
  5. KL divergence vs clean for the final matrix.

Outputs:
  <out>_stats.npz    : per-matrix diagnostics for the train and test sets
  <out>_accuracy.txt : classifier accuracy summary
  <test-cov-output>/<mock_type>/test/*.fits : denoised test covariances
                                              (only with --test-cov-output)

Memory: each worker holds several n x n float64 matrices (~4.5 GB for
n = 7500). Use --nproc 64 or fewer on a 512 GB node.

Example:
    picca_denoise_pipeline.py --model denoise_model_dr1.npz \\
        --out denoise_run_dr1 --nproc 64 --test-cov-output denoised/
"""
import argparse
import multiprocessing as mp
import os
import time

import numpy as np

from picca.cov_denoise.core import (GroupClassifier, corr_matrix,
                                    fix_positive_definite,
                                    initial_reconstruction, kl_divergence,
                                    read_model_metadata, residual_correction,
                                    scale_to_cov)
from picca.cov_denoise.io import (WORKFLOW_HELP, HelpFormatter,
                                  add_paths_config_args,
                                  glob_noisy_paths, load_sample_paths,
                                  read_cov, split_train_test,
                                  write_denoised_cov)
from picca.utils import userprint

# Set before the pool is created; workers inherit them through fork
_BASIS = None
_CLF = None
_CLEAN_CORR = None
_CLEAN_VAR = None
_CLEAN_RESID = None
_SMOOTH_PEV = False
_TEST_COV_OUTDIR = None

VECTOR_FIELDS = ["pev", "pev_smooth", "resid_pev", "resid_pev_smooth"]
SCALAR_FIELDS = ["power_ratio", "resid_power_ratio", "KL_initial", "KL_final",
                 "frob_initial", "frob_final", "mse_initial", "mse_final",
                 "frob_noisy", "mse_noisy"]
FLAG_FIELDS = ["npd_corrected_initial", "npd_corrected_final"]


def _frob_mse(corr, clean_corr):
    """Frobenius norm and MSE of `corr` - `clean_corr`."""
    diff = corr - clean_corr
    return float(np.linalg.norm(diff, "fro")), float(np.mean(diff ** 2))


def _fix_if_not_pd(corr, kl_value, noisy_var, clean_cov, clean_corr, n,
                   label, index, path):
    """Rebuild `corr` with clipped eigenvalues when its KL computation failed.

    A failed KL (-1) means the matrix is singular or not positive definite
    relative to the clean covariance.

    Returns:
        (corr, (KL, frob, mse), was_fixed)
    """
    if kl_value != -1.0:
        return corr, (kl_value,) + _frob_mse(corr, clean_corr), 0
    corr, _, n_neg, min_eig = fix_positive_definite(corr, force=True)
    userprint(f"WARNING: non-positive definite {label} at index {index} "
              f"({path}): {n_neg} eigenvalue(s) shifted, lowest eigenvalue "
              f"= {min_eig:.6f}")
    kl_value = kl_divergence(clean_cov, scale_to_cov(corr, noisy_var), n)
    if kl_value == -1.0:
        userprint(f"WARNING: {label} still not positive definite after "
                  f"correction at index {index}")
    return corr, (kl_value,) + _frob_mse(corr, clean_corr), 1


def _pipeline_worker(args):
    """Denoise one matrix and compute diagnostics against the clean matrix."""
    index, path, key, split = args
    n = _BASIS.shape[0]

    cov = read_cov(path)
    noisy_var = np.diagonal(cov).copy()
    noisy_corr = corr_matrix(cov)
    clean_corr = _CLEAN_CORR[key]
    clean_cov = scale_to_cov(clean_corr, _CLEAN_VAR[key])

    result = {"path": path}
    pev, pev_smooth, power_ratio, initial_recon = initial_reconstruction(
        noisy_corr, _BASIS, _SMOOTH_PEV)
    result.update(pev=pev, pev_smooth=pev_smooth, power_ratio=power_ratio)
    result["frob_noisy"], result["mse_noisy"] = _frob_mse(noisy_corr, clean_corr)

    kl_initial = kl_divergence(clean_cov, scale_to_cov(initial_recon, noisy_var), n)
    initial_recon, stats, npd = _fix_if_not_pd(
        initial_recon, kl_initial, noisy_var, clean_cov, clean_corr, n,
        "initial recon", index, path)
    result.update(KL_initial=stats[0], frob_initial=stats[1],
                  mse_initial=stats[2], npd_corrected_initial=npd)

    corrected = residual_correction(noisy_corr, initial_recon, pev_smooth,
                                    _CLF, _CLEAN_RESID, _SMOOTH_PEV)
    final_corr = corrected.pop("final_corr")
    result.update(corrected)

    kl_final = kl_divergence(clean_cov, scale_to_cov(final_corr, noisy_var), n)
    final_corr, stats, npd = _fix_if_not_pd(
        final_corr, kl_final, noisy_var, clean_cov, clean_corr, n,
        "final corr", index, path)
    result.update(KL_final=stats[0], frob_final=stats[1],
                  mse_final=stats[2], npd_corrected_final=npd)

    result["out_path"] = ""
    if split == "test" and _TEST_COV_OUTDIR is not None:
        out_dir = os.path.join(_TEST_COV_OUTDIR, key, split)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, os.path.basename(path))
        write_denoised_cov(out_path, scale_to_cov(final_corr, noisy_var),
                           scale_to_cov(initial_recon, noisy_var))
        result["out_path"] = out_path
    return split, index, result


def _accuracy_lines(split_name, hard, true_keys, unique_keys):
    n_correct = sum(h == t for h, t in zip(hard, true_keys))
    lines = [f"{split_name} accuracy: {n_correct}/{len(true_keys)} "
             f"({100 * n_correct / len(true_keys):.1f}%)"]
    for key in unique_keys:
        idx = [j for j, t in enumerate(true_keys) if t == key]
        if not idx:
            continue
        correct = sum(hard[j] == key for j in idx)
        lines.append(f"  {key}: {correct}/{len(idx)} "
                     f"({100 * correct / len(idx):.1f}%)")
    return lines


def _stack_results(results, keys, unique_keys, prefix, include_out_paths):
    """Turn per-matrix result dicts into the arrays saved in the stats file."""
    arrays = {
        f"{prefix}_paths": np.array([r["path"] for r in results], dtype=object),
        f"{prefix}_true_keys": np.array(keys, dtype=object),
    }
    if include_out_paths:
        arrays[f"{prefix}_out_paths"] = np.array(
            [r["out_path"] for r in results], dtype=object)
    for field in VECTOR_FIELDS + SCALAR_FIELDS:
        arrays[f"{prefix}_{field}"] = np.array([r[field] for r in results],
                                               dtype=np.float64)
    for field in FLAG_FIELDS:
        arrays[f"{prefix}_{field}"] = np.array([r[field] for r in results],
                                               dtype=np.int32)
    for field in ("distances", "weights"):
        arrays[f"{prefix}_{field}"] = np.array(
            [[r[field][k] for k in unique_keys] for r in results])
    return arrays


def main(cmdargs=None):
    """Run the covariance-denoising pipeline."""
    parser = argparse.ArgumentParser(
        formatter_class=HelpFormatter, epilog=WORKFLOW_HELP,
        description="Denoise all noisy mock covariance matrices of a sample "
                    "with a fitted model and compute diagnostics.")
    add_paths_config_args(parser)
    parser.add_argument("--model", type=str, required=True,
                        help="Model .npz from picca_denoise_fit_classifier.py")
    parser.add_argument("--out", type=str, required=True,
                        help="Output prefix for the stats and accuracy files")
    parser.add_argument("--nproc", type=int, default=64,
                        help="Number of worker processes (see memory note in "
                             "the script docstring)")
    parser.add_argument("--test-cov-output", type=str, default=None,
                        help="Directory for denoised test covariances, written "
                             "to <dir>/<mock_type>/test/. Not saved if unset")
    parser.add_argument("--print-every", type=int, default=10,
                        help="Progress print interval")
    args = parser.parse_args(cmdargs)

    t_total = time.time()
    n_workers = args.nproc or os.cpu_count()

    # Model
    userprint(f"Loading model from: {args.model}")
    model_data = np.load(args.model, allow_pickle=True)
    meta = read_model_metadata(model_data)
    if "sample" not in meta:
        raise ValueError(f"{args.model} has no run metadata. Re-fit it with "
                         "picca_denoise_fit_classifier.py")
    sample = str(meta["sample"])
    smooth_pev = bool(meta["smooth_pev"])
    basis = model_data["V"]
    clf = GroupClassifier.from_model_data(model_data)
    clean_resid = {k: model_data[f"resid__{k}"] for k in clf.unique_keys}
    n = basis.shape[0]
    userprint(f"  Sample: {sample}  |  V shape: {basis.shape}  |  "
              f"Groups: {clf.unique_keys}  |  smooth_pev: {smooth_pev}\n")

    sample_paths = load_sample_paths(sample, args.paths_config)
    if sample_paths["mock_types"] != clf.unique_keys:
        raise ValueError(f"Mock types in paths config {sample_paths['mock_types']}"
                         f" do not match the model groups {clf.unique_keys}")

    userprint("Loading clean covariance matrices...")
    clean_corr, clean_var = {}, {}
    for key in clf.unique_keys:
        cov = read_cov(sample_paths["clean_cov"][key])
        clean_var[key] = np.diagonal(cov).copy()
        clean_corr[key] = corr_matrix(cov)
        userprint(f"  {key}: {sample_paths['clean_cov'][key]}")
    userprint()

    # Same split as when fitting, checked against the stored training set
    paths_per_key = glob_noisy_paths(sample_paths)
    train, test = split_train_test(paths_per_key, str(meta["train_only_regex"]),
                                   float(meta["test_fraction"]), int(meta["seed"]))
    train_files = sorted(f"{key}/{os.path.basename(p)}" for p, key in train)
    if train_files != [str(f) for f in meta["train_files"]]:
        raise ValueError("The training set rebuilt from the paths config does "
                         "not match the one stored in the model. Check that "
                         "the noisy files have not changed since fitting.")
    n_train, n_test = len(train), len(test)
    userprint(f"Train: {n_train}  |  Test: {n_test}\n")

    if args.test_cov_output is not None:
        userprint(f"Denoised test covariances -> {args.test_cov_output}\n")

    global _BASIS, _CLF, _CLEAN_CORR, _CLEAN_VAR, _CLEAN_RESID
    global _SMOOTH_PEV, _TEST_COV_OUTDIR
    _BASIS, _CLF, _CLEAN_RESID = basis, clf, clean_resid
    _CLEAN_CORR, _CLEAN_VAR = clean_corr, clean_var
    _SMOOTH_PEV, _TEST_COV_OUTDIR = smooth_pev, args.test_cov_output

    # Denoise train + test in one pass
    work = ([(i, p, k, "train") for i, (p, k) in enumerate(train)]
            + [(i, p, k, "test") for i, (p, k) in enumerate(test)])
    results = {"train": [None] * n_train, "test": [None] * n_test}
    n_total = len(work)
    userprint(f"Denoising {n_total} matrices ({n_workers} workers)...")
    t_start = time.time()
    with mp.get_context("fork").Pool(processes=n_workers) as pool:
        for ndone, (split, i, res) in enumerate(
                pool.imap_unordered(_pipeline_worker, work), start=1):
            results[split][i] = res
            if args.print_every > 0 and (ndone % args.print_every == 0
                                         or ndone == n_total):
                elapsed = time.time() - t_start
                rate = ndone / elapsed if elapsed > 0 else 0
                eta = (n_total - ndone) / rate if rate > 0 else float("inf")
                userprint(f"  [{ndone:4d}/{n_total}]  {elapsed:.1f}s  "
                          f"{rate:.2f} mat/s  ETA={eta:.0f}s")
    userprint(f"  Done in {time.time() - t_start:.1f}s\n")

    unique_keys = clf.unique_keys
    arrays = {}
    arrays.update(_stack_results(results["train"], [k for _, k in train],
                                 unique_keys, "tr", include_out_paths=False))
    arrays.update(_stack_results(results["test"], [k for _, k in test],
                                 unique_keys, "te", include_out_paths=True))

    # Classifier accuracy (on the same pseudo-eigenvalues used to classify)
    accuracy_lines = []
    for prefix, split_name in (("tr", "Training"), ("te", "Testing")):
        hard = clf.predict_hard(arrays[f"{prefix}_pev_smooth"])
        arrays[f"{prefix}_hard_pred"] = np.array(hard, dtype=object)
        accuracy_lines += _accuracy_lines(split_name, hard,
                                          list(arrays[f"{prefix}_true_keys"]),
                                          unique_keys) + [""]
    userprint("\n".join(accuracy_lines))

    userprint("Non-positive-definite matrices (corrected):")
    for prefix, split_name in (("tr", "Training"), ("te", "Testing")):
        userprint(f"  {split_name}: "
                  f"{int(arrays[f'{prefix}_npd_corrected_initial'].sum())} "
                  f"initial recon, "
                  f"{int(arrays[f'{prefix}_npd_corrected_final'].sum())} "
                  "final corr")
    userprint()

    arrays.update(group_key_order=np.array(unique_keys, dtype=object),
                  clf_weights=clf.weights, V=basis,
                  smooth_pev=np.array(smooth_pev), sample=np.array(sample),
                  n=np.array(n))
    np.savez(args.out + "_stats.npz", **arrays)
    userprint(f"Stats saved to {args.out}_stats.npz")

    with open(args.out + "_accuracy.txt", "w", encoding="utf-8") as file:
        file.write("\n".join(accuracy_lines) + "\n")
    userprint(f"Accuracy saved to {args.out}_accuracy.txt")
    userprint(f"\nTotal wall time: {time.time() - t_total:.1f}s")


if __name__ == "__main__":
    main()
