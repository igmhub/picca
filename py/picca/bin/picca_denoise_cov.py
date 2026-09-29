#!/usr/bin/env python3
"""Denoise one covariance matrix with a fitted denoising model.

Reads a noisy covariance from a FITS file (e.g. the data covariance), denoises
it with a model from picca_denoise_fit_classifier.py, and writes:
  HDU 1 'COVMAT'      column 'COV'      : denoised covariance
  HDU 2 'COVMAT_INIT' column 'COV-INIT' : initial reconstruction
The header of HDU 1 records the sample, model and classification weights.

Example:
    picca_denoise_cov.py --sample DR1 --in data_cov.fits --out denoised.fits
    picca_denoise_cov.py --model model_dr1.npz --in data_cov.fits \\
        --out denoised.fits
"""
import argparse
import os

from picca.cov_denoise.core import DenoiseModel
from picca.cov_denoise.io import (WORKFLOW_HELP, HelpFormatter,
                                  add_paths_config_args, load_sample_paths,
                                  write_denoised_cov)
from picca.utils import userprint


def main(cmdargs=None):
    """Denoise one covariance matrix."""
    parser = argparse.ArgumentParser(
        formatter_class=HelpFormatter, epilog=WORKFLOW_HELP,
        description="Denoise one covariance matrix with a fitted model.")
    add_paths_config_args(parser)
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--sample", type=str, default=None,
                             help="Use the model listed for this sample "
                                  "(e.g. DR1) in the paths config")
    model_group.add_argument("--model", type=str, default=None,
                             help="Path to a model .npz")
    parser.add_argument("--in", dest="in_path", type=str, required=True,
                        help="Noisy covariance FITS file (column 'COV' or "
                             "'COVMAT' in HDU 1)")
    parser.add_argument("--out", type=str, required=True,
                        help="Output FITS file for the denoised covariance")
    args = parser.parse_args(cmdargs)

    if os.path.abspath(args.in_path) == os.path.abspath(args.out):
        parser.error("--out must differ from --in")

    if args.sample is not None:
        model_path = load_sample_paths(args.sample, args.paths_config)["model"]
        model = DenoiseModel.load_sample(args.sample, args.paths_config)
    else:
        model_path = args.model
        model = DenoiseModel.load(args.model)

    userprint(f"Denoising {args.in_path}")
    result = model.denoise(args.in_path)

    weights = ", ".join(f"{k}: {w:.3f}" for k, w in result["weights"].items())
    userprint(f"  Classified as {result['hard_pred']} (weights {weights})")
    if result["npd_corrected_initial"] or result["npd_corrected_final"]:
        userprint("  Eigenvalues were clipped to make the result positive "
                  "definite (see warnings above)")

    header = [
        {"name": "DNSMODEL", "value": os.path.basename(model_path),
         "comment": "denoising model file"},
        {"name": "DNSINPUT", "value": os.path.basename(args.in_path),
         "comment": "noisy input covariance"},
        {"name": "DNSSMOOT", "value": model.smooth_pev,
         "comment": "pseudo-eigenvalues smoothed in log-k"},
        {"name": "DNSGROUP", "value": result["hard_pred"],
         "comment": "nearest mock type"},
        {"name": "DNSNPDI", "value": result["npd_corrected_initial"],
         "comment": "initial recon made positive definite"},
        {"name": "DNSNPDF", "value": result["npd_corrected_final"],
         "comment": "final cov made positive definite"},
    ]
    if args.sample is not None:
        header.insert(0, {"name": "DNSSAMPL", "value": args.sample.upper(),
                          "comment": "denoising sample"})
    for i, (key, weight) in enumerate(result["weights"].items()):
        header += [{"name": f"DNSKEY{i}", "value": key, "comment": "mock type"},
                   {"name": f"DNSWGT{i}", "value": float(weight),
                    "comment": f"classification weight of {key}"}]

    write_denoised_cov(args.out, result["final_cov"],
                       result["initial_recon_cov"], header=header)
    userprint(f"Denoised covariance written to {args.out}")


if __name__ == "__main__":
    main()
