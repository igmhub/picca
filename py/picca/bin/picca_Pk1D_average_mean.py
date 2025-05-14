#!/usr/bin/env python
"""Compute the averaged 1D power spectrum
"""

import sys, glob, argparse, ast
from picca.pk1d import postproc_pk1d
from picca.utils import userprint


def main(cmdargs):
    """Compute the averaged 1D power spectrum averages"""

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Compute the averaged 1D power spectrum averages",
    )

    parser.add_argument(
        "--in-dir",
        type=str,
        default=None,
        required=False,
        help="String to glob all mean p1d to average",
    )

    parser.add_argument(
        "--in-filenames",
        type=str,
        default=None,
        required=False,
        help="String of all mean p1d to average",
    )

    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        required=False,
        help="Output path",
    )

    parser.add_argument(
        "--weighted-mean",
        action="store_true",
        default=False,
        required=False,
    )

    args = parser.parse_args(cmdargs)

    if (args.in_dir is None and args.in_filenames is None) or (
        args.in_dir is not None and args.in_filenames is not None
    ):
        userprint(
            (
                "ERROR: No transmisson input files or both 'in_dir' and "
                "'in_filenames' given"
            )
        )
        sys.exit()
    elif args.in_dir is not None:
        mean_p1d_names = glob.glob(args.in_dir)
    else:
        mean_p1d_names = ast.literal_eval(args.in_filenames)
    userprint("INFO: Found {} files".format(len(mean_p1d_names)))

    postproc_pk1d.average_mean_pk1d_files(
        mean_p1d_names,
        args.output_path,
        weighted_mean=False,
    )


if __name__ == "__main__":
    cmdargs = sys.argv[1:]
    main(cmdargs)
