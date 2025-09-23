#!/usr/bin/env python

"""
Smooth the covariance matrix calculated from the 'write_full_covariance' script.
This script is an updated version of:
/global/cfs/projectdirs/desi/science/lya/y1-kp6/iron-tests/correlations/scripts/write_smooth_covariance_flex_size.py
"""

import argparse
import time

import fitsio
import numpy as np


def smooth_corrmat_asym(
    corrmat,
    rp0,
    rt0,
    rp1,
    rt1,
    drt=4.0,
    drp=4.0,
):

    n0 = corrmat.shape[0]
    n1 = corrmat.shape[1]
    smooth_corrmat = np.zeros((n0, n1))

    # if one of the two array has both positive and negative
    # rp but not the other, we take the absolute value of both
    # (this is for the cross-covariance LyaxQSO - LyaxLya)

    if (np.any(rp0 < 0) and np.all(rp1 >= 0)) or (np.all(rp0 >= 0) and np.any(rp1 < 0)):
        arp0 = np.abs(rp0)
        arp1 = np.abs(rp1)
    else:
        arp0 = rp0
        arp1 = rp1

    # average the correlation from bins with similar separations in
    # parallel and perpendicular distances
    sum_correlation = {}
    counts_correlation = {}
    for i0 in range(n0):
        print("\rsmoothing {}".format(i0), end="")
        for i1 in range(n1):
            if corrmat[i0, i1] > 0.999:  # ignore the diagonal
                continue
            idrp = round(abs(arp1[i1] - arp0[i0]) / drp)
            idrt = round(abs(rt1[i1] - rt0[i0]) / drt)
            if (idrp, idrt) not in sum_correlation:
                sum_correlation[(idrp, idrt)] = 0.0
                counts_correlation[(idrp, idrt)] = 0
            sum_correlation[(idrp, idrt)] += corrmat[i0, i1]
            counts_correlation[(idrp, idrt)] += 1
    # second loop
    for i0 in range(n0):
        print("\rsmoothing {}".format(i0), end="")
        for i1 in range(n1):
            if corrmat[i0, i1] > 0.999:  # ignore the diagonal
                smooth_corrmat[i0, i1] = 1.0
                continue
            idrp = round(abs(arp1[i1] - arp0[i0]) / drp)
            idrt = round(abs(rt1[i1] - rt0[i0]) / drt)
            smooth_corrmat[i0, i1] = (
                sum_correlation[(idrp, idrt)] / counts_correlation[(idrp, idrt)]
            )

    print("\n")
    return smooth_corrmat


def main(cmdargs=None):
    
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Smoothes the covariance matrix.",
    )

    parser.add_argument(
        "-i",
        "--input-cov",
        type=str,
        default=None,
        required=True,
        help="Path to input covariance file.",
    )

    parser.add_argument(
        "-o",
        "--output-cov",
        type=str,
        default=None,
        required=True,
        help="Path to output covariance file.",
    )

    parser.add_argument(
        "--correlation-types",
        type=str,
        required=True,
        nargs="+",
        default=None,
        choices=["auto", "cross"],
        help="Whether each block in the covariance is an auto or cross correlation. "
        "Formerly '--block-types'. Choices are auto or cross. Default is now None. "
        "Enter as a list separated by spaces. MUST BE IN THE SAME ORDER AS THE "
        "CORRELATIONS IN THE WRITE_FULL_COVARIANCE SCRIPT "
        "Example: --correlation-types auto auto cross cross for the DR2 BAO analysis",
    )

    parser.add_argument(
        "--rp-min-auto",
        type=float,
        default=0.0,
        required=False,
        help="Min r-parallel [h^-1 Mpc] for auto-correlations",
    )

    parser.add_argument(
        "--rp-max-auto",
        type=float,
        default=200.0,
        required=False,
        help="Max r-parallel [h^-1 Mpc] for auto-correlations",
    )

    parser.add_argument(
        "--rp-min-cross",
        type=float,
        default=-200.0,
        required=False,
        help="Min r-parallel [h^-1 Mpc] for cross-correlations",
    )

    parser.add_argument(
        "--rp-max-cross",
        type=float,
        default=200.0,
        required=False,
        help="Max r-parallel [h^-1 Mpc] for cross-correlations",
    )

    parser.add_argument(
        "--rt-min-auto",
        type=float,
        default=0.0,
        required=False,
        help="Min r-transverse [h^-1 Mpc] for auto-correlations",
    )

    parser.add_argument(
        "--rt-max-auto",
        type=float,
        default=200.0,
        required=False,
        help="Max r-transverse [h^-1 Mpc] for auto-correlations",
    )

    parser.add_argument(
        "--rt-min-cross",
        type=float,
        default=0.0,
        required=False,
        help="Min r-transverse [h^-1 Mpc] for cross-correlations",
    )

    parser.add_argument(
        "--rt-max-cross",
        type=float,
        default=200.0,
        required=False,
        help="Max r-transverse [h^-1 Mpc] for cross-correlations",
    )

    parser.add_argument(
        "--np-auto",
        type=int,
        default=50,
        required=False,
        help="Number of r-parallel bins for auto-correlations",
    )

    parser.add_argument(
        "--np-cross",
        type=int,
        default=100,
        required=False,
        help="Number of r-parallel bins for cross-correlations",
    )

    parser.add_argument(
        "--nt-auto",
        type=int,
        default=50,
        required=False,
        help="Number of r-transverse bins for auto-correlations",
    )

    parser.add_argument(
        "--nt-cross",
        type=int,
        default=50,
        required=False,
        help="Number of r-transverse bins for cross-correlations",
    )

    args = parser.parse_args()
    # start time
    t1 = time.time()

    table = fitsio.read(args.input_cov)  # like iron-5-2-1-global-cov.fits"
    # print(table.dtype.names)
    cov = table["COV"]
    print(f"Covariance shape: {cov.shape}")

    block_edges = [
        0,
    ]
    for block_type in args.correlation_types:
        if block_type == "auto":
            block_edges.append(block_edges[-1] + int(args.np_auto * args.nt_auto))
        elif block_type == "cross":
            block_edges.append(block_edges[-1] + int(args.np_cross * args.nt_cross))
        else:
            raise ValueError("correlation type must be 'auto' or 'cross'", block_type)

    blocks = [
        [edgemin, edgemax]
        for edgemin, edgemax in zip(block_edges[:-1], block_edges[1:])
    ]

    print(
        f"WE NEED A ({block_edges[-1]},{block_edges[-1]}) COVARIANCE AND THE ORDER MUST "
        "MATCH WHAT WAS INPUT TO THE WRITE_FULL_COVARIANCE SCRIPT"
    )
    print(f"OF SIZE {blocks}.")
    assert cov.shape == (block_edges[-1], block_edges[-1])

    var = np.diagonal(cov)
    corrmat = cov / np.sqrt(var[None, :] * var[:, None])

    rp = []
    rt = []

    for block_type in args.correlation_types:
        if block_type == "auto":
            rpmin = args.rp_min_auto
            rpmax = args.rp_max_auto
            rtmin = args.rt_min_auto
            rtmax = args.rt_max_auto

            n_p = args.np_auto
            n_t = args.nt_auto
        else:
            rpmin = args.rp_min_cross
            rpmax = args.rp_max_cross
            rtmin = args.rt_min_cross
            rtmax = args.rt_max_cross

            n_p = args.np_cross
            n_t = args.nt_cross

        # bin-widths
        delta_rp = (rpmax - rpmin) / n_p
        delta_rt = (rtmax - rtmin) / n_t

        # rp and rt here are arrays of the bin centers
        rp.append(
            np.tile(
                np.linspace(rpmin + delta_rp / 2, rpmax - delta_rp / 2, n_p), (n_t, 1)
            ).T.ravel()
        )
        rt.append(
            np.tile(
                np.linspace(rtmin + delta_rt / 2, rtmax - delta_rt / 2, n_t), (n_p, 1)
            ).ravel()
        )

    print("Smoothing covariance")
    t2 = time.time()

    for i in range(len(blocks)):
        for j in range(i, len(blocks)):
            print("block", i, j)
            # beg is the beginning of the block
            # end is the end of the block
            beg_i = blocks[i][0]
            end_i = blocks[i][1]
            beg_j = blocks[j][0]
            end_j = blocks[j][1]
            # rp0 and rp1 (and rt0 and rt1) are the bin centers (rp (and rt) above) of bins indexed in [i,j]
            corrmat[beg_i:end_i, beg_j:end_j] = smooth_corrmat_asym(
                corrmat=corrmat[beg_i:end_i, beg_j:end_j],
                rp0=rp[i],
                rt0=rt[i],
                rp1=rp[j],
                rt1=rt[j],
                drt=delta_rt,
                drp=delta_rp,
            )
            corrmat[beg_j:end_j, beg_i:end_i] = corrmat[beg_i:end_i, beg_j:end_j].T

    cov = corrmat * np.sqrt(var[None, :] * var[:, None])

    table["COV"] = cov
    fitsio.write(args.output_cov, table, clobber=True)

    t3 = time.time()

    print(f"Time spent smoothing covariance: {(t3 - t2)/60:.3f} minutes")
    print(f"Total time: {(t3 - t1)/60:.3f} minutes")
