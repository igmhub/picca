#!/usr/bin/env python
"""Compute the averaged 1D power spectrum
"""

import sys, os, argparse
import numpy as np
import fitsio
from picca.pk1d import postproc_pk1d



def define_wavenumber_array(k_min, k_max, k_dist, velunits, pixsize, rebinfac):
    """ Define the wavenumber array limits and binning
        Default binning defined linearly with assumed average redshift of 3.4,
        and a forest defined between 1050 and 1200 Angstrom.
        Default velocity binning with same number of pixels"""

    k_min_lin_default = 2 * np.pi / ((1200 - 1050) * (1 + 3.4) / rebinfac)
    k_max_lin_default = np.pi / pixsize
    nb_k_bin_lin = int(k_max_lin_default / k_min_lin_default / 4)
    k_dist_lin_default = (k_max_lin_default - k_min_lin_default) / nb_k_bin_lin

    k_min_vel_default = 0.000813
    k_dist_vel_default = 0.000542 * rebinfac
    k_max_vel_default = k_min_vel_default + nb_k_bin_lin * k_dist_vel_default

    if velunits:
        if k_min is None: k_min = k_min_vel_default
        if k_max is None: k_max = k_max_vel_default
        if k_dist is None: k_dist = k_dist_vel_default
    else:
        if k_min is None: k_min = k_min_lin_default
        if k_max is None: k_max = k_max_lin_default
        if k_dist is None: k_dist = k_dist_lin_default

    return k_min, k_max, k_dist



def main(cmdargs):
    """Compute the averaged 1D power spectrum"""

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Compute the averaged 1D power spectrum')


    parser.add_argument('--in-dir',
                        type=str,
                        default=None,
                        required=True,
                        help='Directory to individual P1D files')

    parser.add_argument('--zedge-min',
                        type=float,
                        default='2.1',
                        required=False,
                        help='Minimal value of the redshift edge array,'
                             'Default value: 2.1')

    parser.add_argument('--zedge-max',
                        type=float,
                        default='6.3',
                        required=False,
                        help='Maximal value of the redshift edge array,'
                             'Default value: 6.3')

    parser.add_argument('--zedge-bin',
                        type=float,
                        default='0.2',
                        required=False,
                        help='Number of bins of the redshift edge array,'
                             'Default value: 0.2')

    parser.add_argument('--kedge-min',
                        type=float,
                        default=None,
                        required=False,
                        help='Minimal value of the wavenumber edges array,'
                             'Default value defined as function of --rebinfac, '
                             '--pixsize, and --velunits arguments')

    parser.add_argument('--kedge-max',
                        type=float,
                        default=None,
                        required=False,
                        help='Maximal value of the wavenumber edges array,'
                             'Default value defined as function of --rebinfac, '
                             '--pixsize, and --velunits arguments')

    parser.add_argument('--kedge-bin',
                        type=float,
                        default=None,
                        required=False,
                        help='Number of bins of the wavenumber edges array,'
                             'Default value defined as function of --rebinfac, '
                             '--pixsize, and --velunits arguments')

    parser.add_argument('--rebinfac',
                        type=int,
                        default=1,
                        required=False,
                        help='Rebinning factor used to define the binning of '
                             'the output wavenumber array')

    parser.add_argument('--pixsize',
                        type=float,
                        default=0.8,
                        required=False,
                        help='Size of a spectrum pixel in Angstrom, used to'
                             'define the binning of the output wavenumber array')

    parser.add_argument('--weights-method',
                        type=str,
                        default='no_weights',
                        required=False,
                        help='Weighted scheme for the averaging.'
                             'Possible options: no_weights, simple_snr')

    parser.add_argument('--apply-mean-snr-cut',
                        action='store_true',
                        default=False,
                        required=False,
                        help='Apply a redshift-dependent SNR quality cut')

    parser.add_argument('--snr-cut-scheme',
                        type=str,
                        default='eboss',
                        required=False,
                        help='Choice of SNR cut type, '
                             'Possible options: eboss')

    parser.add_argument('--overwrite',
                        action='store_true',
                        default=True,
                        required=False,
                        help='Overwrite the output')

    parser.add_argument('--velunits',
                        action='store_true',
                        default=False,
                        required=False,
                        help='Make the calculation in velocity units')

    parser.add_argument('--no-median',
                        action='store_true',
                        default=False,
                        required=False,
                        help='Use averages instead of medians')


    args = parser.parse_args(sys.argv[1:])

    if (args.weights_method != "no_weights") & (args.apply_mean_snr_cut):
        raise ValueError("""You are using a weighting method with a
                            redshift-dependent SNR quality cut, this is not
                            tested and should bias the result""")

    if args.apply_mean_snr_cut:
        if args.snr_cut_scheme == "eboss":
            snr_cut_mean =       [4.1, 3.9, 3.6, 3.2, 2.9, 2.6, 2.2, 2.0, 2.0,
                                  2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                                  2.0, 2.0, 2.0]
            zbins_snr_cut_mean = [2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8,
                                  4.0, 4.2, 4.4, 4.6, 4.8, 5.0, 5.2, 5.4, 5.6,
                                  5.8, 6.0, 6.2]
        else:
            raise ValueError("Please choose the SNR cutting scheme to be eboss, "
                             "or turn off the --apply-mean-snr-cut parameter, or "
                             "add here in the code a specific SNR cutting scheme")
    else:
        snr_cut_mean = None
        zbins_snr_cut_mean = None





    kedge_min, kedge_max, kedge_bin = define_wavenumber_array(args.kedge_min,
                                                               args.kedge_max,
                                                               args.kedge_bin,
                                                               args.velunits,
                                                               args.pixsize,
                                                               args.rebinfac)


    k_edges = np.arange(kedge_min,kedge_max,kedge_bin)
    z_edges = np.around(np.arange(args.zedge_min, args.zedge_max, args.zedge_bin),5)


    data = postproc_pk1d.parallelize_p1d_comp(args.in_dir,
                                              z_edges,
                                              k_edges,
                                              weights_method=args.weights_method,
                                              snr_cut_mean=snr_cut_mean,
                                              zbins=zbins_snr_cut_mean,
                                              nomedians=args.no_median,
                                              velunits=args.velunits,
                                              overwrite=args.overwrite)

if __name__ == '__main__':
    cmdargs = sys.argv[1:]
    main(cmdargs)
