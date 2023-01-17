#!/usr/bin/env python
"""Compute the averaged 1D power spectrum
"""

import sys, os, argparse
import numpy as np
import fitsio
from picca.pk1d import postproc_pk1d, utils


def define_wavenumber_array(k_min, k_max, k_dist, velunits, pixsize, rebinfac):
    """ Define the wavenumber array limits and binning
        Default binning defined linearly with assumed average redshift of 3.4,
        and a forest defined between 1050 and 1200 Angstrom.
        Default velocity binning with same number of pixels"""

    k_min_lin_default = utils.DEFAULT_K_MIN_LIN * rebinfac
    k_max_lin_default = np.pi / pixsize
    nb_k_bins = int(k_max_lin_default / k_min_lin_default / utils.DEFAULT_K_BINNING_FACTOR)
    k_dist_lin_default = (k_max_lin_default - k_min_lin_default) / nb_k_bins

    k_dist_vel_default = utils.DEFAULT_K_BIN_VEL * rebinfac
    k_max_vel_default = utils.DEFAULT_K_MIN_VEL + nb_k_bins * k_dist_vel_default

    if velunits:
        if k_min is None: k_min = utils.DEFAULT_K_MIN_VEL
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
    
    parser.add_argument('--output-file',
                        type=str,
                        default=None,
                        required=False,
                        help='Output file name,' 
                             'If set to None, file name is set to --in-dir/mean_Pk1d_[weight_method]_[snr_cut]_[vel].fits.gz')

    parser.add_argument('--zedge-min',
                        type=float,
                        default='2.1',
                        required=False,
                        help='Minimal value of the redshift edge array,'
                             'Default value: 2.1')

    parser.add_argument('--zedge-max',
                        type=float,
                        default='6.5',
                        required=False,
                        help='Maximal value of the redshift edge array,'
                             'Default value: 6.5')

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

    parser.add_argument('--weight-method',
                        type=str,
                        default='no_weights',
                        help='Weighting scheme for the mean P1D computation,'
                             'Possible options: no_weights, simple_snr, fit_snr')

    parser.add_argument('--output-snrfit',
                        type=str,
                        default=None,
                        help='Name of the ASCII file where SNR fit results are stored,'
                        'if weight-method is fit_snr')

    parser.add_argument('--snr-cut-scheme',
                        type=str,
                        default=None,
                        help='Choice of SNR cut type, '
                             'Possible options: eboss (varying cut vs z, as in eBOSS - DR14); fixed;' 
                             'None (obligatory when --weight-method != no_weights)')

    parser.add_argument('--snrcut',
                        type=float,
                        default=None,
                        help='Value of the SNR cut if snr-cut-scheme=fixed')

    parser.add_argument('--overwrite',
                        action='store_true',
                        default=False,
                        required=False,
                        help='Overwrite the output')

    parser.add_argument('--velunits',
                        action='store_true',
                        default=False,
                        required=False,
                        help='Compute mean P1D in velocity units')

    parser.add_argument('--no-median',
                        action='store_true',
                        default=False,
                        required=False,
                        help='Skip computation of median quantities')

    parser.add_argument('--ncpu',
                        type=int,
                        default=8,
                        required=False,
                        help='Number of CPUs used to read input P1D files')

    args = parser.parse_args(sys.argv[1:])

    if (args.weight_method != 'no_weights') and (args.snr_cut_scheme is not None):
        raise ValueError("""You are using a weighting method with a
                            redshift-dependent SNR quality cut, this is not
                            tested and should bias the result""")

    if args.snr_cut_scheme == 'eboss':
        snrcut = np.array([4.1, 3.9, 3.6, 3.2, 2.9, 2.6, 2.2, 2.0, 2.0,
                              2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                              2.0, 2.0, 2.0])
        zbins_snrcut = np.arange(2.2, 6.4, 0.2)
    elif args.snr_cut_scheme == 'fixed':
        snrcut = np.array([args.snrcut])
        zbins_snrcut = None
    elif args.snr_cut_scheme == None:
        snrcut = None
        zbins_snrcut = None
    else:
        raise ValueError("Unknown value for option --snr-cut-scheme"
                         "You may add here in the code a specific SNR cutting scheme")

    kedge_min, kedge_max, kedge_bin = define_wavenumber_array(args.kedge_min,
                                                               args.kedge_max,
                                                               args.kedge_bin,
                                                               args.velunits,
                                                               args.pixsize,
                                                               args.rebinfac)
    k_edges = np.arange(kedge_min, kedge_max, kedge_bin)
    z_edges = np.around(np.arange(args.zedge_min, args.zedge_max, args.zedge_bin), 5)

    if args.output_file is None:
        med_ext = "" if args.no_median else "_medians"
        snr_ext = "_snr_cut" if snrcut is not None else ""
        vel_ext = "_vel" if args.velunits else ""
        output_file = os.path.join(args.in_dir,
                f'mean_Pk1d_{args.weight_method}{med_ext}{snr_ext}{vel_ext}.fits.gz')
    else:
        output_file = args.output_file

    postproc_pk1d.run_postproc_pk1d(args.in_dir, output_file,
                                    z_edges,
                                    k_edges,
                                    weight_method=args.weight_method,
                                    output_snrfit=args.output_snrfit,
                                    snrcut=snrcut,
                                    zbins_snrcut=zbins_snrcut,
                                    nomedians=args.no_median,
                                    velunits=args.velunits,
                                    overwrite=args.overwrite,
                                    ncpu = args.ncpu)


if __name__ == '__main__':
    cmdargs = sys.argv[1:]
    main(cmdargs)
