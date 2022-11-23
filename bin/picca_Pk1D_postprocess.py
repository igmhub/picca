#!/usr/bin/env python
"""Compute the averaged 1D power spectrum
"""

import sys, os, argparse
import numpy as np
import fitsio
from picca.pk1d import postproc_pk1d




def define_wavevector_limits(args,velunits,pixsize):
    if velunits:
            k_inf=k_inf_vel
            k_sup=k_sup_vel
            k_dist=k_bin_dist_vel
        else:
            k_inf=0.000813
            k_dist=0.000542*args["rebinfac"]
            k_inf_lin=2*np.pi/((1200-1050)*(1+3.4)/args["rebinfac"])
            k_sup_lin=np.pi/pixsize
            nb_k_bin=int(k_sup_lin/k_inf_lin/4)
            k_sup=k_inf + nb_k_bin*k_dist
    else:
        if("k_inf_lin" in args.keys()):
            k_inf=args["k_inf_lin"]
            k_sup=args["k_sup_lin"]
            k_dist=args["k_bin_dist_lin"]
        else:
            k_inf=2*np.pi/((1200-1050)*(1+3.4)/args["rebinfac"])
            k_sup=np.pi/pixsize
            nb_k_bin=int(k_sup/k_inf/4)
            k_dist=(k_sup-k_inf)/nb_k_bin
    return(k_inf,k_sup,k_dist)



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
                        help='')

    parser.add_argument('--zedge-max',
                        type=float,
                        default='6.3',
                        required=False,
                        help='')

    parser.add_argument('--zbin-size',
                        type=float,
                        default='0.2',
                        required=False,
                        help='')

    parser.add_argument('--weights-method',
                        type=str,
                        default='no_weights',
                        required=False,
                        help='Weighted scheme for the averaging.'
                             'Possible options: no_weights, simple_snr')

    parser.add_argument('--apply-mean-snr-cut',
                        action='store_true',
                        default=True,
                        required=False,
                        help='')

    parser.add_argument('--snr-cut-scheme',
                        type=str,
                        default='eboss',
                        required=False,
                        help='')

    parser.add_argument('--overwrite',
                        action='store_true',
                        default=True,
                        required=False,
                        help='')

    parser.add_argument('--velunits',
                        action='store_true',
                        default=False,
                        required=False,
                        help='Make the calculation in velocity units')

    parser.add_argument('--no-median',
                        action='store_true',
                        default=True,
                        required=False,
                        help='')





    args = parser.parse_args(sys.argv[1:])

    if args.apply_mean_snr_cut:
        if args.snr_cut_scheme is "eboss":
            snr_cut_mean = [4.1, 3.9, 3.6, 3.2, 2.9, 2.6, 2.2, 2.0, 2.0, 2.0, 2.0,
                            2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
            zbins =        [2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2,
                            4.4, 4.6, 4.8, 5.0, 5.2, 5.4, 5.6, 5.8, 6.0, 6.2]
        else:
            raise ValueError("Please choose the snr cutting scheme to be eboss, "
                             "or turn of the --apply-mean-snr-cut parameter, or "
                             "add here in the code a specific snr cutting scheme")
    else:
        snr_cut_mean = None
        zbins = None

    zbin_edges = np.arange(args.zedge_min, args.zedge_max, args.zbin_size)


    #
    # # kbin_edges computation
    # rebinfac=3
    # pixsize=0.8
    # k_inf_lin=2*np.pi/((1200-1050)*(1+3.4)/rebinfac)
    # k_sup_lin=np.pi/pixsize
    # nb_k_bin=int(k_sup_lin/k_inf_lin/4)
    # k_bin_dist_lin=(k_sup_lin-k_inf_lin)/nb_k_bin
    # k_inf_vel=0.000813
    # k_bin_dist_vel=0.000542*rebinfac
    # k_sup_vel=k_inf_vel + nb_k_bin*k_bin_dist_vel
    # args={}
    # args['z_binsize']=0.2
    # args['k_inf_lin']=k_inf_lin
    # args['k_sup_lin']=k_sup_lin
    # args['k_bin_dist_lin']=k_bin_dist_lin
    # args['k_inf_vel']=k_inf_vel
    # args['k_sup_vel']=k_sup_vel
    # args['k_bin_dist_vel']=k_bin_dist_vel
    # k_inf, k_sup, k_dist = define_wavevector_limits(args,velunits,pixsize)
    # kbin_edges=np.arange(k_inf,k_sup,k_dist)
    # print(kbin_edges)



    data = postproc_pk1d.parallelize_p1d_comp(args.in_dir,
                                              zbin_edges,
                                              kbin_edges,
                                              weights_method=args.weights_method,
                                              snr_cut_mean=snr_cut_mean,
                                              zbins=zbins,
                                              nomedians=args.no_medians,
                                              velunits=args.velunits,
                                              overwrite=args.overwrite)

if __name__ == '__main__':
    cmdargs = sys.argv[1:]
    main(cmdargs)
