#!/usr/bin/env python
"""Compute the auto and cross-correlation of delta fields.

This module follow the procedure described in sections 3.1 and 3.2 of du Mas des
Bourboux et al. 2020 (In prep) to compute the 3D Lyman-alpha auto-correlation.
"""
import sys
import time
import argparse
import multiprocessing
from multiprocessing import Pool, Lock, cpu_count, Value
import numpy as np
import fitsio

from picca import constants, cf, utils, io
from picca.utils import userprint


def corr_func(healpixs):
    """Computes the correlation function.

    To optimize the computation, first compute a list of neighbours for each of
    the healpix. This is an auxiliar function to split the computational load
    using several CPUs.

    Args:
        healpixs: array of ints
            List of healpix numbers

    Returns:
        The correlation function data
    """
    cf.fill_neighs(healpixs)
    correlation_function_data = cf.compute_xi(healpixs)
    return correlation_function_data


def main(cmdargs):
    """Compute the auto and cross-correlation of delta fields"""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Compute the auto and cross-correlation of delta fields')

    parser.add_argument('--out',
                        type=str,
                        default=None,
                        required=True,
                        help='Output file name')

    parser.add_argument('--in-dir',
                        type=str,
                        default=None,
                        required=True,
                        help='Directory to delta files')

    parser.add_argument('--from-image',
                        action="store_true",
                        help='Read delta from image format')

    parser.add_argument('--in-dir2',
                        type=str,
                        default=None,
                        required=False,
                        help='Directory to 2nd delta files')

    parser.add_argument('--rp-min',
                        type=float,
                        default=0.,
                        required=False,
                        help='Min r-parallel [h^-1 Mpc]')

    parser.add_argument('--rp-max',
                        type=float,
                        default=200.,
                        required=False,
                        help='Max r-parallel [h^-1 Mpc]')

    parser.add_argument('--rt-max',
                        type=float,
                        default=200.,
                        required=False,
                        help='Max r-transverse [h^-1 Mpc]')

    parser.add_argument('--np',
                        type=int,
                        default=50,
                        required=False,
                        help='Number of r-parallel bins')

    parser.add_argument('--nt',
                        type=int,
                        default=50,
                        required=False,
                        help='Number of r-transverse bins')

    parser.add_argument('--z-cut-min',
                        type=float,
                        default=0.,
                        required=False,
                        help=('Use only pairs of forest x object with the mean '
                              'of the last absorber redshift and the object '
                              'redshift larger than z-cut-min'))

    parser.add_argument('--z-cut-max',
                        type=float,
                        default=10.,
                        required=False,
                        help=('Use only pairs of forest x object with the mean '
                              'of the last absorber redshift and the object '
                              'redshift smaller than z-cut-max'))

    parser.add_argument('--lambda-abs',
                        type=str,
                        default='LYA',
                        required=False,
                        help=('Name of the absorption in picca.constants '
                              'defining the redshift of the delta'))

    parser.add_argument('--lambda-abs2',
                        type=str,
                        default=None,
                        required=False,
                        help=('Name of the absorption in picca.constants '
                              'defining the redshift of the 2nd delta'))

    parser.add_argument('--z-ref',
                        type=float,
                        default=2.25,
                        required=False,
                        help='Reference redshift')

    parser.add_argument('--z-evol',
                        type=float,
                        default=2.9,
                        required=False,
                        help=('Exponent of the redshift evolution of the delta '
                              'field'))

    parser.add_argument('--z-evol2',
                        type=float,
                        default=2.9,
                        required=False,
                        help=('Exponent of the redshift evolution of the 2nd '
                              'delta field'))

    parser.add_argument('--fid-Om',
                        type=float,
                        default=0.315,
                        required=False,
                        help=('Omega_matter(z=0) of fiducial LambdaCDM '
                              'cosmology'))

    parser.add_argument('--fid-Or',
                        type=float,
                        default=0.,
                        required=False,
                        help=('Omega_radiation(z=0) of fiducial LambdaCDM '
                              'cosmology'))

    parser.add_argument('--fid-Ok',
                        type=float,
                        default=0.,
                        required=False,
                        help='Omega_k(z=0) of fiducial LambdaCDM cosmology')

    parser.add_argument('--fid-wl',
                        type=float,
                        default=-1.,
                        required=False,
                        help=('Equation of state of dark energy of fiducial '
                              'LambdaCDM cosmology'))

    parser.add_argument('--no-project',
                        action='store_true',
                        required=False,
                        help='Do not project out continuum fitting modes')

    parser.add_argument('--remove-same-half-plate-close-pairs',
                        action='store_true',
                        required=False,
                        help=('Reject pairs in the first bin in r-parallel '
                              'from same half plate'))

    parser.add_argument('--nside',
                        type=int,
                        default=16,
                        required=False,
                        help='Healpix nside')

    parser.add_argument('--nproc',
                        type=int,
                        default=None,
                        required=False,
                        help='Number of processors')

    parser.add_argument('--nspec',
                        type=int,
                        default=None,
                        required=False,
                        help='Maximum number of spectra to read')

    parser.add_argument('--unfold-cf',
                        action='store_true',
                        required=False,
                        help=('rp can be positive or negative depending on the '
                              'relative position between absorber1 and '
                              'absorber2'))

    parser.add_argument('--shuffle-distrib-forest-seed',
                        type=int,
                        default=None,
                        required=False,
                        help=('Shuffle the distribution of forests on the sky '
                              'following the given seed. Do not shuffle if '
                              'None'))

    parser.add_argument('--rebin-factor',
                        type=int,
                        default=None,
                        required=False,
                        help='Rebin factor for deltas. If not None, deltas will '
                             'be rebinned by that factor')

    args = parser.parse_args(cmdargs)

    if args.nproc is None:
        args.nproc = cpu_count() // 2

    # setup variables in module cf
    cf.r_par_max = args.rp_max
    cf.r_trans_max = args.rt_max
    cf.r_par_min = args.rp_min
    cf.z_cut_max = args.z_cut_max
    cf.z_cut_min = args.z_cut_min
    cf.num_bins_r_par = args.np
    cf.num_bins_r_trans = args.nt
    cf.nside = args.nside
    cf.z_ref = args.z_ref
    cf.alpha = args.z_evol
    cf.lambda_abs = constants.ABSORBER_IGM[args.lambda_abs]
    cf.remove_same_half_plate_close_pairs = args.remove_same_half_plate_close_pairs

    # read blinding keyword
    blinding = io.read_blinding(args.in_dir)

    # load fiducial cosmology
    cosmo = constants.Cosmo(Om=args.fid_Om,
                            Or=args.fid_Or,
                            Ok=args.fid_Ok,
                            wl=args.fid_wl,
                            blinding=blinding)

    t0 = time.time()

    ### Read data 1
    data, num_data, z_min, z_max = io.read_deltas(args.in_dir,
                                                  cf.nside,
                                                  cf.lambda_abs,
                                                  cf.alpha,
                                                  cf.z_ref,
                                                  cosmo,
                                                  max_num_spec=args.nspec,
                                                  no_project=args.no_project,
                                                  from_image=args.from_image,
                                                  nproc=args.nproc,
                                                  rebin_factor=args.rebin_factor)
    del z_max
    cf.data = data
    cf.num_data = num_data
    cf.ang_max = utils.compute_ang_max(cosmo, cf.r_trans_max, z_min)
    userprint("")
    userprint("done, npix = {}".format(len(data)))

    ### Read data 2
    if args.in_dir2 or args.lambda_abs2:
        if args.lambda_abs2 or args.unfold_cf:
            cf.x_correlation = True
        cf.alpha2 = args.z_evol2
        if args.in_dir2 is None:
            args.in_dir2 = args.in_dir
        if args.lambda_abs2:
            cf.lambda_abs2 = constants.ABSORBER_IGM[args.lambda_abs2]
        else:
            cf.lambda_abs2 = cf.lambda_abs

        data2, num_data2, z_min2, z_max2 = io.read_deltas(
            args.in_dir2,
            cf.nside,
            cf.lambda_abs2,
            cf.alpha2,
            cf.z_ref,
            cosmo,
            max_num_spec=args.nspec,
            no_project=args.no_project,
            from_image=args.from_image,
            nproc=args.nproc,
            rebin_factor=args.rebin_factor)
        del z_max2
        cf.data2 = data2
        cf.num_data2 = num_data2
        cf.ang_max = utils.compute_ang_max(cosmo, cf.r_trans_max, z_min, z_min2)
        userprint("")
        userprint("done, npix = {}".format(len(data2)))

    # shuffle forests
    if args.shuffle_distrib_forest_seed is not None:
        cf.data = utils.shuffle_distrib_forests(
            cf.data, args.shuffle_distrib_forest_seed)

    t1 = time.time()
    userprint(f'picca_cf.py - Time reading data: {(t1-t0)/60:.3f} minutes')
    # compute correlation function, use pool to parallelize
    cf.counter = Value('i', 0)
    cf.lock = Lock()
    cpu_data = {healpix: [healpix] for healpix in data}
    context = multiprocessing.get_context('fork')
    pool = context.Pool(processes=args.nproc)
    correlation_function_data = pool.map(corr_func, sorted(cpu_data.values()))
    pool.close()

    t2 = time.time()
    userprint(f'picca_cf.py - Time computing correlation function: {(t2-t1)/60:.3f} minutes')

    # group data from parallelisation
    correlation_function_data = np.array(correlation_function_data)
    weights_list = correlation_function_data[:, 0, :]
    xi_list = correlation_function_data[:, 1, :]
    r_par_list = correlation_function_data[:, 2, :]
    r_trans_list = correlation_function_data[:, 3, :]
    z_list = correlation_function_data[:, 4, :]
    num_pairs_list = correlation_function_data[:, 5, :].astype(np.int64)
    healpix_list = np.array(sorted(list(cpu_data.keys())))

    # normalize values
    w = (weights_list.sum(axis=0) > 0.)
    r_par = (r_par_list * weights_list).sum(axis=0)
    r_par[w] /= weights_list.sum(axis=0)[w]
    r_trans = (r_trans_list * weights_list).sum(axis=0)
    r_trans[w] /= weights_list.sum(axis=0)[w]
    z = (z_list * weights_list).sum(axis=0)
    z[w] /= weights_list.sum(axis=0)[w]
    num_pairs = num_pairs_list.sum(axis=0)

    # save data
    results = fitsio.FITS(args.out, 'rw', clobber=True)
    header = [{
        'name': 'RPMIN',
        'value': cf.r_par_min,
        'comment': 'Minimum r-parallel [h^-1 Mpc]'
    }, {
        'name': 'RPMAX',
        'value': cf.r_par_max,
        'comment': 'Maximum r-parallel [h^-1 Mpc]'
    }, {
        'name': 'RTMAX',
        'value': cf.r_trans_max,
        'comment': 'Maximum r-transverse [h^-1 Mpc]'
    }, {
        'name': 'NP',
        'value': cf.num_bins_r_par,
        'comment': 'Number of bins in r-parallel'
    }, {
        'name': 'NT',
        'value': cf.num_bins_r_trans,
        'comment': 'Number of bins in r-transverse'
    }, {
        'name': 'ZCUTMIN',
        'value': cf.z_cut_min,
        'comment': 'Minimum redshift of pairs'
    }, {
        'name': 'ZCUTMAX',
        'value': cf.z_cut_max,
        'comment': 'Maximum redshift of pairs'
    }, {
        'name': 'NSIDE',
        'value': cf.nside,
        'comment': 'Healpix nside'
    }, {
        'name': 'OMEGAM',
        'value': args.fid_Om,
        'comment': 'Omega_matter(z=0) of fiducial LambdaCDM cosmology'
    }, {
        'name': 'OMEGAR',
        'value': args.fid_Or,
        'comment': 'Omega_radiation(z=0) of fiducial LambdaCDM cosmology'
    }, {
        'name': 'OMEGAK',
        'value': args.fid_Ok,
        'comment': 'Omega_k(z=0) of fiducial LambdaCDM cosmology'
    }, {
        'name': 'WL',
        'value': args.fid_wl,
        'comment': 'Equation of state of dark energy of fiducial LambdaCDM cosmology'
    }, {
        'name': "BLINDING",
        'value': blinding,
        'comment': 'String specifying the blinding strategy'
    }
    ]
    results.write(
        [r_par, r_trans, z, num_pairs],
        names=['RP', 'RT', 'Z', 'NB'],
        comment=['R-parallel', 'R-transverse', 'Redshift', 'Number of pairs'],
        units=['h^-1 Mpc', 'h^-1 Mpc', '', ''],
        header=header,
        extname='ATTRI')

    header2 = [{
        'name': 'HLPXSCHM',
        'value': 'RING',
        'comment': 'Healpix scheme'
    }]
    xi_list_name = "DA"
    if blinding != "none":
        xi_list_name += "_BLIND"
    results.write([healpix_list, weights_list, xi_list],
                  names=['HEALPID', 'WE', xi_list_name],
                  comment=['Healpix index', 'Sum of weight', 'Correlation'],
                  header=header2,
                  extname='COR')

    results.close()

    t3 = time.time()
    userprint(f'picca_cf.py - Time total : {(t3-t0)/60:.3f} minutes')

if __name__ == '__main__':
    cmdargs=sys.argv[1:]
    main(cmdargs)
