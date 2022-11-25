#!/usr/bin/env python
"""Compute the auto and cross-correlation between a catalog of objects and a
delta field.

This module follow the procedure described in sections 3.1 and 3.3 of du Mas des
Bourboux et al. 2020 (In prep) to compute the 3D Lyman-alpha auto-correlation.
"""
import sys
import time
import argparse
import multiprocessing
from multiprocessing import Pool, Lock, cpu_count, Value
import numpy as np
import fitsio

from picca import constants, xcf, io, prep_del, utils
from picca.data import Forest
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
    xcf.fill_neighs(healpixs)
    correlation_function_data = xcf.compute_xi(healpixs)
    return correlation_function_data


def main(cmdargs):
    """Compute the cross-correlation between a catalog of objects and a delta
    field."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=('Compute the cross-correlation between a catalog of '
                     'objects and a delta field.'))

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
                        type=str,
                        default=None,
                        required=False,
                        help='Read delta from image format',
                        nargs='*')

    parser.add_argument('--drq',
                        type=str,
                        default=None,
                        required=True,
                        help='Catalog of objects in format selected by mode')

    parser.add_argument('--mode',
                        type=str,
                        required=False,
                        choices=['desi','desi_healpix','desi_mocks','sdss'],
                        default='sdss',
                        help='Mode for reading the catalog (default sdss)'
                        )

    parser.add_argument('--rp-min',
                        type=float,
                        default=-200.,
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
                        default=100,
                        required=False,
                        help='Number of r-parallel bins')

    parser.add_argument('--nt',
                        type=int,
                        default=50,
                        required=False,
                        help='Number of r-transverse bins')

    parser.add_argument('--z-min-obj',
                        type=float,
                        default=0,
                        required=False,
                        help='Min redshift for object field')

    parser.add_argument('--z-max-obj',
                        type=float,
                        default=10,
                        required=False,
                        help='Max redshift for object field')

    parser.add_argument(
        '--z-cut-min',
        type=float,
        default=0.,
        required=False,
        help=('Use only pairs of forest x object with the mean of the last '
              'absorber redshift and the object redshift larger than '
              'z-cut-min'))

    parser.add_argument(
        '--z-cut-max',
        type=float,
        default=10.,
        required=False,
        help=('Use only pairs of forest x object with the mean of the last '
              'absorber redshift and the object redshift smaller than '
              'z-cut-max'))

    parser.add_argument(
        '--lambda-abs',
        type=str,
        default='LYA',
        required=False,
        help=('Name of the absorption in picca.constants defining the redshift '
              'of the delta'))

    parser.add_argument('--z-ref',
                        type=float,
                        default=2.25,
                        required=False,
                        help='Reference redshift')

    parser.add_argument(
        '--z-evol-del',
        type=float,
        default=2.9,
        required=False,
        help='Exponent of the redshift evolution of the delta field')

    parser.add_argument(
        '--z-evol-obj',
        type=float,
        default=1.,
        required=False,
        help='Exponent of the redshift evolution of the object field')

    parser.add_argument(
        '--fid-Om',
        type=float,
        default=0.315,
        required=False,
        help='Omega_matter(z=0) of fiducial LambdaCDM cosmology')

    parser.add_argument(
        '--fid-Or',
        type=float,
        default=0.,
        required=False,
        help='Omega_radiation(z=0) of fiducial LambdaCDM cosmology')

    parser.add_argument('--fid-Ok',
                        type=float,
                        default=0.,
                        required=False,
                        help='Omega_k(z=0) of fiducial LambdaCDM cosmology')

    parser.add_argument(
        '--fid-wl',
        type=float,
        default=-1.,
        required=False,
        help='Equation of state of dark energy of fiducial LambdaCDM cosmology')

    parser.add_argument('--no-project',
                        action='store_true',
                        required=False,
                        help='Do not project out continuum fitting modes')

    parser.add_argument('--no-remove-mean-lambda-obs',
                        action='store_true',
                        required=False,
                        help='Do not remove mean delta versus lambda_obs')

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

    parser.add_argument(
        '--shuffle-distrib-obj-seed',
        type=int,
        default=None,
        required=False,
        help=('Shuffle the distribution of objects on the sky following the '
              'given seed. Do not shuffle if None'))

    parser.add_argument(
        '--shuffle-distrib-forest-seed',
        type=int,
        default=None,
        required=False,
        help=('Shuffle the distribution of forests on the sky following the '
              'given seed. Do not shuffle if None'))

    parser.add_argument('--rebin-factor',
                        type=int,
                        default=None,
                        required=False,
                        help='Rebin factor for deltas. If not None, deltas will '
                             'be rebinned by that factor')


    args = parser.parse_args(cmdargs)
    if args.nproc is None:
        args.nproc = cpu_count() // 2

    # setup variables in module xcf
    xcf.r_par_max = args.rp_max
    xcf.r_par_min = args.rp_min
    xcf.z_cut_max = args.z_cut_max
    xcf.z_cut_min = args.z_cut_min
    xcf.r_trans_max = args.rt_max
    xcf.num_bins_r_par = args.np
    xcf.num_bins_r_trans = args.nt
    xcf.nside = args.nside
    xcf.lambda_abs = constants.ABSORBER_IGM[args.lambda_abs]

    # read blinding keyword
    blinding = io.read_blinding(args.in_dir)

    # load fiducial cosmology
    cosmo = constants.Cosmo(Om=args.fid_Om,
                            Or=args.fid_Or,
                            Ok=args.fid_Ok,
                            wl=args.fid_wl,
                            blinding=blinding)

    t0 = time.time()

    # Find the redshift range
    if args.z_min_obj is None:
        r_comov_min = cosmo.get_r_comov(z_min)
        r_comov_min = max(0., r_comov_min + xcf.r_par_min)
        args.z_min_obj = cosmo.distance_to_redshift(r_comov_min)
        userprint("z_min_obj = {}".format(args.z_min_obj), end="")
    if args.z_max_obj is None:
        r_comov_max = cosmo.get_r_comov(z_max)
        r_comov_max = max(0., r_comov_max + xcf.r_par_max)
        args.z_max_obj = cosmo.distance_to_redshift(r_comov_max)
        userprint("z_max_obj = {}".format(args.z_max_obj), end="")

    ### Read objects
    objs, z_min2 = io.read_objects(args.drq, args.nside, args.z_min_obj,
                                   args.z_max_obj, args.z_evol_obj, args.z_ref,
                                   cosmo, mode=args.mode)
    xcf.objs = objs

    ### Read deltas
    data, num_data, z_min, z_max = io.read_deltas(args.in_dir,
                                                  args.nside,
                                                  xcf.lambda_abs,
                                                  args.z_evol_del,
                                                  args.z_ref,
                                                  cosmo=cosmo,
                                                  max_num_spec=args.nspec,
                                                  no_project=args.no_project,
                                                  from_image=args.from_image,
                                                  nproc=args.nproc,
                                                  rebin_factor=args.rebin_factor)
    xcf.data = data
    xcf.num_data = num_data
    userprint("")
    userprint("done, npix = {}\n".format(len(data)))
    ### Remove <delta> vs. lambda_obs
    if not args.no_remove_mean_lambda_obs:
        Forest.delta_log_lambda = None
        for healpix in xcf.data:
            for delta in xcf.data[healpix]:
                delta_log_lambda = np.asarray([
                    delta.log_lambda[index] - delta.log_lambda[index - 1]
                    for index in range(1, delta.log_lambda.size)
                ]).min()
                if Forest.delta_log_lambda is None:
                    Forest.delta_log_lambda = delta_log_lambda
                else:
                    Forest.delta_log_lambda = min(delta_log_lambda,
                                                  Forest.delta_log_lambda)
        Forest.log_lambda_min = (np.log10(
            (z_min + 1.) * xcf.lambda_abs) - Forest.delta_log_lambda / 2.)
        Forest.log_lambda_max = (np.log10(
            (z_max + 1.) * xcf.lambda_abs) + Forest.delta_log_lambda / 2.)
        log_lambda, mean_delta, stack_weight = prep_del.stack(
            xcf.data, stack_from_deltas=True)
        del log_lambda, stack_weight
        for healpix in xcf.data:
            for delta in xcf.data[healpix]:
                bins = ((delta.log_lambda - Forest.log_lambda_min) /
                        Forest.delta_log_lambda + 0.5).astype(int)
                delta.delta -= mean_delta[bins]

    # shuffle forests and objects
    if not args.shuffle_distrib_obj_seed is None:
        xcf.objs = utils.shuffle_distrib_forests(objs,
                                                 args.shuffle_distrib_obj_seed)
    if not args.shuffle_distrib_forest_seed is None:
        xcf.data = utils.shuffle_distrib_forests(
            xcf.data, args.shuffle_distrib_forest_seed)

    userprint("")

    # compute maximum angular separation
    xcf.ang_max = utils.compute_ang_max(cosmo, xcf.r_trans_max, z_min, z_min2)

    t1 = time.time()
    userprint(f'picca_xcf.py - Time reading data: {(t1-t0)/60:.3f} minutes')

    # compute correlation function, use pool to parallelize
    xcf.counter = Value('i', 0)
    xcf.lock = Lock()
    cpu_data = {healpix: [healpix] for healpix in data}
    context = multiprocessing.get_context('fork')
    pool = context.Pool(processes=args.nproc)
    correlation_function_data = pool.map(corr_func, sorted(cpu_data.values()))
    pool.close()

    t2 = time.time()
    userprint(f'picca_xcf.py - Time computing cross-correlation function: {(t2-t1)/60:.3f} minutes')

    # group data from parallelisation
    correlation_function_data = np.array(correlation_function_data)
    weights_list = correlation_function_data[:, 0, :]
    xi_list = correlation_function_data[:, 1, :]
    r_par_list = correlation_function_data[:, 2, :]
    r_trans_list = correlation_function_data[:, 3, :]
    z_list = correlation_function_data[:, 4, :]
    num_pairs_list = correlation_function_data[:, 5, :].astype(np.int64)
    healpix_list = np.array(sorted(list(cpu_data.keys())))

    w = (weights_list.sum(axis=0) > 0.)
    r_par = (r_par_list * weights_list).sum(axis=0)
    r_par[w] /= weights_list.sum(axis=0)[w]
    r_trans = (r_trans_list * weights_list).sum(axis=0)
    r_trans[w] /= weights_list.sum(axis=0)[w]
    z = (z_list * weights_list).sum(axis=0)
    z[w] /= weights_list.sum(axis=0)[w]
    num_pairs = num_pairs_list.sum(axis=0)

    results = fitsio.FITS(args.out, 'rw', clobber=True)
    header = [{
        'name': 'RPMIN',
        'value': xcf.r_par_min,
        'comment': 'Minimum r-parallel [h^-1 Mpc]'
    }, {
        'name': 'RPMAX',
        'value': xcf.r_par_max,
        'comment': 'Maximum r-parallel [h^-1 Mpc]'
    }, {
        'name': 'RTMAX',
        'value': xcf.r_trans_max,
        'comment': 'Maximum r-transverse [h^-1 Mpc]'
    }, {
        'name': 'NP',
        'value': xcf.num_bins_r_par,
        'comment': 'Number of bins in r-parallel'
    }, {
        'name': 'NT',
        'value': xcf.num_bins_r_trans,
        'comment': 'Number of bins in r-transverse'
    }, {
        'name': 'ZCUTMIN',
        'value': xcf.z_cut_min,
        'comment': 'Minimum redshift of pairs'
    }, {
        'name': 'ZCUTMAX',
        'value': xcf.z_cut_max,
        'comment': 'Maximum redshift of pairs'
    }, {
        'name': 'NSIDE',
        'value': xcf.nside,
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
    da_name = "DA"
    if blinding != "none":
        da_name += "_BLIND"
    results.write([healpix_list, weights_list, xi_list],
                  names=['HEALPID', 'WE', da_name],
                  comment=['Healpix index', 'Sum of weight', 'Correlation'],
                  header=header2,
                  extname='COR')

    results.close()

    t3 = time.time()
    userprint(f'picca_xcf.py - Time total: {(t3-t0)/60:.3f} minutes')

if __name__ == '__main__':
    cmdargs=sys.argv[1:]
    main(cmdargs)
