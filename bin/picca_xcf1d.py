#!/usr/bin/env python
"""Compute the 1D auto or cross-correlation between a catalog of objects and a
delta field as a function of wavelength ratio
"""
import argparse
import sys
import multiprocessing
from multiprocessing import Pool, cpu_count
import numpy as np
import fitsio

from picca import constants, xcf, io, prep_del
from picca.data import Forest
from picca.utils import userprint


def corr_func(healpixs):
    """Compute the 1D cross-correlation for a given list of healpixs
    Args:
        healpixs: array of ints
            List of healpix numbers

    Returns:
        The correlation function
    """
    correlation_function_data = xcf.compute_xi_1d(healpixs)
    return correlation_function_data


def main(cmdargs):
    """Compute the 1D cross-correlation between a catalog of objects and a delta
    field as a function of wavelength ratio"""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=('Compute the 1D cross-correlation between a catalog of '
                     'objects and a delta field as a function of wavelength '
                     'ratio'))

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

    parser.add_argument('--drq',
                        type=str,
                        default=None,
                        required=True,
                        help='Catalog of objects in DRQ format')

    parser.add_argument(
                        '--mode',
                        type=str,
                        default='sdss',
                        choices=['sdss','desi'],
                        required=False,
                        help='type of catalog supplied, default sdss')

    parser.add_argument('--wr-min',
                        type=float,
                        default=0.9,
                        required=False,
                        help='Min of wavelength ratio')

    parser.add_argument('--wr-max',
                        type=float,
                        default=1.1,
                        required=False,
                        help='Max of wavelength ratio')

    parser.add_argument('--np',
                        type=int,
                        default=100,
                        required=False,
                        help='Number of wavelength ratio bins')

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

    parser.add_argument(
        '--lambda-abs-obj',
        type=str,
        default='LYA',
        required=False,
        help=('Name of the absorption in picca.constants the object is '
              'considered as'))

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

    args = parser.parse_args(cmdargs)
    if args.nproc is None:
        args.nproc = cpu_count() // 2

    # setup variables in module xcf
    xcf.r_par_min = args.wr_min
    xcf.r_par_max = args.wr_max
    xcf.r_trans_max = 1.e-6
    xcf.z_cut_min = args.z_cut_min
    xcf.z_cut_max = args.z_cut_max
    xcf.num_bins_r_par = args.np
    xcf.nt = 1
    xcf.nside = args.nside
    xcf.ang_correlation = True

    lambda_abs = constants.ABSORBER_IGM[args.lambda_abs]

    ### Read deltas
    data, num_data, z_min, z_max = io.read_deltas(args.in_dir,
                                                  args.nside,
                                                  lambda_abs,
                                                  args.z_evol_del,
                                                  args.z_ref,
                                                  cosmo=None,
                                                  max_num_spec=args.nspec,
                                                  no_project=args.no_project)
    xcf.data = data
    xcf.num_data = num_data
    sys.stderr.write("\n")
    userprint("done, npix = {}".format(len(data)))

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
            (z_min + 1.) * lambda_abs) - Forest.delta_log_lambda / 2.)
        Forest.log_lambda_max = (np.log10(
            (z_max + 1.) * lambda_abs) + Forest.delta_log_lambda / 2.)
        log_lambda, mean_delta, stack_weight = prep_del.stack(
            xcf.data, stack_from_deltas=True)
        del log_lambda, stack_weight
        for healpix in xcf.data:
            for delta in xcf.data[healpix]:
                bins = ((delta.log_lambda - Forest.log_lambda_min) /
                        Forest.delta_log_lambda + 0.5).astype(int)
                delta.delta -= mean_delta[bins]

    ### Read objects
    objs, z_min2 = io.read_objects(args.drq,
                                   args.nside,
                                   args.z_min_obj,
                                   args.z_max_obj,
                                   args.z_evol_obj,
                                   args.z_ref,
                                   cosmo=None,
                                   mode=args.mode)
    del z_min2
    xcf.objs = objs
    for healpix in xcf.objs:
        for obj in xcf.objs[healpix]:
            obj.log_lambda = np.log10(
                (1. + obj.z_qso) * constants.ABSORBER_IGM[args.lambda_abs_obj])
    sys.stderr.write("\n")

    # Compute the correlation function, use pool to parallelize
    context = multiprocessing.get_context('fork')
    pool = context.Pool(processes=args.nproc)
    healpixs = [[healpix] for healpix in sorted(data) if healpix in xcf.objs]
    correlation_function_data = pool.map(corr_func, healpixs)
    pool.close()

    # group data from parallelisation
    correlation_function_data = np.array(correlation_function_data)
    weights_list = correlation_function_data[:, 0, :]
    xi_list = correlation_function_data[:, 1, :]
    r_par_list = correlation_function_data[:, 2, :]
    z_list = correlation_function_data[:, 3, :]
    num_pairs_list = correlation_function_data[:, 4, :].astype(np.int64)
    healpix_list = np.array(
        [healpix for healpix in sorted(data) if healpix in xcf.objs])

    w = (weights_list.sum(axis=0) > 0.)
    r_par = (r_par_list * weights_list).sum(axis=0)
    r_par[w] /= weights_list.sum(axis=0)[w]
    z = (z_list * weights_list).sum(axis=0)
    z[w] /= weights_list.sum(axis=0)[w]
    num_pairs = num_pairs_list.sum(axis=0)

    results = fitsio.FITS(args.out, 'rw', clobber=True)
    header = [{
        'name': 'RPMIN',
        'value': xcf.r_par_min,
        'comment': 'Minimum wavelength ratio'
    }, {
        'name': 'RPMAX',
        'value': xcf.r_par_max,
        'comment': 'Maximum wavelength ratio'
    }, {
        'name': 'NP',
        'value': xcf.num_bins_r_par,
        'comment': 'Number of bins in wavelength ratio'
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
    }]
    results.write([r_par, z, num_pairs],
                  names=['RP', 'Z', 'NB'],
                  units=['', '', ''],
                  comment=['Wavelength ratio', 'Redshift', 'Number of pairs'],
                  header=header,
                  extname='ATTRI')

    header2 = [{
        'name': 'HLPXSCHM',
        'value': 'RING',
        'comment': 'Healpix scheme'
    }]
    results.write([healpix_list, weights_list, xi_list],
                  names=['HEALPID', 'WE', 'DA'],
                  comment=['Healpix index', 'Sum of weight', 'Correlation'],
                  header=header2,
                  extname='COR')

    results.close()


if __name__ == '__main__':
    cmdargs=sys.argv[1:]
    main(cmdargs)
