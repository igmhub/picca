#!/usr/bin/env python
"""Compute the auto and cross-correlation of delta fields as a function of
angle and wavelength ratio
"""
import sys
import argparse
import multiprocessing
from multiprocessing import Pool, Lock, cpu_count, Value
import numpy as np
import fitsio

from picca import constants, cf, io
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
    """Compute the auto and cross-correlation of delta fields as a function of
    angle and wavelength ratio"""

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=('Compute the auto and cross-correlation of delta fields '
                     'as a function of angle and wavelength ratio'))

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

    parser.add_argument('--in-dir2',
                        type=str,
                        default=None,
                        required=False,
                        help='Directory to 2nd delta files')

    parser.add_argument('--wr-min',
                        type=float,
                        default=1.,
                        required=False,
                        help='Min of wavelength ratio')

    parser.add_argument('--wr-max',
                        type=float,
                        default=1.1,
                        required=False,
                        help='Max of wavelength ratio')

    parser.add_argument('--ang-max',
                        type=float,
                        default=0.02,
                        required=False,
                        help='Max angle (rad)')

    parser.add_argument('--np',
                        type=int,
                        default=50,
                        required=False,
                        help='Number of wavelength ratio bins')

    parser.add_argument('--nt',
                        type=int,
                        default=50,
                        required=False,
                        help='Number of angular bins')

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
        '--lambda-abs2',
        type=str,
        default=None,
        required=False,
        help=('Name of the absorption in picca.constants defining the redshift '
              'of the 2nd delta'))

    parser.add_argument('--z-ref',
                        type=float,
                        default=2.25,
                        required=False,
                        help='Reference redshift')

    parser.add_argument(
        '--z-evol',
        type=float,
        default=2.9,
        required=False,
        help='Exponent of the redshift evolution of the delta field')

    parser.add_argument(
        '--z-evol2',
        type=float,
        default=2.9,
        required=False,
        help='Exponent of the redshift evolution of the 2nd delta field')

    parser.add_argument('--no-project',
                        action='store_true',
                        required=False,
                        help='Do not project out continuum fitting modes')

    parser.add_argument(
        '--remove-same-half-plate-close-pairs',
        action='store_true',
        required=False,
        help='Reject pairs in the first bin in r-parallel from same half plate')

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

    # setup variables in module cf
    cf.r_par_min = args.wr_min
    cf.r_par_max = args.wr_max
    cf.r_trans_max = args.ang_max
    cf.z_cut_max = args.z_cut_max
    cf.z_cut_min = args.z_cut_min
    cf.num_bins_r_par = args.np
    cf.num_bins_r_trans = args.nt
    cf.nside = args.nside
    cf.z_ref = args.z_ref
    cf.alpha = args.z_evol
    cf.x_correlation = False
    cf.ang_correlation = True
    cf.ang_max = args.ang_max
    cf.lambda_abs = constants.ABSORBER_IGM[args.lambda_abs]
    cf.remove_same_half_plate_close_pairs = args.remove_same_half_plate_close_pairs

    ### Read data 1
    data, num_data, z_min, z_max = io.read_deltas(args.in_dir,
                                                  cf.nside,
                                                  cf.lambda_abs,
                                                  cf.alpha,
                                                  cf.z_ref,
                                                  cosmo=None,
                                                  max_num_spec=args.nspec,
                                                  no_project=args.no_project)
    cf.data = data
    cf.num_data = num_data
    del z_min, z_max
    userprint("")
    userprint("done, npix = {}".format(len(data)))

    ### Read data 2
    if args.in_dir2 or args.lambda_abs2:
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
            cosmo=None,
            max_num_spec=args.nspec,
            no_project=args.no_project)
        cf.data2 = data2
        cf.num_data2 = num_data2
        del z_min2, z_max2
        userprint("")
        userprint("done, npix = {}".format(len(data2)))

    # compute correlation function, use pool to parallelize
    cf.counter = Value('i', 0)
    cf.lock = Lock()
    cpu_data = {healpix: [healpix] for healpix in data}
    context = multiprocessing.get_context('fork')
    pool = context.Pool(processes=args.nproc)
    correlation_function_data = pool.map(corr_func, sorted(cpu_data.values()))
    pool.close()

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

    results = fitsio.FITS(args.out, 'rw', clobber=True)
    header = [{
        'name': 'RPMIN',
        'value': cf.r_par_min,
        'comment': 'Minimum wavelength ratio'
    }, {
        'name': 'RPMAX',
        'value': cf.r_par_max,
        'comment': 'Maximum wavelength ratio'
    }, {
        'name': 'RTMAX',
        'value': cf.r_trans_max,
        'comment': 'Maximum angle [rad]'
    }, {
        'name': 'NP',
        'value': cf.num_bins_r_par,
        'comment': 'Number of bins in wavelength ratio'
    }, {
        'name': 'NT',
        'value': cf.num_bins_r_trans,
        'comment': 'Number of bins in angle'
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
    }]
    results.write(
        [r_par, r_trans, z, num_pairs],
        names=['RP', 'RT', 'Z', 'NB'],
        units=['', 'rad', '', ''],
        comment=['Wavelength ratio', 'Angle', 'Redshift', 'Number of pairs'],
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
