#!/usr/bin/env python
"""Compute the 1D auto or cross-correlation between delta field from the same
forest.
"""
import sys
import argparse
import multiprocessing
from multiprocessing import Pool, Lock, cpu_count, Value
import numpy as np
import fitsio

from picca import constants, cf, io
from picca.utils import userprint


def corr_func(p):
    """Compute the 1D auto- or the cross-correlation for a given healpix

    Args:
        healpix: int
            The healpix number

    Returns:
        The correlation function
    """
    if cf.x_correlation:
        correlation_function_data = cf.compute_xi_1d_cross(p)
    else:
        correlation_function_data = cf.compute_xi_1d(p)
    with cf.lock:
        cf.counter.value += 1
    return correlation_function_data


def main(cmdargs):
    # pylint: disable-msg=too-many-locals,too-many-branches,too-many-statements
    """Compute the 1D auto or cross-correlation between delta field from the same
    forest."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=('Compute the 1D auto or cross-correlation between delta '
                     'field from the same forest.'))

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

    parser.add_argument('--in-dir2',
                        type=str,
                        default=None,
                        required=False,
                        help='Directory to 2nd delta files')

    parser.add_argument('--lambda-min',
                        type=float,
                        default=3600.,
                        required=False,
                        help='Lower limit on observed wavelength [Angstrom]')

    parser.add_argument('--lambda-max',
                        type=float,
                        default=5500.,
                        required=False,
                        help='Upper limit on observed wavelength [Angstrom]')

    parser.add_argument('--dll',
                        type=float,
                        default=3.e-4,
                        required=False,
                        help='Loglam bin size')

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
              'of the 2nd delta (if not give, same as 1st delta)'))

    parser.add_argument('--z-ref',
                        type=float,
                        default=2.25,
                        required=False,
                        help='Reference redshift')

    parser.add_argument(
        '--z-evol',
        type=float,
        default=1.,
        required=False,
        help='Exponent of the redshift evolution of the delta field')

    parser.add_argument(
        '--z-evol2',
        type=float,
        default=1.,
        required=False,
        help='Exponent of the redshift evolution of the 2nd delta field')

    parser.add_argument('--no-project',
                        action='store_true',
                        required=False,
                        help='Do not project out continuum fitting modes')

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

    # setup variables in cf
    cf.nside = args.nside
    cf.log_lambda_min = np.log10(args.lambda_min)
    cf.log_lambda_max = np.log10(args.lambda_max)
    cf.delta_log_lambda = args.dll
    cf.num_pixels = int((cf.log_lambda_max - cf.log_lambda_min) /
                        cf.delta_log_lambda + 1)
    cf.x_correlation = False

    cf.lambda_abs = constants.ABSORBER_IGM[args.lambda_abs]
    if args.lambda_abs2:
        cf.lambda_abs2 = constants.ABSORBER_IGM[args.lambda_abs2]
    else:
        cf.lambda_abs2 = constants.ABSORBER_IGM[args.lambda_abs]

    ### Read data 1
    data, num_data, z_min, z_max = io.read_deltas(args.in_dir,
                                                  cf.nside,
                                                  cf.lambda_abs,
                                                  args.z_evol,
                                                  args.z_ref,
                                                  cosmo=None,
                                                  max_num_spec=args.nspec,
                                                  no_project=args.no_project)
    cf.data = data
    cf.num_data = num_data
    del z_min, z_max
    userprint("")
    userprint("done, npix = {}\n".format(len(data)))

    ### Read data 2
    if args.in_dir2:
        cf.x_correlation = True
        data2, num_data2, z_min2, z_max2 = io.read_deltas(
            args.in_dir2,
            cf.nside,
            cf.lambda_abs2,
            args.z_evol2,
            args.z_ref,
            cosmo=None,
            max_num_spec=args.nspec,
            no_project=args.no_project)
        cf.data2 = data2
        cf.num_data2 = num_data2
        del z_min2, z_max2
        userprint("")
        userprint("done, npix = {}\n".format(len(data2)))
    elif cf.lambda_abs != cf.lambda_abs2:
        cf.x_correlation = True
        data2, num_data2, z_min2, z_max2 = io.read_deltas(
            args.in_dir,
            cf.nside,
            cf.lambda_abs2,
            args.z_evol2,
            args.z_ref,
            cosmo=None,
            max_num_spec=args.nspec,
            no_project=args.no_project)
        cf.data2 = data2
        cf.num_data2 = num_data2
        del z_min2, z_max2

    # Convert lists to arrays
    cf.data = {key: np.array(value) for key, value in cf.data.items()}
    if cf.x_correlation:
        cf.data2 = {key: np.array(value) for key, value in cf.data2.items()}

    # Compute the correlation function, use pool to parallelize
    cf.counter = Value('i', 0)
    cf.lock = Lock()
    context = multiprocessing.get_context('fork')
    pool = context.Pool(processes=args.nproc)

    if cf.x_correlation:
        healpixs = sorted([
            key for key in list(cf.data.keys()) if key in list(cf.data2.keys())
        ])
    else:
        healpixs = sorted(list(cf.data.keys()))
    correlation_function_data = pool.map(corr_func, healpixs)
    pool.close()
    userprint('\n')

    # group data from parallelisation
    correlation_function_data = np.array(correlation_function_data)
    weights_list = correlation_function_data[:, 0, :]
    xi_list = correlation_function_data[:, 1, :]
    num_pairs_list = correlation_function_data[:, 2, :]
    weights_list = np.array(weights_list)
    xi_list = np.array(xi_list)
    num_pairs_list = np.array(num_pairs_list).astype(np.int64)

    userprint("multiplying")
    xi_list *= weights_list
    xi_list = xi_list.sum(axis=0)
    weights_list = weights_list.sum(axis=0)
    num_pairs_list = num_pairs_list.sum(axis=0)
    userprint("done")

    xi_list = xi_list.reshape(cf.num_pixels, cf.num_pixels)
    weights_list = weights_list.reshape(cf.num_pixels, cf.num_pixels)
    num_pairs_list = num_pairs_list.reshape(cf.num_pixels, cf.num_pixels)

    w = weights_list > 0
    xi_list[w] /= weights_list[w]

    ### Make copies of the 2D arrays that will be saved in the output file
    xi_list_2d = xi_list.copy()
    weights_list_2d = weights_list.copy()
    num_pairs_list_2d = num_pairs_list.copy()

    # collapse measured correlation into 1D arrays
    variance_1d = np.diag(xi_list).copy()
    weights_variance_1d = np.diag(weights_list).copy()
    num_pairs_variance_1d = np.diag(num_pairs_list).copy()
    xi = xi_list.copy()
    norm = np.sqrt(variance_1d * variance_1d[:, None])
    w = norm > 0
    xi[w] /= norm[w]

    userprint("rebinning")
    xi_1d = np.zeros(cf.num_pixels)
    weights_1d = np.zeros(cf.num_pixels)
    num_pairs1d = np.zeros(cf.num_pixels, dtype=np.int64)
    bins = np.arange(cf.num_pixels)

    dbin = bins - bins[:, None]
    w = dbin >= 0
    dbin = dbin[w]
    xi = xi[w]
    weights_list = weights_list[w]
    num_pairs_list = num_pairs_list[w]

    rebin = np.bincount(dbin, weights=xi * weights_list)
    xi_1d[:len(rebin)] = rebin
    rebin = np.bincount(dbin, weights=weights_list)
    weights_1d[:len(rebin)] = rebin
    rebin = np.bincount(dbin, weights=num_pairs_list)
    num_pairs1d[:len(rebin)] = rebin

    w = weights_1d > 0
    xi_1d[w] /= weights_1d[w]

    # Save results
    userprint("writing")

    results = fitsio.FITS(args.out, 'rw', clobber=True)
    header = [
        {
            'name': 'LLMIN',
            'value': cf.log_lambda_min,
            'comment': 'Minimum log10 lambda [log Angstrom]'
        },
        {
            'name': 'LLMAX',
            'value': cf.log_lambda_max,
            'comment': 'Maximum log10 lambda [log Angstrom]'
        },
        {
            'name': 'DLL',
            'value': cf.delta_log_lambda,
            'comment': 'Loglam bin size [log Angstrom]'
        },
    ]
    comment = [
        'Variance', 'Sum of weight for variance', 'Sum of pairs for variance',
        'Correlation', 'Sum of weight for correlation',
        'Sum of pairs for correlation'
    ]
    results.write([
        variance_1d, weights_variance_1d, num_pairs_variance_1d, xi_1d,
        weights_1d, num_pairs1d
    ],
                  names=['v1d', 'wv1d', 'nv1d', 'c1d', 'nc1d', 'nb1d'],
                  header=header,
                  comment=comment,
                  extname='1DCOR')

    comment = ['Covariance', 'Sum of weight', 'Number of pairs']
    results.write([xi_list_2d, weights_list_2d, num_pairs_list_2d],
                  names=['DA', 'WE', 'NB'],
                  comment=comment,
                  extname='2DCOR')
    results.close()

    userprint("all done")


if __name__ == '__main__':
    cmdargs=sys.argv[1:]
    main(cmdargs)
