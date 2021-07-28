#!/usr/bin/env python
"""Compute the auto and cross-correlation between catalogs of objects
"""
import argparse
import sys
import multiprocessing
from multiprocessing import Pool, Lock, cpu_count, Value
import numpy as np
import fitsio

from picca import constants, co, io, utils
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
    co.fill_neighs(healpixs)
    correlation_function_data = co.compute_xi(healpixs)
    return correlation_function_data


def main(cmdargs):
    """Compute the auto and cross-correlation between catalogs of objects"""

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=('Compute the auto and cross-correlation between catalogs '
                     'of objects'))

    parser.add_argument('--out',
                        type=str,
                        default=None,
                        required=True,
                        help='Output file name')

    parser.add_argument('--drq',
                        type=str,
                        default=None,
                        required=True,
                        help='Catalog of objects in DRQ format')

    parser.add_argument('--drq2',
                        type=str,
                        default=None,
                        required=False,
                        help='Catalog of objects 2 in DRQ format')
    parser.add_argument(
                        '--mode',
                        type=str,
                        default='sdss',
                        choices=['sdss','desi'],
                        required=False,
                        help='type of catalog supplied, default sdss')

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

    parser.add_argument(
        '--z-cut-min',
        type=float,
        default=0.,
        required=False,
        help=('Use only pairs of object x object with the mean redshift larger '
              'than z-cut-min'))

    parser.add_argument(
        '--z-cut-max',
        type=float,
        default=10.,
        required=False,
        help=('Use only pairs of object x object with the mean redshift lower '
              'than z-cut-max'))

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

    parser.add_argument('--z-ref',
                        type=float,
                        default=2.25,
                        required=False,
                        help='Reference redshift')

    parser.add_argument(
        '--z-evol-obj',
        type=float,
        default=1.,
        required=False,
        help='Exponent of the redshift evolution of the object field')

    parser.add_argument(
        '--z-evol-obj2',
        type=float,
        default=1.,
        required=False,
        help='Exponent of the redshift evolution of the object 2 field')

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

    parser.add_argument(
        '--type-corr',
        type=str,
        default='DD',
        required=False,
        help='type of correlation: DD, RR, DR, RD, xDD, xRR, xD1R2, xR1D2')

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

    args = parser.parse_args(cmdargs)

    if args.nproc is None:
        args.nproc = cpu_count() // 2

    # setup variables in module co
    co.r_par_max = args.rp_max
    co.r_par_min = args.rp_min
    co.r_trans_max = args.rt_max
    co.z_cut_min = args.z_cut_min
    co.z_cut_max = args.z_cut_max
    co.num_bins_r_par = args.np
    co.num_bins_r_trans = args.nt
    co.nside = args.nside
    co.type_corr = args.type_corr
    if co.type_corr not in [
            'DD', 'RR', 'DR', 'RD', 'xDD', 'xRR', 'xD1R2', 'xR1D2'
    ]:
        userprint(("ERROR: type-corr not in ['DD', 'RR', 'DR', 'RD', 'xDD', "
                   "'xRR', 'xD1R2', 'xR1D2']"))
        sys.exit()
    if args.drq2 is None:
        co.x_correlation = False
    else:
        co.x_correlation = True

    # load fiducial cosmology
    cosmo = constants.Cosmo(Om=args.fid_Om,
                            Or=args.fid_Or,
                            Ok=args.fid_Ok,
                            wl=args.fid_wl)

    ### Read objects 1
    objs, z_min = io.read_objects(args.drq, args.nside, args.z_min_obj,
                                  args.z_max_obj, args.z_evol_obj, args.z_ref,
                                  cosmo,mode=args.mode)
    userprint("")
    co.objs = objs
    co.num_data = len([obj for healpix in co.objs for obj in co.objs[healpix]])
    # compute maximum angular separation
    co.ang_max = utils.compute_ang_max(cosmo, co.r_trans_max, z_min)

    ### Read objects 2
    if co.x_correlation:
        objs2, z_min2 = io.read_objects(args.drq2, args.nside, args.z_min_obj,
                                        args.z_max_obj, args.z_evol_obj2,
                                        args.z_ref, cosmo,mode=args.mode)
        userprint("")
        co.objs2 = objs2
        # recompute maximum angular separation
        co.ang_max = utils.compute_ang_max(cosmo, co.r_trans_max, z_min, z_min2)

    # compute correlation function, use pool to parallelize
    co.counter = Value('i', 0)
    co.lock = Lock()
    cpu_data = {healpix: [healpix] for healpix in co.objs}
    context = multiprocessing.get_context('fork')
    pool = context.Pool(processes=args.nproc)
    correlation_function_data = pool.map(corr_func,
                                         sorted(list(cpu_data.values())))
    pool.close()

    # group data from parallelisation
    correlation_function_data = np.array(correlation_function_data)
    weights_list = correlation_function_data[:, 0, :]
    r_par_list = correlation_function_data[:, 1, :]
    r_trans_list = correlation_function_data[:, 2, :]
    z_list = correlation_function_data[:, 3, :]
    num_pairs_list = correlation_function_data[:, 4, :].astype(np.int64)
    healpix_list = np.array(sorted(list(cpu_data)))

    w = (weights_list.sum(axis=0) > 0.)
    r_par = (r_par_list * weights_list).sum(axis=0)
    r_par[w] /= weights_list.sum(axis=0)[w]
    r_trans = (r_trans_list * weights_list).sum(axis=0)
    r_trans[w] /= weights_list.sum(axis=0)[w]
    z = (z_list * weights_list).sum(axis=0)
    z[w] /= weights_list.sum(axis=0)[w]
    num_pairs = num_pairs_list.sum(axis=0)

    results = fitsio.FITS(args.out, 'rw', clobber=True)
    header = [
        {
            'name': 'RPMIN',
            'value': co.r_par_min,
            'comment': 'Minimum r-parallel [h^-1 Mpc]'
        },
        {
            'name': 'RPMAX',
            'value': co.r_par_max,
            'comment': 'Maximum r-parallel [h^-1 Mpc]'
        },
        {
            'name': 'RTMAX',
            'value': co.r_trans_max,
            'comment': 'Maximum r-transverse [h^-1 Mpc]'
        },
        {
            'name': 'NP',
            'value': co.num_bins_r_par,
            'comment': 'Number of bins in r-parallel'
        },
        {
            'name': 'NT',
            'value': co.num_bins_r_trans,
            'comment': 'Number of bins in r-transverse'
        },
        {
            'name': 'NSIDE',
            'value': co.nside,
            'comment': 'Healpix nside'
        },
        {
            'name': 'TYPECORR',
            'value': co.type_corr,
            'comment': 'Correlation type'
        },
        {
            'name':
                'NOBJ',
            'value':
                len([obj for healpix in co.objs for obj in co.objs[healpix]]),
            'comment':
                'Number of objects'
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
        }
        ]
    if co.x_correlation:
        header += [{
            'name':
                'NOBJ2',
            'value':
                len([
                    obj2 for healpix in co.objs2 for obj2 in co.objs2[healpix]
                ]),
            'comment':
                'Number of objects 2'
        }]

    comment = ['R-parallel', 'R-transverse', 'Redshift', 'Number of pairs']
    units = ['h^-1 Mpc', 'h^-1 Mpc', '', '']
    results.write([r_par, r_trans, z, num_pairs],
                  names=['RP', 'RT', 'Z', 'NB'],
                  header=header,
                  comment=comment,
                  units=units,
                  extname='ATTRI')

    comment = ['Healpix index', 'Sum of weight', 'Number of pairs']
    header2 = [{
        'name': 'HLPXSCHM',
        'value': 'RING',
        'comment': 'healpix scheme'
    }]
    results.write([healpix_list, weights_list, num_pairs_list],
                  names=['HEALPID', 'WE', 'NB'],
                  header=header2,
                  comment=comment,
                  extname='COR')
    results.close()

    userprint("\nFinished")


if __name__ == '__main__':
    cmdargs=sys.argv[1:]
    main(cmdargs)
