#!/usr/bin/env python
"""Compute the wick covariance for the cross-correlation of object x forests.

The wick covariance is computed as explained in Delubac et al. 2015
"""
import sys
import argparse
import multiprocessing
from multiprocessing import Pool, Lock, cpu_count, Value
import fitsio
import numpy as np
from scipy.interpolate import interp1d

from picca import constants, io, utils, xcf, cf
from picca.utils import userprint


def calc_wick_terms(healpixs):
    """Computes the wick expansion terms of the covariance matrix.

    To optimize the computation, first compute a list of neighbours for each of
    the healpix. This is an auxiliar function to split the computational load
    using several CPUs.

    Args:
        healpixs: array of ints
            List of healpix numbers

    Returns:
        The results of the Wick computation
    """
    np.random.seed(healpixs[0])
    wick_data = xcf.compute_wick_terms(healpixs)
    return wick_data


def main(cmdargs):
    # pylint: disable-msg=too-many-locals,too-many-branches,too-many-statements
    """Computes the wick covariance for the cross-correlation of object x
    forests."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=('Compute the wick covariance for the cross-correlation of '
                     'object x forests.'))

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
                        help='Catalog of objects in DRQ format')

    parser.add_argument(
                        '--mode',
                        type=str,
                        default='sdss',
                        choices=['sdss','desi'],
                        required=False,
                        help='type of catalog supplied, default sdss')

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

    parser.add_argument('--max-diagram',
                        type=int,
                        default=4,
                        required=False,
                        help='Maximum diagram to compute')

    parser.add_argument(
        '--cf1d',
        type=str,
        required=True,
        help=('1D auto-correlation of pixels from the same forest file: '
              'picca_cf1d.py'))

    parser.add_argument(
        '--cf',
        type=str,
        default=None,
        required=False,
        help=('3D auto-correlation of pixels from different forests: '
              'picca_cf.py'))

    parser.add_argument(
        '--rej',
        type=float,
        default=1.,
        required=False,
        help=('Fraction of rejected object-forests pairs: -1=no rejection, '
              '1=all rejection'))

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
    xcf.r_par_min = args.rp_min
    xcf.r_par_max = args.rp_max
    xcf.r_trans_max = args.rt_max
    xcf.z_cut_min = args.z_cut_min
    xcf.z_cut_max = args.z_cut_max
    xcf.num_bins_r_par = args.np
    xcf.num_bins_r_trans = args.nt
    xcf.nside = args.nside
    xcf.reject = args.rej
    xcf.z_ref = args.z_ref
    xcf.alpha = args.z_evol_del
    xcf.alpha_obj = args.z_evol_obj
    xcf.lambda_abs = constants.ABSORBER_IGM[args.lambda_abs]
    xcf.max_diagram = args.max_diagram

    # read blinding keyword
    blinding = io.read_blinding(args.in_dir)

    # load fiducial cosmology
    if (args.fid_Or != 0.) or (args.fid_Ok != 0.) or (args.fid_wl != -1.):
        userprint(("ERROR: Cosmology with other than Omega_m set are not yet "
                   "implemented"))
        sys.exit()
    cosmo = constants.Cosmo(Om=args.fid_Om,
                            Or=args.fid_Or,
                            Ok=args.fid_Ok,
                            wl=args.fid_wl,
                            blinding=blinding)

    ### Read deltas
    data, num_data, z_min, z_max = io.read_deltas(args.in_dir,
                                                  args.nside,
                                                  xcf.lambda_abs,
                                                  args.z_evol_del,
                                                  args.z_ref,
                                                  cosmo=cosmo,
                                                  max_num_spec=args.nspec,
                                                  nproc=args.nproc,
                                                  rebin_factor=args.rebin_factor)
    for deltas in data.values():
        for delta in deltas:
            delta.fname = 'D1'
            for item in [
                    'cont', 'delta', 'order', 'ivar', 'exposures_diff',
                    'mean_snr', 'mean_reso', 'mean_z', 'delta_log_lambda'
            ]:
                setattr(delta, item, None)
    xcf.data = data
    xcf.num_data = num_data
    sys.stderr.write("\n")
    userprint("done, npix = {}, ndels = {}".format(len(data), xcf.num_data))
    sys.stderr.write("\n")

    ### Find the redshift range
    if args.z_min_obj is None:
        r_comov_min = cosmo.get_r_comov(z_min)
        r_comov_min = max(0., r_comov_min + xcf.r_par_min)
        args.z_min_obj = cosmo.distance_to_redshift(r_comov_min)
        sys.stderr.write("\r z_min_obj = {}\r".format(args.z_min_obj))
    if args.z_max_obj is None:
        r_comov_max = cosmo.get_r_comov(z_max)
        r_comov_max = max(0., r_comov_max + xcf.r_par_max)
        args.z_max_obj = cosmo.distance_to_redshift(r_comov_max)
        sys.stderr.write("\r z_max_obj = {}\r".format(args.z_max_obj))

    ### Read objects
    objs, z_min2 = io.read_objects(args.drq, args.nside, args.z_min_obj,
                                   args.z_max_obj, args.z_evol_obj, args.z_ref,
                                   cosmo, mode=args.mode)
    xcf.objs = objs
    sys.stderr.write("\n")
    userprint("done, npix = {}".format(len(objs)))
    sys.stderr.write("\n")

    # compute maximum angular separation
    xcf.ang_max = utils.compute_ang_max(cosmo, xcf.r_trans_max, z_min, z_min2)

    # Load 1d correlation functions
    hdul = fitsio.FITS(args.cf1d)
    header = hdul[1].read_header()
    log_lambda_min = header['LLMIN']
    delta_log_lambda = header['DLL']
    num_pairs_variance_1d = hdul[1]['nv1d'][:]
    variance_1d = hdul[1]['v1d'][:]
    log_lambda = log_lambda_min + delta_log_lambda * np.arange(variance_1d.size)
    xcf.get_variance_1d['D1'] = interp1d(log_lambda[num_pairs_variance_1d > 0],
                                         variance_1d[num_pairs_variance_1d > 0],
                                         kind='nearest',
                                         fill_value='extrapolate')

    num_pairs1d = hdul[1]['nb1d'][:]
    xi_1d = hdul[1]['c1d'][:]
    xcf.xi_1d['D1'] = interp1d((log_lambda - log_lambda_min)[num_pairs1d > 0],
                               xi_1d[num_pairs1d > 0],
                               kind='nearest',
                               fill_value='extrapolate')
    hdul.close()

    # Load correlation functions
    if not args.cf is None:
        hdul = fitsio.FITS(args.cf)
        header = hdul[1].read_header()
        assert cf.num_bins_r_par == header['NP']
        assert cf.num_bins_r_trans == header['NT']
        assert cf.r_par_min == header['RPMIN']
        assert cf.r_par_max == header['RPMAX']
        assert cf.r_trans_max == header['RTMAX']
        xi = hdul[2]['DA'][:]
        weights = hdul[2]['WE'][:]
        xi = (xi * weights).sum(axis=0)
        weights = weights.sum(axis=0)
        w = weights > 0.
        xi[w] /= weights[w]
        xcf.xi_wick = xi.copy()
        hdul.close()

        cf.data = xcf.data
        cf.ang_max = xcf.ang_max
        cf.nside = xcf.nside
        cf.z_cut_max = xcf.z_cut_max
        cf.z_cut_min = xcf.z_cut_min

    ### Send
    xcf.counter = Value('i', 0)
    xcf.lock = Lock()

    cpu_data = {}
    for index, healpix in enumerate(sorted(data)):
        num_processor = index % args.nproc
        if not num_processor in cpu_data:
            cpu_data[num_processor] = []
        cpu_data[num_processor].append(healpix)

    # Find neighbours
    for healpixs in cpu_data.values():
        xcf.fill_neighs(healpixs)
        if not xcf.xi_wick is None:
            cf.fill_neighs(healpixs)

    # compute the covariance matrix
    context = multiprocessing.get_context('fork')
    pool = context.Pool(processes=min(args.nproc, len(cpu_data.values())))
    userprint(" \nStarting\n")
    wick_data = pool.map(calc_wick_terms, sorted(cpu_data.values()))
    userprint(" \nFinished\n")
    pool.close()

    wick_data = np.array(wick_data)
    weights_wick = wick_data[:, 0].sum(axis=0)
    num_pairs_wick = wick_data[:, 1].sum(axis=0)
    npairs = wick_data[:, 2].sum(axis=0)
    npairs_used = wick_data[:, 3].sum(axis=0)
    t1 = wick_data[:, 4].sum(axis=0)
    t2 = wick_data[:, 5].sum(axis=0)
    t3 = wick_data[:, 6].sum(axis=0)
    t4 = wick_data[:, 7].sum(axis=0)
    t5 = wick_data[:, 8].sum(axis=0)
    t6 = wick_data[:, 9].sum(axis=0)
    weights = weights_wick * weights_wick[:, None]
    w = weights > 0.
    t1[w] /= weights[w]
    t2[w] /= weights[w]
    t3[w] /= weights[w]
    t4[w] /= weights[w]
    t5[w] /= weights[w]
    t6[w] /= weights[w]
    t1 *= 1. * npairs_used / npairs
    t2 *= 1. * npairs_used / npairs
    t3 *= 1. * npairs_used / npairs
    t4 *= 1. * npairs_used / npairs
    t5 *= 1. * npairs_used / npairs
    t6 *= 1. * npairs_used / npairs
    t_tot = t1 + t2 + t3 + t4 + t5 + t6

    results = fitsio.FITS(args.out, 'rw', clobber=True)
    header = [
        {
            'name': 'RPMIN',
            'value': xcf.r_par_min,
            'comment': 'Minimum r-parallel [h^-1 Mpc]'
        },
        {
            'name': 'RPMAX',
            'value': xcf.r_par_max,
            'comment': 'Maximum r-parallel [h^-1 Mpc]'
        },
        {
            'name': 'RTMAX',
            'value': xcf.r_trans_max,
            'comment': 'Maximum r-transverse [h^-1 Mpc]'
        },
        {
            'name': 'NP',
            'value': xcf.num_bins_r_par,
            'comment': 'Number of bins in r-parallel'
        },
        {
            'name': 'NT',
            'value': xcf.num_bins_r_trans,
            'comment': 'Number of bins in r-transverse'
        },
        {
            'name': 'ZCUTMIN',
            'value': xcf.z_cut_min,
            'comment': 'Minimum redshift of pairs'
        },
        {
            'name': 'ZCUTMAX',
            'value': xcf.z_cut_max,
            'comment': 'Maximum redshift of pairs'
        },
        {
            'name': 'REJ',
            'value': xcf.reject,
            'comment': 'Rejection factor'
        },
        {
            'name': 'NPALL',
            'value': npairs,
            'comment': 'Number of pairs'
        },
        {
            'name': 'NPUSED',
            'value': npairs_used,
            'comment': 'Number of used pairs'
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
    comment = [
        'Sum of weight', 'Covariance', 'Nomber of pairs', 'T1', 'T2', 'T3',
        'T4', 'T5', 'T6'
    ]
    results.write(
        [t_tot, weights_wick, num_pairs_wick, t1, t2, t3, t4, t5, t6],
        names=['CO', 'WALL', 'NB', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6'],
        comment=comment,
        header=header,
        extname='COV')
    results.close()


if __name__ == '__main__':
    cmdargs=sys.argv[1:]
    main(cmdargs)
