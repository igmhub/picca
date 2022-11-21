#!/usr/bin/env python
"""Computes the distortion matrix between of the cross-correlation delta x
object

This module follow the procedure described in sections 3.5 of du Mas des
Bourboux et al. 2020 (In prep) to compute the distortion matrix
"""
import sys
import time
import argparse
import multiprocessing
from multiprocessing import Pool, Lock, cpu_count, Value
import numpy as np
import fitsio

from picca import constants, xcf, io, utils
from picca.utils import userprint


def calc_dmat(healpixs):
    """Computes the distortion matrix.

    To optimize the computation, first compute a list of neighbours for each of
    the healpix. This is an auxiliar function to split the computational load
    using several CPUs.

    Args:
        healpixs: array of ints
            List of healpix numbers

    Returns:
        The distortion matrix data
    """
    xcf.fill_neighs(healpixs)
    np.random.seed(healpixs[0])
    dmat_data = xcf.compute_dmat(healpixs)
    return dmat_data


def main(cmdargs):
    """Computes the distortion matrix of the cross-correlation delta x
    object."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=('Compute the distortion matrix of the cross-correlation '
                     'delta x object.'))

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
                        choices=['sdss','desi','desi_mocks','desi_healpix'],
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

    parser.add_argument(
        '--coef-binning-model',
        type=int,
        default=1,
        required=False,
        help=('Coefficient multiplying np and nt to get finner binning for the '
              'model'))

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

    parser.add_argument(
        '--rej',
        type=float,
        default=1.,
        required=False,
        help='Fraction of rejected pairs: -1=no rejection, 1=all rejection')

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

    userprint("nproc", args.nproc)

    # setup variables in module xcf
    xcf.r_par_max = args.rp_max
    xcf.r_par_min = args.rp_min
    xcf.r_trans_max = args.rt_max
    xcf.z_cut_max = args.z_cut_max
    xcf.z_cut_min = args.z_cut_min
    xcf.num_bins_r_par = args.np
    xcf.num_bins_r_trans = args.nt
    xcf.num_model_bins_r_par = args.np * args.coef_binning_model
    xcf.num_model_bins_r_trans = args.nt * args.coef_binning_model
    xcf.nside = args.nside
    xcf.z_ref = args.z_ref
    xcf.alpha = args.z_evol_del
    xcf.lambda_abs = constants.ABSORBER_IGM[args.lambda_abs]
    xcf.reject = args.rej

    # read blinding keyword
    blinding = io.read_blinding(args.in_dir)

    # load fiducial cosmology
    cosmo = constants.Cosmo(Om=args.fid_Om,
                            Or=args.fid_Or,
                            Ok=args.fid_Ok,
                            wl=args.fid_wl,
                            blinding=blinding)

    t0 = time.time()

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
    xcf.data = data
    xcf.num_data = num_data
    userprint("\n")
    userprint("done, npix = {}\n".format(len(data)))

    ### Find the redshift range
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
    userprint("\n")
    xcf.objs = objs

    # compute maximum angular separation
    xcf.ang_max = utils.compute_ang_max(cosmo, xcf.r_trans_max, z_min, z_min2)

    t1 = time.time()
    userprint(f'picca_xdmat.py - Time reading data: {(t1-t0)/60:.3f} minutes')

    xcf.counter = Value('i', 0)
    xcf.lock = Lock()
    cpu_data = {}
    for index, healpix in enumerate(sorted(data)):
        num_processor = index % args.nproc
        if num_processor not in cpu_data:
            cpu_data[num_processor] = []
        cpu_data[num_processor].append(healpix)

    # compute the distortion matrix
    if args.nproc > 1:
        context = multiprocessing.get_context('fork')
        pool = context.Pool(processes=args.nproc)
        dmat_data = pool.map(calc_dmat, sorted(cpu_data.values()))
        pool.close()
    elif args.nproc == 1:
        dmat_data = map(calc_dmat, sorted(cpu_data.values()))
        dmat_data = list(dmat_data)

    t2 = time.time()
    userprint(f'picca_xdmat.py - Time computing distortion matrix: {(t2-t1)/60:.3f} minutes')

    # merge the results from different CPUs
    dmat_data = np.array(dmat_data)
    weights_dmat = dmat_data[:, 0].sum(axis=0)
    dmat = dmat_data[:, 1].sum(axis=0)
    r_par = dmat_data[:, 2].sum(axis=0)
    r_trans = dmat_data[:, 3].sum(axis=0)
    z = dmat_data[:, 4].sum(axis=0)
    weights = dmat_data[:, 5].sum(axis=0)
    num_pairs = dmat_data[:, 6].sum(axis=0)
    num_pairs_used = dmat_data[:, 7].sum(axis=0)

    # normalize values
    w = weights > 0.
    r_par[w] /= weights[w]
    r_trans[w] /= weights[w]
    z[w] /= weights[w]
    w = weights_dmat > 0.
    dmat[w, :] /= weights_dmat[w, None]

    # save results
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
            'name': 'COEFMOD',
            'value': args.coef_binning_model,
            'comment': 'Coefficient for model binning'
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
            'value': num_pairs,
            'comment': 'Number of pairs'
        },
        {
            'name': 'NPUSED',
            'value': num_pairs_used,
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
    dmat_name = "DM"
    if blinding != "none":
        dmat_name += "_BLIND"
    results.write([weights_dmat, dmat],
                  names=['WDM', dmat_name],
                  comment=['Sum of weight', 'Distortion matrix'],
                  units=['', ''],
                  header=header,
                  extname='DMAT')
    results.write([r_par, r_trans, z],
                  names=['RP', 'RT', 'Z'],
                  comment=['R-parallel', 'R-transverse', 'Redshift'],
                  units=['h^-1 Mpc', 'h^-1 Mpc', ''],
                  extname='ATTRI')
    results.close()

    t3 = time.time()
    userprint(f'picca_xdmat.py - Time total: {(t3-t0)/60:.3f} minutes')

if __name__ == '__main__':
    cmdargs=sys.argv[1:]
    main(cmdargs)
