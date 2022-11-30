#!/usr/bin/env python
"""Compute the distortion matrix of the cross-correlation delta x object for a
list of IGM absorption.

This module follow the procedure described in sections 4.3 of du Mas des
Bourboux et al. 2020 (In prep) to compute the distortion matrix
"""
import sys
import time
import argparse
from functools import partial
import multiprocessing
from multiprocessing import Pool, Lock, cpu_count, Value
import numpy as np
import fitsio

from picca import constants, xcf, io, utils
from picca.utils import userprint


def calc_metal_dmat(abs_igm, healpixs):
    """Computes the metal distortion matrix.

    To optimize the computation, first compute a list of neighbours for each of
    the healpix. This is an auxiliar function to split the computational load
    using several CPUs.

    Args:
        abs_igm: string
            Name of the absorption in picca.constants defining the
            redshift of the forest pixels
        healpixs: array of ints
            List of healpix numbers

    Returns:
        The distortion matrix data
    """
    xcf.fill_neighs(healpixs)
    np.random.seed(healpixs[0])
    dmat_data = xcf.compute_metal_dmat(healpixs, abs_igm=abs_igm)
    return dmat_data


def main(cmdargs):
    """Compute the distortion matrix of the cross-correlation delta x object for
     a list of IGM absorption."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=('Compute the distortion matrix of the cross-correlation '
                     'delta x object for a list of IGM absorption.'))

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

    parser.add_argument('--obj-name',
                        type=str,
                        default='QSO',
                        required=False,
                        help='Name of the object tracer')

    parser.add_argument(
        '--abs-igm',
        type=str,
        default=None,
        required=False,
        nargs='*',
        help='List of names of metal absorption in picca.constants')

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
        '--metal-alpha',
        type=float,
        default=1.,
        required=False,
        help='Exponent of the redshift evolution of the metal delta field')

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
    xcf.r_par_max = args.rp_max
    xcf.r_par_min = args.rp_min
    xcf.r_trans_max = args.rt_max
    xcf.z_cut_max = args.z_cut_max
    xcf.z_cut_min = args.z_cut_min
    xcf.num_bins_r_par = args.np * args.coef_binning_model
    xcf.num_bins_r_trans = args.nt * args.coef_binning_model
    xcf.num_model_bins_r_par = args.np * args.coef_binning_model
    xcf.num_model_bins_r_trans = args.nt * args.coef_binning_model
    xcf.nside = args.nside
    xcf.z_ref = args.z_ref
    xcf.lambda_abs = constants.ABSORBER_IGM[args.lambda_abs]
    xcf.reject = args.rej

    xcf.alpha_abs = {}
    xcf.alpha_abs[args.lambda_abs] = args.z_evol_del
    for metal in args.abs_igm:
        xcf.alpha_abs[metal] = args.metal_alpha

    # read blinding keyword
    blinding = io.read_blinding(args.in_dir)

    # load fiducial cosmology
    cosmo = constants.Cosmo(Om=args.fid_Om,
                            Or=args.fid_Or,
                            Ok=args.fid_Ok,
                            wl=args.fid_wl,
                            blinding=blinding)
    xcf.cosmo = cosmo

    t0 = time.time()

    # read deltas
    data, num_data, z_min, z_max = io.read_deltas(args.in_dir,
                                                  args.nside,
                                                  xcf.lambda_abs,
                                                  args.z_evol_del,
                                                  args.z_ref,
                                                  cosmo,
                                                  max_num_spec=args.nspec,
                                                  nproc=args.nproc,
                                                  rebin_factor=args.rebin_factor)

    xcf.data = data
    xcf.num_data = num_data

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

    # read objets
    objs, z_min2 = io.read_objects(args.drq, args.nside, args.z_min_obj,
                                   args.z_max_obj, args.z_evol_obj, args.z_ref,
                                   cosmo, mode=args.mode)
    xcf.objs = objs

    # compute maximum angular separation
    xcf.ang_max = utils.compute_ang_max(cosmo, xcf.r_trans_max, z_min, z_min2)

    t1 = time.time()
    userprint(f'picca_metal_xdmat.py - Time reading data: {(t1-t0)/60:.3f} minutes')

    xcf.counter = Value('i', 0)
    xcf.lock = Lock()
    cpu_data = {}
    for index, healpix in enumerate(sorted(list(data.keys()))):
        num_processor = index % args.nproc
        if not num_processor in cpu_data:
            cpu_data[num_processor] = []
        cpu_data[num_processor].append(healpix)

    # intiialize arrays to store the results for the different metal absorption
    dmat_all = []
    weights_dmat_all = []
    r_par_all = []
    r_trans_all = []
    z_all = []
    names = []
    num_pairs_all = []
    num_pairs_used_all = []

    # loop over metals
    for index, abs_igm in enumerate(args.abs_igm):
        xcf.counter.value = 0
        calc_metal_dmat_wrapper = partial(calc_metal_dmat, abs_igm)
        userprint("")

        if args.nproc > 1:
            context = multiprocessing.get_context('fork')
            pool = context.Pool(processes=args.nproc)
            dmat_data = pool.map(calc_metal_dmat_wrapper,
                                 sorted(cpu_data.values()))
            pool.close()
        elif args.nproc == 1:
            dmat_data = map(calc_metal_dmat_wrapper, sorted(cpu_data.values()))
            dmat_data = list(dmat_data)

        # merge the results from different CPUs
        dmat_data = np.array(dmat_data)
        weights_dmat = dmat_data[:, 0].sum(axis=0)
        dmat = dmat_data[:, 1].sum(axis=0)
        r_par = dmat_data[:, 2].sum(axis=0)
        r_trans = dmat_data[:, 3].sum(axis=0)
        z = dmat_data[:, 4].sum(axis=0)
        weights = dmat_data[:, 5].sum(axis=0)

        # normalize_values
        w = weights > 0
        r_par[w] /= weights[w]
        r_trans[w] /= weights[w]
        z[w] /= weights[w]
        num_pairs = dmat_data[:, 6].sum(axis=0)
        num_pairs_used = dmat_data[:, 7].sum(axis=0)
        w = weights_dmat > 0
        dmat[w, :] /= weights_dmat[w, None]

        # add these results to the list ofor the different metal absorption
        dmat_all.append(dmat)
        weights_dmat_all.append(weights_dmat)
        r_par_all.append(r_par)
        r_trans_all.append(r_trans)
        z_all.append(z)
        names.append(abs_igm)
        num_pairs_all.append(num_pairs)
        num_pairs_used_all.append(num_pairs_used)

    t2 = time.time()
    userprint(f'picca_metal_xdmat.py - Time computing all metal matrix: {(t2-t1)/60:.3f} minutes')

    # save the results
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
    len_names = np.array([len(name) for name in names]).max()
    names = np.array(names, dtype='S' + str(len_names))
    results.write(
        [
            np.array(num_pairs_all),
            np.array(num_pairs_used_all),
            np.array(names)
        ],
        names=['NPALL', 'NPUSED', 'ABS_IGM'],
        header=header,
        comment=['Number of pairs', 'Number of used pairs', 'Absorption name'],
        extname='ATTRI')

    dmat_name = "DM_"
    if blinding != "none":
        dmat_name += "BLIND_"
    names = names.astype(str)
    out_list = []
    out_names = []
    out_comment = []
    out_units = []
    for index, name in enumerate(names):
        out_names += ['RP_' + args.obj_name + '_' + name]
        out_list += [r_par_all[index]]
        out_comment += ['R-parallel']
        out_units += ['h^-1 Mpc']

        out_names += ['RT_' + args.obj_name + '_' + name]
        out_list += [r_trans_all[index]]
        out_comment += ['R-transverse']
        out_units += ['h^-1 Mpc']

        out_names += ['Z_' + args.obj_name + '_' + name]
        out_list += [z_all[index]]
        out_comment += ['Redshift']
        out_units += ['']

        out_names += [dmat_name + args.obj_name + '_' + name]
        out_list += [dmat_all[index]]
        out_comment += ['Distortion matrix']
        out_units += ['']

        out_names += ['WDM_' + args.obj_name + '_' + name]
        out_list += [weights_dmat_all[index]]
        out_comment += ['Sum of weight']
        out_units += ['']

    results.write(out_list,
                  names=out_names,
                  comment=out_comment,
                  units=out_units,
                  extname='MDMAT')
    results.close()

    t3 = time.time()
    userprint(f'picca_metal_xdmat.py - Time total: {(t3-t0)/60:.3f} minutes')


if __name__ == '__main__':
    cmdargs=sys.argv[1:]
    main(cmdargs)
