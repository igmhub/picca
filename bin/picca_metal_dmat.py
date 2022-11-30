#!/usr/bin/env python
"""Compute the auto and cross-correlation of delta fields for a list of IGM
absorption.

This module follow the procedure described in sections 4.3 of du Mas des
Bourboux et al. 2020 (In prep) to compute the distortion matrix
"""
import sys
import time
import argparse
import multiprocessing
from multiprocessing import Pool, Lock, cpu_count, Value
from functools import partial
import numpy as np
import fitsio

from picca import constants, cf, utils, io
from picca.utils import userprint


def calc_metal_dmat(abs_igm1, abs_igm2, healpixs):
    """Computes the metal distortion matrix.

    To optimize the computation, first compute a list of neighbours for each of
    the healpix. This is an auxiliar function to split the computational load
    using several CPUs.

    Args:
        abs_igm1: str
            Name of the absorption in picca.constants defining the
            redshift of the forest pixels
        abs_igm2: str
            Name of the second absorption in picca.constants defining the
            redshift of the forest pixels
        healpixs: array of ints
            List of healpix numbers

    Returns:
        The distortion matrix data
    """
    cf.fill_neighs(healpixs)
    np.random.seed(healpixs[0])
    dmat_data = cf.compute_metal_dmat(healpixs,
                                      abs_igm1=abs_igm1,
                                      abs_igm2=abs_igm2)
    return dmat_data


def main(cmdargs):
    # pylint: disable-msg=too-many-locals,too-many-branches,too-many-statements
    """Compute the auto and cross-correlation of delta fields for a list of IGM
    absorption."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=('Compute the auto and cross-correlation of delta fields '
                     'for a list of IGM absorption.'))

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
        '--coef-binning-model',
        type=int,
        default=1,
        required=False,
        help=('Coefficient multiplying np and nt to get finner binning for the '
              'model'))

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

    parser.add_argument(
        '--abs-igm',
        type=str,
        default=[],
        required=False,
        nargs='*',
        help=('List of names of metal absorption in picca.constants present in '
              'forest'))

    parser.add_argument(
        '--abs-igm2',
        type=str,
        default=[],
        required=False,
        nargs='*',
        help=('List of names of metal absorption in picca.constants present in '
              '2nd forest'))

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
        '--remove-same-half-plate-close-pairs',
        action='store_true',
        required=False,
        help='Reject pairs in the first bin in r-parallel from same half plate')

    parser.add_argument(
        '--rej',
        type=float,
        default=1.,
        required=False,
        help=('Fraction of rejected forest-forest pairs: -1=no rejection, '
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

    parser.add_argument(
        '--unfold-cf',
        action='store_true',
        required=False,
        help=('rp can be positive or negative depending on the relative '
              'position between absorber1 and absorber2'))

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

    # setup variables in module cf
    cf.r_par_max = args.rp_max
    cf.r_trans_max = args.rt_max
    cf.r_par_min = args.rp_min
    cf.z_cut_max = args.z_cut_max
    cf.z_cut_min = args.z_cut_min
    cf.num_bins_r_par = args.np * args.coef_binning_model
    cf.num_bins_r_trans = args.nt * args.coef_binning_model
    cf.num_model_bins_r_par = args.np * args.coef_binning_model
    cf.num_model_bins_r_trans = args.nt * args.coef_binning_model
    cf.nside = args.nside
    cf.z_ref = args.z_ref
    cf.alpha = args.z_evol
    cf.reject = args.rej
    cf.lambda_abs = constants.ABSORBER_IGM[args.lambda_abs]
    cf.remove_same_half_plate_close_pairs = args.remove_same_half_plate_close_pairs

    cf.alpha_abs = {}
    cf.alpha_abs[args.lambda_abs] = cf.alpha
    for metal in args.abs_igm:
        cf.alpha_abs[metal] = args.metal_alpha

    # read blinding keyword
    blinding = io.read_blinding(args.in_dir)

    # load fiducial cosmology
    cf.cosmo = constants.Cosmo(Om=args.fid_Om,
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
                                                  cf.cosmo,
                                                  max_num_spec=args.nspec,
                                                  nproc=args.nproc,
                                                  rebin_factor=args.rebin_factor)
    del z_max
    cf.data = data
    cf.num_data = num_data
    cf.ang_max = utils.compute_ang_max(cf.cosmo, cf.r_trans_max, z_min)
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
        cf.alpha_abs[args.lambda_abs2] = cf.alpha2
        for m in args.abs_igm2:
            cf.alpha_abs[m] = args.metal_alpha

        data2, num_data2, z_min2, z_max2 = io.read_deltas(
            args.in_dir2,
            cf.nside,
            cf.lambda_abs2,
            cf.alpha2,
            cf.z_ref,
            cf.cosmo,
            max_num_spec=args.nspec,
            nproc=args.nproc,
            rebin_factor=args.rebin_factor)
        del z_max2
        cf.data2 = data2
        cf.num_data2 = num_data2
        cf.ang_max = utils.compute_ang_max(cf.cosmo, cf.r_trans_max, z_min,
                                           z_min2)
        userprint("")
        userprint("done, npix = {}".format(len(data2)))

    t1 = time.time()
    userprint(f'picca_metal_dmat.py - Time reading data: {(t1-t0)/60:.3f} minutes')

    cf.counter = Value('i', 0)
    cf.lock = Lock()
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

    abs_igm = [args.lambda_abs] + args.abs_igm
    userprint("abs_igm = {}".format(abs_igm))

    if args.lambda_abs2 is None:
        args.lambda_abs2 = args.lambda_abs
        args.abs_igm2 = args.abs_igm

    abs_igm_2 = [args.lambda_abs2] + args.abs_igm2

    if cf.x_correlation:
        userprint("abs_igm2 = {}".format(abs_igm_2))

    # loop over metals
    for index1, abs_igm1 in enumerate(abs_igm):
        index0 = index1
        if args.lambda_abs != args.lambda_abs2:
            index0 = 0
        for index2, abs_igm2 in enumerate(abs_igm_2[index0:]):
            if index1 == 0 and index2 == 0:
                continue
            cf.counter.value = 0
            calc_metal_dmat_wrapper = partial(calc_metal_dmat, abs_igm1,
                                              abs_igm2)
            userprint("")

            # compute the distortion matrix
            if args.nproc > 1:
                context = multiprocessing.get_context('fork')
                pool = context.Pool(processes=args.nproc)
                dmat_data = pool.map(calc_metal_dmat_wrapper,
                                     sorted(cpu_data.values()))
                pool.close()
            elif args.nproc == 1:
                dmat_data = map(calc_metal_dmat_wrapper,
                                sorted(cpu_data.values()))
                dmat_data = list(dmat_data)

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

            # normalize_values
            w = weights > 0
            r_par[w] /= weights[w]
            r_trans[w] /= weights[w]
            z[w] /= weights[w]
            w = weights_dmat > 0
            dmat[w, :] /= weights_dmat[w, None]

            # add these results to the list ofor the different metal absorption
            dmat_all.append(dmat)
            weights_dmat_all.append(weights_dmat)
            r_par_all.append(r_par)
            r_trans_all.append(r_trans)
            z_all.append(z)
            names.append(abs_igm1 + "_" + abs_igm2)
            num_pairs_all.append(num_pairs)
            num_pairs_used_all.append(num_pairs_used)

    t2 = time.time()
    userprint(f'picca_metal_dmat.py - Time computing all metal matrices : {(t2-t1)/60:.3f} minutes')

    # save the results
    results = fitsio.FITS(args.out, 'rw', clobber=True)
    header = [
        {
            'name': 'RPMIN',
            'value': cf.r_par_min,
            'comment': 'Minimum r-parallel [h^-1 Mpc]'
        },
        {
            'name': 'RPMAX',
            'value': cf.r_par_max,
            'comment': 'Maximum r-parallel [h^-1 Mpc]'
        },
        {
            'name': 'RTMAX',
            'value': cf.r_trans_max,
            'comment': 'Maximum r-transverse [h^-1 Mpc]'
        },
        {
            'name': 'NP',
            'value': cf.num_bins_r_par,
            'comment': 'Number of bins in r-parallel'
        },
        {
            'name': 'NT',
            'value': cf.num_bins_r_trans,
            'comment': ' Number of bins in r-transverse'
        },
        {
            'name': 'COEFMOD',
            'value': args.coef_binning_model,
            'comment': 'Coefficient for model binning'
        },
        {
            'name': 'ZCUTMIN',
            'value': cf.z_cut_min,
            'comment': 'Minimum redshift of pairs'
        },
        {
            'name': 'ZCUTMAX',
            'value': cf.z_cut_max,
            'comment': 'Maximum redshift of pairs'
        },
        {
            'name': 'REJ',
            'value': cf.reject,
            'comment': 'Rejection factor'
        },
        {
            'name': 'ALPHAMET',
            'value': args.metal_alpha,
            'comment': 'Evolution of metal bias'
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
        out_names += ['RP_' + name]
        out_list += [r_par_all[index]]
        out_comment += ['R-parallel']
        out_units += ['h^-1 Mpc']

        out_names += ['RT_' + name]
        out_list += [r_trans_all[index]]
        out_comment += ['R-transverse']
        out_units += ['h^-1 Mpc']

        out_names += ['Z_' + name]
        out_list += [z_all[index]]
        out_comment += ['Redshift']
        out_units += ['']

        out_names += [dmat_name + name]
        out_list += [dmat_all[index]]
        out_comment += ['Distortion matrix']
        out_units += ['']

        out_names += ['WDM_' + name]
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
    userprint(f'picca_metal_dmat.py - Time total : {(t3-t0)/60:.3f} minutes')

if __name__ == '__main__':
    cmdargs=sys.argv[1:]
    main(cmdargs)
