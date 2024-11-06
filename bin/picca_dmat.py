#!/usr/bin/env python
"""Computes the distortion matrix between two delta fields.

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

from picca import constants, cf, utils, io
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
    cf.fill_neighs(healpixs)
    np.random.seed(healpixs[0])
    dmat_data = cf.compute_dmat(healpixs)
    return dmat_data


def main(cmdargs):
    # pylint: disable-msg=too-many-locals,too-many-branches,too-many-statements
    """Computes the distortion matrix"""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=('Compute the distortion matrix of the auto and '
                     'cross-correlation of delta fields'))

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
        '--z-min-sources',
        type=float,
        default=0.,
        required=False,
        help=('Limit the minimum redshift of the quasars '
                'used as sources for spectra'))

    parser.add_argument(
        '--z-max-sources',
        type=float,
        default=10.,
        required=False,
        help=('Limit the maximum redshift of the quasars '
                'used as sources for spectra'))

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

    # remove this option because it has no effect on the output
    # and hence can lead to confusions
    # parser.add_argument('--z-ref',
    #                    type=float,
    #                    default=2.25,
    #                    required=False,
    #                    help='Reference redshift')

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

    parser.add_argument('--no-redshift-evolution',
                        action='store_true',
                        help='Ignore redshift evolution when computing distortion matrix')

    args = parser.parse_args(cmdargs)

    if args.nproc is None:
        args.nproc = cpu_count() // 2

    userprint("nproc", args.nproc)

    # setup variables in module cf
    cf.r_par_max = args.rp_max
    cf.r_par_min = args.rp_min
    cf.r_trans_max = args.rt_max
    cf.z_cut_max = args.z_cut_max
    cf.z_cut_min = args.z_cut_min
    cf.num_bins_r_par = args.np
    cf.num_bins_r_trans = args.nt
    cf.num_model_bins_r_par = args.np * args.coef_binning_model
    cf.num_model_bins_r_trans = args.nt * args.coef_binning_model
    cf.nside = args.nside

    if args.no_redshift_evolution :
        cf.redshift_evolution_in_distortion_matrix = False
        userprint("ignore redshift evolution in the distortion matrix")

    # this value has no effect because it scales the weights that are both at the numerator and denominator of the estimator
    # it is also used as a TEMPORARY VARIABLE to compute the distortion matrix scaling
    # but this is rescaled to the actual effective redshift of the data (zeff) in the following
    cf.z_ref = 2.25

    cf.alpha  = args.z_evol
    # by default, for autocorrelations, we are setting cf.alpha2 to the same value
    # it is overwritten by the value of args.z_evol2 if a second set of delta is given in input
    cf.alpha2 = args.z_evol

    cf.reject = args.rej
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
                                                  nproc=args.nproc,
                                                  rebin_factor=args.rebin_factor,
                                                  z_min_qso=args.z_min_sources,
                                                  z_max_qso=args.z_max_sources)
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
            nproc=args.nproc,
            rebin_factor=args.rebin_factor,
            z_min_qso=args.z_min_sources,
            z_max_qso=args.z_max_sources)
        del z_max2
        cf.data2 = data2
        cf.num_data2 = num_data2
        cf.ang_max = utils.compute_ang_max(cosmo, cf.r_trans_max, z_min, z_min2)
        userprint("")
        userprint("done, npix = {}".format(len(data2)))

    t1 = time.time()
    userprint(f'picca_dmat.py - Time reading data: {(t1-t0)/60:.3f} minutes')

    cf.counter = Value('i', 0)
    cf.lock = Lock()
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

    t2 = time.time()
    userprint(f'picca_dmat.py - Time computing distortion matrix: {(t2-t1)/60:.3f} minutes')


    # merge the results from different CPUs
    dmat_data = list(dmat_data)
    weights_dmat = np.array([item[0] for item in dmat_data]).sum(axis=0)
    dmat = np.array([item[1] for item in dmat_data]).sum(axis=0)
    r_par = np.array([item[2] for item in dmat_data]).sum(axis=0)
    r_trans = np.array([item[3] for item in dmat_data]).sum(axis=0)
    zeff = np.array([item[4] for item in dmat_data]).sum(axis=0)
    weights = np.array([item[5] for item in dmat_data]).sum(axis=0)
    num_pairs = np.array([item[6] for item in dmat_data]).sum(axis=0)
    num_pairs_used = np.array([item[7] for item in dmat_data]).sum(axis=0)

    # normalize values
    w = weights > 0.
    r_par[w] /= weights[w]
    r_trans[w] /= weights[w]
    zeff[w] /= weights[w]
    mean_zeff = np.mean(zeff[w])

    w = weights_dmat > 0
    dmat[w] /= weights_dmat[w, None]

    if cf.redshift_evolution_in_distortion_matrix :
        # now that we have the effective redshift of the input model considered
        # for the distortion matrix, we do rescale the whole matrix
        # we first consider the same effective redshift for all the model bins
        zeff[:]   = mean_zeff
        zfac = ((1+cf.z_ref)/(1+mean_zeff))**((cf.alpha-1)+(cf.alpha2-1))
        dmat *= zfac

    # save results
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
            'comment': 'Number of bins in r-transverse'
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
    results.write([r_par, r_trans, zeff],
                  names=['RP', 'RT', 'Z'],
                  comment=['R-parallel', 'R-transverse', 'Redshift'],
                  units=['h^-1 Mpc', 'h^-1 Mpc', ''],
                  extname='ATTRI')
    results.close()

    t3 = time.time()
    userprint(f'picca_dmat.py - Time total : {(t3-t0)/60:.3f} minutes')


if __name__ == '__main__':
    cmdargs=sys.argv[1:]
    main(cmdargs)
