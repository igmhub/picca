#!/usr/bin/env python
"""Compute the wick covariance for the auto-correlation of forests

The wick covariance is computed as explained in Delubac et al. 2015
"""
import sys
import argparse
import multiprocessing
from multiprocessing import Pool, Lock, cpu_count, Value
import fitsio
import numpy as np
from scipy.interpolate import interp1d

from picca import constants, cf, utils, io
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
    cf.fill_neighs(healpixs)
    np.random.seed(healpixs[0])
    wick_data = cf.compute_wick_terms(healpixs)
    return wick_data


def main(cmdargs):
    # pylint: disable-msg=too-many-locals,too-many-branches,too-many-statements
    """Computes the wick covariance for the auto-correlation of forests"""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=('Compute the wick covariance for the auto-correlation of '
                     'forests'))

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

    parser.add_argument('--max-diagram',
                        type=int,
                        default=3,
                        required=False,
                        help='Maximum diagram to compute')

    parser.add_argument(
        '--cf1d',
        type=str,
        required=True,
        help=('1D auto-correlation of pixels from the same forest file: '
              'picca_cf1d.py'))

    parser.add_argument(
        '--cf1d2',
        type=str,
        default=None,
        required=False,
        help=('1D auto-correlation of pixels from the same forest file of the '
              '2nd delta field: picca_cf1d.py'))

    parser.add_argument(
        '--cf',
        type=str,
        default=None,
        required=False,
        help=('3D auto-correlation of pixels from different forests: '
              'picca_cf.py'))

    parser.add_argument(
        '--cf2',
        type=str,
        default=None,
        required=False,
        help=('3D auto-correlation of pixels from different forests for 2nd '
              'catalog: picca_cf.py'))

    parser.add_argument(
        '--cf12',
        type=str,
        default=None,
        required=False,
        help=('3D auto-correlation of pixels from different forests for cross '
              '1st and 2nd catalog: picca_cf.py'))

    parser.add_argument(
        '--unfold-cf',
        action='store_true',
        required=False,
        help=('rp can be positive or negative depending on the relative '
              'position between absorber1 and absorber2'))

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

    # setup variables in module cf
    cf.r_par_max = args.rp_max
    cf.r_trans_max = args.rt_max
    cf.r_par_min = args.rp_min
    cf.z_cut_max = args.z_cut_max
    cf.z_cut_min = args.z_cut_min
    cf.num_bins_r_par = args.np
    cf.num_bins_r_trans = args.nt
    cf.nside = args.nside
    cf.z_ref = args.z_ref
    cf.alpha = args.z_evol
    cf.alpha2 = args.z_evol
    cf.lambda_abs = constants.ABSORBER_IGM[args.lambda_abs]
    cf.reject = args.rej
    cf.max_diagram = args.max_diagram

    # read blinding keyword
    blinding = io.read_blinding(args.in_dir)

    # load cosmology
    if (args.fid_Or != 0.) or (args.fid_Ok != 0.) or (args.fid_wl != -1.):
        userprint(("ERROR: Cosmology with other than Omega_m set are not yet "
                   "implemented"))
        sys.exit()
    cosmo = constants.Cosmo(Om=args.fid_Om,
                            Or=args.fid_Or,
                            Ok=args.fid_Ok,
                            wl=args.fid_wl,
                            blinding=blinding)

    # read data 1
    data, num_data, z_min, z_max = io.read_deltas(args.in_dir,
                                                  cf.nside,
                                                  cf.lambda_abs,
                                                  cf.alpha,
                                                  cf.z_ref,
                                                  cosmo,
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
    del z_max
    cf.data = data
    cf.num_data = num_data
    cf.ang_max = utils.compute_ang_max(cosmo, cf.r_trans_max, z_min)
    sys.stderr.write("\n")
    userprint("done, npix = {}".format(len(data)))

    # Load 1d correlation functions
    dic_xi1d = {"D1": args.cf1d, "D2": args.cf1d2}
    for fname, filename in dic_xi1d.items():
        if filename is None:
            continue
        hdul = fitsio.FITS(filename)
        header = hdul[1].read_header()
        log_lambda_min = header['LLMIN']
        delta_log_lambda = header['DLL']
        num_pairs_variance_1d = hdul[1]['nv1d'][:]
        variance_1d = hdul[1]['v1d'][:]
        log_lambda = (log_lambda_min +
                      delta_log_lambda * np.arange(len(variance_1d)))
        cf.get_variance_1d[fname] = interp1d(
            log_lambda[num_pairs_variance_1d > 0],
            variance_1d[num_pairs_variance_1d > 0],
            kind='nearest',
            fill_value='extrapolate')

        num_pairs1d = hdul[1]['nb1d'][:]
        xi_1d = hdul[1]['c1d'][:]
        cf.xi_1d[fname] = interp1d(
            (log_lambda - log_lambda_min)[num_pairs1d > 0],
            xi_1d[num_pairs1d > 0],
            kind='nearest',
            fill_value='extrapolate')
        hdul.close()

    # Load correlation functions
    dic_xi = {
        "D1_D1": args.cf,
        "D2_D2": args.cf2,
        "D1_D2": args.cf12,
        "D2_D1": args.cf12
    }
    for fname, filename in dic_xi.items():
        if filename is None:
            continue
        hdul = fitsio.FITS(filename)
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
        cf.xi_wick[fname] = xi.copy()
        hdul.close()

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
            rebin_factor=args.rebin_factor)
        for deltas in data.values():
            for delta in deltas:
                delta.fname = 'D2'
                for item in [
                        'cont', 'delta', 'order', 'ivar', 'exposures_diff',
                        'mean_snr', 'mean_reso', 'mean_z', 'delta_log_lambda'
                ]:
                    setattr(delta, item, None)
        del z_max2
        cf.data2 = data2
        cf.num_data2 = num_data2
        cf.ang_max = utils.compute_ang_max(cosmo, cf.r_trans_max, z_min, z_min2)
        userprint("")
        userprint("done, npix = {}".format(len(data2)))

    cf.counter = Value('i', 0)
    cf.lock = Lock()

    cpu_data = {}
    for index, healpix in enumerate(sorted(data)):
        num_processor = index % args.nproc
        if not num_processor in cpu_data:
            cpu_data[num_processor] = []
        cpu_data[num_processor].append(healpix)

    # compute the covariance matrix
    context = multiprocessing.get_context('fork')
    pool = context.Pool(processes=min(args.nproc, len(cpu_data.values())))
    userprint(" \nStarting\n")
    if args.nproc>1:
        wick_data = pool.map(calc_wick_terms, sorted(cpu_data.values()))
    else:
        wick_data = [calc_wick_terms(arg) for arg in sorted(cpu_data.values())]
    userprint(" \nFinished\n")
    pool.close()

    # merge the results from the different CPUs
    wick_data = np.array(wick_data)
    weights_wick = wick_data[:, 0].sum(axis=0)
    num_pairs_wick = wick_data[:, 1].sum(axis=0)
    num_pairs = wick_data[:, 2].sum(axis=0)
    num_pairs_used = wick_data[:, 3].sum(axis=0)
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
    t1 *= 1. * num_pairs_used / num_pairs
    t2 *= 1. * num_pairs_used / num_pairs
    t3 *= 1. * num_pairs_used / num_pairs
    t4 *= 1. * num_pairs_used / num_pairs
    t5 *= 1. * num_pairs_used / num_pairs
    t6 *= 1. * num_pairs_used / num_pairs
    t_tot = t1 + t2 + t3 + t4 + t5 + t6

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
