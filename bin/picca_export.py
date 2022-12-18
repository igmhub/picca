#!/usr/bin/env python
"""Export auto and cross-correlation for the fitter."""
import sys
import argparse
import fitsio
import numpy as np
import scipy.interpolate
import scipy.linalg
import h5py
import os.path

from picca.utils import smooth_cov, compute_cov
from picca.utils import userprint


def main(cmdargs):
    """Export auto and cross-correlation for the fitter."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Export auto and cross-correlation for the fitter.')

    parser.add_argument(
        '--data',
        type=str,
        default=None,
        required=True,
        help='Correlation produced via picca_cf.py, picca_xcf.py, ...')

    parser.add_argument(
        '--out',
        type=str,
        default=None,
        required=True,
        help='Output file name')

    parser.add_argument(
        '--dmat',
        type=str,
        default=None,
        required=False,
        help=('Distortion matrix produced via picca_dmat.py, picca_xdmat.py... '
              '(if not provided will be identity)'))

    parser.add_argument(
        '--cov',
        type=str,
        default=None,
        required=False,
        help=('Covariance matrix (if not provided will be calculated by '
              'subsampling)'))

    parser.add_argument(
        '--cor',
        type=str,
        default=None,
        required=False,
        help=('Correlation matrix (if not provided will be calculated by '
              'subsampling)'))

    parser.add_argument(
        '--remove-shuffled-correlation',
        type=str,
        default=None,
        required=False,
        help='Remove a correlation from shuffling the distribution of los')

    parser.add_argument(
        '--do-not-smooth-cov',
        action='store_true',
        default=False,
        help='Do not smooth the covariance matrix')

    parser.add_argument(
        '--blind-corr-type',
        default=None,
        choices=['lyaxlya', 'lyaxlyb', 'qsoxlya', 'qsoxlyb'],
        help='Type of correlation. Required to apply blinding in DESI')

    args = parser.parse_args(cmdargs)

    hdul = fitsio.FITS(args.data)

    r_par = np.array(hdul[1]['RP'][:])
    r_trans = np.array(hdul[1]['RT'][:])
    z = np.array(hdul[1]['Z'][:])
    num_pairs = np.array(hdul[1]['NB'][:])
    weights = np.array(hdul[2]['WE'][:])

    if 'DA_BLIND' in hdul[2].get_colnames():
        xi = np.array(hdul[2]['DA_BLIND'][:])
        data_name = 'DA_BLIND'
    else:
        xi = np.array(hdul[2]['DA'][:])
        data_name = 'DA'

    head = hdul[1].read_header()
    num_bins_r_par = head['NP']
    num_bins_r_trans = head['NT']
    r_trans_max = head['RTMAX']
    r_par_min = head['RPMIN']
    r_par_max = head['RPMAX']

    if "BLINDING" in head:
        blinding = head["BLINDING"]
        if blinding == 'minimal':
            blinding = 'corr_yshift'
            userprint("The minimal strategy is no longer supported."
                      "Automatically switch to corr_yshift.")
    else:
        # if BLINDING keyword not present (old file), ignore blinding
        blinding = "none"
    hdul.close()

    if args.remove_shuffled_correlation is not None:
        hdul = fitsio.FITS(args.remove_shuffled_correlation)
        xi_shuffled = hdul['COR'][data_name][:]
        weight_shuffled = hdul['COR']['WE'][:]
        xi_shuffled = (xi_shuffled * weight_shuffled).sum(axis=1)
        weight_shuffled = weight_shuffled.sum(axis=1)
        w = weight_shuffled > 0.
        xi_shuffled[w] /= weight_shuffled[w]
        hdul.close()
        xi -= xi_shuffled[:, None]

    if args.cov is not None:
        userprint(("INFO: The covariance-matrix will be read from file: "
                   "{}").format(args.cov))
        hdul = fitsio.FITS(args.cov)
        covariance = hdul[1]['CO'][:]
        hdul.close()
    elif args.cor is not None:
        userprint(("INFO: The correlation-matrix will be read from file: "
                   "{}").format(args.cor))
        hdul = fitsio.FITS(args.cor)
        correlation = hdul[1]['CO'][:]
        hdul.close()
        if ((correlation.min() < -1.) or (correlation.min() > 1.) or
                (correlation.max() < -1.) or (correlation.max() > 1.) or
                np.any(np.diag(correlation) != 1.)):
            userprint(("WARNING: The correlation-matrix has some incorrect "
                       "values"))
        var = np.diagonal(correlation)
        correlation = correlation / np.sqrt(var * var[:, None])
        covariance = compute_cov(xi, weights)
        var = np.diagonal(covariance)
        covariance = correlation * np.sqrt(var * var[:, None])
    else:
        delta_r_par = (r_par_max - r_par_min) / num_bins_r_par
        delta_r_trans = (r_trans_max - 0.) / num_bins_r_trans
        if not args.do_not_smooth_cov:
            userprint("INFO: The covariance will be smoothed")
            covariance = smooth_cov(xi,
                                    weights,
                                    r_par,
                                    r_trans,
                                    delta_r_trans=delta_r_trans,
                                    delta_r_par=delta_r_par)
        else:
            userprint("INFO: The covariance will not be smoothed")
            covariance = compute_cov(xi, weights)

    xi = (xi * weights).sum(axis=0)
    weights = weights.sum(axis=0)
    w = weights > 0
    xi[w] /= weights[w]

    try:
        scipy.linalg.cholesky(covariance)
    except scipy.linalg.LinAlgError:
        userprint("WARNING: Matrix is not positive definite")

    if args.dmat is not None:
        hdul = fitsio.FITS(args.dmat)
        if data_name == "DA_BLIND" and 'DM_BLIND' in hdul[1].get_colnames():
            dmat = np.array(hdul[1]['DM_BLIND'][:])
            dmat_name = 'DM_BLIND'
        elif data_name == "DA_BlIND":
            userprint("Blinded correlations were given but distortion matrix "
                      "is unblinded. These files should not mix. Exiting...")
            sys.exit(1)
        elif 'DM_BLIND' in hdul[1].get_colnames():
            userprint("Non-blinded correlations were given but distortion matrix "
                      "is blinded. These files should not mix. Exiting...")
            sys.exit(1)
        else:
            dmat = hdul[1]['DM'][:]
            dmat_name = 'DM'

        try:
            r_par_dmat = hdul[2]['RP'][:]
            r_trans_dmat = hdul[2]['RT'][:]
            z_dmat = hdul[2]['Z'][:]
        except IOError:
            r_par_dmat = r_par.copy()
            r_trans_dmat = r_trans.copy()
            z_dmat = z.copy()
        if dmat.shape == (xi.size, xi.size):
            r_par_dmat = r_par.copy()
            r_trans_dmat = r_trans.copy()
            z_dmat = z.copy()
        hdul.close()
    else:
        dmat = np.eye(len(xi))
        r_par_dmat = r_par.copy()
        r_trans_dmat = r_trans.copy()
        z_dmat = z.copy()
        dmat_name = 'DM_EMPTY'

    results = fitsio.FITS(args.out, 'rw', clobber=True)
    header = [
    {
        'name': "BLINDING",
        'value': blinding,
        'comment': 'String specifying the blinding strategy'
    },
    {
        'name': 'RPMIN',
        'value': r_par_min,
        'comment': 'Minimum r-parallel'
    }, {
        'name': 'RPMAX',
        'value': r_par_max,
        'comment': 'Maximum r-parallel'
    }, {
        'name': 'RTMAX',
        'value': r_trans_max,
        'comment': 'Maximum r-transverse'
    }, {
        'name': 'NP',
        'value': num_bins_r_par,
        'comment': 'Number of bins in r-parallel'
    }, {
        'name': 'NT',
        'value': num_bins_r_trans,
        'comment': 'Number of bins in r-transverse'
    }, {
        'name': 'OMEGAM',
        'value': head['OMEGAM'],
        'comment': 'Omega_matter(z=0) of fiducial LambdaCDM cosmology'
    }, {
        'name': 'OMEGAR',
        'value': head['OMEGAR'],
        'comment': 'Omega_radiation(z=0) of fiducial LambdaCDM cosmology'
    }, {
        'name': 'OMEGAK',
        'value': head['OMEGAK'],
        'comment': 'Omega_k(z=0) of fiducial LambdaCDM cosmology'
    }, {
        'name': 'WL',
        'value': head['WL'],
        'comment': 'Equation of state of dark energy of fiducial LambdaCDM cosmology'
    },
    ]
    comment = [
        'R-parallel', 'R-transverse', 'Redshift', 'Correlation',
        'Covariance matrix', 'Distortion matrix', 'Number of pairs'
    ]

    # Check if we need blinding and apply it
    if 'BLIND' in data_name or blinding != 'none':
        blinding_dir = '/global/cfs/projectdirs/desi/science/lya/y1-kp6/blinding/'
        blinding_templates = {'desi_m2': {'standard': 'm2_blinding_v1.2_standard_29_03_2022.h5',
                                          'grid': 'm2_blinding_v1.2_regular_grid_29_03_2022.h5'},
                              'desi_y1': {'standard': 'y1_blinding_v2_standard_17_12_2022.h5',
                                          'grid': 'y1_blinding_v2_regular_grid_17_12_2022.h5'},
                              'desi_y3': {'standard': 'y3_blinding_v3_standard_18_12_2022.h5',
                                          'grid': 'y3_blinding_v3_regular_grid_18_12_2022.h5'}}

        if blinding in blinding_templates:
            userprint(f"Blinding using seed for {blinding}")
        else:
            raise ValueError(f"Expected blinding to be one of {blinding_templates.keys()}."
                             f" Found {blinding}.")

        if args.blind_corr_type is None:
            raise ValueError("Blinding requires argument --blind_corr_type.")

        # Check type of correlation and get size and regular binning
        if args.blind_corr_type in ['lyaxlya', 'lyaxlyb']:
            corr_size = 2500
            rp_interp_grid = np.arange(2., 202., 4)
            rt_interp_grid = np.arange(2., 202., 4)
        elif args.blind_corr_type in ['qsoxlya', 'qsoxlyb']:
            corr_size = 5000
            rp_interp_grid = np.arange(-197.99, 202.01, 4)
            rt_interp_grid = np.arange(2., 202, 4)
        else:
            raise ValueError("Unknown correlation type: {}".format(args.blind_corr_type))

        if corr_size == len(xi):
            # Read the blinding file and get the right template
            blinding_filename = blinding_dir + blinding_templates[blinding]['standard']
        else:
            # Read the regular grid blinding file and get the right template
            blinding_filename = blinding_dir + blinding_templates[blinding]['grid']

        if not os.path.isfile(blinding_filename):
            raise RuntimeError("Missing blinding file. Make sure you are running at"
                               " NERSC or contact picca developers")
        blinding_file = h5py.File(blinding_filename, 'r')
        hex_diff = np.array(blinding_file['blinding'][args.blind_corr_type]).astype(str)
        diff_grid = np.array([float.fromhex(x) for x in hex_diff])

        if corr_size == len(xi):
            diff = diff_grid
        else:
            # Interpolate the blinding template on the regular grid
            interp = scipy.interpolate.RectBivariateSpline(
                    rp_interp_grid, rt_interp_grid,
                    diff_grid.reshape(len(rp_interp_grid), len(rt_interp_grid)), kx=3, ky=3)
            diff = interp.ev(r_par, r_trans)

        # Check that the shapes match
        if np.shape(xi) != np.shape(diff):
            raise RuntimeError("Unknown binning or wrong correlation type. Cannot blind."
                               " Please raise an issue or contact picca developers.")

        # Add blinding
        xi = xi + diff

    results.write([xi, r_par, r_trans, z, covariance, dmat, num_pairs],
                  names=[data_name, 'RP', 'RT', 'Z', 'CO', dmat_name, 'NB'],
                  comment=comment,
                  header=header,
                  extname='COR')
    comment = ['R-parallel model', 'R-transverse model', 'Redshift model']
    results.write([r_par_dmat, r_trans_dmat, z_dmat],
                  names=['DMRP', 'DMRT', 'DMZ'],
                  comment=comment,
                  extname='DMATTRI')
    results.close()


if __name__ == '__main__':
    cmdargs = sys.argv[1:]
    main(cmdargs)
