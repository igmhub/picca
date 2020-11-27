#!/usr/bin/env python3
"""Export auto and cross-correlation for the fitter."""
import argparse
import fitsio
import numpy as np
import scipy.linalg

from picca.utils import smooth_cov, compute_cov
from picca.utils import userprint


def main():
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

    parser.add_argument('--out',
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

    parser.add_argument('--do-not-smooth-cov',
                        action='store_true',
                        default=False,
                        help='Do not smooth the covariance matrix')

    args = parser.parse_args()

    hdul = fitsio.FITS(args.data)

    r_par = np.array(hdul[1]['RP'][:])
    r_trans = np.array(hdul[1]['RT'][:])
    z = np.array(hdul[1]['Z'][:])
    num_pairs = np.array(hdul[1]['NB'][:])
    xi = np.array(hdul[2]['DA'][:])
    weights = np.array(hdul[2]['WE'][:])

    head = hdul[1].read_header()
    num_bins_r_par = head['NP']
    num_bins_r_trans = head['NT']
    r_trans_max = head['RTMAX']
    r_par_min = head['RPMIN']
    r_par_max = head['RPMAX']
    hdul.close()

    if not args.remove_shuffled_correlation is None:
        hdul = fitsio.FITS(args.remove_shuffled_correlation)
        xi_shuffled = hdul['COR']['DA'][:]
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
        dmat = hdul[1]['DM'][:]
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

    results = fitsio.FITS(args.out, 'rw', clobber=True)
    header = [{
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
    }
    ]
    comment = [
        'R-parallel', 'R-transverse', 'Redshift', 'Correlation',
        'Covariance matrix', 'Distortion matrix', 'Number of pairs'
    ]
    results.write([r_par, r_trans, z, xi, covariance, dmat, num_pairs],
                  names=['RP', 'RT', 'Z', 'DA', 'CO', 'DM', 'NB'],
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
    main()
