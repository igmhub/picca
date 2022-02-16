#!/usr/bin/env python
"""Compute the cross-covariance matrix between two correlations."""
import argparse
import sys
import numpy as np
import scipy.linalg
import fitsio

from picca.utils import compute_cov, userprint

def main(cmdargs):

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=('Compute the cross-covariance matrix between two '
                     'correlations'))

    parser.add_argument(
        '--data1',
        type=str,
        default=None,
        required=True,
        help='Correlation 1 produced via picca_cf.py, picca_xcf.py, ...')

    parser.add_argument(
        '--data2',
        type=str,
        default=None,
        required=True,
        help='Correlation 2 produced via picca_cf.py, picca_xcf.py, ...')

    parser.add_argument('--out',
                        type=str,
                        default=None,
                        required=True,
                        help='Output file name')

    args = parser.parse_args(cmdargs)

    data = {}

    # Read data
    for index, filename in enumerate([args.data1, args.data2]):
        hdul = fitsio.FITS(filename)
        header = hdul[1].read_header()
        nside = header['NSIDE']
        header2 = hdul[2].read_header()
        healpix_scheme = header2['HLPXSCHM']
        weights = np.array(hdul[2]['WE'][:])
        healpix_list = np.array(hdul[2]['HEALPID'][:])

        if 'DA_BLIND' in hdul[2].get_colnames():
            xi = np.array(hdul[2]['DA_BLIND'][:])
        else:
            xi = np.array(hdul[2]['DA'][:])

        data[index] = {
            'DA': xi,
            'WE': weights,
            'HEALPID': healpix_list,
            'NSIDE': nside,
            'HLPXSCHM': healpix_scheme
        }
        hdul.close()

    # exit if NSIDE1 != NSIDE2
    if data[0]['NSIDE'] != data[1]['NSIDE']:
        userprint(("ERROR: NSIDE are different: {} != "
                   "{}").format(data[0]['NSIDE'], data[1]['NSIDE']))
        sys.exit()
    # exit if HLPXSCHM1 != HLPXSCHM2
    if data[0]['HLPXSCHM'] != data[1]['HLPXSCHM']:
        userprint(("ERROR: HLPXSCHM are different: {} != "
                   "{}").format(data[0]['HLPXSCHM'], data[1]['HLPXSCHM']))
        sys.exit()

    # Add unshared healpix as empty data
    for key in sorted(list(data.keys())):
        key2 = (key + 1) % 2
        w = np.logical_not(np.in1d(data[key2]['HEALPID'], data[key]['HEALPID']))
        if w.sum() > 0:
            new_healpix = data[key2]['HEALPID'][w]
            num_new_healpix = new_healpix.size
            num_bins = data[key]['DA'].shape[1]
            userprint(("Some healpix are unshared in data {}: "
                       "{}").format(key, new_healpix))
            data[key]['DA'] = np.append(data[key]['DA'],
                                        np.zeros((num_new_healpix, num_bins)),
                                        axis=0)
            data[key]['WE'] = np.append(data[key]['WE'],
                                        np.zeros((num_new_healpix, num_bins)),
                                        axis=0)
            data[key]['HEALPID'] = np.append(data[key]['HEALPID'], new_healpix)

    # Sort the data by the healpix values
    for key in sorted(list(data.keys())):
        sort = np.array(data[key]['HEALPID']).argsort()
        data[key]['DA'] = data[key]['DA'][sort]
        data[key]['WE'] = data[key]['WE'][sort]
        data[key]['HEALPID'] = data[key]['HEALPID'][sort]

    # Append the data
    xi = np.append(data[0]['DA'], data[1]['DA'], axis=1)
    weights = np.append(data[0]['WE'], data[1]['WE'], axis=1)

    # Compute the covariance
    covariance = compute_cov(xi, weights)

    # Get the cross-covariance
    num_bins = data[0]['DA'].shape[1]
    cross_covariance = covariance.copy()
    cross_covariance = cross_covariance[:, num_bins:]
    cross_covariance = cross_covariance[:num_bins, :]

    ### Get the cross-correlation
    var = np.diagonal(covariance)
    cor = covariance / np.sqrt(var * var[:, None])
    cross_correlation = cor.copy()
    cross_correlation = cross_correlation[:, num_bins:]
    cross_correlation = cross_correlation[:num_bins, :]

    ### Test if valid
    try:
        scipy.linalg.cholesky(covariance)
    except scipy.linalg.LinAlgError:
        userprint('WARNING: Matrix is not positive definite')

    ### Save
    results = fitsio.FITS(args.out, 'rw', clobber=True)
    results.write([cross_covariance, cross_correlation],
                  names=['CO', 'COR'],
                  comment=['Covariance matrix', 'Correlation matrix'],
                  extname='COVAR')
    results.close()


if __name__ == '__main__':
    cmdargs=sys.argv[1:]
    main(cmdargs)