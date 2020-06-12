#!/usr/bin/python3
"""Coadd correlation function from different redshift intervals"""
import os
import argparse
import fitsio
import numpy as np
import scipy.linalg

from picca.utils import smooth_cov, userprint

def main():
    """Coadds correlation function from different redshift intervals"""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        type=str,
        nargs="*",
        required=True,
        help="the (x)cf_z_....fits files to be coadded")

    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="output file")

    parser.add_argument(
        "--no-dmat",
        action=
        'store_true',
        default=False,
        required=False,
        help='Use an identity matrix as the distortion matrix.')

    args = parser.parse_args()

    for file in args.data:
        if not os.path.isfile(file):
            args.data.remove(file)

    # initialize coadd arrays, fill them with zeros
    hdul = fitsio.FITS(args.data[0])
    header = hdul[1].read_header()
    r_par = hdul[1]['RP'][:]*0
    r_trans = hdul[1]['RT'][:]*0
    num_pairs = hdul[1]['NB'][:]*0
    z = hdul[1]['Z'][:]*0
    weights_total = r_par*0
    hdul.close()

    xi = {}
    weights = {}

    # initialize distortion matrix array, fill them with zeros
    if not args.no_dmat:
        hdul = fitsio.FITS(args.data[0].replace('cf', 'dmat'))
        dmat = hdul[1]['DM'][:]*0
        try:
            r_par_dmat = np.zeros(hdul[2]['RP'][:].size)
            r_trans_dmat = np.zeros(hdul[2]['RT'][:].size)
            z_dmat = np.zeros(hdul[2]['Z'][:].size)
            num_pairs_dmat = 0.
        except IOError:
            pass
        hdul.close()
    else:
        dmat = np.eye(num_pairs.shape[0])

    # loop over files
    for file in args.data:
        # add correlation function
        if not (args.no_dmat or os.path.isfile(file.replace('cf', 'dmat'))):
            continue
        hdul = fitsio.FITS(file)
        weights_aux = hdul[2]["WE"][:]
        weights_total_aux = weights_aux.sum(axis=0)
        r_par += hdul[1]['RP'][:]*weights_total_aux
        r_trans += hdul[1]['RT'][:]*weights_total_aux
        z += hdul[1]['Z'][:]*weights_total_aux
        num_pairs += hdul[1]['NB'][:]
        weights_total += weights_total_aux

        healpixs = hdul[2]['HEALPID'][:]
        for index, healpix in enumerate(healpixs):
            userprint("\rcoadding healpix {} in file {}".format(healpix,
                                                                file), end="")
            if healpix in xi:
                xi[healpix] += hdul[2]["DA"][:][index]*weights_aux[index]
                weights[healpix] += weights_aux[index, :]
            else:
                xi[healpix] = hdul[2]["DA"][:][index]*weights_aux[index]
                weights[healpix] = weights_aux[index]
        hdul.close()

        # add distortion matrix
        if not args.no_dmat:
            hdul = fitsio.FITS(file.replace('cf', 'dmat'))
            dmat += hdul[1]['DM'][:]*weights_total_aux[:, None]
            if 'r_par_dmat' in locals():
                # TODO: get the weights
                r_par_dmat += hdul[2]['RP'][:]
                r_trans_dmat += hdul[2]['RT'][:]
                z_dmat += hdul[2]['Z'][:]
                num_pairs_dmat += 1.
            hdul.close()

    # combine healpixs
    for healpix in xi:
        w = weights[healpix] > 0
        xi[healpix][w] /= weights[healpix][w]
    xi = np.vstack(list(xi.values()))
    weights = np.vstack(list(weights.values()))

    # compute covariance matrix
    covariance = smooth_cov(xi, weights, r_par, r_trans)

    # normalize
    r_par /= weights_total
    r_trans /= weights_total
    z /= weights_total
    if not args.no_dmat:
        dmat /= weights_total[:, None]

    xi = (xi*weights).sum(axis=0)
    xi /= weights_total

    if 'r_par_dmat' in locals():
        r_par_dmat /= num_pairs_dmat
        r_trans_dmat /= num_pairs_dmat
        z_dmat /= num_pairs_dmat
    if ('r_par_dmat' not in locals()) or (r_par_dmat.size == r_par.size):
        r_par_dmat = r_par.copy()
        r_trans_dmat = r_trans.copy()
        z_dmat = z.copy()

    try:
        scipy.linalg.cholesky(covariance)
    except scipy.linalg.LinAlgError:
        userprint('WARNING: Matrix is not positive definite')

    # save results
    results = fitsio.FITS(args.out, "rw", clobber=True)
    comment = ['R-parallel', 'R-transverse', 'Redshift', 'Correlation',
               'Covariance matrix', 'Distortion matrix', 'Number of pairs']
    results.write([r_par, r_trans, z, xi, covariance, dmat, num_pairs],
                  names=['RP', 'RT', 'Z', 'DA', 'CO', 'DM', 'NB'],
                  comment=comment,
                  header=header,
                  extname='COR')
    comment = ['R-parallel model', 'R-transverse model', 'Redshift model']
    results.write(
        [r_par_dmat, r_trans_dmat, z_dmat],
        names=['DMRP', 'DMRT', 'DMZ'],
        comment=comment,
        extname='DMATTRI')
    results.close()

if __name__ == "__main__":
    main()
