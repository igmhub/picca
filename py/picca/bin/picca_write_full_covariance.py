#!/usr/bin/env python3

"""
Compute and write out the full (unsmoothed) covariance matrix given some correlation functions. 
This script is an updated version of: /global/cfs/projectdirs/desi/science/lya/y1-kp6/iron-tests/correlations/scripts/write_full_covariance_matrix_flex_size.py
"""

import fitsio
import argparse
import numpy as np
import time

from functools import reduce
from picca.utils import compute_cov


def read_corr(files):
    xi = []
    weights = []
    hp_ids = []
    for file in files:
        if file is not None:
            with fitsio.FITS(file) as hdul:
                txi=None
                if 'DA' in hdul[2].get_colnames():
                    txi=hdul[2]['DA'][:]
                else:
                    txi=hdul[2]['DA_BLIND'][:]
                print(file,"correlation shape=",txi.shape)
                xi.append(txi)
                weights.append(hdul[2]['WE'][:])
                hp_ids.append(hdul[2]["HEALPID"][:])

    
    common_hp = reduce(np.intersect1d, hp_ids)
    masks = [np.isin(hp_ids_i, common_hp) for hp_ids_i in hp_ids]
    
    xi = np.hstack([xi_i[mask] for xi_i, mask in zip(xi, masks)])
    weights = np.hstack([weights_i[mask] for weights_i, mask in zip(weights, masks)])

    return xi, weights


if __name__=='__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Writes the full covariance matrix for the given correlation functions.')
    
    parser.add_argument('-c', '--correlations', 
                        type=str, 
                        required = True, 
                        nargs = '+', 
                        help = "Path(s) to correlation file(s) separated by spaces. Can be any number of correlations but must be at least one. The input order here will be the order the they are computed in the covariance and must match the order in the Vega ini files and in the --correlation-types argument in the write_smooth_covariance script.")
    
    parser.add_argument('-o', '--output', 
                        type=str, 
                        required=True, 
                        help="Path of output fits file")
    
    args = parser.parse_args()
    
    #start time
    t1 = time.time()

    print()
    print('IMPORTANT')
    print(f'The order the correlations will be computed in the covariance is \n{args.correlations}. \nIf this is not okay, terminate this script and adjust your input to the --correlations argument.')

    #starting computation
    print()
    print('Computing covariance')
    t2 = time.time()
    #args.correlations is already a list so it can be directly input to the read_corr function
    xi, weights = read_corr(args.correlations)

    cov = compute_cov(xi, weights)
    #done with computation
    t3 = time.time()
    
    print('Writing covariance')
    results = fitsio.FITS(args.output, 'rw', clobber=True)
    results.write([cov], names=['COV'], units=[''], extname='COVMAT')
    results.close()

    t4 = time.time()
    print()
    print(f'Time spent computing covariance: {(t3 - t2)/60:.3f} minutes')
    print(f'Total time: {(t4-t1)/60:.3f} minutes')
