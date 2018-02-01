#!/usr/bin/env python

import scipy as sp
import scipy.linalg
import fitsio
import argparse

from picca.utils import cov

if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data1', type = str, default = None, required=True,
                        help = 'data file #1')

    parser.add_argument('--data2', type = str, default = None, required=True,
                        help = 'data file #2')

    parser.add_argument('--out', type = str, default = None, required=True,
                        help = 'output file')
    args = parser.parse_args()


    data = {}
    
    ### Read data
    for i,p in enumerate([args.data1,args.data2]):
        h = fitsio.FITS(p)
        da  = sp.array(h[2]['DA'][:])
        we  = sp.array(h[2]['WE'][:])
        hep = sp.array(h[2]['HEALPID'][:])
        data[i] = {'DA':da, 'WE':we, 'HEALPID':hep}
        h.close()

    ### Remove unshared healpix
    for i in sorted(list(data.keys())):
        w = sp.in1d(data[i]['HEALPID'],data[(i+1)%2]['HEALPID'])
        if w.sum()!=w.size:
            print("Some healpix are unshared in data {}: {}".format(i,data[i]['HEALPID'][sp.logical_not(w)]))
            data[i]['DA']      = data[i]['DA'][w,:]
            data[i]['WE']      = data[i]['WE'][w,:]
            data[i]['HEALPID'] = data[i]['HEALPID'][w]

    ### Sort the data by the healpix values
    for i in sorted(list(data.keys())):
        sort = sp.array(data[i]['HEALPID']).argsort()
        data[i]['DA']      = data[i]['DA'][sort]
        data[i]['WE']      = data[i]['WE'][sort]
        data[i]['HEALPID'] = data[i]['HEALPID'][sort]
        
    ### Append the data
    da  = sp.append(data[0]['DA'],data[1]['DA'],axis=1)
    we  = sp.append(data[0]['WE'],data[1]['WE'],axis=1)
    hep = data[0]['HEALPID'].copy()
    
    ### Compute the covariance
    co = cov(da,we)
    
    ### Get the cross-covariance
    size1 = data[0]['DA'].shape[1]
    cross_co = co.copy()
    cross_co = cross_co[:,size1:]
    cross_co = cross_co[:size1,:]
    
    ### Get the cross-correlation
    var = sp.diagonal(co)
    cor = co/sp.sqrt(var*var[:,None])
    cross_cor = cor.copy()
    cross_cor = cross_cor[:,size1:]
    cross_cor = cross_cor[:size1,:]

    ### Test if valid
    try:
        scipy.linalg.cholesky(co)
    except:
        print("Matrix is not positive definite")

    ### Save
    h = fitsio.FITS(args.out,'rw',overwrite=True)
    h.write([cross_co,cross_cor],names=['CO','COR'])
    h.close()
