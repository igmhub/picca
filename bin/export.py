#!/usr/bin/env python

import fitsio
import scipy as sp
import scipy.linalg
import argparse

from picca.utils import smooth_cov


if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data', type = str, default = None, required=True,
                        help = 'data file')

    parser.add_argument('--out', type = str, default = None, required=True,
                        help = 'output file')

    parser.add_argument('--dmat', type = str, default = None, required=False,
                        help = 'distorsion matrix file')

    parser.add_argument('--cov', type = str, default = None, required=False,
                        help = 'covariance matrix file (if not provided it will be calculated by subsampling)')


    args = parser.parse_args()

    h = fitsio.FITS(args.data)

    rp = sp.array(h[1]['RP'][:])
    rt = sp.array(h[1]['RT'][:])
    z  = sp.array(h[1]['Z'][:])
    nb = sp.array(h[1]['NB'][:])
    da = sp.array(h[2]['DA'][:])
    we = sp.array(h[2]['WE'][:])
    hep = sp.array(h[2]['HEALPID'][:])

    if args.cov is not None:
        hh = fitsio.FITS(args.cov)
        co = hh[1]['CO'][:]
        hh.close()
    else:
        head = h[1].read_header()
        nt = head['NT']
        np = head['NP']
        rt_min = 0.
        rt_max = head['RTMAX']
        rp_min = head['RPMIN']
        rp_max = head['RPMAX']
        binSizeP = (rp_max-rp_min) / np
        binSizeT = (rt_max-rt_min) / nt
        co = smooth_cov(da,we,rp,rt,drt=binSizeT,drp=binSizeP)

    da = (da*we).sum(axis=0)
    we = we.sum(axis=0)
    w = we>0
    da[w]/=we[w]

    h.close()

    try:
        scipy.linalg.cholesky(co)
    except:
        print("Matrix is not positive definite")

    if args.dmat is not None:
        h = fitsio.FITS(args.dmat)
        dm = h[1]['DM'][:]
        h.close()
    else:
        dm = sp.eye(len(da))

    h = fitsio.FITS(args.out,'rw',clobber=True)

    h.write([rp,rt,z,da,co,dm,nb],names=['RP','RT','Z','DA','CO','DM','NB'])
    h.close()
