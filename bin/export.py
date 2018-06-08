#!/usr/bin/env python

from __future__ import print_function
import fitsio
import scipy as sp
import scipy.linalg
import argparse

from picca.utils import smooth_cov, cov


if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Export auto and cross-correlation for the fitter.')

    parser.add_argument('--data', type=str, default=None, required=True,
        help='Correlation produced via do_cf.py, do_xcf.py, ...')

    parser.add_argument('--out', type=str, default=None, required=True,
        help='Output file name')

    parser.add_argument('--dmat', type=str, default=None, required=False,
        help='Distortion matrix produced via do_dmat.py, do_xdmat.py... (if not provided will be identity)')

    parser.add_argument('--cov', type=str, default=None, required=False,
        help='Covariance matrix (if not provided will be calculated by subsampling)')

    parser.add_argument('--do-not-smooth-cov', action='store_true', default=False,
        help='Do not smooth the covariance matrix')


    args = parser.parse_args()

    h = fitsio.FITS(args.data)

    rp = sp.array(h[1]['RP'][:])
    rt = sp.array(h[1]['RT'][:])
    z  = sp.array(h[1]['Z'][:])
    nb = sp.array(h[1]['NB'][:])
    da = sp.array(h[2]['DA'][:])
    we = sp.array(h[2]['WE'][:])
    hep = sp.array(h[2]['HEALPID'][:])

    head = h[1].read_header()
    nt = head['NT']
    np = head['NP']
    rt_max = head['RTMAX']
    rp_min = head['RPMIN']
    rp_max = head['RPMAX']
    h.close()

    if args.cov is not None:
        hh = fitsio.FITS(args.cov)
        co = hh[1]['CO'][:]
        hh.close()
    else:
        binSizeP = (rp_max-rp_min) / np
        binSizeT = (rt_max-0.) / nt
        if not args.do_not_smooth_cov:
            print('INFO: The covariance will be smoothed')
            co = smooth_cov(da,we,rp,rt,drt=binSizeT,drp=binSizeP)
        else:
            print('INFO: The covariance will not be smoothed')
            co = cov(da,we)

    da = (da*we).sum(axis=0)
    we = we.sum(axis=0)
    w = we>0
    da[w]/=we[w]

    try:
        scipy.linalg.cholesky(co)
    except:
        print("Warning: Matrix is not positive definite")

    if args.dmat is not None:
        h = fitsio.FITS(args.dmat)
        dm = h[1]['DM'][:]
        h.close()
    else:
        dm = sp.eye(len(da))

    h = fitsio.FITS(args.out,'rw',clobber=True)
    head = [ {'name':'RPMIN','value':rp_min,'comment':'Minimum r-parallel'},
        {'name':'RPMAX','value':rp_max,'comment':'Maximum r-parallel'},
        {'name':'RTMAX','value':rt_max,'comment':'Maximum r-transverse'},
        {'name':'NP','value':np,'comment':'Number of bins in r-parallel'},
        {'name':'NT','value':nt,'comment':'Number of bins in r-transverse'}
    ]
    comment = ['R-parallel','R-transverse','Redshift','Correlation','Covariance matrix','Distortion matrix','Number of pairs']
    h.write([rp,rt,z,da,co,dm,nb],names=['RP','RT','Z','DA','CO','DM','NB'],comment=comment,header=head,extname='COR')
    h.close()
