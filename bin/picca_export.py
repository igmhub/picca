#!/usr/bin/env python

from __future__ import print_function
import fitsio
import scipy as sp
import scipy.linalg
import argparse

from picca.utils import smooth_cov, cov
from picca.utils import print

if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Export auto and cross-correlation for the fitter.')

    parser.add_argument('--data', type=str, default=None, required=True,
        help='Correlation produced via picca_cf.py, picca_xcf.py, ...')

    parser.add_argument('--out', type=str, default=None, required=True,
        help='Output file name')

    parser.add_argument('--dmat', type=str, default=None, required=False,
        help='Distortion matrix produced via picca_dmat.py, picca_xdmat.py... (if not provided will be identity)')

    parser.add_argument('--cov', type=str, default=None, required=False,
        help='Covariance matrix (if not provided will be calculated by subsampling)')

    parser.add_argument('--cor', type=str, default=None, required=False,
        help='Correlation matrix (if not provided will be calculated by subsampling)')

    parser.add_argument('--remove-shuffled-correlation', type=str, default=None, required=False,
        help='Remove a correlation from shuffling the distribution of los')

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

    if not args.remove_shuffled_correlation is None:
        th = fitsio.FITS(args.remove_shuffled_correlation)
        da_s = th['COR']['DA'][:]
        we_s = th['COR']['WE'][:]
        da_s = (da_s*we_s).sum(axis=1)
        we_s = we_s.sum(axis=1)
        w = we_s>0.
        da_s[w] /= we_s[w]
        th.close()
        da -= da_s[:,None]

    if args.cov is not None:
        print('INFO: The covariance-matrix will be read from file: {}'.format(args.cov))
        hh = fitsio.FITS(args.cov)
        co = hh[1]['CO'][:]
        hh.close()
    elif args.cor is not None:
        print('INFO: The correlation-matrix will be read from file: {}'.format(args.cor))
        hh = fitsio.FITS(args.cor)
        cor = hh[1]['CO'][:]
        hh.close()
        if (cor.min()<-1.) | (cor.min()>1.) | (cor.max()<-1.) | (cor.max()>1.) | sp.any(sp.diag(cor)!=1.):
            print('WARNING: The correlation-matrix has some incorrect values')
        tvar = sp.diagonal(cor)
        cor = cor/sp.sqrt(tvar*tvar[:,None])
        co = cov(da,we)
        var = sp.diagonal(co)
        co = cor * sp.sqrt(var*var[:,None])
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
    except scipy.linalg.LinAlgError:
        print('WARNING: Matrix is not positive definite')

    if args.dmat is not None:
        h = fitsio.FITS(args.dmat)
        dm = h[1]['DM'][:]
        try:
            dmrp = h[2]['RP'][:]
            dmrt = h[2]['RT'][:]
            dmz = h[2]['Z'][:]
        except IOError:
            dmrp = rp.copy()
            dmrt = rt.copy()
            dmz = z.copy()
        if dm.shape==(da.size,da.size):
            dmrp = rp.copy()
            dmrt = rt.copy()
            dmz = z.copy()
        h.close()
    else:
        dm = sp.eye(len(da))
        dmrp = rp.copy()
        dmrt = rt.copy()
        dmz = z.copy()

    h = fitsio.FITS(args.out,'rw',clobber=True)
    head = [ {'name':'RPMIN','value':rp_min,'comment':'Minimum r-parallel'},
        {'name':'RPMAX','value':rp_max,'comment':'Maximum r-parallel'},
        {'name':'RTMAX','value':rt_max,'comment':'Maximum r-transverse'},
        {'name':'NP','value':np,'comment':'Number of bins in r-parallel'},
        {'name':'NT','value':nt,'comment':'Number of bins in r-transverse'}
    ]
    comment = ['R-parallel','R-transverse','Redshift','Correlation','Covariance matrix','Distortion matrix','Number of pairs']
    h.write([rp,rt,z,da,co,dm,nb],names=['RP','RT','Z','DA','CO','DM','NB'],comment=comment,header=head,extname='COR')
    comment = ['R-parallel model','R-transverse model','Redshift model']
    h.write([dmrp,dmrt,dmz],names=['DMRP','DMRT','DMZ'],comment=comment,extname='DMATTRI')
    h.close()
