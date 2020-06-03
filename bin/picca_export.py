#!/usr/bin/env python


import fitsio
import scipy as sp
import scipy.linalg
import argparse

from picca.utils import smooth_cov, compute_covariance
from picca.utils import userprint

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

    r_par = sp.array(h[1]['RP'][:])
    r_trans = sp.array(h[1]['RT'][:])
    z  = sp.array(h[1]['Z'][:])
    nb = sp.array(h[1]['NB'][:])
    da = sp.array(h[2]['DA'][:])
    weights = sp.array(h[2]['WE'][:])
    hep = sp.array(h[2]['HEALPID'][:])

    head = h[1].read_header()
    num_bins_r_par = head['NP']
    num_bins_r_trans = head['NT']
    r_trans_max = head['RTMAX']
    r_par_min = head['RPMIN']
    r_par_max = head['RPMAX']
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
        userprint('INFO: The covariance-matrix will be read from file: {}'.format(args.cov))
        hh = fitsio.FITS(args.cov)
        covariance = hh[1]['CO'][:]
        hh.close()
    elif args.cor is not None:
        userprint('INFO: The correlation-matrix will be read from file: {}'.format(args.cor))
        hh = fitsio.FITS(args.cor)
        cor = hh[1]['CO'][:]
        hh.close()
        if (cor.min()<-1.) | (cor.min()>1.) | (cor.max()<-1.) | (cor.max()>1.) | sp.any(sp.diag(cor)!=1.):
            userprint('WARNING: The correlation-matrix has some incorrect values')
        tvar = sp.diagonal(cor)
        cor = cor/sp.sqrt(tvar*tvar[:,None])
        covariance = compute_covariance(da,weights)
        var = sp.diagonal(covariance)
        covariance = cor * sp.sqrt(var*var[:,None])
    else:
        binSizeP = (r_par_max-r_par_min) / num_bins_r_par
        binSizeT = (r_trans_max-0.) / num_bins_r_trans
        if not args.do_not_smooth_cov:
            userprint('INFO: The covariance will be smoothed')
            covariance = smooth_cov(da,weights,r_par,r_trans,drt=binSizeT,drp=binSizeP)
        else:
            userprint('INFO: The covariance will not be smoothed')
            covariance = compute_covariance(da,weights)

    da = (da*weights).sum(axis=0)
    weights = weights.sum(axis=0)
    w = weights>0
    da[w]/=weights[w]

    try:
        scipy.linalg.cholesky(covariance)
    except scipy.linalg.LinAlgError:
        userprint('WARNING: Matrix is not positive definite')

    if args.dmat is not None:
        h = fitsio.FITS(args.dmat)
        dm = h[1]['DM'][:]
        try:
            dmrp = h[2]['RP'][:]
            dmrt = h[2]['RT'][:]
            dmz = h[2]['Z'][:]
        except IOError:
            dmrp = r_par.copy()
            dmrt = r_trans.copy()
            dmz = z.copy()
        if dm.shape==(da.size,da.size):
            dmrp = r_par.copy()
            dmrt = r_trans.copy()
            dmz = z.copy()
        h.close()
    else:
        dm = sp.eye(len(da))
        dmrp = r_par.copy()
        dmrt = r_trans.copy()
        dmz = z.copy()

    h = fitsio.FITS(args.out,'rw',clobber=True)
    head = [ {'name':'RPMIN','value':r_par_min,'comment':'Minimum r-parallel'},
        {'name':'RPMAX','value':r_par_max,'comment':'Maximum r-parallel'},
        {'name':'RTMAX','value':r_trans_max,'comment':'Maximum r-transverse'},
        {'name':'NP','value':num_bins_r_par,'comment':'Number of bins in r-parallel'},
        {'name':'NT','value':num_bins_r_trans,'comment':'Number of bins in r-transverse'}
    ]
    comment = ['R-parallel','R-transverse','Redshift','Correlation','Covariance matrix','Distortion matrix','Number of pairs']
    h.write([r_par,r_trans,z,da,covariance,dm,nb],names=['RP','RT','Z','DA','CO','DM','NB'],comment=comment,header=head,extname='COR')
    comment = ['R-parallel model','R-transverse model','Redshift model']
    h.write([dmrp,dmrt,dmz],names=['DMRP','DMRT','DMZ'],comment=comment,extname='DMATTRI')
    h.close()
