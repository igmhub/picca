#!/usr/bin/env python

import os
import fitsio
import argparse
import numpy as np
import scipy as sp
import scipy.linalg

from picca.utils import smooth_cov, userprint

parser = argparse.ArgumentParser()

parser.add_argument("--data",type=str,nargs="*",required=True,
        help="the (x)cf_z_....fits files to be coadded")

parser.add_argument("--out",type=str,required=True,
        help="output file")

parser.add_argument("--no-dmat",action='store_true',default=False,required=False,
        help='Use an identity matrix as the distortion matrix.')

args=parser.parse_args()

for f in args.data:
    if not os.path.isfile(f):
        args.data.remove(f)

h=fitsio.FITS(args.data[0])

head = h[1].read_header()
r_par = h[1]['RP'][:]*0
r_trans = h[1]['RT'][:]*0
nb = h[1]['NB'][:]*0
z = h[1]['Z'][:]*0
wet = r_par*0
h.close()

da = {}
we = {}

if not args.no_dmat:
    h = fitsio.FITS(args.data[0].replace('cf','dmat'))
    dm = h[1]['DM'][:]*0
    try:
        dmrp = np.zeros(h[2]['RP'][:].size)
        dmrt = np.zeros(h[2]['RT'][:].size)
        dmz = np.zeros(h[2]['Z'][:].size)
        nbdm = 0.
    except IOError:
        pass
    h.close()
else:
    dm = sp.eye(nb.shape[0])

for f in args.data:
    if not (os.path.isfile(f.replace('cf','dmat')) or args.no_dmat):
        continue
    h = fitsio.FITS(f)
    we_aux = h[2]["WE"][:]
    wet_aux = we_aux.sum(axis=0)
    r_par += h[1]['RP'][:]*wet_aux
    r_trans += h[1]['RT'][:]*wet_aux
    z  += h[1]['Z'][:]*wet_aux
    nb += h[1]['NB'][:]
    wet += wet_aux

    hid = h[2]['HEALPID'][:]
    for i,p in enumerate(hid):
        userprint("\rcoadding healpix {} in file {}".format(p,f),end="")
        if p in da:
            da[p] += h[2]["DA"][:][i]*we_aux[i]
            we[p] += we_aux[i,:]
        else:
            da[p] = h[2]["DA"][:][i]*we_aux[i]
            we[p] = we_aux[i]
    h.close()

    if not args.no_dmat:
        h = fitsio.FITS(f.replace('cf','dmat'))
        dm += h[1]['DM'][:]*wet_aux[:,None]
        if 'dmrp' in locals():
            ## TODO: get the weights
            dmrp += h[2]['RP'][:]
            dmrt += h[2]['RT'][:]
            dmz += h[2]['Z'][:]
            nbdm += 1.
        h.close()

for p in da:
    w=we[p]>0
    da[p][w]/=we[p][w]

r_par /= wet
r_trans /= wet
z /= wet
if not args.no_dmat:
    dm /= wet[:,None]

da = sp.vstack(list(da.values()))
we = sp.vstack(list(we.values()))

co = smooth_cov(da,we,r_par,r_trans)
da = (da*we).sum(axis=0)
da /= wet


if 'dmrp' in locals():
    dmrp /= nbdm
    dmrt /= nbdm
    dmz /= nbdm
if ('dmrp' not in locals()) or (dmrp.size==r_par.size):
    dmrp = r_par.copy()
    dmrt = r_trans.copy()
    dmz = z.copy()

try:
    scipy.linalg.cholesky(co)
except scipy.linalg.LinAlgError:
    userprint('WARNING: Matrix is not positive definite')


h = fitsio.FITS(args.out,"rw",clobber=True)
comment = ['R-parallel','R-transverse','Redshift','Correlation','Covariance matrix','Distortion matrix','Number of pairs']
h.write([r_par,r_trans,z,da,co,dm,nb],names=['RP','RT','Z','DA','CO','DM','NB'],comment=comment,header=head,extname='COR')
comment = ['R-parallel model','R-transverse model','Redshift model']
h.write([dmrp,dmrt,dmz],names=['DMRP','DMRT','DMZ'],comment=comment,extname='DMATTRI')
h.close()
