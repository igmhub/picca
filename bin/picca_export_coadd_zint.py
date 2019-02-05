#!/usr/bin/env python

from __future__ import print_function
import os
import fitsio
import argparse
import scipy as sp
import scipy.linalg

from picca.utils import smooth_cov, print

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
rp = h[1]['RP'][:]*0
rt = h[1]['RT'][:]*0
nb = h[1]['NB'][:]*0
z = h[1]['Z'][:]*0
wet = rp*0
h.close()

da = {}
we = {}

if not args.no_dmat:
    h = fitsio.FITS(args.data[0].replace('cf','dmat'))
    dm = h[1]['DM'][:]*0
    h.close()
else:
    dm = sp.eye(nb.shape[0])

for f in args.data:
    if not (os.path.isfile(f.replace('cf','dmat')) or args.no_dmat):
        continue
    h = fitsio.FITS(f)
    we_aux = h[2]["WE"][:]
    wet_aux = we_aux.sum(axis=0)
    rp += h[1]['RP'][:]*wet_aux
    rt += h[1]['RT'][:]*wet_aux
    z  += h[1]['Z'][:]*wet_aux
    nb += h[1]['NB'][:]
    wet += wet_aux

    hid = h[2]['HEALPID'][:]
    for i,p in enumerate(hid):
        print("\rcoadding healpix {} in file {}".format(p,f),end="")
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
        h.close()

for p in da:
    w=we[p]>0
    da[p][w]/=we[p][w]

rp /= wet
rt /= wet
z /= wet
if not args.no_dmat:
    dm /= wet[:,None]

da = sp.vstack(list(da.values()))
we = sp.vstack(list(we.values()))

co = smooth_cov(da,we,rp,rt)
da = (da*we).sum(axis=0)
da /= wet

try:
    scipy.linalg.cholesky(co)
except scipy.linalg.LinAlgError:
    print('WARNING: Matrix is not positive definite')

h = fitsio.FITS(args.out,"rw",clobber=True)
h.write([rp,rt,z,nb,da,dm,co],names=["RP","RT","Z","NB","DA","DM","CO"],header=head,extname='COR')
h.close()
