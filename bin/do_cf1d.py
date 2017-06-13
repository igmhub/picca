#!/usr/bin/env python

import scipy as sp
import fitsio
import argparse
import glob
import healpy
import sys
from scipy import random 
import traceback
import pylab 
import numpy as np 
import math


from pylya import constants
from pylya import cf
from pylya.data import delta
from pylya.data import forest
from multiprocessing import Pool,Process,Lock,Manager,cpu_count,Value


def cf1d(p):
    try:
        if x_correlation: tmp = cf.x_forest_cf1d(p)
        else: tmp = cf.cf1d(p)
    except:
        traceback.print_exc()
    with cf.lock:
        cf.counter.value += 1
    sys.stderr.write("\rcomputing xi: {}%".format(round(cf.counter.value*100./cf.npix,2)))
    return tmp

if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--out', type = str, default = None, required=True,
                        help = 'output file name')

    parser.add_argument('--in-dir', type = str, default = None, required=True,
                        help = 'data directory')

    parser.add_argument('--in-dir2', type = str, default = None, required=False,
                        help = 'second delta directory')

    parser.add_argument('--nproc', type = int, default = None, required=False,
                    help = 'number of processors')

    parser.add_argument('--no-project', action="store_true", required=False,
                    help = 'do not project out continuum fitting modes')

    parser.add_argument('--nspec', type=int,default=None, required=False,
                    help = 'maximum spectra to read')

    parser.add_argument('--ll-min', type=float,default=3600., required=False,
                    help = 'minumin loglam')

    parser.add_argument('--ll-max', type=float,default=5500., required=False,
                    help = 'maximum loglam')

    parser.add_argument('--dll', type=float,default=3e-4, required=False,
                    help = 'loglam bin size')

    args = parser.parse_args()

    if args.nproc is None:
        args.nproc = cpu_count()/2

    forest.lmax = sp.log10(args.ll_max)
    forest.lmin = sp.log10(args.ll_min)
    forest.dll = args.dll
    n1d = int((forest.lmax-forest.lmin)/forest.dll+1)
    cf.n1d = n1d
    cf.nside = 16

    data = {}
    ndata = 0
    dels = []
    fi = glob.glob(args.in_dir+"/*.fits.gz")
    for i,f in enumerate(fi):
        sys.stderr.write("\rread {} of {} {}".format(i,len(fi),ndata))
        hdus = fitsio.FITS(f)
        dels += [delta.from_fitsio(h) for h in hdus[1:]]
        ndata+=len(hdus[1:])
        hdus.close()
        if not args.nspec is None:
            if ndata>args.nspec:break

    phi = [d.ra for d in dels]
    th = [sp.pi/2-d.dec for d in dels]
    pix = healpy.ang2pix(cf.nside,th,phi)
    for d,p in zip(dels,pix):
        if not p in data:
            data[p]=[]
        data[p].append(d)
        if not args.no_project:
            d.project()

    cf.npix = len(data)
    cf.data = data

    x_correlation=False
    if args.in_dir2: 
        x_correlation=True
        data2 = {}
        ndata2 = 0
        dels2 = []
        fi = glob.glob(args.in_dir2+"/*.fits.gz")
        for i,f in enumerate(fi):
            sys.stderr.write("\rread {} of {} {}".format(i,len(fi),ndata2))
            hdus = fitsio.FITS(f)
            dels2 += [delta.from_fitsio(h) for h in hdus[1:]]
            ndata2+=len(hdus[1:])
            hdus.close()
            if not args.nspec is None:
                if ndata2>args.nspec:break
        phi2 = [d.ra for d in dels2]
        th2 = [sp.pi/2-d.dec for d in dels2]
        pix2 = healpy.ang2pix(cf.nside,th2,phi2)
        for d,p in zip(dels2,pix2):
            if not p in data2:
                data2[p]=[]
            data2[p].append(d)
            if not args.no_project:
                d.project()
        cf.npix2 = len(data2)
        cf.data2 = data2

    print 
    print 'ndata = ',ndata 
    if x_correlation: print 'ndata2 = ',ndata2
 
    cf.counter = Value('i',0)

    cf.lock = Lock()

    if x_correlation: 
        keys = []
        for i in data.keys(): 
            if i in data2.keys(): 
                keys.append(i)
        cfs = map(cf1d,keys)
    else: cfs = map(cf1d,data.keys())
       
    cfs=sp.array(cfs)
    we_i=cfs[:,0,:]
    nb_i=cfs[:,2,:]
    cf_i=cfs[:,1,:]
    we_i = sp.array(we_i)
    cf_i = sp.array(cf_i)
    nb_i = sp.array(nb_i).astype(sp.int64)      

    print 'cf_i.shape = ',cf_i.shape

    print "multiplying"
    cfs = cf_i*we_i
    cfs = cfs.sum(axis=0)
    wes = we_i.sum(axis=0)
    nbs = nb_i.sum(axis=0)

    cfs = cfs.reshape(n1d,n1d)
    wes = wes.reshape(n1d,n1d)
    nbs = nbs.reshape(n1d,n1d)
 
    w = wes>0
    cfs[w]/=wes[w]
    v1d = sp.diag(cfs).copy()
    wv1d = sp.diag(wes).copy()
    nv1d = sp.diag(nbs).copy()
    cor = cfs

    if 0: 
        norm = sp.sqrt(v1d*v1d[:,None])
        w = norm>0
        cor[w]/=norm[w]

    print "rebinning"
    c1d = sp.zeros(n1d)
    nc1d = sp.zeros(n1d)
    nb1d = sp.zeros(n1d,dtype=sp.int64)
    bins = sp.arange(n1d)

    dbin = bins-bins[:,None]

    w = dbin>=0 

    dbin = dbin[w]
    cor = cor[w] 
    wes = wes[w]
    nbs = nbs[w]
    c = sp.bincount(dbin,weights = cor*wes)
    c1d[:len(c)] = c
    c = sp.bincount(dbin,weights=wes)
    nc1d[:len(c)] = c
    c = sp.bincount(dbin,weights=nbs)
    nb1d[:len(c)] = c

    w1=nc1d>0
    c1d[w1]/=nc1d[w1]
    print "computing cov mat ..."
    we_i = sp.array(we_i)
    cf_i = sp.array(cf_i)
    print 
    c1d_i = sp.zeros((cf_i.shape[0],n1d))
    nc1d_i = sp.zeros((cf_i.shape[0],n1d))


    print 'cf_i.shape[0] = ',cf_i.shape[0]

    for p in range(cf_i.shape[0]): 
        tmp_cf_i=cf_i[p].reshape((n1d,n1d))
        tmp_we_i=we_i[p].reshape((n1d,n1d))
        tmp_cf_i=tmp_cf_i[w]
        tmp_we_i=tmp_we_i[w]
        c = sp.bincount(dbin,weights = tmp_cf_i*tmp_we_i)
        c1d_i[p][:len(c)] = c
        c = sp.bincount(dbin,weights=tmp_we_i)
        nc1d_i[p][:len(c)] = c
        w1=nc1d_i[p]>0
        c1d_i[p][w1]/=nc1d_i[p][w1]


    if 0: 
        new_c1d=np.sum(c1d_i*nc1d_i,axis=0)
        w=np.sum(nc1d_i,axis=0)>0
        new_c1d[w]/=np.sum(nc1d_i,axis=0)[w]
        print 'new_c1d.shape = ', new_c1d.shape
        print 'nc1d_i.shape = ',nc1d_i.shape 
        pylab.plot(10**(forest.dll*np.arange(614)),new_c1d,label='sum(total)')

        #for p in range(cf_i.shape[0]): 
        #    pylab.plot(10**(forest.dll*np.arange(614)),c1d_i[p],label=p)

        pylab.plot(10**(forest.dll*np.arange(614)),c1d,label='total')
        pylab.legend()
        pylab.show()
        sys.exit(12)


    cov=np.zeros((n1d,n1d))
    for p in range(cf_i.shape[0]): 
        wres=nc1d_i[p]*(c1d_i[p]-c1d)
        cov+= np.outer(wres,wres)
    #swmat_v=np.outer(nb1d,nb1d).ravel()
    swmat_v=np.outer(nc1d,nc1d).ravel()
    covmat_v = cov.ravel()/(swmat_v+(swmat_v==0))
    cov      = covmat_v.reshape(n1d,n1d)


    err=np.sqrt(np.diag(cov.copy()))

    print "writing"
    out = fitsio.FITS(args.out,'rw',clobber=True)
    head = {}
    head['LLMAX']=forest.lmax
    head['LLMIN']=forest.lmin
    head['DLL']=forest.dll

    out.write([v1d,wv1d,nv1d,c1d,nc1d,nb1d,cov,err],names=['v1d','wv1d','nv1d','c1d','nc1d','nb1d','cov','err'],header=head)
    out.close()

    print "all done"

    
