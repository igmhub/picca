#!/usr/bin/env python

import scipy as sp
import fitsio
import argparse
import glob
import healpy
import sys
from scipy import random 
import traceback

from pylya import constants
from pylya import cf
from pylya.data import delta
from pylya.data import forest

from multiprocessing import Pool,Process,Lock,Manager,cpu_count,Value


def cf1d(p):
    try:
        tmp = cf.cf1d(p)
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

    fi = glob.glob(args.in_dir+"/*.fits.gz")
    data = {}
    ndata = 0
    for i,f in enumerate(fi):
        if i%10==0:
            sys.stderr.write("\rread {} of {} {}".format(i,len(fi),ndata))
        hdus = fitsio.FITS(f)
        dels = [delta.from_fitsio(h) for h in hdus[1:]]
        ndata+=len(dels)
        for d in dels:
            p = ndata%args.nproc
            if not p in data:
                data[p]=[]
            data[p].append(d)

            if not args.no_project:
                d.project()

        if not args.nspec is None:
            if ndata>args.nspec:break

    sys.stderr.write("\n")

    cf.npix = len(data)
    cf.data = data
    print "done"

    cf.counter = Value('i',0)

    cf.lock = Lock()
    #pool = Pool(processes=args.nproc)

    cfs = map(cf1d,data.keys())
    #pool.close()

    cfs=sp.array(cfs)
    wes=cfs[:,0,:]
    nbs=cfs[:,2,:]
    cfs=cfs[:,1,:]
    wes = sp.array(wes)
    cfs = sp.array(cfs)
    nbs = sp.array(nbs).astype(sp.int64)

    print "multiplying"
    cfs *= wes
    cfs = cfs.sum(axis=0)
    wes = wes.sum(axis=0)
    nbs = nbs.sum(axis=0)

    print "done",cfs.shape,wes.shape

    cfs = cfs.reshape(n1d,n1d)
    wes = wes.reshape(n1d,n1d)
    nbs = nbs.reshape(n1d,n1d)

    print "rebinning"
 
    w = wes>0
    cfs[w]/=wes[w]
    v1d = sp.diag(cfs).copy()
    wv1d = sp.diag(wes).copy()
    nv1d = sp.diag(nbs).copy()
    cor = cfs
    norm = sp.sqrt(v1d*v1d[:,None])
    w = norm>0
    cor[w]/=norm[w]

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

    w=nc1d>0
    c1d[w]/=nc1d[w]

    print "writing"

    out = fitsio.FITS(args.out,'rw',clobber=True)
    head = {}
    head['LLMAX']=forest.lmax
    head['LLMIN']=forest.lmin
    head['DLL']=forest.dll

    out.write([v1d,wv1d,nv1d,c1d,nc1d,nb1d],names=['v1d','wv1d','nv1d','c1d','nc1d','nb1d'],header=head)
    out.close()

    print "all done"

    
