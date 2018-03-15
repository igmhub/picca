#!/usr/bin/env python

import scipy as sp
import fitsio
import argparse
import glob
import healpy
import sys
from scipy import random
import traceback

from picca import constants
from picca import cf
from picca.data import delta
from picca.data import forest

from multiprocessing import Pool,Process,Lock,Manager,cpu_count,Value


def cf1d(p):
    try:
        if x_correlation:
            tmp = cf.x_forest_cf1d(p)
        else :
            tmp = cf.cf1d(p)
    except:
        traceback.print_exc()
    with cf.lock:
        cf.counter.value += 1
    sys.stderr.write("\rcomputing xi: {}%".format(round(cf.counter.value*100./cf.npix/2.,2)))
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

    parser.add_argument('--lambda-min',type = float,default=3600.,required=False,
            help='lower limit on observed wavelength [Angstrom]')

    parser.add_argument('--lambda-max',type = float,default=5500.,required=False,
            help='upper limit on observed wavelength [Angstrom]')

    parser.add_argument('--dll', type=float,default=3.e-4, required=False,
                    help = 'loglam bin size')

    args = parser.parse_args()

    if args.nproc is None:
        args.nproc = cpu_count()//2

    forest.lmin = sp.log10(args.lambda_min)
    forest.lmax = sp.log10(args.lambda_max)
    forest.dll = args.dll
    n1d = int((forest.lmax-forest.lmin)/forest.dll+1)
    cf.n1d = n1d

    if (len(args.in_dir)>8) and (args.in_dir[-8:]==".fits.gz"):
        fi = glob.glob(args.in_dir)
    else:
        fi = glob.glob(args.in_dir+"/*.fits.gz")
    fi = sorted(fi)
    data = {}
    ndata = 0
    for i,f in enumerate(fi):
        if i%1==0:
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
    cf.ndata = ndata

    x_correlation=False
    if args.in_dir2:
        x_correlation=True
        if (len(args.in_dir2)>8) and (args.in_dir2[-8:]==".fits.gz"):
            fi = glob.glob(args.in_dir2)
        else:
            fi = glob.glob(args.in_dir2+"/*.fits.gz")
        fi = sorted(fi)
        data2 = {}
        ndata2 = 0
        dels2=[]
        for i,f in enumerate(fi):
            if i%1==0:
                sys.stderr.write("\rread {} of {} {}".format(i,len(fi),ndata2))
            hdus = fitsio.FITS(f)
            dels2 = [delta.from_fitsio(h) for h in hdus[1:]]
            ndata2+=len(dels2)
            for d in dels2:
                p = ndata2%args.nproc
                if not p in data2:
                    data2[p]=[]
                data2[p].append(d)
                if not args.no_project:
                    d.project()
            if args.nspec:
                if ndata2>args.nspec:break
    print("done")

    if x_correlation:
        cf.data2  = data2
        cf.ndata2 = ndata2

    cf.counter = Value('i',0)

    cf.lock = Lock()
    pool = Pool(processes=args.nproc)

    if x_correlation:
        keys = []
        for i in list(data.keys()):
            if i in list(data2.keys()):
                keys.append(i)
        cfs = pool.map(cf1d,sorted(keys))
    else: cfs = pool.map(cf1d,sorted(list(data.keys())))

    pool.close()

    cfs=sp.array(cfs)
    wes=cfs[:,0,:]
    nbs=cfs[:,2,:]
    cfs=cfs[:,1,:]
    wes = sp.array(wes)
    cfs = sp.array(cfs)
    nbs = sp.array(nbs).astype(sp.int64)

    print("multiplying")
    cfs *= wes
    cfs = cfs.sum(axis=0)
    wes = wes.sum(axis=0)
    nbs = nbs.sum(axis=0)

    print("done")

    cfs = cfs.reshape(n1d,n1d)
    wes = wes.reshape(n1d,n1d)
    nbs = nbs.reshape(n1d,n1d)

    print("rebinning")

    w = wes>0
    cfs[w]/=wes[w]

    ### Make copies of the 2D arrays that will be saved in the output file
    cfs_2d = cfs.copy()
    wes_2d = wes.copy()
    nbs_2d = nbs.copy()

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

    print("writing")

    out = fitsio.FITS(args.out,'rw',clobber=True)
    head = {}
    head['LLMAX']=forest.lmax
    head['LLMIN']=forest.lmin
    head['DLL']=forest.dll

    out.write([v1d,wv1d,nv1d,c1d,nc1d,nb1d],names=['v1d','wv1d','nv1d','c1d','nc1d','nb1d'],header=head)
    out.write([cfs_2d,wes_2d,nbs_2d],names=['DA','WE','NB'])
    out.close()

    print("all done")

    
