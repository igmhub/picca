#!/usr/bin/env python

import scipy as sp
import fitsio
import argparse
import sys
import traceback
from multiprocessing import Pool,Lock,cpu_count,Value

from picca import constants, cf, io


def cf1d(p):
    try:
        if cf.x_correlation:
            tmp = cf.x_forest_cf1d(p)
        else :
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

    parser.add_argument('--lambda-abs', type = str, default = 'LYA', required=False,
                        help = 'name of the absorption in picca.constants')

    parser.add_argument('--lambda-abs2', type = str, default = None, required=False,
                        help = 'name of the 2nd absorption in picca.constants')

    parser.add_argument('--z-ref', type = float, default = 2.25, required=False,
                    help = 'reference redshift')

    parser.add_argument('--z-evol', type = float, default = 1., required=False,
                    help = 'exponent of the redshift evolution of the delta field')

    parser.add_argument('--z-evol2', type = float, default = 1., required=False,
                    help = 'exponent of the redshift evolution of the 2nd delta field')

    args = parser.parse_args()

    if args.nproc is None:
        args.nproc = cpu_count()//2


    ###
    cf.nside = 16
    cf.lmin = sp.log10(args.lambda_min)
    cf.lmax = sp.log10(args.lambda_max)
    cf.dll = args.dll
    cf.n1d = int((cf.lmax-cf.lmin)/cf.dll+1)
    cf.x_correlation = False

    cf.lambda_abs = constants.absorber_IGM[args.lambda_abs]
    if args.lambda_abs2:
        cf.lambda_abs2 = constants.absorber_IGM[args.lambda_abs2]
    else:
        cf.lambda_abs2 = constants.absorber_IGM[args.lambda_abs]


    ### Read data 1
    data, ndata, zmin_pix, zmax_pix = io.read_deltas(args.in_dir, cf.nside, cf.lambda_abs,args.z_evol, args.z_ref, cosmo=None,nspec=args.nspec,no_project=args.no_project)
    cf.npix  = len(data)
    cf.data  = data
    cf.ndata = ndata
    sys.stderr.write("\n")
    print("done, npix = {}\n".format(cf.npix))

    ### Read data 2
    if args.in_dir2:
        cf.x_correlation = True
        data2, ndata2, zmin_pix2, zmax_pix2 = io.read_deltas(args.in_dir2, cf.nside, cf.lambda_abs2,args.z_evol2, args.z_ref, cosmo=None,nspec=args.nspec,no_project=args.no_project)
        cf.data2  = data2
        cf.ndata2 = ndata2
        sys.stderr.write("\n")
        print("done, npix = {}\n".format(len(data2)))
    elif cf.lambda_abs != cf.lambda_abs2:
        cf.x_correlation = True
        data2, ndata2, zmin_pix2, zmax_pix2 = io.read_deltas(args.in_dir, cf.nside, cf.lambda_abs2,args.z_evol2, args.z_ref, cosmo=None,nspec=args.nspec,no_project=args.no_project)
        cf.data2  = data2
        cf.ndata2 = ndata2


    ###
    cf.counter = Value('i',0)
    cf.lock = Lock()
    pool = Pool(processes=args.nproc)

    if cf.x_correlation:
        keys = []
        for i in list(data.keys()):
            if i in list(data2.keys()):
                keys.append(i)
        cfs = pool.map(cf1d,sorted(keys))
    else: cfs = pool.map(cf1d,sorted(list(data.keys())))
    pool.close()
    print('\n')


    ###
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

    cfs = cfs.reshape(cf.n1d,cf.n1d)
    wes = wes.reshape(cf.n1d,cf.n1d)
    nbs = nbs.reshape(cf.n1d,cf.n1d)

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

    c1d = sp.zeros(cf.n1d)
    nc1d = sp.zeros(cf.n1d)
    nb1d = sp.zeros(cf.n1d,dtype=sp.int64)
    bins = sp.arange(cf.n1d)

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


    ###
    print("writing")

    out = fitsio.FITS(args.out,'rw',clobber=True)
    head = {}
    head['LLMIN'] = cf.lmin
    head['LLMAX'] = cf.lmax
    head['DLL'] = cf.dll

    out.write([v1d,wv1d,nv1d,c1d,nc1d,nb1d],names=['v1d','wv1d','nv1d','c1d','nc1d','nb1d'],header=head)
    out.write([cfs_2d,wes_2d,nbs_2d],names=['DA','WE','NB'])
    out.close()

    print("all done")
