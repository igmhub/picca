#!/usr/bin/env python

import scipy as sp
import fitsio
import argparse
import glob
import healpy
import sys
from scipy import random 
import copy 

from picca import constants
from picca import cf
from picca import io
from picca.data import delta

from multiprocessing import Pool,Process,Lock,Manager,cpu_count,Value


def corr_func(p):
    if cf.x_correlation: 
        cf.fill_neighs_x_correlation(p)
    else: 
        cf.fill_neighs(p)
    tmp = cf.cf(p)
    return tmp

if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--out', type = str, default = None, required=True,
                        help = 'output file name')

    parser.add_argument('--in-dir', type = str, default = None, required=True,
                        help = 'data directory')

    parser.add_argument('--in-dir2', type = str, default = None, required=False,
                        help = 'data directory #2, for forest x-correlation')

    parser.add_argument('--wr-min', type = float, default = 1., required=False,
                        help = 'min of wavelength ratio')

    parser.add_argument('--wr-max', type = float, default = 1.1, required=False,
                        help = 'max of wavelength ratio')

    parser.add_argument('--ang-max', type = float, default = 0.02, required=False,
                        help = 'max angle')

    parser.add_argument('--np', type = int, default = 50, required=False,
                        help = 'number of parallel bins')

    parser.add_argument('--nt', type = int, default = 50, required=False,
                        help = 'number of transverse bins')

    parser.add_argument('--lambda-abs', type = float, default = constants.absorber_IGM['LYA'], required=False,
                        help = 'wavelength of absorption [Angstrom]')

    parser.add_argument('--lambda-abs2', type = float, default = constants.absorber_IGM['LYA'], required=False,
                        help = 'wavelength of absorption in forest 2 [Angstrom]')

    parser.add_argument('--nside', type = int, default = 16, required=False,
                    help = 'healpix nside')

    parser.add_argument('--nproc', type = int, default = None, required=False,
                    help = 'number of processors')

    parser.add_argument('--z-ref', type = float, default = 2.25, required=False,
                    help = 'reference redshift')

    parser.add_argument('--z-evol', type = float, default = 2.9, required=False,
                    help = 'exponent of the redshift evolution of the delta field')

    parser.add_argument('--z-evol2', type = float, default = 2.9, required=False,
                    help = 'exponent of the redshift evolution of the 2nd delta field')

    parser.add_argument('--no-project', action="store_true", required=False,
                    help = 'do not project out continuum fitting modes')

    parser.add_argument('--from-image', action="store_true", required=False,
                    help = 'use image format to read deltas')

    parser.add_argument('--nspec', type=int,default=None, required=False,
                    help = 'maximum spectra to read')

    args = parser.parse_args()

    if args.nproc is None:
        args.nproc = cpu_count()/2

    cf.rp_min          = args.wr_min 
    cf.rp_max          = args.wr_max
    cf.rt_max          = args.ang_max
    cf.np              = args.np
    cf.nt              = args.nt
    cf.nside           = args.nside
    cf.zref            = args.z_ref
    cf.alpha           = args.z_evol
    cf.x_correlation   = False
    cf.ang_correlation = True
    cf.angmax          = args.ang_max
    
    ### Read data 1
    data, ndata, zmin_pix, zmax_pix = io.read_deltas(args.in_dir, args.nside, args.lambda_abs,args.z_evol, args.z_ref, cosmo=None,nspec=args.nspec,no_project=args.no_project)
    cf.npix  = len(data)
    cf.data  = data
    cf.ndata = ndata
    sys.stderr.write("\n")
    print("done, npix = {}".format(cf.npix))
    
    ### Read data 2
    if args.in_dir2:
        cf.x_correlation = True
        data2, ndata2, zmin_pix2, zmax_pix2 = io.read_deltas(args.in_dir2, args.nside, args.lambda_abs2,args.z_evol2, args.z_ref, cosmo=None,nspec=args.nspec,no_project=args.no_project)
        cf.data2  = data2
        cf.ndata2 = ndata2 
        sys.stderr.write("\n") 
        print("done, npix = {}".format(len(data2)))
    elif args.lambda_abs != args.lambda_abs2:
        cf.x_correlation = True
        cf.data2  = copy.deepcopy(data)
        cf.ndata2 = copy.deepcopy(ndata)




    ### Send
    cf.counter = Value('i',0)

    cf.lock = Lock()
    cpu_data = {}
    for p in list(data.keys()):
        cpu_data[p] = [p]

    pool = Pool(processes=args.nproc)

    cfs = pool.map(corr_func,sorted(cpu_data.values()))
    pool.close()
    
    
    
    
    
    ### Store
    cfs=sp.array(cfs)
    wes=cfs[:,0,:]
    rps=cfs[:,2,:]
    rts=cfs[:,3,:]
    zs=cfs[:,4,:]
    nbs=cfs[:,5,:].astype(sp.int64)
    cfs=cfs[:,1,:]
    hep=sp.array(sorted(list(cpu_data.keys())))

    cut      = (wes.sum(axis=0)>0.)
    rp       = (rps*wes).sum(axis=0)
    rp[cut] /= wes.sum(axis=0)[cut]
    rt       = (rts*wes).sum(axis=0)
    rt[cut] /= wes.sum(axis=0)[cut]
    z        = (zs*wes).sum(axis=0)
    z[cut]  /= wes.sum(axis=0)[cut]
    nb       = nbs.sum(axis=0)

    out = fitsio.FITS(args.out,'rw',clobber=True)
    head = {}
    head['RPMIN']=cf.rp_min
    head['RPMAX']=cf.rp_max
    head['RTMAX']=cf.rt_max
    head['NT']=cf.nt
    head['NP']=cf.np

    out.write([rp,rt,z,nb],names=['RP','RT','Z','NB'],header=head)
    ## use the default scheme in healpy => RING
    head2 = [{'name':'HLPXSCHM','value':'RING','comment':'healpix scheme'}]
    out.write([hep,wes,cfs],names=['HEALPID','WE','DA'],header=head2)
    out.close()

    
