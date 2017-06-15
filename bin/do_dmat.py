#!/usr/bin/env python

import scipy as sp
import fitsio
import argparse
import glob
import healpy
import sys
import copy
from scipy import random 
from scipy.interpolate import interp1d

from pylya import constants
from pylya import cf
from pylya.data import delta

from multiprocessing import Pool,Process,Lock,Manager,cpu_count,Value


def calc_dmat(p):
    if x_correlation: 
        cf.fill_neighs_x_correlation(p)
    else: 
        cf.fill_neighs(p)
    if args.neg_rp: 
        tmp = cf.neg_rp_dmat(p)
    else: 
        tmp = cf.dmat(p)

    return tmp

if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--out', type = str, default = None, required=True,
                        help = 'output file name')

    parser.add_argument('--in-dir', type = str, default = None, required=True,
                        help = 'data directory')

    parser.add_argument('--in-dir2', type = str, default = None, required=False,
                        help = 'second delta directory')

    parser.add_argument('--rp-max', type = float, default = 200, required=False,
                        help = 'max rp')

    parser.add_argument('--rt-max', type = float, default = 200, required=False,
                        help = 'max rt')

    parser.add_argument('--np', type = int, default = 50, required=False,
                        help = 'number of r-parallel bins')

    parser.add_argument('--nt', type = int, default = 50, required=False,
                        help = 'number of r-transverse bins')

    parser.add_argument('--lambda-abs', type = float, default = constants.lya, required=False,
                        help = 'wavelength of absorption')

    parser.add_argument('--lambda-abs2', type = float, default = constants.lya, required=False,
                        help = 'wavelength of absorption in forest 2')

    parser.add_argument('--fid-Om', type = float, default = 0.315, required=False,
                    help = 'Om of fiducial cosmology')

    parser.add_argument('--nside', type = int, default = 8, required=False,
                    help = 'healpix nside')

    parser.add_argument('--nproc', type = int, default = None, required=False,
                    help = 'number of processors')

    parser.add_argument('--z-ref', type = float, default = 2.25, required=False,
                    help = 'reference redshift')

    parser.add_argument('--rej', type = float, default = .95, required=False,
                    help = 'reference redshift')

    parser.add_argument('--z-evol', type = float, default = 2.9, required=False,
                    help = 'exponent of the redshift evolution of the delta field')

    parser.add_argument('--nspec', type=int,default=None, required=False,
                    help = 'maximum spectra to read')

    parser.add_argument('--no-project', action="store_true", required=False,
                    help = 'do not project out continuum fitting modes')

    parser.add_argument('--neg-rp', action="store_true", required=False,
                         help = 'Compute the cf for rp in [-rp_max,rp_rmax]')
    args = parser.parse_args()

    if args.nproc is None:
        args.nproc = cpu_count()/2

    print "nproc",args.nproc

    cf.rp_max = args.rp_max
    cf.rt_max = args.rt_max
    
    cf.np = args.np
    cf.nt = args.nt
    cf.nside = args.nside
    cf.zref = args.z_ref
    cf.alpha = args.z_evol
    cf.lambda_abs = args.lambda_abs
    cf.rej = args.rej

    cosmo = constants.cosmo(args.fid_Om)

    z_min_pix = 1.e6

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

    x_correlation=False
    if args.in_dir2: 
        x_correlation=True
        data2 = {}
        ndata2 = 0
        dels2 = []
        fi = glob.glob(args.in_dir2+"/*.fits.gz")
        for i,f in enumerate(fi):
            sys.stderr.write("\rread {} of {} {}".format(i,len(fi),ndata))
            hdus = fitsio.FITS(f)
            dels2 += [delta.from_fitsio(h) for h in hdus[1:]]
            ndata2+=len(hdus[1:])
            hdus.close()
            if not args.nspec is None:
                if ndata2>args.nspec:break
    elif args.lambda_abs != args.lambda_abs2:   
        x_correlation=True
        data2  = copy.deepcopy(data)
        ndata2 = copy.deepcopy(ndata)
        dels2  = copy.deepcopy(dels)


    z_min_pix = 10**dels[0].ll[0]/args.lambda_abs-1
    phi = [d.ra for d in dels]
    th = [sp.pi/2-d.dec for d in dels]
    pix = healpy.ang2pix(cf.nside,th,phi)
    for d,p in zip(dels,pix):
        if not p in data:
            data[p]=[]
        data[p].append(d)

        z = 10**d.ll/args.lambda_abs-1
        z_min_pix = sp.amin( sp.append([z_min_pix],z) )
        d.z = z
        d.r_comov = cosmo.r_comoving(z)
        d.we *= ((1+z)/(1+args.z_ref))**(cf.alpha-1)
        if not args.no_project:
            d.project()
    
    if x_correlation: 
        z_min_pix2 = 10**dels2[0].ll[0]/args.lambda_abs2-1
        z_min_pix=sp.amin(sp.append(z_min_pix,z_min_pix2))
        phi2 = [d.ra for d in dels2]
        th2 = [sp.pi/2-d.dec for d in dels2]
        pix2 = healpy.ang2pix(cf.nside,th2,phi2)

        for d,p in zip(dels2,pix2):
            if not p in data2:
                data2[p]=[]
            data2[p].append(d)

            z = 10**d.ll/args.lambda_abs2-1
            z_min_pix2 = sp.amin(sp.append([z_min_pix2],z) )
            d.z = z
            d.r_comov = cosmo.r_comoving(z)
            d.we *= ((1+z)/(1+args.z_ref))**(cf.alpha-1)
            if not args.no_project:
                d.project()
            if not args.nspec is None:
                if ndata2>args.nspec:break

    sys.stderr.write("read {}\n".format(ndata))

    cf.angmax = 2.*sp.arcsin(cf.rt_max/(2.*cosmo.r_comoving(z_min_pix)))

    cf.npix = len(data)
    print "done, npix = {}".format(cf.npix)

    cf.data = data
    cf.ndata=ndata

    if x_correlation:
        cf.data2 = data2
        cf.ndata2=ndata2

    cf.counter = Value('i',0)

    cf.lock = Lock()
    
    cpu_data = {}
    for i,p in enumerate(data.keys()):
        ip = i%args.nproc
        if not ip in cpu_data:
            cpu_data[ip] = []
        cpu_data[ip].append(p)

    random.seed(0)
    pool = Pool(processes=args.nproc)
    dm = pool.map(calc_dmat,cpu_data.values())
    pool.close()

    dm = sp.array(dm)
    wdm =dm[:,0].sum(axis=0)
    npairs=dm[:,2].sum(axis=0)
    npairs_used=dm[:,3].sum(axis=0)
    dm=dm[:,1].sum(axis=0)

    w = wdm>0.
    dm[:,w] /= wdm[w]

    out = fitsio.FITS(args.out,'rw',clobber=True)
    head = {}
    head['REJ']=args.rej
    head['RPMAX']=cf.rp_max
    head['RTMAX']=cf.rt_max
    head['NT']=cf.nt
    head['NP']=cf.np
    head['NPROR']=npairs
    head['NPUSED']=npairs_used

    out.write([wdm,dm],names=['WDM','DM'],header=head)
    out.close()

    
