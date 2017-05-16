#!/usr/bin/env python

import scipy as sp
import fitsio
import argparse
import glob
import healpy
import sys
from scipy import random 
import copy
from pylya import constants
from pylya import xcf_forest
from pylya.data import delta

from multiprocessing import Pool,Process,Lock,Manager,cpu_count,Value


def corr_func(p):
    xcf_forest.fill_neighs(p)
    tmp = xcf_forest.cf(p)
    return tmp

if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--out', type = str, default = None, required=True,
                        help = 'output file name')

    parser.add_argument('--in-dir1', type = str, default = None, required=True,
                        help = 'first delta directory')

    parser.add_argument('--in-dir2', type = str, default = None, required=False,
                        help = 'second delta directory')

    parser.add_argument('--line1', type = float, default = constants.lya, required=False,
                        help = 'type of absorption in forest 1')

    parser.add_argument('--line2', type = float, default = constants.lya, required=False,
                        help = 'type of absorption in forest 2')

    parser.add_argument('--rp-max', type = float, default = 200, required=False,
                        help = 'max rp')

    parser.add_argument('--rt-max', type = float, default = 200, required=False,
                        help = 'max rt')

    parser.add_argument('--np', type = int, default = 50, required=False,
                        help = 'number of r-parallel bins')

    parser.add_argument('--nt', type = int, default = 50, required=False,
                        help = 'number of r-transverse bins')

    parser.add_argument('--fid-Om', type = float, default = 0.315, required=False,
                    help = 'Om of fiducial cosmology')

    parser.add_argument('--nside', type = int, default = 16, required=False,
                    help = 'healpix nside')

    parser.add_argument('--nproc', type = int, default = None, required=False,
                    help = 'number of processors')

    parser.add_argument('--z-ref', type = float, default = 2.25, required=False,
                    help = 'reference redshift')

    parser.add_argument('--z-evol', type = float, default = 2.9, required=False,
                    help = 'exponent of the redshift evolution of the delta field')

    parser.add_argument('--no-project', action="store_true", required=False,
                    help = 'do not project out continuum fitting modes')

    parser.add_argument('--order_0_1', action="store_true", required=False,
                    help = 'For lyA/lyB cross-correlation, deltas #1 have been comptued with a continuum with zero-order polynomial in log(lambda)')

    parser.add_argument('--order_0_2', action="store_true", required=False,
                    help = 'For lyA/lyB cross-correlation, deltas #2 have been comptued with a continuum with zero-order polynomial in log(lambda)')

    parser.add_argument('--from-image', action="store_true", required=False,
                    help = 'use image format to read deltas')

    parser.add_argument('--nspec', type=int,default=None, required=False,
                    help = 'maximum spectra to read')

    args = parser.parse_args()

    if args.nproc is None:
        args.nproc = cpu_count()/2

    xcf_forest.rp_max = args.rp_max
    xcf_forest.rt_max = args.rt_max
    xcf_forest.np     = args.np
    xcf_forest.nt     = args.nt
    xcf_forest.nside  = args.nside
    xcf_forest.zref   = args.z_ref
    xcf_forest.alpha  = args.z_evol    

    if args.line1 is None:
        lambda_abs1=constants.lya
    else :  
        lambda_abs1=args.line1

    print 'lambda_abs1 = ', lambda_abs1

    if args.line2 is None:
        lambda_abs2=constants.lya
    else :  
        lambda_abs2=args.line2

    print 'lambda_abs2 = ', lambda_abs2

    cosmo = constants.cosmo(args.fid_Om)

    dir_name1 = args.in_dir1
    if args.in_dir2 is None:
        dir_name2 = dir_name1
    else: 
        dir_name2 = args.in_dir2
    
    print "dir_name1 = {}".format(dir_name1)
    print "dir_name2 = {}".format(dir_name2)
    print 

    print 'opening ',dir_name1
    data1  = {}
    ndata1 = 0
    dels1  = []
    if not args.from_image:
        fi = glob.glob(dir_name1+"/*.fits.gz")
        for i,f in enumerate(fi):
            sys.stderr.write("\rread {} of {} {}".format(i,len(fi),ndata1))
            hdus = fitsio.FITS(f)
            dels1 += [delta.from_fitsio(h) for h in hdus[1:]]
            ndata1+=len(hdus[1:])
            hdus.close()
            if not args.nspec is None:
                if ndata1>args.nspec:break
    else:
        dels1 = delta.from_image(in_dir1)
    print 'done with ',dir_name1

    if dir_name2==dir_name1: 
        data2  = copy.deepcopy(data1)
        ndata2 = copy.deepcopy(ndata1)
        dels2  = copy.deepcopy(dels1)
    else: 
        print 'opening ',dir_name2
        data2  = {}
        ndata2 = 0
        dels2  = []
        if not args.from_image:
            fi = glob.glob(dir_name2+"/*.fits.gz")
            for i,f in enumerate(fi):
                sys.stderr.write("\rread {} of {} {}".format(i,len(fi),ndata2))
                hdus = fitsio.FITS(f)
                dels2 += [delta.from_fitsio(h) for h in hdus[1:]]
                ndata2+=len(hdus[1:])
                hdus.close()
                if not args.nspec is None:
                    if ndata2>args.nspec:break
        else:
            dels2 = delta.from_image(in_dir2)
        print 'done with ',dir_name2


    z_min_pix1 = 10**dels1[0].ll[0]/lambda_abs1-1
    phi1 = [d.ra for d in dels1]
    th1 = [sp.pi/2-d.dec for d in dels1]
    pix1 = healpy.ang2pix(xcf_forest.nside,th1,phi1)

    for d,p in zip(dels1,pix1):
        if not p in data1:
            data1[p]=[]
        data1[p].append(d)

        z = 10**d.ll/lambda_abs1-1
        z_min_pix1 = sp.amin( sp.append([z_min_pix1],z) )
        d.z = z
        d.r_comov = cosmo.r_comoving(z)
        d.we *= ((1+z)/(1+args.z_ref))**(xcf_forest.alpha-1)
        if not args.no_project:
            if args.order_0_1: 
                d.project_0()
            else:
                d.project()

    z_min_pix2 = 10**dels2[0].ll[0]/lambda_abs2-1
    phi2 = [d.ra for d in dels2]
    th2 = [sp.pi/2-d.dec for d in dels2]
    pix2 = healpy.ang2pix(xcf_forest.nside,th2,phi2)

    for d,p in zip(dels2,pix2):
        if not p in data2:
            data2[p]=[]
        data2[p].append(d)

        z = 10**d.ll/lambda_abs2-1
        z_min_pix2 = sp.amin( sp.append([z_min_pix2],z) )
        d.z = z
        d.r_comov = cosmo.r_comoving(z)
        d.we *= ((1+z)/(1+args.z_ref))**(xcf_forest.alpha-1)
        if not args.no_project:
            if args.order_0_2: 
                d.project_0()
            else:
                d.project()
    
    
    z_min_pix=sp.amin(sp.append(z_min_pix1,z_min_pix2))
    xcf_forest.angmax = 2.*sp.arcsin(xcf_forest.rt_max/(2.*cosmo.r_comoving(z_min_pix)))

    sys.stderr.write("\n")
    
    xcf_forest.npix1 = len(data1)
    xcf_forest.data1 = data1
    xcf_forest.ndata1=ndata1
    print "done, npix1 = {}".format(xcf_forest.npix1)

    xcf_forest.npix2 = len(data2)
    xcf_forest.data2 = data2
    xcf_forest.ndata2=ndata2
    print "done, npix1 = {}".format(xcf_forest.npix2)
    
    xcf_forest.counter = Value('i',0)

    xcf_forest.lock = Lock()

    cpu_data1 = {}
    for p in data1.keys():
        cpu_data1[p] = [p]

    pool = Pool(processes=args.nproc)
    cfs = pool.map(corr_func,cpu_data1.values())
    pool.close()

    cfs=sp.array(cfs)
    wes=cfs[:,0,:]
    rps=cfs[:,2,:]
    rts=cfs[:,3,:]
    zs=cfs[:,4,:]
    cfs=cfs[:,1,:]

    cut      = (wes.sum(axis=0)>0.)
    rp       = (rps*wes).sum(axis=0)
    rp[cut] /= wes.sum(axis=0)[cut]
    rt       = (rts*wes).sum(axis=0)
    rt[cut] /= wes.sum(axis=0)[cut]
    z        = (zs*wes).sum(axis=0)
    z[cut]  /= wes.sum(axis=0)[cut]

    out = fitsio.FITS(args.out,'rw',clobber=True)
    head = {}
    head['RPMAX']=xcf_forest.rp_max
    head['RTMAX']=xcf_forest.rt_max
    head['NT']=xcf_forest.nt
    head['NP']=xcf_forest.np

    out.write([rp,rt,z],names=['RP','RT','Z'],header=head)
    out.write([wes,cfs],names=['WE','DA'])
    out.close()
