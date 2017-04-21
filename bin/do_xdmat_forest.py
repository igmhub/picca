#!/usr/bin/env python

import scipy as sp
import fitsio
import argparse
import glob
import healpy
import sys
from scipy import random 
from scipy.interpolate import interp1d
import copy 
from pylya import constants
from pylya import xcf_forest
from pylya.data import delta

from multiprocessing import Pool,Process,Lock,Manager,cpu_count,Value


def calc_dmat(p):
    xcf_forest.fill_neighs(p)
    tmp = xcf_forest.dmat(p)
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

    parser.add_argument('--nside', type = int, default = 8, required=False,
                    help = 'healpix nside')

    parser.add_argument('--nproc', type = int, default = None, required=False,
                    help = 'number of processors')

    parser.add_argument('--z-ref', type = float, default = 2.25, required=False,
                    help = 'reference redshift')

    parser.add_argument('--rej', type = float, default = 1., required=False,
                    help = 'reference redshift')

    parser.add_argument('--z-evol', type = float, default = 2.9, required=False,
                    help = 'exponent of the redshift evolution of the delta field')

    parser.add_argument('--nspec', type=int,default=None, required=False,
                    help = 'maximum spectra to read')

    parser.add_argument('--no-project', action="store_true", required=False,
                    help = 'do not project out continuum fitting modes')

    args = parser.parse_args()

    if args.nproc is None:
        args.nproc = cpu_count()/2

    print "nproc",args.nproc

    xcf_forest.rp_max = args.rp_max
    xcf_forest.rt_max = args.rt_max
    xcf_forest.np     = args.np
    xcf_forest.nt     = args.nt
    xcf_forest.nside  = args.nside
    xcf_forest.zref   = args.z_ref
    xcf_forest.alpha  = args.z_evol   
    xcf_forest.rej = args.rej

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

    z_min_pix = 1.e6

    print 'opening dir1 = {}'.format(args.in_dir1)
    fi = glob.glob(args.in_dir1+"/*.fits.gz")
    data1 = {}
    ndata1 = 0
    for i,f in enumerate(fi):
        sys.stderr.write("\rread {} of {} {}".format(i,len(fi),ndata1))
        hdus = fitsio.FITS(f)
        dels = [delta.from_fitsio(h) for h in hdus[1:]]
        ndata1+=len(dels)
        phi1 = [d.ra for d in dels]
        th1 = [sp.pi/2-d.dec for d in dels]
        pix1 = healpy.ang2pix(xcf_forest.nside,th1,phi1)
        for d,p in zip(dels,pix1):
            if not p in data1:
                data1[p]=[]
            data1[p].append(d)

            z = 10**d.ll/lambda_abs1-1.
            z_min_pix1 = sp.amin( sp.append([z_min_pix],z) )
            d.r_comov = cosmo.r_comoving(z)
            d.we *= ((1+z)/(1+args.z_ref))**(xcf_forest.alpha-1)
            if not args.no_project:
                d.project()
        if not args.nspec is None:
            if ndata1>args.nspec:break
    sys.stderr.write("\n")
    
    if dir_name2==dir_name1: 
        data2  = copy.deepcopy(data1)
        ndata2 = copy.deepcopy(ndata1)
        z_min_pix2 = copy.deepcopy(z_min_pix1)
    else: 
        print 'opening dir2 = {}'.format(args.in_dir2)
        fi = glob.glob(args.in_dir2+"/*.fits.gz")
        data2 = {}
        ndata2 = 0
        for i,f in enumerate(fi):
            sys.stderr.write("\rread {} of {} {}".format(i,len(fi),ndata2))
            hdus = fitsio.FITS(f)
            dels = [delta.from_fitsio(h) for h in hdus[1:]]
            ndata2+=len(dels)
            phi2 = [d.ra for d in dels]
            th2 = [sp.pi/2-d.dec for d in dels]
            pix2 = healpy.ang2pix(xcf_forest.nside,th2,phi2)
            for d,p in zip(dels,pix2):
                if not p in data2:
                    data2[p]=[]
                data2[p].append(d)

                z = 10**d.ll/lambda_abs2-1.
                z_min_pix2 = sp.amin( sp.append([z_min_pix],z) )
                d.r_comov = cosmo.r_comoving(z)
                d.we *= ((1+z)/(1+args.z_ref))**(xcf_forest.alpha-1)
                if not args.no_project:
                    d.project()
            if not args.nspec is None:
                if ndata2>args.nspec:break
        sys.stderr.write("\n")


  
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
    print "done, npix2 = {}".format(xcf_forest.npix2)
    xcf_forest.counter = Value('i',0)

    xcf_forest.lock = Lock()
    
    cpu_data1 = {}
    for i,p in enumerate(data1.keys()):
        ip = i%args.nproc
        if not ip in cpu_data1:
            cpu_data1[ip] = []
        cpu_data1[ip].append(p)

    random.seed(0)
    pool = Pool(processes=args.nproc)
    dm = pool.map(calc_dmat,cpu_data1.values())
    pool.close()

    dm = sp.array(dm)
    wdm =dm[:,0].sum(axis=0)
    npairs=dm[:,2].sum(axis=0)
    npairs_used=dm[:,3].sum(axis=0)
    dm=dm[:,1].sum(axis=0)

    dm*=(wdm !=0)/(wdm+(wdm ==0))


    out = fitsio.FITS(args.out,'rw',clobber=True)
    head = {}
    head['REJ']=args.rej
    head['RPMAX']=xcf_forest.rp_max
    head['RTMAX']=xcf_forest.rt_max
    head['NT']=xcf_forest.nt
    head['NP']=xcf_forest.np
    head['NPROR']=npairs
    head['NPUSED']=npairs_used

    out.write([wdm,dm],names=['WDM','DM'],header=head)
    out.close()

    
