#!/usr/bin/env python

import scipy as sp
import fitsio
import argparse
import glob
import healpy
import sys
from functools import partial
import copy 

from scipy import random 
from scipy.interpolate import interp1d

from picca import constants
from picca import xcf
from picca import io

from multiprocessing import Pool,Process,Lock,Manager,cpu_count,Value

def calc_metal_xdmat(abs_igm,p):
    xcf.fill_neighs(p)
    tmp = xcf.metal_dmat(p,abs_igm=abs_igm)
    return tmp

if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--out', type = str, default = None, required=True,
                        help = 'output file name')

    parser.add_argument('--drq', type = str, default = None, required=True,
                        help = 'drq')

    parser.add_argument('--in-dir', type = str, default = None, required=True,
                        help = 'data directory')

    parser.add_argument('--rp-max', type = float, default = 200., required=False,
                        help = 'max rp [h^-1 Mpc]')

    parser.add_argument('--rp-min', type = float, default = -200., required=False,
                        help = 'min rp [h^-1 Mpc]')

    parser.add_argument('--rt-max', type = float, default = 200., required=False,
                        help = 'max rt [h^-1 Mpc]')

    parser.add_argument('--np', type = int, default = 100, required=False,
                        help = 'number of r-parallel bins')

    parser.add_argument('--nt', type = int, default = 50, required=False,
                        help = 'number of r-transverse bins')

    parser.add_argument('--lambda-abs', type = float, default = constants.absorber_IGM['LYA'], required=False,
                        help = 'wavelength of absorption [Angstrom]')

    parser.add_argument('--lambda-abs-name', type = str, default = 'LYA', required=False,
                        help = 'name of the absorption transistion')

    parser.add_argument('--obj-name', type = str, default = 'QSO', required=False,
                        help = 'name of the object tracer')

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

    parser.add_argument('--z-evol-del', type = float, default = 2.9, required=False,
                    help = 'exponent of the redshift evolution of the delta field')

    parser.add_argument('--z-evol-obj', type = float, default = 1., required=False,
                    help = 'exponent of the redshift evolution of the object field')

    parser.add_argument('--z-min-obj', type = float, default = None, required=False,
                    help = 'min redshift for object field')

    parser.add_argument('--z-max-obj', type = float, default = None, required=False,
                        help = 'max redshift for object field')

    parser.add_argument('--nspec', type=int,default=None, required=False,
                    help = 'maximum spectra to read')

    parser.add_argument('--abs-igm', type=str,default=None, required=False,nargs="*",
                    help = 'list of metals')

    args = parser.parse_args()

    if args.nproc is None:
        args.nproc = cpu_count()/2

    xcf.rp_max = args.rp_max
    xcf.rp_min = args.rp_min
    xcf.rt_max = args.rt_max
    xcf.np = args.np
    xcf.nt = args.nt
    xcf.nside = args.nside
    xcf.zref = args.z_ref
    xcf.lambda_abs = args.lambda_abs
    xcf.rej = args.rej

    ## use a metal grid equal to the lya grid
    xcf.npm = args.np
    xcf.ntm = args.nt

    cosmo = constants.cosmo(args.fid_Om)
    xcf.cosmo=cosmo

    dels, ndels, zmin_pix, zmax_pix = io.read_deltas(args.in_dir, args.nside, args.lambda_abs,\
                            args.z_evol_del, args.z_ref, cosmo,nspec=args.nspec)

    xcf.npix = len(dels)
    xcf.dels = dels
    xcf.ndels = ndels

    
    ### Find the redshift range
    if (args.z_min_obj is None):
        dmin_pix = cosmo.r_comoving(zmin_pix)
        dmin_obj = max(0.,dmin_pix+xcf.rp_min)
        args.z_min_obj = cosmo.r_2_z(dmin_obj)
        sys.stderr.write("\r z_min_obj = {}\r".format(args.z_min_obj))
    if (args.z_max_obj is None):
        dmax_pix = cosmo.r_comoving(zmax_pix)
        dmax_obj = max(0.,dmax_pix+xcf.rp_max)
        args.z_max_obj = cosmo.r_2_z(dmax_obj)
        sys.stderr.write("\r z_max_obj = {}\r".format(args.z_max_obj))

    objs,zmin_obj = io.read_objects(args.drq, args.nside, args.z_min_obj, args.z_max_obj,\
                                args.z_evol_obj, args.z_ref,cosmo)
    xcf.objs = objs

    xcf.angmax = 2*sp.arcsin(xcf.rt_max/(cosmo.r_comoving(zmin_pix)+cosmo.r_comoving(zmin_obj)))

    xcf.counter = Value('i',0)
    xcf.lock = Lock()
    
    cpu_data = {}
    for i,p in enumerate(dels.keys()):
        ip = i%args.nproc
        if not ip in cpu_data:
            cpu_data[ip] = []
        cpu_data[ip].append(p)

    random.seed(0)

    dm_all=[]
    wdm_all=[]
    rp_all=[]
    rt_all=[]
    z_all=[]
    names=[]
    npairs_all=[]
    npairs_used_all=[]
 

    for i,abs_igm in enumerate(args.abs_igm):
        xcf.counter.value=0
        f=partial(calc_metal_xdmat,abs_igm)
        sys.stderr.write("\n")
        pool = Pool(processes=args.nproc)
        dm = pool.map(f,sorted(cpu_data.values()))
        pool.close()
        dm = sp.array(dm)
        wdm =dm[:,0].sum(axis=0)
        rp = dm[:,2].sum(axis=0)
        rt = dm[:,3].sum(axis=0)
        z = dm[:,4].sum(axis=0)
        we = dm[:,5].sum(axis=0)
        w=we>0
        rp[w]/=we[w]
        rt[w]/=we[w]
        z[w]/=we[w]
        npairs=dm[:,6].sum(axis=0)
        npairs_used=dm[:,7].sum(axis=0)
        dm=dm[:,1].sum(axis=0)
        w=wdm>0
        dm[w,:]/=wdm[w,None]

        dm_all.append(dm)
        wdm_all.append(wdm)
        rp_all.append(rp)
        rt_all.append(rt)
        z_all.append(z)
        names.append(abs_igm)

        npairs_all.append(npairs)
        npairs_used_all.append(npairs_used)

    out = fitsio.FITS(args.out,'rw',clobber=True)
    head = {}
    head['REJ']=args.rej
    head['RPMAX']=xcf.rp_max
    head['RPMIN']=xcf.rp_min
    head['RTMAX']=xcf.rt_max
    head['NT']=xcf.nt
    head['NP']=xcf.np

    out.write([sp.array(npairs_all),sp.array(npairs_used_all),sp.array(names)],names=["NPALL","NPUSED","ABS_IGM"],header=head)

    out_list = []
    out_names=[]
    for i,ai in enumerate(names):
        out_names=out_names + ["RP_"+args.obj_name+"_"+ai]
        out_list = out_list + [rp_all[i]]

        out_names=out_names + ["RT_"+args.obj_name+"_"+ai]
        out_list = out_list + [rt_all[i]]

        out_names=out_names + ["Z_"+args.obj_name+"_"+ai]
        out_list = out_list + [z_all[i]]

        out_names = out_names + ["DM_"+args.obj_name+"_"+ai]
        out_list = out_list + [dm_all[i]]

        out_names=out_names+["WDM_"+args.obj_name+"_"+ai]
        out_list = out_list+[wdm_all[i]]

    out.write(out_list,names=out_names)
    out.close()

    
