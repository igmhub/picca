#!/usr/bin/env python

import sys
import argparse
import fitsio
import scipy as sp
from scipy import random
from multiprocessing import Pool,Process,Lock,Manager,cpu_count,Value

from picca import constants, io, utils
import xcf

def calc_wickT(p):
    xcf.fill_neighs(p)
    tmp = xcf.wickT(p)
    return tmp

if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--out', type = str, default = None, required=True,
                    help = 'output file name')

    parser.add_argument('--in-dir', type = str, default = None, required=True,
                    help = 'data directory')

    parser.add_argument('--drq', type = str, default = None, required=True,
                    help = 'drq')

    parser.add_argument('--np', type = int, default = 100, required=False,
                    help = 'number of r-parallel bins')

    parser.add_argument('--nt', type = int, default = 50, required=False,
                    help = 'number of r-transverse bins')

    parser.add_argument('--rp-min', type = float, default = -200., required=False,
                    help = 'min rp [h^-1 Mpc]')

    parser.add_argument('--rp-max', type = float, default = 200., required=False,
                    help = 'max rp [h^-1 Mpc]')

    parser.add_argument('--rt-max', type = float, default = 200., required=False,
                    help = 'max rt [h^-1 Mpc]')

    parser.add_argument('--lambda-abs', type = str, default = 'LYA', required=False,
                    help = 'name of the absorption in picca.constants')

    parser.add_argument('--fid-Om', type = float, default = 0.315, required=False,
                    help = 'Om of fiducial cosmology')

    parser.add_argument('--nside', type = int, default = 16, required=False,
                    help = 'healpix nside')

    parser.add_argument('--nproc', type = int, default = None, required=False,
                    help = 'number of processors')

    parser.add_argument('--z-min-obj', type = float, default = None, required=False,
                    help = 'min redshift for object field')

    parser.add_argument('--z-max-obj', type = float, default = None, required=False,
                    help = 'max redshift for object field')

    parser.add_argument('--z-ref', type = float, default = 2.25, required=False,
                    help = 'reference redshift')

    parser.add_argument('--z-cut-min', type = float, default = 0., required=False,
                    help = 'use only pairs of forest/qso with the mean of the last absorber redshift and the qso redshift higher than z-cut-min')

    parser.add_argument('--z-cut-max', type = float, default = 10., required=False,
                    help = 'use only pairs of forest/qso with the mean of the last absorber redshift and the qso redshift smaller than z-cut-min')

    parser.add_argument('--z-evol-del', type = float, default = 2.9, required=False,
                    help = 'exponent of the redshift evolution of the delta field')

    parser.add_argument('--z-evol-obj', type = float, default = 1., required=False,
                    help = 'exponent of the redshift evolution of the object field')

    parser.add_argument('--nspec', type=int,default=None, required=False,
                    help = 'maximum spectra to read')

    parser.add_argument('--cf1d', type=str, required=True,
                    help = 'cf1d file')

    args = parser.parse_args()

    if args.nproc is None:
        args.nproc = cpu_count()//2

    ### Parameters
    xcf.np = args.np
    xcf.nt = args.nt
    xcf.rp_min = args.rp_min
    xcf.rp_max = args.rp_max
    xcf.rt_max = args.rt_max
    xcf.z_cut_min = args.z_cut_min
    xcf.z_cut_max = args.z_cut_max
    xcf.nside = args.nside
    xcf.lambda_abs = constants.absorber_IGM[args.lambda_abs]

    ### Cosmo    
    cosmo = constants.cosmo(args.fid_Om)

    ### Read deltas
    dels, ndels, zmin_pix, zmax_pix = io.read_deltas(args.in_dir, args.nside, xcf.lambda_abs, args.z_evol_del, args.z_ref, cosmo=cosmo,nspec=args.nspec)
    xcf.npix = len(dels)
    xcf.dels = dels
    xcf.ndels = ndels
    sys.stderr.write("\n")
    print("done, npix = {}".format(xcf.npix))

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

    ### Read objects
    objs,zmin_obj = io.read_objects(args.drq, args.nside, args.z_min_obj, args.z_max_obj,\
                                args.z_evol_obj, args.z_ref,cosmo)
    xcf.objs = objs
    sys.stderr.write("\n")
    print("done, npix = {}".format(len(objs)))

    ### Maximum angle
    xcf.angmax = utils.compute_ang_max(cosmo,xcf.rt_max,zmin_pix,zmin_obj)

    ### Load cf1d
    h = fitsio.FITS(args.cf1d)
    head = h[1].read_header()
    xcf.lmin = head['LLMIN']
    xcf.dll  = head['DLL']
    xcf.cf1d = h[2]['DA'][:]
    h.close()

    ### Send
    xcf.counter = Value('i',0)
    xcf.lock = Lock()

    cpu_data = {}
    for i,p in enumerate(list(xcf.dels.keys())):
        ip = i%args.nproc
        if not ip in cpu_data:
            cpu_data[ip] = []
        cpu_data[ip].append(p)

    random.seed(0)
    pool = Pool(processes=args.nproc)
    print(" Starting")
    wickT = pool.map(calc_wickT,sorted(list(cpu_data.values())))
    print(" Finished")
    pool.close()

    print(wickT)
