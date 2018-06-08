#!/usr/bin/env python

import sys
import argparse
import fitsio
import scipy as sp
from multiprocessing import Pool,Lock,cpu_count,Value

from picca import constants, io, utils, xcf

def calc_wickT(args):
    p = args[0]
    seed = args[1]
    xcf.fill_neighs(p)
    sp.random.seed(seed)
    tmp = xcf.wickT(p)
    return tmp

if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Compute the wick covariance for the cross-correlation of object x forests.')

    parser.add_argument('--out', type=str, default=None, required=True,
        help='Output file name')

    parser.add_argument('--in-dir', type=str, default=None, required=True,
        help='Directory to delta files')

    parser.add_argument('--from-image', type=str, default=None, required=False,
        help='Read delta from image format', nargs='*')

    parser.add_argument('--drq', type=str, default=None, required=True,
        help='Catalog of objects in DRQ format')

    parser.add_argument('--rp-min', type=float, default=-200., required=False,
        help='Min r-parallel [h^-1 Mpc]')

    parser.add_argument('--rp-max', type=float, default=200., required=False,
        help='Max r-parallel [h^-1 Mpc]')

    parser.add_argument('--rt-max', type=float, default=200., required=False,
        help='Max r-transverse [h^-1 Mpc]')

    parser.add_argument('--np', type=int, default=100, required=False,
        help='Number of r-parallel bins')

    parser.add_argument('--nt', type=int, default=50, required=False,
        help='Number of r-transverse bins')

    parser.add_argument('--z-min-obj', type=float, default=None, required=False,
        help='Min redshift for object field')

    parser.add_argument('--z-max-obj', type=float, default=None, required=False,
        help='Max redshift for object field')

    parser.add_argument('--z-cut-min', type=float, default=0., required=False,
        help='Use only pairs of forest x object with the mean of the last absorber \
        redshift and the object redshift larger than z-cut-min')

    parser.add_argument('--z-cut-max', type=float, default=10., required=False,
        help='Use only pairs of forest x object with the mean of the last absorber \
        redshift and the object redshift smaller than z-cut-max')

    parser.add_argument('--lambda-abs', type=str, default='LYA', required=False,
        help='Name of the absorption in picca.constants defining the redshift of the delta')

    parser.add_argument('--z-ref', type=float, default=2.25, required=False,
        help='Reference redshift')

    parser.add_argument('--z-evol-del', type=float, default=2.9, required=False,
        help='Exponent of the redshift evolution of the delta field')

    parser.add_argument('--z-evol-obj', type=float, default=1., required=False,
        help='Exponent of the redshift evolution of the object field')

    parser.add_argument('--fid-Om', type=float, default=0.315, required=False,
        help='Omega_matter(z=0) of fiducial LambdaCDM cosmology')

    parser.add_argument('--cf1d', type=str, required=True,
        help='1D auto-correlation of pixels from the same forest file: do_cf1d.py')

    parser.add_argument('--cf', type=str, default=None, required=False,
        help='3D auto-correlation of pixels from different forests')

    parser.add_argument('--rej', type=float, default=1., required=False,
        help='Fraction of rejected object-forests pairs: -1=no rejection, 1=all rejection')

    parser.add_argument('--nside', type=int, default=16, required=False,
        help='Healpix nside')

    parser.add_argument('--nproc', type=int, default=None, required=False,
        help='Number of processors')

    parser.add_argument('--nspec', type=int, default=None, required=False,
        help='Maximum number of spectra to read')


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
    xcf.rej = args.rej
    xcf.lambda_abs = constants.absorber_IGM[args.lambda_abs]

    ### Cosmo
    cosmo = constants.cosmo(args.fid_Om)

    ### Read deltas
    dels, ndels, zmin_pix, zmax_pix = io.read_deltas(args.in_dir, args.nside, xcf.lambda_abs, args.z_evol_del, args.z_ref, cosmo=cosmo,nspec=args.nspec)
    xcf.npix = len(dels)
    xcf.dels = dels
    xcf.ndels = ndels
    sys.stderr.write("\n")
    print("done, npix = {}, ndels = {}".format(xcf.npix,xcf.ndels))
    sys.stderr.write("\n")

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
    sys.stderr.write("\n")

    ### Maximum angle
    xcf.angmax = utils.compute_ang_max(cosmo,xcf.rt_max,zmin_pix,zmin_obj)

    ### Load cf1d
    h = fitsio.FITS(args.cf1d)
    head = h[1].read_header()
    xcf.lmin = head['LLMIN']
    xcf.dll  = head['DLL']
    xcf.cf1d = h[2]['DA'][:]
    h.close()

    ### Load cf
    if not args.cf is None:
        h = fitsio.FITS(args.cf)
        head = h[1].read_header()
        xcf.cf_np = head['NP']
        xcf.cf_nt = head['NT']
        xcf.cf_rp_min = head['RPMIN']
        xcf.cf_rp_max = head['RPMAX']
        xcf.cf_rt_max = head['RTMAX']
        xcf.cf_angmax = utils.compute_ang_max(cosmo,xcf.cf_rt_max,zmin_pix)
        da = h[2]['DA'][:]
        we = h[2]['WE'][:]
        da = (da*we).sum(axis=0)
        we = we.sum(axis=0)
        w = we>0.
        da[w] /= we[w]
        xcf.cf = da.copy()
        h.close()

    ### Send
    xcf.counter = Value('i',0)
    xcf.lock = Lock()

    cpu_data = {}
    for i,p in enumerate(sorted(list(xcf.dels.keys()))):
        ip = i%args.nproc
        if not ip in cpu_data:
            cpu_data[ip] = []
        cpu_data[ip].append(p)

    sp.random.seed(0)
    table_of_seed = sp.unique((100000.*sp.random.rand(10*len(cpu_data))).astype(int))

    pool = Pool(processes=args.nproc)
    print(" \nStarting\n")
    to_send = [ (v,table_of_seed[j]) for j, v in enumerate( sorted(list(cpu_data.values())) ) ]
    wickT = pool.map(calc_wickT,to_send)
    print(" \nFinished\n")
    pool.close()

    wickT       = sp.array(wickT)
    wAll        = wickT[:,0].sum(axis=0)
    nb          = wickT[:,1].sum(axis=0)
    npairs      = wickT[:,2].sum(axis=0)
    npairs_used = wickT[:,3].sum(axis=0)
    T1          = wickT[:,4].sum(axis=0)
    T2          = wickT[:,5].sum(axis=0)
    T3          = wickT[:,6].sum(axis=0)
    T4          = wickT[:,7].sum(axis=0)
    T5          = wickT[:,8].sum(axis=0)
    T6          = wickT[:,9].sum(axis=0)
    we     = wAll*wAll[:,None]
    w      = we>0.
    T1[w] /= we[w]
    T2[w] /= we[w]
    T3[w] /= we[w]
    T4[w] /= we[w]
    T5[w] /= we[w]
    T6[w] /= we[w]
    T1    *= 1.*npairs_used/npairs
    T2    *= 1.*npairs_used/npairs
    T3    *= 1.*npairs_used/npairs
    T4    *= 1.*npairs_used/npairs
    T5    *= 1.*npairs_used/npairs
    T6    *= 1.*npairs_used/npairs
    Ttot   = T1+T2+T3+T4+T5+T6

    out = fitsio.FITS(args.out,'rw',clobber=True)
    head = [ {'name':'RPMIN','value':xcf.rp_min,'comment':'Minimum r-parallel [h^-1 Mpc]'},
        {'name':'RPMAX','value':xcf.rp_max,'comment':'Maximum r-parallel [h^-1 Mpc]'},
        {'name':'RTMAX','value':xcf.rt_max,'comment':'Maximum r-transverse [h^-1 Mpc]'},
        {'name':'NP','value':xcf.np,'comment':'Number of bins in r-parallel'},
        {'name':'NT','value':xcf.nt,'comment':'Number of bins in r-transverse'},
        {'name':'ZCUTMIN','value':xcf.z_cut_min,'comment':'Minimum redshift of pairs'},
        {'name':'ZCUTMAX','value':xcf.z_cut_max,'comment':'Maximum redshift of pairs'},
        {'name':'REJ','value':xcf.rej,'comment':'Rejection factor'},
        {'name':'NPALL','value':npairs,'comment':'Number of pairs'},
        {'name':'NPUSED','value':npairs_used,'comment':'Number of used pairs'},
    ]
    comment = ['Sum of weight','Covariance','Nomber of pairs','T1','T2','T3','T4','T5','T6']
    out.write([Ttot,wAll,nb,T1,T2,T3,T4,T5,T6],names=['CO','WALL','NB','T1','T2','T3','T4','T5','T6'],comment=comment,header=head,extname='COV')
    out.close()
