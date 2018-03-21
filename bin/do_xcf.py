#!/usr/bin/env python

import scipy as sp
import fitsio
import argparse
import sys
from scipy import random

from picca import constants
from picca import xcf
from picca import io
from picca.data import forest
from picca import prep_del
from picca import utils

from multiprocessing import Pool,Process,Lock,Manager,cpu_count,Value


def corr_func(p):
    xcf.fill_neighs(p)
    tmp = xcf.xcf(p)
    return tmp

if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--out', type = str, default = None, required=True,
                        help = 'output file name')

    parser.add_argument('--in-dir', type = str, default = None, required=True,
                        help = 'data directory')

    parser.add_argument('--drq', type = str, default = None, required=True,
                        help = 'drq')

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

    parser.add_argument('--no-project', action="store_true", required=False,
                    help = 'do not project out continuum fitting modes')

    parser.add_argument('--nspec', type=int,default=None, required=False,
                    help = 'maximum spectra to read')

    parser.add_argument('--no-remove-mean-lambda-obs', action="store_true", required=False,
                    help = 'Do not remove mean delta versus lambda_obs')

    parser.add_argument('--from-image', type = str, default = None, required=False,
                    help = 'use image format to read deltas', nargs='*')

    args = parser.parse_args()

    if args.nproc is None:
        args.nproc = cpu_count()//2

    xcf.rp_max = args.rp_max
    xcf.rp_min = args.rp_min
    xcf.z_cut_max = args.z_cut_max
    xcf.z_cut_min = args.z_cut_min
    xcf.rt_max = args.rt_max
    xcf.np = args.np
    xcf.nt = args.nt
    xcf.nside = args.nside
    xcf.lambda_abs = constants.absorber_IGM[args.lambda_abs]

    cosmo = constants.cosmo(args.fid_Om)

    ### Read deltas
    dels, ndels, zmin_pix, zmax_pix = io.read_deltas(args.in_dir, args.nside, xcf.lambda_abs,
        args.z_evol_del, args.z_ref, cosmo=cosmo,nspec=args.nspec,no_project=args.no_project,
        from_image=args.from_image)
    xcf.npix = len(dels)
    xcf.dels = dels
    xcf.ndels = ndels
    sys.stderr.write("\n")
    print("done, npix = {}\n".format(xcf.npix))

    ### Remove <delta> vs. lambda_obs
    if not args.no_remove_mean_lambda_obs:
        forest.dll = None
        for p in xcf.dels:
            for d in xcf.dels[p]:
                dll = sp.asarray([d.ll[ii]-d.ll[ii-1] for ii in range(1,d.ll.size)]).min()
                if forest.dll is None:
                    forest.dll = dll
                else:
                    forest.dll = min(dll,forest.dll)
        forest.lmin  = sp.log10( (zmin_pix+1.)*xcf.lambda_abs )-forest.dll/2.
        forest.lmax  = sp.log10( (zmax_pix+1.)*xcf.lambda_abs )+forest.dll/2.
        ll,st, wst   = prep_del.stack(xcf.dels,delta=True)
        for p in xcf.dels:
            for d in xcf.dels[p]:
                bins = ((d.ll-forest.lmin)/forest.dll+0.5).astype(int)
                d.de -= st[bins]

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
    sys.stderr.write("\n")
    xcf.objs = objs

    ###
    xcf.angmax = utils.compute_ang_max(cosmo,xcf.rt_max,zmin_pix,zmin_obj)



    xcf.counter = Value('i',0)

    xcf.lock = Lock()
    cpu_data = {}
    for p in list(dels.keys()):
        cpu_data[p] = [p]

    pool = Pool(processes=args.nproc)

    cfs = pool.map(corr_func,sorted(list(cpu_data.values())))
    pool.close()

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
    nb = nbs.sum(axis=0)

    out = fitsio.FITS(args.out,'rw',clobber=True)
    head = {}
    head['RPMIN']=xcf.rp_min
    head['RPMAX']=xcf.rp_max
    head['RTMAX']=xcf.rt_max
    head['Z_CUT_MIN']=xcf.z_cut_min
    head['Z_CUT_MAX']=xcf.z_cut_max
    head['NT']=xcf.nt
    head['NP']=xcf.np
    head['NSIDE']=xcf.nside

    out.write([rp,rt,z,nb],names=['RP','RT','Z','NB'],header=head)
    head2 = [{'name':'HLPXSCHM','value':'RING','comment':'healpix scheme'}]
    out.write([hep,wes,cfs],names=['HEALPID','WE','DA'],header=head2)
    out.close()
