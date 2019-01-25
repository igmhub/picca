#!/usr/bin/env python
from __future__ import print_function
import scipy as sp
import fitsio
import argparse
from functools import partial
from multiprocessing import Pool,Lock,cpu_count,Value

from picca import constants, xcf, io, utils
from picca.utils import print

def calc_metal_xdmat(abs_igm,p):
    xcf.fill_neighs(p)
    sp.random.seed(p[0])
    tmp = xcf.metal_dmat(p,abs_igm=abs_igm)
    return tmp

if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Compute the distortion matrix of the cross-correlation delta x object for a list of IGM absorption.')

    parser.add_argument('--out', type=str, default=None, required=True,
        help='Output file name')

    parser.add_argument('--in-dir', type=str, default=None, required=True,
        help='Directory to delta files')

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

    parser.add_argument('--obj-name', type=str, default='QSO', required=False,
        help='Name of the object tracer')

    parser.add_argument('--abs-igm', type=str,default=None, required=False, nargs='*',
        help='List of names of metal absorption in picca.constants')

    parser.add_argument('--z-ref', type=float, default=2.25, required=False,
        help='Reference redshift')

    parser.add_argument('--z-evol-del', type=float, default=2.9, required=False,
        help='Exponent of the redshift evolution of the delta field')

    parser.add_argument('--z-evol-obj', type=float, default=1., required=False,
        help='Exponent of the redshift evolution of the object field')

    parser.add_argument('--fid-Om', type=float, default=0.315, required=False,
        help='Omega_matter(z=0) of fiducial LambdaCDM cosmology')

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

    xcf.rp_max = args.rp_max
    xcf.rp_min = args.rp_min
    xcf.rt_max = args.rt_max
    xcf.z_cut_max = args.z_cut_max
    xcf.z_cut_min = args.z_cut_min
    xcf.np = args.np
    xcf.nt = args.nt
    xcf.nside = args.nside
    xcf.zref = args.z_ref
    xcf.lambda_abs = constants.absorber_IGM[args.lambda_abs]
    xcf.rej = args.rej

    ## use a metal grid equal to the lya grid
    xcf.npm = args.np
    xcf.ntm = args.nt

    cosmo = constants.cosmo(args.fid_Om)
    xcf.cosmo=cosmo

    dels, ndels, zmin_pix, zmax_pix = io.read_deltas(args.in_dir, args.nside, xcf.lambda_abs,\
                            args.z_evol_del, args.z_ref, cosmo,nspec=args.nspec)

    xcf.npix = len(dels)
    xcf.dels = dels
    xcf.ndels = ndels


    ### Find the redshift range
    if (args.z_min_obj is None):
        dmin_pix = cosmo.r_comoving(zmin_pix)
        dmin_obj = max(0.,dmin_pix+xcf.rp_min)
        args.z_min_obj = cosmo.r_2_z(dmin_obj)
        print("\r z_min_obj = {}\r".format(args.z_min_obj),end="")
    if (args.z_max_obj is None):
        dmax_pix = cosmo.r_comoving(zmax_pix)
        dmax_obj = max(0.,dmax_pix+xcf.rp_max)
        args.z_max_obj = cosmo.r_2_z(dmax_obj)
        print("\r z_max_obj = {}\r".format(args.z_max_obj),end="")

    objs,zmin_obj = io.read_objects(args.drq, args.nside, args.z_min_obj, args.z_max_obj,\
                                args.z_evol_obj, args.z_ref,cosmo)
    xcf.objs = objs

    xcf.angmax = utils.compute_ang_max(cosmo,xcf.rt_max,zmin_pix,zmin_obj)

    xcf.counter = Value('i',0)
    xcf.lock = Lock()

    cpu_data = {}
    for i,p in enumerate(sorted(list(dels.keys()))):
        ip = i%args.nproc
        if not ip in cpu_data:
            cpu_data[ip] = []
        cpu_data[ip].append(p)

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
        print("")
        pool = Pool(processes=args.nproc)
        dm = pool.map(f,sorted(list(cpu_data.values())))
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
    head = [ {'name':'RPMIN','value':xcf.rp_min,'comment':'Minimum r-parallel [h^-1 Mpc]'},
        {'name':'RPMAX','value':xcf.rp_max,'comment':'Maximum r-parallel [h^-1 Mpc]'},
        {'name':'RTMAX','value':xcf.rt_max,'comment':'Maximum r-transverse [h^-1 Mpc]'},
        {'name':'NP','value':xcf.np,'comment':'Number of bins in r-parallel'},
        {'name':'NT','value':xcf.nt,'comment':'Number of bins in r-transverse'},
        {'name':'ZCUTMIN','value':xcf.z_cut_min,'comment':'Minimum redshift of pairs'},
        {'name':'ZCUTMAX','value':xcf.z_cut_max,'comment':'Maximum redshift of pairs'},
        {'name':'REJ','value':xcf.rej,'comment':'Rejection factor'},
    ]

    len_names = sp.array([ len(s) for s in names ]).max()
    names = sp.array(names, dtype='S'+str(len_names))
    out.write([sp.array(npairs_all),sp.array(npairs_used_all),sp.array(names)],names=['NPALL','NPUSED','ABS_IGM'],header=head,
        comment=['Number of pairs','Number of used pairs','Absorption name'],extname='ATTRI')

    names = names.astype(str)
    out_list = []
    out_names = []
    out_comment = []
    out_units = []
    for i,ai in enumerate(names):
        out_names += ['RP_'+args.obj_name+'_'+ai]
        out_list += [rp_all[i]]
        out_comment += ['R-parallel']
        out_units += ['h^-1 Mpc']

        out_names += ['RT_'+args.obj_name+'_'+ai]
        out_list += [rt_all[i]]
        out_comment += ['R-transverse']
        out_units += ['h^-1 Mpc']

        out_names += ['Z_'+args.obj_name+'_'+ai]
        out_list += [z_all[i]]
        out_comment += ['Redshift']
        out_units += ['']

        out_names += ['DM_'+args.obj_name+'_'+ai]
        out_list += [dm_all[i]]
        out_comment += ['Distortion matrix']
        out_units += ['']

        out_names += ['WDM_'+args.obj_name+'_'+ai]
        out_list += [wdm_all[i]]
        out_comment += ['Sum of weight']
        out_units += ['']

    out.write(out_list,names=out_names,comment=out_comment,units=out_units,extname='MDMAT')
    out.close()
