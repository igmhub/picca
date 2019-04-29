#!/usr/bin/env python
from __future__ import print_function
import scipy as sp
import fitsio
import argparse
from multiprocessing import Pool,Lock,cpu_count,Value

from picca import constants, xcf, io, utils
from picca.utils import print

def calc_dmat(p):
    xcf.fill_neighs(p)
    sp.random.seed(p[0])
    tmp = xcf.dmat(p)
    return tmp

if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Compute the distortion matrix of the cross-correlation delta x object.')

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

    parser.add_argument('--coef-binning-model', type=int, default=1, required=False,
        help='Coefficient multiplying np and nt to get finner binning for the model')

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

    parser.add_argument('--fid-Or', type=float, default=0., required=False,
        help='Omega_radiation(z=0) of fiducial LambdaCDM cosmology')

    parser.add_argument('--fid-Ok', type=float, default=0., required=False,
        help='Omega_k(z=0) of fiducial LambdaCDM cosmology')

    parser.add_argument('--fid-wl', type=float, default=-1., required=False,
        help='Equation of state of dark energy of fiducial LambdaCDM cosmology')

    parser.add_argument('--rej', type=float, default=1., required=False,
        help='Fraction of rejected pairs: -1=no rejection, 1=all rejection')

    parser.add_argument('--nside', type=int, default=16, required=False,
        help='Healpix nside')

    parser.add_argument('--nproc', type=int, default=None, required=False,
        help='Number of processors')

    parser.add_argument('--nspec', type=int, default=None, required=False,
        help='Maximum number of spectra to read')


    args = parser.parse_args()

    if args.nproc is None:
        args.nproc = cpu_count()//2

    print("nproc",args.nproc)

    xcf.rp_max = args.rp_max
    xcf.rp_min = args.rp_min
    xcf.rt_max = args.rt_max
    xcf.z_cut_max = args.z_cut_max
    xcf.z_cut_min = args.z_cut_min
    xcf.np = args.np
    xcf.nt = args.nt
    xcf.npm = args.np*args.coef_binning_model
    xcf.ntm = args.nt*args.coef_binning_model
    xcf.nside = args.nside
    xcf.zref = args.z_ref
    xcf.alpha = args.z_evol_del
    xcf.lambda_abs = constants.absorber_IGM[args.lambda_abs]
    xcf.rej = args.rej

    cosmo = constants.cosmo(Om=args.fid_Om,Or=args.fid_Or,Ok=args.fid_Ok,wl=args.fid_wl)

    ### Read deltas
    dels, ndels, zmin_pix, zmax_pix = io.read_deltas(args.in_dir, args.nside, xcf.lambda_abs,
        args.z_evol_del, args.z_ref, cosmo=cosmo,nspec=args.nspec)
    xcf.npix = len(dels)
    xcf.dels = dels
    xcf.ndels = ndels
    print("\n")
    print("done, npix = {}\n".format(xcf.npix))

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

    ### Read objects
    objs,zmin_obj = io.read_objects(args.drq, args.nside, args.z_min_obj, args.z_max_obj,\
                                args.z_evol_obj, args.z_ref,cosmo)
    print("\n")
    xcf.objs = objs

    ###
    xcf.angmax = utils.compute_ang_max(cosmo,xcf.rt_max,zmin_pix,zmin_obj)




    xcf.counter = Value('i',0)

    xcf.lock = Lock()

    cpu_data = {}
    for i,p in enumerate(sorted(list(dels.keys()))):
        ip = i%args.nproc
        if not ip in cpu_data:
            cpu_data[ip] = []
        cpu_data[ip].append(p)

    if args.nproc>1:
        pool = Pool(processes=args.nproc)
        dm = pool.map(calc_dmat,sorted(list(cpu_data.values())))
        pool.close()
    elif args.nproc==1:
        dm = map(calc_dmat,sorted(list(cpu_data.values())))
        dm = list(dm)

    dm = sp.array(dm)
    wdm =dm[:,0].sum(axis=0)
    rp = dm[:,2].sum(axis=0)
    rt = dm[:,3].sum(axis=0)
    z = dm[:,4].sum(axis=0)
    we = dm[:,5].sum(axis=0)
    npairs = dm[:,6].sum(axis=0)
    npairs_used = dm[:,7].sum(axis=0)
    dm=dm[:,1].sum(axis=0)

    w = we>0.
    rp[w] /= we[w]
    rt[w] /= we[w]
    z[w] /= we[w]
    w = wdm>0.
    dm[w,:] /= wdm[w,None]

    out = fitsio.FITS(args.out,'rw',clobber=True)
    head = [ {'name':'RPMIN','value':xcf.rp_min,'comment':'Minimum r-parallel [h^-1 Mpc]'},
        {'name':'RPMAX','value':xcf.rp_max,'comment':'Maximum r-parallel [h^-1 Mpc]'},
        {'name':'RTMAX','value':xcf.rt_max,'comment':'Maximum r-transverse [h^-1 Mpc]'},
        {'name':'NP','value':xcf.np,'comment':'Number of bins in r-parallel'},
        {'name':'NT','value':xcf.nt,'comment':'Number of bins in r-transverse'},
        {'name':'COEFMOD','value':args.coef_binning_model,'comment':'Coefficient for model binning'},
        {'name':'ZCUTMIN','value':xcf.z_cut_min,'comment':'Minimum redshift of pairs'},
        {'name':'ZCUTMAX','value':xcf.z_cut_max,'comment':'Maximum redshift of pairs'},
        {'name':'REJ','value':xcf.rej,'comment':'Rejection factor'},
        {'name':'NPALL','value':npairs,'comment':'Number of pairs'},
        {'name':'NPUSED','value':npairs_used,'comment':'Number of used pairs'},
    ]
    out.write([wdm,dm],
        names=['WDM','DM'],
        comment=['Sum of weight','Distortion matrix'],
        units=['',''],
        header=head,extname='DMAT')
    out.write([rp,rt,z],
        names=['RP','RT','Z'],
        comment=['R-parallel','R-transverse','Redshift'],
        units=['h^-1 Mpc','h^-1 Mpc','',],
        extname='ATTRI')
    out.close()
