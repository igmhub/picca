#!/usr/bin/env python
import scipy as sp
import fitsio
import argparse
from multiprocessing import Pool,Lock,cpu_count,Value

from picca import constants, xcf, io, utils
from picca.utils import userprint

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

    userprint("nproc",args.nproc)

    xcf.r_par_max = args.rp_max
    xcf.r_par_min = args.rp_min
    xcf.r_trans_max = args.rt_max
    xcf.z_cut_max = args.z_cut_max
    xcf.z_cut_min = args.z_cut_min
    xcf.num_bins_r_par = args.np
    xcf.num_bins_r_trans = args.nt
    xcf.num_model_bins_r_par = args.np*args.coef_binning_model
    xcf.num_model_bins_r_trans = args.nt*args.coef_binning_model
    xcf.nside = args.nside
    xcf.z_ref = args.z_ref
    xcf.alpha = args.z_evol_del
    xcf.lambda_abs = constants.ABSORBER_IGM[args.lambda_abs]
    xcf.reject = args.rej

    cosmo = constants.Cosmo(Om=args.fid_Om,Or=args.fid_Or,Ok=args.fid_Ok,wl=args.fid_wl)

    ### Read deltas
    dels, ndels, zmin_pix, zmax_pix = io.read_deltas(args.in_dir, args.nside, xcf.lambda_abs,
        args.z_evol_del, args.z_ref, cosmo=cosmo,max_num_spec=args.nspec)
    xcf.npix = len(dels)
    xcf.dels = dels
    xcf.ndels = ndels
    userprint("\n")
    userprint("done, npix = {}\n".format(xcf.npix))

    ### Find the redshift range
    if (args.z_min_obj is None):
        dmin_pix = cosmo.get_r_comov(zmin_pix)
        dmin_obj = max(0.,dmin_pix+xcf.r_par_min)
        args.z_min_obj = cosmo.distance_to_redshift(dmin_obj)
        userprint("\r z_min_obj = {}\r".format(args.z_min_obj),end="")
    if (args.z_max_obj is None):
        dmax_pix = cosmo.get_r_comov(zmax_pix)
        dmax_obj = max(0.,dmax_pix+xcf.r_par_max)
        args.z_max_obj = cosmo.distance_to_redshift(dmax_obj)
        userprint("\r z_max_obj = {}\r".format(args.z_max_obj),end="")

    ### Read objects
    objs,zmin_obj = io.read_objects(args.drq, args.nside, args.z_min_obj, args.z_max_obj,\
                                args.z_evol_obj, args.z_ref,cosmo)
    userprint("\n")
    xcf.objs = objs

    ###
    xcf.ang_max = utils.compute_ang_max(cosmo,xcf.r_trans_max,zmin_pix,zmin_obj)




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
        dmat_data = pool.map(calc_dmat,sorted(list(cpu_data.values())))
        pool.close()
    elif args.nproc==1:
        dmat_data = map(calc_dmat,sorted(list(cpu_data.values())))
        dmat_data = list(dmat_data)

    dmat_data = sp.array(dmat_data)
    weights_dmat = dmat_data[:,0].sum(axis=0)
    r_par = dmat_data[:,2].sum(axis=0)
    r_trans = dmat_data[:,3].sum(axis=0)
    z = dmat_data[:,4].sum(axis=0)
    weights = dmat_data[:,5].sum(axis=0)
    npairs = dmat_data[:,6].sum(axis=0)
    npairs_used = dmat_data[:,7].sum(axis=0)
    dmat = dmat_data[:,1].sum(axis=0)

    w = weights>0.
    r_par[w] /= weights[w]
    r_trans[w] /= weights[w]
    z[w] /= weights[w]
    w = weights_dmat>0.
    dmat[w,:] /= weights_dmat[w,None]

    out = fitsio.FITS(args.out,'rw',clobber=True)
    head = [ {'name':'RPMIN','value':xcf.r_par_min,'comment':'Minimum r-parallel [h^-1 Mpc]'},
        {'name':'RPMAX','value':xcf.r_par_max,'comment':'Maximum r-parallel [h^-1 Mpc]'},
        {'name':'RTMAX','value':xcf.r_trans_max,'comment':'Maximum r-transverse [h^-1 Mpc]'},
        {'name':'NP','value':xcf.num_bins_r_par,'comment':'Number of bins in r-parallel'},
        {'name':'NT','value':xcf.num_bins_r_trans,'comment':'Number of bins in r-transverse'},
        {'name':'COEFMOD','value':args.coef_binning_model,'comment':'Coefficient for model binning'},
        {'name':'ZCUTMIN','value':xcf.z_cut_min,'comment':'Minimum redshift of pairs'},
        {'name':'ZCUTMAX','value':xcf.z_cut_max,'comment':'Maximum redshift of pairs'},
        {'name':'REJ','value':xcf.reject,'comment':'Rejection factor'},
        {'name':'NPALL','value':npairs,'comment':'Number of pairs'},
        {'name':'NPUSED','value':npairs_used,'comment':'Number of used pairs'},
    ]
    out.write([weights_dmat,dmat],
        names=['WDM','DM'],
        comment=['Sum of weight','Distortion matrix'],
        units=['',''],
        header=head,extname='DMAT')
    out.write([r_par,r_trans,z],
        names=['RP','RT','Z'],
        comment=['R-parallel','R-transverse','Redshift'],
        units=['h^-1 Mpc','h^-1 Mpc','',],
        extname='ATTRI')
    out.close()
