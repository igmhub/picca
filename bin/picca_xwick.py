#!/usr/bin/env python

from __future__ import print_function
import sys
import argparse
import fitsio
import scipy as sp
from scipy.interpolate import interp1d
from multiprocessing import Pool,Lock,cpu_count,Value

from picca import constants, io, utils, xcf, cf
from picca.utils import print

def calc_wickT(p):
    """Send the Wick computation for a set of pixels

    Args:
        p (lst): list of HEALpix pixels

    Returns:
        (tuple): results of the Wick computation

    """
    sp.random.seed(p[0])
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

    parser.add_argument('--fid-Or', type=float, default=0., required=False,
        help='Omega_radiation(z=0) of fiducial LambdaCDM cosmology')

    parser.add_argument('--fid-Ok', type=float, default=0., required=False,
        help='Omega_k(z=0) of fiducial LambdaCDM cosmology')

    parser.add_argument('--fid-wl', type=float, default=-1., required=False,
        help='Equation of state of dark energy of fiducial LambdaCDM cosmology')

    parser.add_argument('--max-diagram', type=int, default=4, required=False,
        help='Maximum diagram to compute')

    parser.add_argument('--cf1d', type=str, required=True,
        help='1D auto-correlation of pixels from the same forest file: picca_cf1d.py')

    parser.add_argument('--cf', type=str, default=None, required=False,
        help='3D auto-correlation of pixels from different forests: picca_cf.py')

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
    xcf.zref = args.z_ref
    xcf.z_evol_del = args.z_evol_del
    xcf.z_evol_obj = args.z_evol_obj
    xcf.lambda_abs = constants.absorber_IGM[args.lambda_abs]
    xcf.max_diagram = args.max_diagram

    ### Cosmo
    if (args.fid_Or!=0.) or (args.fid_Ok!=0.) or (args.fid_wl!=-1.):
        print("ERROR: Cosmology with other than Omega_m set are not yet implemented")
        sys.exit()
    cosmo = constants.cosmo(Om=args.fid_Om,Or=args.fid_Or,Ok=args.fid_Ok,wl=args.fid_wl)

    ### Read deltas
    dels, ndels, zmin_pix, zmax_pix = io.read_deltas(args.in_dir, args.nside, xcf.lambda_abs, args.z_evol_del, args.z_ref, cosmo=cosmo,nspec=args.nspec)
    for p,delsp in dels.items():
        for d in delsp:
            d.fname = 'D1'
            for k in ['co','de','order','iv','diff','m_SNR','m_reso','m_z','dll']:
                setattr(d,k,None)
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
    llmin = head['LLMIN']
    llmax = head['LLMAX']
    dll = head['DLL']
    nv1d = h[1]['nv1d'][:]
    v1d = h[1]['v1d'][:]
    ll = llmin + dll*sp.arange(v1d.size)
    xcf.v1d['D1'] = interp1d(ll[nv1d>0],v1d[nv1d>0],kind='nearest',fill_value='extrapolate')

    nb1d = h[1]['nb1d'][:]
    c1d = h[1]['c1d'][:]
    xcf.c1d['D1'] = interp1d((ll-llmin)[nb1d>0],c1d[nb1d>0],kind='nearest',fill_value='extrapolate')
    h.close()

    ### Load cf
    if not args.cf is None:
        h = fitsio.FITS(args.cf)
        head = h[1].read_header()
        xcf.cfWick_np = head['NP']
        xcf.cfWick_nt = head['NT']
        xcf.cfWick_rp_min = head['RPMIN']
        xcf.cfWick_rp_max = head['RPMAX']
        xcf.cfWick_rt_max = head['RTMAX']
        xcf.cfWick_angmax = utils.compute_ang_max(cosmo,xcf.cfWick_rt_max,zmin_pix)
        da = h[2]['DA'][:]
        we = h[2]['WE'][:]
        da = (da*we).sum(axis=0)
        we = we.sum(axis=0)
        w = we>0.
        da[w] /= we[w]
        xcf.cfWick = da.copy()
        h.close()

        cf.data = xcf.dels
        cf.angmax = xcf.cfWick_angmax
        cf.nside = xcf.nside
        cf.z_cut_max = xcf.z_cut_max
        cf.z_cut_min = xcf.z_cut_min

    ### Send
    xcf.counter = Value('i',0)
    xcf.lock = Lock()

    cpu_data = {}
    for i,p in enumerate(sorted(xcf.dels.keys())):
        ip = i%args.nproc
        if not ip in cpu_data:
            cpu_data[ip] = []
        cpu_data[ip].append(p)

    ### Get neighbours
    for p in cpu_data.values():
        xcf.fill_neighs(p)
        if not xcf.cfWick is None:
            cf.fill_neighs(p)

    pool = Pool(processes=min(args.nproc,len(cpu_data.values())))
    print(" \nStarting\n")
    wickT = pool.map(calc_wickT,sorted(cpu_data.values()))
    print(" \nFinished\n")
    pool.close()

    wickT = sp.array(wickT)
    wAll = wickT[:,0].sum(axis=0)
    nb = wickT[:,1].sum(axis=0)
    npairs = wickT[:,2].sum(axis=0)
    npairs_used = wickT[:,3].sum(axis=0)
    T1 = wickT[:,4].sum(axis=0)
    T2 = wickT[:,5].sum(axis=0)
    T3 = wickT[:,6].sum(axis=0)
    T4 = wickT[:,7].sum(axis=0)
    T5 = wickT[:,8].sum(axis=0)
    T6 = wickT[:,9].sum(axis=0)
    we = wAll*wAll[:,None]
    w = we>0.
    T1[w] /= we[w]
    T2[w] /= we[w]
    T3[w] /= we[w]
    T4[w] /= we[w]
    T5[w] /= we[w]
    T6[w] /= we[w]
    T1 *= 1.*npairs_used/npairs
    T2 *= 1.*npairs_used/npairs
    T3 *= 1.*npairs_used/npairs
    T4 *= 1.*npairs_used/npairs
    T5 *= 1.*npairs_used/npairs
    T6 *= 1.*npairs_used/npairs
    Ttot = T1+T2+T3+T4+T5+T6

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
