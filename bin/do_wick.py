#!/usr/bin/env python
from __future__ import print_function
import scipy as sp
import fitsio
import argparse
import glob
import healpy
from scipy.interpolate import interp1d
from multiprocessing import Pool,Lock,cpu_count,Value

from picca import constants, cf, utils, io
from picca.data import delta
from picca.utils import print


def calc_wickT(p):
    cf.fill_neighs(p)
    sp.random.seed(p[0])
    tmp = cf.t123(p)
    return tmp

if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Compute the wick covariance for the auto-correlation of forests')

    parser.add_argument('--out', type=str, default=None, required=True,
        help='Output file name')

    parser.add_argument('--in-dir', type=str, default=None, required=True,
        help='Directory to delta files')

    parser.add_argument('--rp-min', type=float, default=0., required=False,
        help='Min r-parallel [h^-1 Mpc]')

    parser.add_argument('--rp-max', type=float, default=200., required=False,
        help='Max r-parallel [h^-1 Mpc]')

    parser.add_argument('--rt-max', type=float, default=200., required=False,
        help='Max r-transverse [h^-1 Mpc]')

    parser.add_argument('--np', type=int, default=50, required=False,
        help='Number of r-parallel bins')

    parser.add_argument('--nt', type=int, default=50, required=False,
        help='Number of r-transverse bins')

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

    parser.add_argument('--z-evol', type=float, default=2.9, required=False,
        help='Exponent of the redshift evolution of the delta field')

    parser.add_argument('--fid-Om', type=float, default=0.315, required=False,
        help='Omega_matter(z=0) of fiducial LambdaCDM cosmology')

    parser.add_argument('--cf1d', type=str, required=True,
        help='1D auto-correlation of pixels from the same forest file: do_cf1d.py')

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

    cf.rp_max = args.rp_max
    cf.rt_max = args.rt_max
    cf.rp_min = args.rp_min
    cf.z_cut_max = args.z_cut_max
    cf.z_cut_min = args.z_cut_min
    cf.np = args.np
    cf.nt = args.nt
    cf.nside = args.nside
    cf.zref = args.z_ref
    cf.alpha = args.z_evol
    cf.lambda_abs = constants.absorber_IGM[args.lambda_abs]
    cf.rej = args.rej

    cosmo = constants.cosmo(args.fid_Om)

    ### Load cf1d
    h = fitsio.FITS(args.cf1d)
    head = h[1].read_header()
    llmin = head['LLMIN']
    llmax = head['LLMAX']
    dll = head['DLL']
    nv1d = h[1]['nv1d'][:]
    cf.v1d = h[1]['v1d'][:]
    ll = llmin + dll*sp.arange(len(cf.v1d))
    cf.v1d = interp1d(ll[nv1d>0],cf.v1d[nv1d>0],kind='nearest',fill_value='extrapolate')

    nb1d   = h[1]['nb1d'][:]
    cf.c1d = h[1]['c1d'][:]
    cf.c1d = interp1d((ll-llmin)[nb1d>0],cf.c1d[nb1d>0],kind='nearest')
    h.close()

    ### Read data
    data, ndata, zmin_pix, zmax_pix = io.read_deltas(args.in_dir, args.nside, cf.lambda_abs, args.z_evol, args.z_ref, cosmo, nspec=args.nspec)
    for p,datap in data.items():
        for d in datap:
            for k in ['co','de','order','iv','diff','m_SNR','m_reso','m_z','dll']:
                setattr(d,k,None)
    cf.npix = len(data)
    cf.data = data
    cf.ndata = ndata
    sys.stderr.write("\n")
    print("done, npix = {}".format(cf.npix))

    cf.angmax = utils.compute_ang_max(cosmo,cf.rt_max,zmin_pix)

    ###
    cf.counter = Value('i',0)
    cf.lock = Lock()
    cpu_data = {}
    for i,p in enumerate(sorted(list(data.keys()))):
        ip = i%args.nproc
        if not ip in cpu_data:
            cpu_data[ip] = []
        cpu_data[ip].append(p)

    pool = Pool(processes=args.nproc)
    print(" \nStarting\n")
    wickT = pool.map(calc_wickT,sorted(list(cpu_data.values())))
    print(" \nFinished\n")
    pool.close()

    wickT = sp.array(wickT)
    wAll = wickT[:,0].sum(axis=0)
    nb = wickT[:,1].sum(axis=0)
    npairs = wickT[:,2].sum(axis=0)
    npairs_used = wickT[:,3].sum(axis=0)
    T1 = wickT[:,4].sum(axis=0)
    T2 = wickT[:,5].sum(axis=0)
    we = wAll*wAll[:,None]
    w = we>0.
    T1[w] /= we[w]
    T2[w] /= we[w]
    T1 *= 1.*npairs_used/npairs
    T2 *= 1.*npairs_used/npairs
    Ttot = T1+T2

    out = fitsio.FITS(args.out,'rw',clobber=True)
    head = [
        {'name':'RPMIN','value':cf.rp_min,'comment':'Minimum r-parallel [h^-1 Mpc]'},
        {'name':'RPMAX','value':cf.rp_max,'comment':'Maximum r-parallel [h^-1 Mpc]'},
        {'name':'RTMAX','value':cf.rt_max,'comment':'Maximum r-transverse [h^-1 Mpc]'},
        {'name':'NP','value':cf.np,'comment':'Number of bins in r-parallel'},
        {'name':'NT','value':cf.nt,'comment':'Number of bins in r-transverse'},
        {'name':'ZCUTMIN','value':cf.z_cut_min,'comment':'Minimum redshift of pairs'},
        {'name':'ZCUTMAX','value':cf.z_cut_max,'comment':'Maximum redshift of pairs'},
        {'name':'REJ','value':cf.rej,'comment':'Rejection factor'},
        {'name':'NPALL','value':npairs,'comment':'Number of pairs'},
        {'name':'NPUSED','value':npairs_used,'comment':'Number of used pairs'},
    ]
    comment = ['Sum of weight','Covariance','Nomber of pairs','T1','T2']
    out.write([Ttot,wAll,nb,T1,T2],names=['CO','WALL','NB','T1','T2'],comment=comment,header=head,extname='COV')
    out.close()
