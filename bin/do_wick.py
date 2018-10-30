#!/usr/bin/env python
from __future__ import print_function
import scipy as sp
import fitsio
import argparse
import glob
import healpy
import sys
from scipy.interpolate import interp1d

from picca import constants
from picca import cf
from picca.data import delta
from picca import utils
from picca.utils import print

from multiprocessing import Pool,Lock,cpu_count,Value


def calc_t123(p):
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

    parser.add_argument('--rp-max', type=float, default=200., required=False,
        help='Max r-parallel [h^-1 Mpc]')

    parser.add_argument('--rt-max', type=float, default=200., required=False,
        help='Max r-transverse [h^-1 Mpc]')

    parser.add_argument('--np', type=int, default=50, required=False,
        help='Number of r-parallel bins')

    parser.add_argument('--nt', type=int, default=50, required=False,
        help='Number of r-transverse bins')

    parser.add_argument('--lambda-abs', type=str, default='LYA', required=False,
        help='Name of the absorption in picca.constants defining the redshift of the delta')

    parser.add_argument('--z-ref', type=float, default=2.25, required=False,
        help='Reference redshift')

    parser.add_argument('--z-evol', type=float, default=2.9, required=False,
        help='Exponent of the redshift evolution of the delta field')

    parser.add_argument('--fid-Om', type=float, default=0.315, required=False,
        help='Omega_matter(z=0) of fiducial LambdaCDM cosmology')

    parser.add_argument('--no-project', action='store_true', required=False,
        help='Do not project out continuum fitting modes')

    parser.add_argument('--cf1d', type=str, required=True,
        help='1D auto-correlation of pixels from the same forest file: do_cf1d.py')

    parser.add_argument('--old-deltas', action='store_true', required=False,
        help='Do not correct weights for redshift evolution')

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
    cf.np = args.np
    cf.nt = args.nt
    cf.nside = args.nside
    cf.zref = args.z_ref
    cf.alpha = args.z_evol
    cf.lambda_abs = constants.absorber_IGM[args.lambda_abs]
    cf.rej = args.rej

    cosmo = constants.cosmo(args.fid_Om)


    h = fitsio.FITS(args.cf1d)
    head = h[1].read_header()
    llmin = head['LLMIN']
    llmax = head['LLMAX']
    dll = head['DLL']
    nv1d   = h[1]['nv1d'][:]
    cf.v1d = h[1]['v1d'][:]
    ll = llmin + dll*sp.arange(len(cf.v1d))
    cf.v1d = interp1d(ll[nv1d>0],cf.v1d[nv1d>0],kind='nearest')

    nb1d   = h[1]['nb1d'][:]
    cf.c1d = h[1]['c1d'][:]
    cf.c1d = interp1d((ll-llmin)[nb1d>0],cf.c1d[nb1d>0],kind='nearest')
    h.close()


    z_min_pix = 1.e6
    if (len(args.in_dir)>8) and (args.in_dir[-8:]==".fits.gz"):
        fi = glob.glob(args.in_dir)
    else:
        fi = glob.glob(args.in_dir+"/*.fits.gz")
    fi = sorted(fi)
    data = {}
    ndata = 0
    for i,f in enumerate(fi):
        if i%10==0:
            print("\rread {} of {} {}".format(i,len(fi),ndata),end="")
        hdus = fitsio.FITS(f)
        dels = [delta.from_fitsio(h) for h in hdus[1:]]
        ndata+=len(dels)
        phi = [d.ra for d in dels]
        th = [sp.pi/2-d.dec for d in dels]
        pix = healpy.ang2pix(cf.nside,th,phi)
        for d,p in zip(dels,pix):
            if not p in data:
                data[p]=[]
            data[p].append(d)

            z = 10**d.ll/args.lambda_abs-1.
            z_min_pix = sp.amin( sp.append([z_min_pix],z) )
            d.r_comov = cosmo.r_comoving(z)
            if not args.old_deltas:
                d.we *= ((1.+z)/(1.+args.z_ref))**(cf.alpha-1.)
            if not args.no_project:
                d.project()
        if not args.nspec is None:
            if ndata>args.nspec:break
    print("")

    cf.angmax = utils.compute_ang_max(cosmo,cf.rt_max,z_min_pix)

    cf.npix = len(data)
    cf.data = data
    cf.ndata = ndata
    print("done")

    cf.counter = Value('i',0)

    cf.lock = Lock()

    cpu_data = {}
    for i,p in enumerate(list(data.keys())):
        ip = i%args.nproc
        if not ip in cpu_data:
            cpu_data[ip] = []
        cpu_data[ip].append(p)

    pool = Pool(processes=args.nproc)
    t123 = pool.map(calc_t123,sorted(list(cpu_data.values())))
    pool.close()

    t123 = sp.array(t123)
    w123=t123[:,0].sum(axis=0)
    npairs=t123[:,2].sum(axis=0)
    npairs_used=t123[:,3].sum(axis=0)
    t123=t123[:,1].sum(axis=0)
    we = w123*w123[:,None]
    w=we>0
    t123[w]/=we[w]
    t123 = npairs_used*t123/npairs


    out = fitsio.FITS(args.out,'rw',clobber=True)
    head = [ {'name':'RPMAX','value':cf.rp_max,'comment':'Maximum r-parallel [h^-1 Mpc]'},
        {'name':'RTMAX','value':cf.rt_max,'comment':'Maximum r-transverse [h^-1 Mpc]'},
        {'name':'NP','value':cf.np,'comment':'Number of bins in r-parallel [h^-1 Mpc]'},
        {'name':'NT','value':cf.nt,'comment':'Number of bins in r-transverse'},
        {'name':'REJ','value':cf.rej,'comment':'Rejection factor'},
        {'name':'NPALL','value':npairs,'comment':'Number of pairs'},
        {'name':'NPUSED','value':npairs_used,'comment':'Number of used pairs'},
    ]
    out.write([w123,t123],names=['WE','CO'],header=head,comment=['Sum of weight','Covariance from T123'],extname='COV')
    out.close()
