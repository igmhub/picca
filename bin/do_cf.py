#!/usr/bin/env python
from __future__ import print_function
import scipy as sp
import fitsio
import argparse
import glob
import healpy
import sys
import copy
from multiprocessing import Pool,Lock,cpu_count,Value

from picca import constants, cf, utils
from picca.data import delta
from picca.utils import print

def corr_func(p):
    if x_correlation:
        cf.fill_neighs_x_correlation(p)
    else:
        cf.fill_neighs(p)
    tmp = cf.cf(p)
    return tmp

if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Compute the auto and cross-correlation of delta fields')

    parser.add_argument('--out', type=str, default=None, required=True,
        help='Output file name')

    parser.add_argument('--in-dir', type=str, default=None, required=True,
        help='Directory to delta files')

    parser.add_argument('--from-image', type=str, default=None, required=False,
        help='Read delta from image format', nargs='*')

    parser.add_argument('--in-dir2', type=str, default=None, required=False,
        help='Directory to 2nd delta files')

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

    parser.add_argument('--lambda-abs2', type=str, default=None, required=False,
        help='Name of the absorption in picca.constants defining the redshift of the 2nd delta')

    parser.add_argument('--z-ref', type=float, default=2.25, required=False,
        help='Reference redshift')

    parser.add_argument('--z-evol', type=float, default=2.9, required=False,
        help='Exponent of the redshift evolution of the delta field')

    parser.add_argument('--z-evol2', type=float, default=2.9, required=False,
        help='Exponent of the redshift evolution of the 2nd delta field')

    parser.add_argument('--fid-Om', type=float, default=0.315, required=False,
        help='Omega_matter(z=0) of fiducial LambdaCDM cosmology')

    parser.add_argument('--no-project', action='store_true', required=False,
        help='Do not project out continuum fitting modes')

    parser.add_argument('--no-same-wavelength-pairs', action='store_true', required=False,
        help='Reject pairs with same wavelength')

    parser.add_argument('--nside', type=int, default=16, required=False,
        help='Healpix nside')

    parser.add_argument('--nproc', type=int, default=None, required=False,
        help='Number of processors')

    parser.add_argument('--nspec', type=int, default=None, required=False,
        help='Maximum number of spectra to read')


    args = parser.parse_args()

    if args.nproc is None:
        args.nproc = cpu_count()//2

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
    cf.no_same_wavelength_pairs = args.no_same_wavelength_pairs

    cosmo = constants.cosmo(args.fid_Om)

    lambda_abs  = constants.absorber_IGM[args.lambda_abs]
    if (args.lambda_abs2) : lambda_abs2 = constants.absorber_IGM[args.lambda_abs2]
    else: lambda_abs2 = constants.absorber_IGM[args.lambda_abs]
    cf.lambda_abs = lambda_abs
    cf.lambda_abs2 = lambda_abs2

    data = {}
    ndata = 0
    dels = []
    fi = []
    if args.from_image == None:
        if (len(args.in_dir)>8) and (args.in_dir[-8:]==".fits.gz"):
            fi += glob.glob(args.in_dir)
        elif (len(args.in_dir)>5) and (args.in_dir[-5:]==".fits"):
            fi += glob.glob(args.in_dir)
        else:
            fi += glob.glob(args.in_dir+"/*.fits") + glob.glob(args.in_dir+"/*.fits.gz")
        fi = sorted(fi)
        for i,f in enumerate(fi):
            sys.stderr.write("\rread {} of {} {}".format(i,len(fi),ndata))
            hdus = fitsio.FITS(f)
            dels += [delta.from_fitsio(h) for h in hdus[1:]]
            ndata+=len(hdus[1:])
            hdus.close()
            if not args.nspec is None:
                if ndata>args.nspec:break
    elif len(args.from_image)>0:
        for arg in args.from_image:
            if (len(arg)>8) and (arg[-8:]==".fits.gz"):
                fi += glob.glob(arg)
            elif (len(arg)>5) and (arg[-5:]==".fits"):
                fi += glob.glob(arg)
            else:
                fi += glob.glob(arg+"/*.fits") + glob.glob(arg+"/*.fits.gz")
        fi = sorted(fi)
        for f in fi:
            d = delta.from_image(f)
            dels += d
        ndata = len(dels)
        print('\nndata = ',ndata)
    else:
        if (len(args.in_dir)>8) and (args.in_dir[-8:]==".fits.gz"):
            fi += glob.glob(args.in_dir)
        elif (len(args.in_dir)>5) and (args.in_dir[-5:]==".fits"):
            fi += glob.glob(args.in_dir)
        else:
            fi += glob.glob(args.in_dir+"/*.fits") + glob.glob(args.in_dir+"/*.fits.gz")
        fi = sorted(fi)
        for f in fi:
            d = delta.from_image(f)
            dels += d
        ndata = len(dels)
        print('\nndata = ',ndata)

    x_correlation=False
    if args.in_dir2:
        x_correlation=True
        data2 = {}
        ndata2 = 0
        dels2 = []
        if not args.from_image:
            if (len(args.in_dir2)>8) and (args.in_dir2[-8:]==".fits.gz"):
                fi = glob.glob(args.in_dir2)
            else:
                fi = glob.glob(args.in_dir2+"/*.fits.gz")
            fi = sorted(fi)
            for i,f in enumerate(fi):
                sys.stderr.write("\rread {} of {} {}".format(i,len(fi),ndata2))
                hdus = fitsio.FITS(f)
                dels2 += [delta.from_fitsio(h) for h in hdus[1:]]
                ndata2+=len(hdus[1:])
                hdus.close()
                if not args.nspec is None:
                    if ndata2>args.nspec:break
        else:
            dels2 = delta.from_image(args.in_dir2)
    elif lambda_abs != lambda_abs2:
        x_correlation=True
        data2  = copy.deepcopy(data)
        ndata2 = copy.deepcopy(ndata)
        dels2  = copy.deepcopy(dels)
    cf.x_correlation = x_correlation
    if x_correlation: print("doing xcorrelation")
    z_min_pix = 10**dels[0].ll[0]/lambda_abs-1.
    phi = [d.ra for d in dels]
    th = [sp.pi/2.-d.dec for d in dels]
    pix = healpy.ang2pix(cf.nside,th,phi)
    for d,p in zip(dels,pix):
        if not p in data:
            data[p]=[]
        data[p].append(d)

        z = 10**d.ll/lambda_abs-1.
        z_min_pix = sp.amin( sp.append([z_min_pix],z) )
        d.z = z
        d.r_comov = cosmo.r_comoving(z)
        d.we *= ((1.+z)/(1.+args.z_ref))**(cf.alpha-1.)
        if not args.no_project:
            d.project()

    cf.angmax = utils.compute_ang_max(cosmo,cf.rt_max,z_min_pix)

    if x_correlation:
        cf.alpha2 = args.z_evol2
        z_min_pix2 = 10**dels2[0].ll[0]/lambda_abs2-1.
        phi2 = [d.ra for d in dels2]
        th2 = [sp.pi/2.-d.dec for d in dels2]
        pix2 = healpy.ang2pix(cf.nside,th2,phi2)

        for d,p in zip(dels2,pix2):
            if not p in data2:
                data2[p]=[]
            data2[p].append(d)

            z = 10**d.ll/lambda_abs2-1.
            z_min_pix2 = sp.amin(sp.append([z_min_pix2],z) )
            d.z = z
            d.r_comov = cosmo.r_comoving(z)
            d.we *= ((1.+z)/(1.+args.z_ref))**(cf.alpha2-1.)
            if not args.no_project:
                d.project()

        cf.angmax = utils.compute_ang_max(cosmo,cf.rt_max,z_min_pix,z_min_pix2)

    sys.stderr.write("\n")

    cf.npix = len(data)
    cf.data = data
    cf.ndata=ndata
    print("done, npix = {}".format(cf.npix))

    if x_correlation:
        cf.data2 = data2
        cf.ndata2=ndata2

    cf.counter = Value('i',0)

    cf.lock = Lock()
    cpu_data = {}
    for p in list(data.keys()):
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
    nb       = nbs.sum(axis=0)


    out = fitsio.FITS(args.out,'rw',clobber=True)
    head = [ {'name':'RPMIN','value':cf.rp_min,'comment':'Minimum r-parallel [h^-1 Mpc]'},
        {'name':'RPMAX','value':cf.rp_max,'comment':'Maximum r-parallel [h^-1 Mpc]'},
        {'name':'RTMAX','value':cf.rt_max,'comment':'Maximum r-transverse [h^-1 Mpc]'},
        {'name':'NP','value':cf.np,'comment':'Number of bins in r-parallel'},
        {'name':'NT','value':cf.nt,'comment':'Number of bins in r-transverse'},
        {'name':'ZCUTMIN','value':cf.z_cut_min,'comment':'Minimum redshift of pairs'},
        {'name':'ZCUTMAX','value':cf.z_cut_max,'comment':'Maximum redshift of pairs'},
        {'name':'NSIDE','value':cf.nside,'comment':'Healpix nside'}
    ]
    out.write([rp,rt,z,nb],names=['RP','RT','Z','NB'],
        comment=['R-parallel','R-transverse','Redshift','Number of pairs'],
        units=['h^-1 Mpc','h^-1 Mpc','',''],
        header=head,extname='ATTRI')

    head2 = [{'name':'HLPXSCHM','value':'RING','comment':'Healpix scheme'}]
    out.write([hep,wes,cfs],names=['HEALPID','WE','DA'],
        comment=['Healpix index', 'Sum of weight', 'Correlation'],
        header=head2,extname='COR')

    out.close()
