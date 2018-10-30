#!/usr/bin/env python
from __future__ import print_function
import scipy as sp
import fitsio
import argparse
import glob
import healpy
import sys
from functools import partial
import copy
from multiprocessing import Pool,Lock,cpu_count,Value

from picca import constants, cf, utils
from picca.data import delta
from picca.utils import print

def calc_metal_dmat(abs_igm1,abs_igm2,p):
    if x_correlation:
        cf.fill_neighs_x_correlation(p)
    else:
        cf.fill_neighs(p)
    sp.random.seed(p[0])
    tmp = cf.metal_dmat(p,abs_igm1=abs_igm1,abs_igm2=abs_igm2)
    return tmp

if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Compute the auto and cross-correlation of delta fields for a list of IGM absorption.')


    parser.add_argument('--out', type=str, default=None, required=True,
        help='Output file name')

    parser.add_argument('--in-dir', type=str, default=None, required=True,
        help='Directory to delta files')

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

    parser.add_argument('--abs-igm', type=str,default=None, required=False,nargs='*',
        help='List of names of metal absorption in picca.constants present in forest')

    parser.add_argument('--abs-igm2', type=str,default=[], required=False,nargs='*',
        help='List of names of metal absorption in picca.constants present in 2nd forest')

    parser.add_argument('--z-ref', type=float, default=2.25, required=False,
        help='Reference redshift')

    parser.add_argument('--z-evol', type=float, default=2.9, required=False,
        help='Exponent of the redshift evolution of the delta field')

    parser.add_argument('--z-evol2', type=float, default=2.9, required=False,
        help='Exponent of the redshift evolution of the 2nd delta field')

    parser.add_argument('--metal-alpha', type=float, default=1., required=False,
        help='Dxponent of the redshift evolution of the metal delta field')

    parser.add_argument('--fid-Om', type=float, default=0.315, required=False,
        help='Omega_matter(z=0) of fiducial LambdaCDM cosmology')

    parser.add_argument('--no-same-wavelength-pairs', action='store_true', required=False,
        help='Reject pairs with same wavelength')

    parser.add_argument('--rej', type=float, default=1., required=False,
        help='Fraction of rejected forest-forest pairs: -1=no rejection, 1=all rejection')

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
    cf.rp_min = args.rp_min
    cf.rt_max = args.rt_max
    cf.z_cut_max = args.z_cut_max
    cf.z_cut_min = args.z_cut_min
    cf.np = args.np
    cf.nt = args.nt

    ## use a metal grid equal to the lya grid
    cf.npm = args.np
    cf.ntm = args.nt

    cf.nside = args.nside
    cf.zref = args.z_ref
    cf.alpha = args.z_evol
    cf.rej = args.rej
    cf.no_same_wavelength_pairs = args.no_same_wavelength_pairs
    cf.alpha_abs = {}
    cf.alpha_abs[args.lambda_abs] = args.z_evol
    if args.lambda_abs2 : cf.alpha_abs[args.lambda_abs2] = args.z_evol2
    for m in args.abs_igm :
        cf.alpha_abs[m] = args.metal_alpha

    for m in args.abs_igm2 :
        cf.alpha_abs[m] = args.metal_alpha

    cosmo = constants.cosmo(args.fid_Om)
    cf.cosmo=cosmo

    lambda_abs  = constants.absorber_IGM[args.lambda_abs]
    if args.lambda_abs2: lambda_abs2 = constants.absorber_IGM[args.lambda_abs2]
    else: lambda_abs2 = constants.absorber_IGM[args.lambda_abs]

    cf.lambda_abs = lambda_abs
    cf.lambda_abs2 = lambda_abs2

    z_min_pix = 1.e6
    ndata=0
    if (len(args.in_dir)>8) and (args.in_dir[-8:]==".fits.gz"):
        fi = glob.glob(args.in_dir)
    else:
        fi = glob.glob(args.in_dir+"/*.fits.gz")
    fi = sorted(fi)
    data = {}
    dels = []
    for i,f in enumerate(fi):
        print("\rread {} of {} {}".format(i,len(fi),ndata),end="")
        hdus = fitsio.FITS(f)
        dels += [delta.from_fitsio(h) for h in hdus[1:]]
        ndata+=len(hdus[1:])
        hdus.close()
        if not args.nspec is None:
            if ndata>args.nspec:break
    print("read {}".format(ndata))

    x_correlation=False
    if args.in_dir2:
        x_correlation=True
        ndata2 = 0
        if (len(args.in_dir2)>8) and (args.in_dir2[-8:]==".fits.gz"):
            fi = glob.glob(args.in_dir2)
        else:
            fi = glob.glob(args.in_dir2+"/*.fits.gz")
        fi = sorted(fi)
        data2 = {}
        dels2 = []
        for i,f in enumerate(fi):
            print("\rread {} of {} {}".format(i,len(fi),ndata),end="")
            hdus = fitsio.FITS(f)
            dels2 += [delta.from_fitsio(h) for h in hdus[1:]]
            ndata2+=len(hdus[1:])
            hdus.close()
            if not args.nspec is None:
                if ndata2>args.nspec:break
        print("read {}\n".format(ndata2),end="")

    elif lambda_abs != lambda_abs2:
        x_correlation=True
        data2  = copy.deepcopy(data)
        ndata2 = copy.deepcopy(ndata)
        dels2  = copy.deepcopy(dels)
    cf.x_correlation=x_correlation

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

    cf.angmax = utils.compute_ang_max(cosmo,cf.rt_max,z_min_pix)

    if x_correlation:
        cf.alpha2 = args.z_evol2
        z_min_pix2 = 10**dels2[0].ll[0]/lambda_abs2-1.
        z_min_pix=sp.amin(sp.append(z_min_pix,z_min_pix2))
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

        cf.angmax = utils.compute_ang_max(cosmo,cf.rt_max,z_min_pix,z_min_pix2)

    cf.npix = len(data)
    cf.data = data
    cf.ndata = ndata

    if x_correlation:
       print("doing cross-correlation ... ")
       cf.data2 = data2
       cf.ndata2 = ndata2
    print("done")


    cf.counter = Value('i',0)

    cf.lock = Lock()

    cpu_data = {}
    for i,p in enumerate(sorted(list(data.keys()))):
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

    abs_igm = [args.lambda_abs]+args.abs_igm
    print("abs_igm = {}".format(abs_igm))

    if args.lambda_abs2 is None :
        args.lambda_abs2 = args.lambda_abs
        args.abs_igm2 = args.abs_igm

    abs_igm_2 = [args.lambda_abs2]+args.abs_igm2

    if x_correlation:
        print("abs_igm2 = {}".format(abs_igm_2))

    for i,abs_igm1 in enumerate(abs_igm):
        i0 = i
        if args.lambda_abs != args.lambda_abs2:
            i0=0
        for j,abs_igm2 in enumerate(abs_igm_2[i0:]):
            if i==0 and j==0:
                continue
            cf.counter.value=0
            f=partial(calc_metal_dmat,abs_igm1,abs_igm2)
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
            names.append(abs_igm1+"_"+abs_igm2)

            npairs_all.append(npairs)
            npairs_used_all.append(npairs_used)

    out = fitsio.FITS(args.out,'rw',clobber=True)
    head = [ {'name':'RPMIN','value':cf.rp_min,'comment':'Minimum r-parallel [h^-1 Mpc]'},
        {'name':'RPMAX','value':cf.rp_max,'comment':'Maximum r-parallel [h^-1 Mpc]'},
        {'name':'RTMAX','value':cf.rt_max,'comment':'Maximum r-transverse [h^-1 Mpc]'},
        {'name':'NP','value':cf.np,'comment':'Number of bins in r-parallel'},
        {'name':'NT','value':cf.nt,'comment':'Number of bins in r-transverse'},
        {'name':'ZCUTMIN','value':cf.z_cut_min,'comment':'Minimum redshift of pairs'},
        {'name':'ZCUTMAX','value':cf.z_cut_max,'comment':'Maximum redshift of pairs'},
        {'name':'REJ','value':cf.rej,'comment':'Rejection factor'},
        {'name':'ALPHAMET','value':args.metal_alpha,'comment':'Evolution of metal bias'},
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
        out_names += ['RP_'+ai]
        out_list += [rp_all[i]]
        out_comment += ['R-parallel']
        out_units += ['h^-1 Mpc']

        out_names += ['RT_'+ai]
        out_list += [rt_all[i]]
        out_comment += ['R-transverse']
        out_units += ['h^-1 Mpc']

        out_names += ['Z_'+ai]
        out_list += [z_all[i]]
        out_comment += ['Redshift']
        out_units += ['']

        out_names += ['DM_'+ai]
        out_list += [dm_all[i]]
        out_comment += ['Distortion matrix']
        out_units += ['']

        out_names += ['WDM_'+ai]
        out_list += [wdm_all[i]]
        out_comment += ['Sum of weight']
        out_units += ['']

    out.write(out_list,names=out_names,comment=out_comment,units=out_units,extname='MDMAT')
    out.close()
