#!/usr/bin/env python
from __future__ import print_function
import scipy as sp
import fitsio
import argparse
from multiprocessing import Pool,Lock,cpu_count,Value

from picca import constants, cf, io
from picca.utils import print


def corr_func(p):
    if cf.x_correlation:
        cf.fill_neighs_x_correlation(p)
    else:
        cf.fill_neighs(p)
    tmp = cf.cf(p)
    return tmp

if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Compute the auto and cross-correlation of delta fields as a function of angle and wavelength ratio')


    parser.add_argument('--out', type=str, default=None, required=True,
        help='Output file name')

    parser.add_argument('--in-dir', type=str, default=None, required=True,
        help='Directory to delta files')

    parser.add_argument('--from-image', type=str, default=None, required=False,
        help='Read delta from image format', nargs='*')

    parser.add_argument('--in-dir2', type=str, default=None, required=False,
        help='Directory to 2nd delta files')

    parser.add_argument('--wr-min', type=float, default=1., required=False,
        help='Min of wavelength ratio')

    parser.add_argument('--wr-max', type=float, default=1.1, required=False,
        help='Max of wavelength ratio')

    parser.add_argument('--ang-max', type=float, default=0.02, required=False,
        help='Max angle (rad)')

    parser.add_argument('--np', type=int, default=50, required=False,
        help='Number of wavelength ratio bins')

    parser.add_argument('--nt', type=int, default=50, required=False,
        help='Number of angular bins')

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

    cf.rp_min          = args.wr_min
    cf.rp_max          = args.wr_max
    cf.rt_max          = args.ang_max
    cf.z_cut_max       = args.z_cut_max
    cf.z_cut_min       = args.z_cut_min
    cf.np              = args.np
    cf.nt              = args.nt
    cf.nside           = args.nside
    cf.zref            = args.z_ref
    cf.alpha           = args.z_evol
    cf.x_correlation   = False
    cf.ang_correlation = True
    cf.angmax          = args.ang_max
    cf.no_same_wavelength_pairs = args.no_same_wavelength_pairs
    cf.lambda_abs = constants.absorber_IGM[args.lambda_abs]


    ### Read data 1
    data, ndata, zmin_pix, zmax_pix = io.read_deltas(args.in_dir, cf.nside, cf.lambda_abs, cf.alpha, cf.zref, cosmo=None, nspec=args.nspec, no_project=args.no_project)
    cf.npix = len(data)
    cf.data = data
    cf.ndata = ndata
    print("")
    print("done, npix = {}".format(cf.npix))

    ### Read data 2
    if args.in_dir2 or args.lambda_abs2:
        cf.x_correlation = True
        cf.alpha2 = args.z_evol2
        if args.in_dir2 is None:
            args.in_dir2 = args.in_dir
        if args.lambda_abs2:
            cf.lambda_abs2 = constants.absorber_IGM[args.lambda_abs2]
        else:
            cf.lambda_abs2 = cf.lambda_abs

        data2, ndata2, zmin_pix2, zmax_pix2 = io.read_deltas(args.in_dir2, cf.nside, cf.lambda_abs2, cf.alpha2, cf.zref, cosmo=None, nspec=args.nspec, no_project=args.no_project)
        cf.data2 = data2
        cf.ndata2 = ndata2
        print("")
        print("done, npix = {}".format(len(data2)))


    ### Send
    cf.counter = Value('i',0)
    cf.lock = Lock()
    cpu_data = {}
    for p in data.keys():
        cpu_data[p] = [p]
    pool = Pool(processes=args.nproc)
    cfs = pool.map(corr_func,sorted(cpu_data.values()))
    pool.close()


    ### Store
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
    head = [ {'name':'RPMIN','value':cf.rp_min,'comment':'Minimum wavelength ratio'},
        {'name':'RPMAX','value':cf.rp_max,'comment':'Maximum wavelength ratio'},
        {'name':'RTMAX','value':cf.rt_max,'comment':'Maximum angle [rad]'},
        {'name':'NP','value':cf.np,'comment':'Number of bins in wavelength ratio'},
        {'name':'NT','value':cf.nt,'comment':'Number of bins in angle'},
        {'name':'ZCUTMIN','value':cf.z_cut_min,'comment':'Minimum redshift of pairs'},
        {'name':'ZCUTMAX','value':cf.z_cut_max,'comment':'Maximum redshift of pairs'},
        {'name':'NSIDE','value':cf.nside,'comment':'Healpix nside'}
    ]
    out.write([rp,rt,z,nb],names=['RP','RT','Z','NB'],
        units=['','rad','',''],
        comment=['Wavelength ratio','Angle','Redshift','Number of pairs'],
        header=head,extname='ATTRI')

    head2 = [{'name':'HLPXSCHM','value':'RING','comment':'Healpix scheme'}]
    out.write([hep,wes,cfs],names=['HEALPID','WE','DA'],
        comment=['Healpix index', 'Sum of weight', 'Correlation'],
        header=head2,extname='COR')

    out.close()
