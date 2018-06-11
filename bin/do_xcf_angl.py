#!/usr/bin/env python

import scipy as sp
import fitsio
import argparse
import sys
from multiprocessing import Pool,Lock,cpu_count,Value

from picca import constants, xcf, io, prep_del
from picca.data import forest

def corr_func(p):
    xcf.fill_neighs(p)
    tmp = xcf.xcf(p)
    return tmp

if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Compute the cross-correlation between a catalog of objects and a delta field as a function of angle and wavelength ratio')

    parser.add_argument('--out', type=str, default=None, required=True,
        help='Output file name')

    parser.add_argument('--in-dir', type=str, default=None, required=True,
        help='Directory to delta files')

    parser.add_argument('--drq', type=str, default=None, required=True,
        help='Catalog of objects in DRQ format')

    parser.add_argument('--wr-min', type=float, default=0.9, required=False,
        help='Min of wavelength ratio')

    parser.add_argument('--wr-max', type=float, default=1.1, required=False,
        help='Max of wavelength ratio')

    parser.add_argument('--ang-max', type=float, default=0.02, required=False,
        help='Max angle (rad)')

    parser.add_argument('--np', type=int, default=100, required=False,
        help='Number of wavelength ratio bins')

    parser.add_argument('--nt', type=int, default=50, required=False,
        help='Number of angular bins')

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

    parser.add_argument('--lambda-abs-obj', type=str, default='LYA', required=False,
        help='Name of the absorption in picca.constants the object is considered as')

    parser.add_argument('--z-ref', type=float, default=2.25, required=False,
        help='Reference redshift')

    parser.add_argument('--z-evol-del', type=float, default=2.9, required=False,
        help='Exponent of the redshift evolution of the delta field')

    parser.add_argument('--z-evol-obj', type=float, default=1., required=False,
        help='Exponent of the redshift evolution of the object field')

    parser.add_argument('--fid-Om', type=float, default=0.315, required=False,
        help='Omega_matter(z=0) of fiducial LambdaCDM cosmology')

    parser.add_argument('--no-project', action='store_true', required=False,
        help='Do not project out continuum fitting modes')

    parser.add_argument('--no-remove-mean-lambda-obs', action='store_true', required=False,
        help='Do not remove mean delta versus lambda_obs')

    parser.add_argument('--nside', type=int, default=16, required=False,
        help='Healpix nside')

    parser.add_argument('--nproc', type=int, default=None, required=False,
        help='Number of processors')

    parser.add_argument('--nspec', type=int, default=None, required=False,
        help='Maximum number of spectra to read')


    args = parser.parse_args()

    if args.nproc is None:
        args.nproc = cpu_count()//2

    xcf.rp_min = args.wr_min
    xcf.rp_max = args.wr_max
    xcf.rt_max = args.ang_max
    xcf.z_cut_min = args.z_cut_min
    xcf.z_cut_max = args.z_cut_max
    xcf.np     = args.np
    xcf.nt     = args.nt
    xcf.nside  = args.nside
    xcf.ang_correlation = True
    xcf.angmax  = args.ang_max

    lambda_abs  = constants.absorber_IGM[args.lambda_abs]
    xcf.lambda_abs = lambda_abs

    cosmo = constants.cosmo(args.fid_Om)

    ### Read deltas
    dels, ndels, zmin_pix, zmax_pix = io.read_deltas(args.in_dir, args.nside, constants.absorber_IGM[args.lambda_abs],args.z_evol_del, args.z_ref, cosmo=cosmo,nspec=args.nspec,no_project=args.no_project)
    xcf.npix = len(dels)
    xcf.dels = dels
    xcf.ndels = ndels
    sys.stderr.write("\n")
    print("done, npix = {}".format(xcf.npix))

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

    ### Read objects
    objs,zmin_obj = io.read_objects(args.drq, args.nside, args.z_min_obj, args.z_max_obj,\
                                args.z_evol_obj, args.z_ref,cosmo)
    for i,ipix in enumerate(sorted(objs.keys())):
        for q in objs[ipix]:
            q.ll = sp.log10( (1.+q.zqso)*constants.absorber_IGM[args.lambda_abs_obj] )
    sys.stderr.write("\n")
    xcf.objs = objs


    ### Send
    xcf.counter = Value('i',0)
    xcf.lock = Lock()
    cpu_data = {}
    for p in list(dels.keys()):
        cpu_data[p] = [p]

    pool = Pool(processes=args.nproc)
    cfs = pool.map(corr_func,sorted(list(cpu_data.values())))
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
    nb = nbs.sum(axis=0)


    out = fitsio.FITS(args.out,'rw',clobber=True)
    head = [ {'name':'RPMIN','value':xcf.rp_min,'comment':'Minimum wavelength ratio'},
        {'name':'RPMAX','value':xcf.rp_max,'comment':'Maximum wavelength ratio'},
        {'name':'RTMAX','value':xcf.rt_max,'comment':'Maximum angle [rad]'},
        {'name':'NP','value':xcf.np,'comment':'Number of bins in wavelength ratio'},
        {'name':'NT','value':xcf.nt,'comment':'Number of bins in angle'},
        {'name':'ZCUTMIN','value':xcf.z_cut_min,'comment':'Minimum redshift of pairs'},
        {'name':'ZCUTMAX','value':xcf.z_cut_max,'comment':'Maximum redshift of pairs'},
        {'name':'NSIDE','value':xcf.nside,'comment':'Healpix nside'}
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
