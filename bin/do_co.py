#!/usr/bin/env python
from __future__ import print_function
import scipy as sp
import fitsio
import argparse
import sys
from multiprocessing import Pool,Lock,cpu_count,Value

from picca import constants, co, io, utils

def corr_func(p):
    if co.x_correlation:
        co.fill_neighs_x_correlation(p)
    else:
        co.fill_neighs(p)
    tmp = co.co(p)
    return tmp

if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Compute the auto and cross-correlation between catalogs of objects')

    parser.add_argument('--out', type=str, default=None, required=True,
        help='Output file name')

    parser.add_argument('--drq', type=str, default=None, required=True,
        help='Catalog of objects in DRQ format')

    parser.add_argument('--drq2', type=str, default=None, required=False,
        help='Catalog of objects 2 in DRQ format')

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
        help='Use only pairs of object x object with the mean redshift larger than z-cut-min')

    parser.add_argument('--z-cut-max', type=float, default=10., required=False,
        help='Use only pairs of object x object with the mean redshift lower than z-cut-max')

    parser.add_argument('--z-min-obj', type=float, default=None, required=False,
        help='Min redshift for object field')

    parser.add_argument('--z-max-obj', type=float, default=None, required=False,
        help='Max redshift for object field')

    parser.add_argument('--z-ref', type=float, default=2.25, required=False,
        help='Reference redshift')

    parser.add_argument('--z-evol-obj', type=float, default=1., required=False,
        help='Exponent of the redshift evolution of the object field')

    parser.add_argument('--z-evol-obj2', type=float, default=1., required=False,
        help='Exponent of the redshift evolution of the object 2 field')

    parser.add_argument('--fid-Om', type=float, default=0.315, required=False,
        help='Omega_matter(z=0) of fiducial LambdaCDM cosmology')

    parser.add_argument('--type-corr', type=str, default='DD', required=False,
        help='type of correlation: DD, RR, DR, RD, xDD, xRR, xD1R2, xD2R1')

    parser.add_argument('--nside', type=int, default=16, required=False,
        help='Healpix nside')

    parser.add_argument('--nproc', type=int, default=None, required=False,
        help='Number of processors')


    args = parser.parse_args()

    if args.nproc is None:
        args.nproc = cpu_count()//2

    co.rp_max = args.rp_max
    co.rp_min = args.rp_min
    co.rt_max = args.rt_max
    co.z_cut_min = args.z_cut_min
    co.z_cut_max = args.z_cut_max
    co.np     = args.np
    co.nt     = args.nt
    co.nside  = args.nside
    co.type_corr = args.type_corr
    if co.type_corr not in ['DD', 'RR', 'DR', 'RD', 'xDD', 'xRR', 'xD1R2', 'xD2R1']:
        print("ERROR: type-corr not in ['DD', 'RR', 'DR', 'RD', 'xDD', 'xRR', 'xD1R2', 'xD2R1']")
        sys.exit()
    if args.drq2 is None:
        co.x_correlation = False
    else:
        co.x_correlation = True

    cosmo = constants.cosmo(args.fid_Om)

    ### Read objects 1
    objs,zmin_obj = io.read_objects(args.drq, args.nside, args.z_min_obj, args.z_max_obj,args.z_evol_obj, args.z_ref, cosmo)
    print("")
    co.objs = objs
    co.ndata = len([o1 for p in co.objs for o1 in co.objs[p]])
    co.angmax = utils.compute_ang_max(cosmo,co.rt_max,zmin_obj)

    ### Read objects 2
    if co.x_correlation:
        objs2,zmin_obj2 = io.read_objects(args.drq2, args.nside, args.z_min_obj, args.z_max_obj, args.z_evol_obj2, args.z_ref,cosmo)
        print("")
        co.objs2 = objs2
        co.angmax = utils.compute_ang_max(cosmo,co.rt_max,zmin_obj,zmin_obj2)

    co.counter = Value('i',0)

    co.lock = Lock()
    cpu_data = {}
    for p in sorted(list(co.objs.keys())):
        cpu_data[p] = [p]
    pool = Pool(processes=args.nproc)

    cfs = pool.map(corr_func,sorted(list(cpu_data.values())))
    pool.close()

    cfs = sp.array(cfs)
    wes = cfs[:,0,:]
    rps = cfs[:,1,:]
    rts = cfs[:,2,:]
    zs  = cfs[:,3,:]
    nbs = cfs[:,4,:].astype(sp.int64)
    hep = sp.array(sorted(list(cpu_data.keys())))

    cut      = (wes.sum(axis=0)>0.)
    rp       = (rps*wes).sum(axis=0)
    rp[cut] /= wes.sum(axis=0)[cut]
    rt       = (rts*wes).sum(axis=0)
    rt[cut] /= wes.sum(axis=0)[cut]
    z        = (zs*wes).sum(axis=0)
    z[cut]  /= wes.sum(axis=0)[cut]
    nb = nbs.sum(axis=0)

    out = fitsio.FITS(args.out,'rw',clobber=True)
    head = [ {'name':'RPMIN','value':co.rp_min,'comment':'Minimum r-parallel [h^-1 Mpc]'},
        {'name':'RPMAX','value':co.rp_max,'comment':'Maximum r-parallel [h^-1 Mpc]'},
        {'name':'RTMAX','value':co.rt_max,'comment':'Maximum r-transverse [h^-1 Mpc]'},
        {'name':'NP','value':co.np,'comment':'Number of bins in r-parallel'},
        {'name':'NT','value':co.nt,'comment':'Number of bins in r-transverse'},
        {'name':'NSIDE','value':co.nside,'comment':'Healpix nside'},
        {'name':'TYPECORR','value':co.type_corr,'comment':'Correlation type'},
        {'name':'NOBJ','value':len([o1 for p in co.objs for o1 in co.objs[p]]),'comment':'Number of objects'},
    ]
    if co.x_correlation:
        head += [{'name':'NOBJ2','value':len([o2 for p in co.objs2 for o2 in co.objs2[p]]),'comment':'Number of objects 2'}]

    comment = ['R-parallel','R-transverse','Redshift','Number of pairs']
    units=['h^-1 Mpc','h^-1 Mpc','','']
    out.write([rp,rt,z,nb],names=['RP','RT','Z','NB'],header=head,comment=comment,units=units,extname='ATTRI')

    comment = ['Healpix index', 'Sum of weight', 'Number of pairs']
    head2 = [{'name':'HLPXSCHM','value':'RING','comment':'healpix scheme'}]
    out.write([hep,wes,nbs],names=['HEALPID','WE','NB'],header=head2,comment=comment,extname='COR')
    out.close()

    print("\nFinished")
