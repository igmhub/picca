#!/usr/bin/env python

import scipy as sp
import fitsio
import argparse
import sys
from multiprocessing import Pool,Lock,cpu_count,Value

from picca import constants, xcf, io, prep_del
from picca.data import forest
from picca.utils import Config, form_cpu_data

def corr_func(p):
    tmp = xcf.xcf(p[0], p[1], p[2])
    return tmp

if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--out', type = str, default = None, required=True,
                    help = 'output file name')

    parser.add_argument('--in-dir', type = str, default = None, required=False,
                    help = 'data directory')
    parser.add_argument('--in-files', type = str, default = None, required=False,
                    help = 'data files', nargs="*")

    parser.add_argument('--drq', type = str, default = None, required=True,
                    help = 'drq')

    parser.add_argument('--wr-min', type = float, default = 0.9, required=False,
                    help = 'min of wavelength ratio')

    parser.add_argument('--wr-max', type = float, default = 1.1, required=False,
                    help = 'max of wavelength ratio')

    parser.add_argument('--ang-max', type = float, default = 0.02, required=False,
                    help = 'max angle')

    parser.add_argument('--np', type = int, default = 100, required=False,
                    help = 'number of parallel bins')

    parser.add_argument('--nt', type = int, default = 50, required=False,
                    help = 'number of transverse bins')

    parser.add_argument('--lambda-abs', type = str, default = 'LYA', required=False,
                    help = 'name of the absorption in picca.constants')

    parser.add_argument('--lambda-abs-obj', type = str, default = 'LYA', required=False,
                    help = 'name of the absorption in picca.constants the object is considered as')

    parser.add_argument('--fid-Om', type = float, default = 0.315, required=False,
                    help = 'Om of fiducial cosmology')

    parser.add_argument('--nside', type = int, default = 16, required=False,
                    help = 'healpix nside')

    parser.add_argument('--nproc', type = int, default = None, required=False,
                    help = 'number of processors')

    parser.add_argument('--z-min-obj', type = float, default = None, required=False,
                    help = 'min redshift for object field')

    parser.add_argument('--z-max-obj', type = float, default = None, required=False,
                    help = 'max redshift for object field')

    parser.add_argument('--z-ref', type = float, default = 2.25, required=False,
                    help = 'reference redshift')

    parser.add_argument('--z-evol-del', type = float, default = 2.9, required=False,
                    help = 'exponent of the redshift evolution of the delta field')

    parser.add_argument('--z-evol-obj', type = float, default = 1., required=False,
                    help = 'exponent of the redshift evolution of the object field')

    parser.add_argument('--z-cut-min', type = float, default = 0., required=False,
                        help = 'use only pairs of forest/qso with the mean of the last absorber redshift and the qso redshift higher than z-cut-min')

    parser.add_argument('--z-cut-max', type = float, default = 10., required=False,
                        help = 'use only pairs of forest/qso with the mean of the last absorber redshift and the qso redshift smaller than z-cut-min')

    parser.add_argument('--no-project', action="store_true", required=False,
                    help = 'do not project out continuum fitting modes')

    parser.add_argument('--nspec', type=int,default=None, required=False,
                    help = 'maximum spectra to read')

    parser.add_argument('--no-remove-mean-lambda-obs', action="store_true", required=False,
                    help = 'Do not remove mean delta versus lambda_obs')

    parser.add_argument('--mpi', action="store_true", required=False,
                    help = 'use mpi')


    args = parser.parse_args()

    if args.nproc is None:
        args.nproc = cpu_count()//2

    config = Config()
    config.rp_min = args.wr_min
    config.rp_max = args.wr_max
    config.rt_max = args.ang_max
    config.z_cut_min = args.z_cut_min
    config.z_cut_max = args.z_cut_max
    config.np = args.np
    config.nt = args.nt
    config.nside = args.nside
    config.ang_correlation = True
    config.angmax = args.ang_max

    lambda_abs = constants.absorber_IGM[args.lambda_abs]

    cosmo = constants.cosmo(args.fid_Om)

    comm = None
    rank = 0
    size = 1
    if args.mpi:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.rank
        size = comm.size
        if args.files is not None:
            args.files = args.files[rank::size]

    ### Read deltas
    dels, ndels, zmin_pix, zmax_pix = io.read_deltas(args.in_dir, 
            args.in_files, args.nside, 
            constants.absorber_IGM[args.lambda_abs],
            args.z_evol_del, args.z_ref, 
            cosmo=cosmo,nspec=args.nspec,no_project=args.no_project)

    print("rank {} read {} deltas".format(rank, ndels))

    ### Remove <delta> vs. lambda_obs
    if not args.no_remove_mean_lambda_obs:
        forest.dll = None
        for p in dels:
            for d in dels[p]:
                dll = sp.asarray([d.ll[ii]-d.ll[ii-1] for ii in range(1,d.ll.size)]).min()
                if forest.dll is None:
                    forest.dll = dll
                else:
                    forest.dll = min(dll,forest.dll)
        forest.lmin  = sp.log10( (zmin_pix+1.)*lambda_abs )-forest.dll/2.
        forest.lmax  = sp.log10( (zmax_pix+1.)*lambda_abs )+forest.dll/2.
        ll,st, wst   = prep_del.stack(dels,delta=True)
        for p in dels:
            for d in dels[p]:
                bins = ((d.ll-forest.lmin)/forest.dll+0.5).astype(int)
                d.de -= st[bins]

    ### Read objects only in rank 0
    if rank == 0:
        objs,zmin_obj = io.read_objects(args.drq, args.nside, 
                args.z_min_obj, args.z_max_obj,
                args.z_evol_obj, args.z_ref,cosmo)
        for i,ipix in enumerate(sorted(objs.keys())):
            for q in objs[ipix]:
                q.ll = sp.log10((1.+q.zqso)*constants.absorber_IGM[args.lambda_abs_obj] )

    ### share data to all nodes:
    if comm is not None:
        all_dels = comm.allgather(dels)
        dels = {}
        for de in all_dels:
            for p in de:
                if not p in dels:
                    dels[p] = []
                dels[p] += de[p]

        ## share objects from rank 0. Somehow using bcast
        ## doesn't work 
        objs = comm.allgather(objs)
        objs = objs[0]

    ### Send
    pix, cpu_data = form_cpu_data(dels, objs, config, rank, size)

    pool = Pool(processes=args.nproc)
    cfs = pool.map(corr_func, cpu_data)
    pool.close()

    if comm is not None:
        cfs = comm.gather(cfs)
        pix = comm.gather(pix)
        if rank == 0:
            cfs = [cf for l in cfs for cf in l]


    ### Store
    if rank == 0:
        cfs=sp.array(cfs)
        wes=cfs[:,0,:]
        rps=cfs[:,2,:]
        rts=cfs[:,3,:]
        zs=cfs[:,4,:]
        nbs=cfs[:,5,:].astype(sp.int64)
        cfs=cfs[:,1,:]
        hep=sp.array(pix)

        cut = (wes.sum(axis=0)>0.)
        rp = (rps*wes).sum(axis=0)
        rp[cut] /= wes.sum(axis=0)[cut]
        rt = (rts*wes).sum(axis=0)
        rt[cut] /= wes.sum(axis=0)[cut]
        z = (zs*wes).sum(axis=0)
        z[cut] /= wes.sum(axis=0)[cut]
        nb = nbs.sum(axis=0)

        out = fitsio.FITS(args.out,'rw',clobber=True)
        head = {}
        head['RPMIN'] = config.rp_min
        head['RPMAX'] = config.rp_max
        head['RTMAX'] = config.rt_max
        head['Z_CUT_MIN'] = config.z_cut_min
        head['Z_CUT_MAX'] = config.z_cut_max
        head['NT'] = config.nt
        head['NP'] = config.np
        head['NSIDE'] = config.nside

        out.write([rp,rt,z,nb],names=['RP','RT','Z','NB'],header=head)
        head2 = [{'name':'HLPXSCHM','value':'RING','comment':'healpix scheme'}]
        out.write([hep,wes,cfs],names=['HEALPID','WE','DA'],header=head2)
        out.close()
