#!/usr/bin/env python

import scipy as sp
import fitsio
import argparse
import sys
from multiprocessing import Pool,Lock,cpu_count,Value
import time
import pickle

from picca import constants, xcf, io, prep_del, utils
from picca.data import forest

import healpy

def corr_func(p):
    print('calling xcf on {} pixels and {} neighs'.format(len(p[0]), len(p[1])))
    sys.stdout.flush()
    tmp = xcf.xcf(p[0], p[1], p[2])
    return tmp

class Config(object):
    pass

if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--out', type = str, default = None, required=True,
                        help = 'output file name')

    parser.add_argument('--in-dir', type = str, default = None, required=True,
                        help = 'data directory')

    parser.add_argument('--drq', type = str, default = None, required=True,
                        help = 'drq')

    parser.add_argument('--rp-max', type = float, default = 200., required=False,
                        help = 'max rp [h^-1 Mpc]')

    parser.add_argument('--rp-min', type = float, default = -200., required=False,
                        help = 'min rp [h^-1 Mpc]')

    parser.add_argument('--rt-max', type = float, default = 200., required=False,
                        help = 'max rt [h^-1 Mpc]')

    parser.add_argument('--np', type = int, default = 100, required=False,
                        help = 'number of r-parallel bins')

    parser.add_argument('--nt', type = int, default = 50, required=False,
                        help = 'number of r-transverse bins')

    parser.add_argument('--lambda-abs', type = str, default = 'LYA', required=False,
                        help = 'name of the absorption in picca.constants')

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

    parser.add_argument('--z-cut-min', type = float, default = 0., required=False,
                        help = 'use only pairs of forest/qso with the mean of the last absorber redshift and the qso redshift higher than z-cut-min')

    parser.add_argument('--z-cut-max', type = float, default = 10., required=False,
                        help = 'use only pairs of forest/qso with the mean of the last absorber redshift and the qso redshift smaller than z-cut-min')

    parser.add_argument('--z-evol-del', type = float, default = 2.9, required=False,
                    help = 'exponent of the redshift evolution of the delta field')

    parser.add_argument('--z-evol-obj', type = float, default = 1., required=False,
                    help = 'exponent of the redshift evolution of the object field')

    parser.add_argument('--no-project', action="store_true", required=False,
                    help = 'do not project out continuum fitting modes')

    parser.add_argument('--nspec', type=int,default=None, required=False,
                    help = 'maximum spectra to read')

    parser.add_argument('--no-remove-mean-lambda-obs', action="store_true", required=False,
                    help = 'Do not remove mean delta versus lambda_obs')

    parser.add_argument('--from-image', type = str, default = None, required=False,
                    help = 'use image format to read deltas', nargs='*')

    parser.add_argument('--mpi', action="store_true", required=False,
                    help = 'use mpi parallelization')

    args = parser.parse_args()

    if args.nproc is None:
        args.nproc = cpu_count()//2

#    xcf.rp_max = args.rp_max
#    xcf.rp_min = args.rp_min
#    xcf.z_cut_max = args.z_cut_max
#    xcf.z_cut_min = args.z_cut_min
#    xcf.rt_max = args.rt_max
#    xcf.np = args.np
#    xcf.nt = args.nt
#    xcf.nside = args.nside
#    xcf.lambda_abs = constants.absorber_IGM[args.lambda_abs]

    rp_min = args.rp_min
    rp_max = args.rp_max
    rt_max = args.rt_max
    np = args.np
    nt = args.nt
    nside = args.nside
    lambda_abs = constants.absorber_IGM[args.lambda_abs]
    z_cut_min = args.z_cut_min
    z_cut_max = args.z_cut_max

    rt_max = args.rt_max
    config = Config()
    config.rp_min = rp_min
    config.rp_max = rp_max
    config.rt_max = rt_max
    config.np = np
    config.nt = nt
    config.nside = nside

    cosmo = constants.cosmo(args.fid_Om)

    t0 = time.time()

    ### Read deltas
    comm = None
    rank = 0
    size = 1
    
    if args.mpi:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.rank
        size = comm.size
 
    dels = {}
    ndels = 0
    zmin_pix = None
    zmax_pix = None
    zmin_obj = None
    objs = None

    if rank == 0:
        print('Starting io in rank 0\n')
        sys.stdout.flush()
        t0 = time.time()
        dels, ndels, zmin_pix, zmax_pix = io.read_deltas(args.in_dir, 
                args.nside, lambda_abs,args.z_evol_del, args.z_ref, 
                cosmo=cosmo,nspec=args.nspec,no_project=args.no_project,
                from_image=args.from_image)
        print('done io in {}\n'.format(time.time()-t0))
        sys.stdout.flush()

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

        ### Find the redshift range
        if args.z_min_obj is None:
            dmin_pix = cosmo.r_comoving(zmin_pix)
            dmin_obj = max(0.,dmin_pix+rp_min)
            args.z_min_obj = cosmo.r_2_z(dmin_obj)
            print("z_min_obj = {}".format(args.z_min_obj))
        if args.z_max_obj is None:
            dmax_pix = cosmo.r_comoving(zmax_pix)
            dmax_obj = max(0.,dmax_pix+rp_max)
            args.z_max_obj = cosmo.r_2_z(dmax_obj)
            print("z_max_obj = {}".format(args.z_max_obj))

        print('read deltas in {}'.format(time.time()-t0))
        sys.stdout.flush()

        t0 = time.time()
        ### Read objects
        objs,zmin_obj = io.read_objects(args.drq, args.nside, args.z_min_obj,
                args.z_max_obj, args.z_evol_obj, args.z_ref,cosmo)
        print('read objects in {}\n'.format(time.time()-t0))
        sys.stdout.flush()


    ## broadcast to all nodes
    if comm is not None:
        ## wait until rank 0 has read all the data
        comm.Barrier()
        t0 = MPI.Wtime()
        ## broadcast deltas in chunks
        ## first send the pixels
        pix = comm.bcast(list(dels.keys()), root=0)
        dels_tmp = {}
        for p in pix:
            if rank==0:
                print('broadcasting {}'.format(p))
                sys.stdout.flush()
            if rank != 0:
                dels[p] = None
            dels[p] = comm.bcast(dels[p], root=0)
            dels_tmp[p] = dels[p]

        dels = dels_tmp
        #dels = comm.bcast(dels, root=0)
        ndels = comm.bcast(ndels, root=0)
        objs = comm.bcast(objs, root=0)
        zmin_pix = comm.bcast(zmin_pix, root=0)
        zmin_obj = comm.bcast(zmin_obj, root=0)
        comm.Barrier()
        if rank == 0:
            print('INFO: broadcasted data in {}'.format(MPI.Wtime()-t0))
            sys.stdout.flush()

    ###
    angmax = utils.compute_ang_max(cosmo,rt_max,zmin_pix,zmin_obj)
    config.angmax = angmax

    print("done, npix = {}".format(len(dels)))
    sys.stdout.flush()

    pix = list(dels.keys())[rank::size]
    pix = sorted(pix)

    neighs = []
    for p in pix:
        center = healpy.pix2vec(args.nside,p)
        neigh = healpy.query_disc(args.nside, center, angmax)
        neigh = [objs[p] for p in neigh if p in objs]
        neighs.append(neigh)
    
    data_pix = [dels[p] for p in pix]
    cpu_data = list(zip(data_pix, neighs, [config]*len(pix)))
    nproc = min(args.nproc, len(pix))
    print('starting pool in rank {} with {} pixels'.format(rank, len(pix)))
    pool = Pool(processes=nproc)
    cfs = pool.map(corr_func, cpu_data)
#    cfs = list(map(corr_func, cpu_data))
    print('done pool in rank {}'.format(rank))
    pool.close()
    
    if comm is not None:
        cfs = comm.gather(cfs)
        cpu_data = comm.gather(cpu_data)
        print(len(cfs))
        if rank == 0:
            cfs = [cf for l in cfs for cf in l]
            cpu_data = [p for l in cpu_data for p in l]

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
        z[cut]  /= wes.sum(axis=0)[cut]
        nb = nbs.sum(axis=0)

        out = fitsio.FITS(args.out,'rw',clobber=True)
        head = {}
        head['RPMIN'] = rp_min
        head['RPMAX'] = rp_max
        head['RTMAX'] = rt_max
        head['Z_CUT_MIN'] = z_cut_min
        head['Z_CUT_MAX'] = z_cut_max
        head['NT'] = nt
        head['NP'] = np
        head['NSIDE'] = nside

        out.write([rp,rt,z,nb],names=['RP','RT','Z','NB'], header=head)
        head2 = [{'name':'HLPXSCHM','value':'RING','comment':'healpix scheme'}]
        out.write([hep,wes,cfs],names=['HEALPID','WE','DA'], header=head2)
        out.close()
