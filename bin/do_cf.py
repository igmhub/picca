#!/usr/bin/env python

import scipy as sp
import fitsio
import argparse
import glob
import healpy
import sys
import copy
from multiprocessing import Pool,Lock,cpu_count,Value
import time

from picca import constants, cf, utils, io
from picca.data import delta

def corr_func(p):
    if x_correlation:
        cf.fill_neighs_x_correlation(p)
    else:
        cf.fill_neighs(p)
    tmp = cf.cf(p)
    return tmp

if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--out', type = str, default = None, required=True,
                        help = 'output file name')

    parser.add_argument('--in-dir', type = str, default = None, required=True,
                        help = 'data directory')

    parser.add_argument('--in-dir2', type = str, default = None, required=False,
                        help = 'data directory #2, for forest x-correlation')

    parser.add_argument('--rp-max', type = float, default = 200., required=False,
                        help = 'max rp [h^-1 Mpc]')

    parser.add_argument('--rp-min', type = float, default = 0., required=False,
                        help = 'min rp., rp can be <0 [h^-1 Mpc]')

    parser.add_argument('--rt-max', type = float, default = 200., required=False,
                        help = 'max rt [h^-1 Mpc]')

    parser.add_argument('--np', type = int, default = 50, required=False,
                        help = 'number of r-parallel bins')

    parser.add_argument('--nt', type = int, default = 50, required=False,
                        help = 'number of r-transverse bins')

    parser.add_argument('--lambda-abs', type = str, default = 'LYA', required=False,
                        help = 'name of the absorption in picca.constants')

    parser.add_argument('--lambda-abs2', type = str, default = None, required=False,
                        help = 'name of the 2nd absorption in picca.constants')

    parser.add_argument('--fid-Om', type = float, default = 0.315, required=False,
                    help = 'Om of fiducial cosmology')

    parser.add_argument('--nside', type = int, default = 16, required=False,
                    help = 'healpix nside')

    parser.add_argument('--nproc', type = int, default = None, required=False,
                    help = 'number of processors')

    parser.add_argument('--z-ref', type = float, default = 2.25, required=False,
                    help = 'reference redshift')

    parser.add_argument('--z-cut-min', type = float, default = 0., required=False,
                        help = 'use only pairs of forests with the mean redshift of the last absorbers higher than z-cut-min')

    parser.add_argument('--z-cut-max', type = float, default = 10., required=False,
                        help = 'use only pairs of forests with the mean redshift of the last absorbers smaller than z-cut-max')

    parser.add_argument('--z-evol', type = float, default = 2.9, required=False,
                    help = 'exponent of the redshift evolution of the delta field')

    parser.add_argument('--z-evol2', type = float, default = 2.9, required=False,
                    help = 'exponent of the redshift evolution of the 2nd delta field')

    parser.add_argument('--no-project', action="store_true", required=False,
                    help = 'do not project out continuum fitting modes')

    parser.add_argument('--from-image', type = str, default = None, required=False,
                    help = 'use image format to read deltas', nargs='*')

    parser.add_argument('--nspec', type=int,default=None, required=False,
                    help = 'maximum spectra to read')

    parser.add_argument('--no-same-wavelength-pairs', action="store_true", required=False,
                    help = 'Reject pairs with same wavelength')

    parser.add_argument('--mpi', action="store_true", required=False,
                    help = 'use mpi')

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
    if args.lambda_abs2 is not None:
        lambda_abs2 = constants.absorber_IGM[args.lambda_abs2]
    else:
        lambda_abs2 = constants.absorber_IGM[args.lambda_abs]

    cf.lambda_abs = lambda_abs
    cf.lambda_abs2 = lambda_abs2

    data = {}
    ndata = 0
    x_correlation = False
    fi = []

    zmin_pix  = None
    zmin_pix2 = None
    

    rank = 0
    size = 1
    comm = None
    if args.mpi:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.rank
        size = comm.size

    if rank == 0:
        t0 = time.time()
        data, ndata, zmin_pix, zmax_pix = io.read_deltas(
                args.in_dir, args.nside, lambda_abs,
                args.z_evol, args.z_ref, cosmo,
                nspec=args.nspec, no_project=args.no_project,
                from_image=args.from_image)

        zmin_pix2 = zmin_pix
        if (args.in_dir2 is not None) or (lambda_abs != lambda_abs2):
            x_correlation = True
            if args.in_dir2 is None:
                args.in_dir2 = args.in_dir

            ## here data might be read twice from disk...
            data2, ndata2, zmin_pix2, zmax_pix2 = io.read_deltas(
                args.in_dir2, args.nside, lambda_abs2, 
                args.z_evol2, args.z_ref, cosmo, 
                nspec=args.nspec, no_project=args.no_project,
                from_image=args.from_image)

    
        print('INFO: finished io in {} sec'.format(time.time()-t0))
        sys.stdout.flush()
    
    sys.stderr.write("\n")

    ## end io
    ## broadcast if needed

    if comm is not None:
        comm.Barrier()
        t0 = MPI.Wtime()
        data = comm.bcast(data, root=0)
        ndata = comm.bcast(ndata, root=0)
        x_correlation = comm.bcast(x_correlation, root=0)
        if x_correlation:
            data2 = comm.bcast(data2, root=0)
            ndata2 = comm.bcast(ndata2, root=0)
        
        zmin_pix  = comm.bcast(zmin_pix, root=0)
        zmin_pix2 = comm.bcast(zmin_pix2, root=0)
        
    
        comm.Barrier()
        if rank == 0:
            print('INFO: Broadcasted input data in {} sec'.format(MPI.Wtime()-t0))
            sys.stdout.flush()

    cf.angmax = utils.compute_ang_max(cosmo,cf.rt_max,zmin_pix, zmin_pix2)
    
    cf.npix = len(data)
    cf.data = data
    cf.ndata = ndata

    if x_correlation:
        cf.data2 = data2
        cf.ndata2 = ndata2
        cf.x_correlation = x_correlation

    cf.counter = Value('i',0)

    cf.lock = Lock()
    cpu_data = list(data.keys())[rank::size]
    cpu_data = sorted(cpu_data)
    cpu_data = [[p] for p in cpu_data]

    #print('INFO: rank {} will compute cf in {} pixels'.format(rank, len(cpu_data)))
    print('INFO: rank {} will compute cf in {} pixels: {}'.format(rank, len(cpu_data), cpu_data))
    sys.stdout.flush()
    
    pool = Pool(processes=args.nproc)

    cfs = pool.map(corr_func, cpu_data)
    pool.close()

    print('INFO: rank {} finished'.format(rank))
    sys.stdout.flush()
    
    if comm is not None:
        cfs = comm.gather(cfs)
        cpu_data = comm.gather(cpu_data)
        if rank == 0:
            cfs = [cf for l in cfs for cf in l]
            cpu_data = [p for l in cpu_data for p in l]

    if rank == 0:
        cfs = sp.array(cfs)
        print("cfs shape", cfs.shape)
        sys.stdout.flush()
        
        wes=cfs[:,0,:]
        rps=cfs[:,2,:]
        rts=cfs[:,3,:]
        zs=cfs[:,4,:]
        nbs=cfs[:,5,:].astype(sp.int64)
        cfs=cfs[:,1,:]

        hep = sp.array(cpu_data)

        cut = (wes.sum(axis=0)>0.)
        rp  = (rps*wes).sum(axis=0)
        rp[cut] /= wes.sum(axis=0)[cut]
        rt = (rts*wes).sum(axis=0)
        rt[cut] /= wes.sum(axis=0)[cut]
        z = (zs*wes).sum(axis=0)
        z[cut]  /= wes.sum(axis=0)[cut]
        nb = nbs.sum(axis=0)

        out = fitsio.FITS(args.out,'rw',clobber=True)
        head = {}
        head['RPMIN']=cf.rp_min
        head['RPMAX']=cf.rp_max
        head['RTMAX']=cf.rt_max
        head['Z_CUT_MIN']=cf.z_cut_min
        head['Z_CUT_MAX']=cf.z_cut_max
        head['NT']=cf.nt
        head['NP']=cf.np
        head['NSIDE']=cf.nside

        out.write([rp,rt,z,nb],names=['RP','RT','Z','NB'],header=head)
        ## use the default scheme in healpy => RING
        head2 = [{'name':'HLPXSCHM','value':'RING','comment':'healpix scheme'}]
        out.write([hep,wes,cfs],names=['HEALPID','WE','DA'],header=head2)
        out.close()
        print("wrote",args.out)

