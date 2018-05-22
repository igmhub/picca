#!/usr/bin/env python

import scipy as sp
import numpy as np
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

comm = None

def corr_func(p):
    if x_correlation:
        cf.fill_neighs_x_correlation(p)
    else:
        cf.fill_neighs(p)
    tmp = cf.cf(p,comm)
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
        
    # rank 0 gets the list of files to read and broadcast the list
    list_of_delta_files = None
    if rank== 0 :
        list_of_delta_files = io.get_list_of_delta_files(args.in_dir,from_image=args.from_image)
    if comm is not None :
        list_of_delta_files = comm.bcast(list_of_delta_files, root=0)
    # each rank has a subsample of the files
    if size==1 : 
        rank_list_of_delta_files = list_of_delta_files
    else :
        rank_list_of_delta_files = np.array_split(list_of_delta_files, size)[rank]
    
    # same thing if second directory
    if  args.in_dir2 is None:
        list_of_delta_files2 = None
        rank_list_of_delta_files2 = None
    else :
        if rank== 0 :
            list_of_delta_files2 = io.get_list_of_delta_files(args.in_dir2,from_image=args.from_image)
        if comm is not None :
            list_of_delta_files2 = comm.bcast(list_of_delta_files2, root=0)
        if size==1 : 
            rank_list_of_delta_files2 = list_of_delta_files2
        else :
            rank_list_of_delta_files2 = np.array_split(list_of_delta_files2, size)[rank]

    if rank==0: 
        t0=time.time()
        
    # read rank_list_of_delta_files
    rank_data, rank_ndata, rank_zmin_pix, rank_zmax_pix = io.read_deltas(
        args.in_dir, args.nside, lambda_abs,
        args.z_evol, args.z_ref, cosmo,
        nspec=args.nspec, no_project=args.no_project,
        from_image=args.from_image,
        list_of_delta_files=rank_list_of_delta_files) # this will read only those delta files
    
    rank_zmin_pix2 = rank_zmin_pix
    if (args.in_dir2 is not None) or (lambda_abs != lambda_abs2):
        x_correlation = True
        if args.in_dir2 is None:
            args.in_dir2 = args.in_dir
        if rank_list_of_delta_files2 is None:
           rank_list_of_delta_files2 = rank_list_of_delta_files
            
        # here data might be read twice from disk...
        rank_data2, rank_ndata2, rank_zmin_pix2, rank_zmax_pix2 = io.read_deltas(
            args.in_dir2, args.nside, lambda_abs2, 
            args.z_evol2, args.z_ref, cosmo, 
            nspec=args.nspec, no_project=args.no_project,
            from_image=args.from_image,
            list_of_delta_files=rank_list_of_delta_files2) # this will read only those delta files
    
    print('INFO rank {} read {} files; npix = {}'.format(rank,len(rank_list_of_delta_files),len(rank_data.keys())))
    sys.stdout.flush()
    
    # just make sure each rank has finished reading its files
    if comm is not None :
        comm.barrier()
    
    if rank==0: 
        t1=time.time()
        print('INFO: finished reading in {} sec'.format(t1-t0))
        print('INFO: now gathering...')
        sys.stdout.flush()
    
    if comm is None :
        data  = rank_data
        ndata = rank_ndata
        data2  = rank_data2
        ndata2 = rank_ndata2
        zmin_pix  = rank_zmin_pix
        zmin_pix2 = rank_zmin_pix2
    else :
        # now gather and hope for the best ...
        # the following is quite awful 
        
        data  = {}
        
        # number of pixels in this rank
        rank_npix = len(rank_data.keys())
        # max number of pixels per rank
        max_rank_npix = np.max(comm.allgather(rank_npix))
        # max number of pixels to exchange with mpi4py without crashing (empirical, 7=ok , 23=not ok)
        max_npix_to_avoid_crash = 5
        nloop = max_rank_npix//max_npix_to_avoid_crash + (max_rank_npix%max_npix_to_avoid_crash>0)
        
        if rank==0 :
            print("nloop of all gather = {}/{}+1 = {}".format(max_rank_npix,max_npix_to_avoid_crash,nloop))
            sys.stdout.flush()

        for loop in range(nloop) :
            if rank==0 :
                print("allgather iter {}/{}".format(loop+1,nloop))
                sys.stdout.flush()
            tmp_rank_data={}
            for k in list(rank_data.keys())[loop::nloop] :
                tmp_rank_data[k]=rank_data[k]
            tmp_data = comm.allgather(tmp_rank_data)
            if tmp_data is not None :
                for tmp_rank_data in tmp_data :
                    for k in tmp_rank_data.keys() : data[k]=tmp_rank_data[k]
        
        ndata = np.sum(comm.allgather(rank_ndata))
        zmin_pix  = np.min(comm.allgather(rank_zmin_pix))
        zmin_pix2 = np.min(comm.allgather(rank_zmin_pix2))


        
        if x_correlation :
            data2={}
            # same
            # number of pixels in this rank
            rank_npix = len(rank_data2.keys())
            # max number of pixels per rank
            max_rank_npix = np.max(comm.allgather(rank_npix))
            # max number of pixels to exchange with mpi4py without crashing (empirical, 7=ok , 23=not ok)
            max_npix_to_avoid_crash = 5
            nloop = max_rank_npix//max_npix_to_avoid_crash + (max_rank_npix%max_npix_to_avoid_crash>0)
        
            for loop in range(nloop) :
                if rank==0 :
                    print("allgather iter {}/{} (sec. data set)".format(loop+1,nloop))
                    sys.stdout.flush()
                tmp_rank_data={}
                for k in list(rank_data2.keys())[loop::nloop] :
                    tmp_rank_data[k]=rank_data2[k]
                tmp_data = comm.allgather(tmp_rank_data)
                if tmp_data is not None :
                    for tmp_rank_data in tmp_data :
                        for k in tmp_rank_data.keys() : data2[k]=tmp_rank_data[k]
            ndata2 = np.sum(comm.allgather(rank_ndata2))
            
        else :
            data2  = None
            ndata2 = None

        if rank==0: 
            print('INFO: finished gathering in {} sec'.format(time.time()-t1))
            print('INFO: zmin_pix = {}'.format(zmin_pix))
            sys.stdout.flush()

        # just make sure we indeed finished gathering data
        comm.barrier()
    
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

    print('INFO: rank {} will compute cf in {} pixels'.format(rank, len(cpu_data)))
    #print('INFO: rank {} will compute cf in {} pixels: {}'.format(rank, len(cpu_data), cpu_data))
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

