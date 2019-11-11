#!/usr/bin/env python

from picca.fitter2 import sample, parser
from mpi4py import MPI
import argparse

if __name__ == '__main__':

    pars = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Fit the correlation function.')

    pars.add_argument('config', type=str, default=None,
        help='Config file')


    args = pars.parse_args()

    dic_init = parser.parse_chi2(args.config)
    sampler = sample.sample(dic_init)

    run_sampler = dic_init['control'].getboolean('sampler', False)
    run_chi2 = dic_init['control'].getboolean('chi2', False)
    run_chi2_parallel = dic_init['control'].getboolean('chi2_parallel', False)
    if run_chi2_parallel:
        run_chi2 = True

    if run_sampler:
        sampler.run_sampler()

    if run_chi2:
        sampler.chi.minimize()

        comm = MPI.COMM_WORLD
        if run_chi2_parallel:
            rank = comm.Get_rank()
            print('I\'m CPU number %i' % rank)        
            comm.barrier()
            sampler.chi.mpi_chi2scan()
            comm.barrier()
            
            if rank == 0:
                sampler.chi.export()
            comm.barrier()

        else:
            size = comm.Get_size()
            if size > 1:
                print('You called the chi2 with MPI but didn\'t ask for \
                    parallelization. Add "chi2_parallel = True" to [control].' )
                raise ValueError

            sampler.chi.chi2scan()
            sampler.chi.export()
        
    


    # chi.minos()
    # chi.fastMC()
