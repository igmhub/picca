from __future__ import print_function, division
import numpy as np
import sys
import copy
from mpi4py import MPI
from scipy.linalg import cholesky

from . import sampler, control


class fitter2_mpi(control.fitter2):
    '''
    Interface for the fitter2 that supports parallel computation.
    Use this to run parallel chi2 scans or fastMC mocks.
    Main interface to running the sampler.
    '''

    def __init__(self, chi2_file):
        super().__init__(chi2_file)

        # Figure out what we need to run
        self.run_sampler = self.control.getboolean('sampler', False)
        self.chi2_parallel = self.control.getboolean('chi2_parallel', False)
        if self.chi2_parallel:
            self.run_chi2 = True

        # Initialize the required objects
        if self.run_sampler:
            self.sampler = sampler.sampler(self.dic_init)

    def run(self):
        ''' Runs the fitter.
        This function is designed to run on multiple cores
        '''

        if not self.chi2_parallel and not self.run_sampler:
            raise ValueError('You called the fitter with MPI but didn\'t ask \
                for parallelization. Add "chi2_parallel = True" to [control] \
                for parallel chi2 or "sampler = True" to run PolyChord.')

        mpi_comm = MPI.COMM_WORLD
        cpu_rank = mpi_comm.Get_rank()
        if self.chi2_parallel:
            # First minimize - same on all CPUs
            self.chi2.minimize()
            mpi_comm.barrier()

            # Compute scan - parallelized
            self._mpi_chi2scan()
            mpi_comm.barrier()

            # Compute fastMC - parallelized
            self._mpi_fastMC()
            mpi_comm.barrier()

            # Export only once
            if cpu_rank == 0:
                self.chi2.export()
            mpi_comm.barrier()

        # Run Sampler
        if self.run_sampler:
            self.sampler.run()

    def _mpi_chi2scan(self):
        '''
        Run a chisq scan in parallel using MPI.
        Works with 1D and 2D grids.
        '''
        if not hasattr(self.chi2, "dic_chi2scan"):
            return

        mpi_comm = MPI.COMM_WORLD
        cpu_rank = mpi_comm.Get_rank()
        num_cpus = mpi_comm.Get_size()

        dim = len(self.chi2.dic_chi2scan)

        # Set all parameters to the minimum and store the current state
        store_data_pars = {}
        for d in self.chi2.data:
            store_d_pars_init = {}
            store_d_par_error = {}
            store_d_par_fixed = {}
            for name in d.pars_init.keys():
                store_d_pars_init[name] = d.pars_init[name]
                d.pars_init[name] = self.chi2.best_fit.values[name]
            for name in d.par_error.keys():
                store_d_par_error[name] = d.par_error[name]
                d.par_error[name] = self.chi2.best_fit.errors[name.split('error_')[1]]
            for name in d.par_fixed.keys():
                store_d_par_fixed[name] = d.par_fixed[name]
            store_data_pars[d.name] = {'init':store_d_pars_init, 'error':store_d_par_error, 'fixed':store_d_par_fixed}

        # Overwrite the run parameters
        for p in self.chi2.dic_chi2scan.keys():
            for d in self.chi2.data:
                if 'error_'+p in d.par_error.keys():
                    d.par_error['error_'+p] = 0.
                if 'fix_'+p in d.par_fixed.keys():
                    d.par_fixed['fix_'+p] = True

        def send_one_fit():
            ''' Minimize the chisq and return the bestfit '''
            try:
                best_fit = self.chi2._minimize()
                chi2_result = best_fit.fval
            except ValueError:
                chi2_result = np.nan
            tresult = []
            for p in sorted(best_fit.values):
                tresult += [best_fit.values[p]]
            tresult += [chi2_result]
            return tresult

        def mpi_gather_data(send_buff):
            ''' Gather the data on CPU #0.
            Assumes each CPU has a send buffer
            '''
            recv_buff = None
            if cpu_rank == 0:
                recv_buff = np.empty([num_cpus, len(send_buff)], dtype=np.float64)
            mpi_comm.barrier()
            mpi_comm.Gather(send_buff, recv_buff, root=0)
            return recv_buff

        def mpi_run_1d_set(num_runs, grid, par):
            '''
            Runs a 1D set in parallel. Grid must be 1D array with each value of par to run.
            Number of computations must be smaller or equal to the number of CPUs
            Results are only stored on CPU #0
            '''
            assert num_runs <= num_cpus
            mpi_comm.barrier()

            for i in range(num_cpus):
                if i == cpu_rank:
                    val = grid[i] if cpu_rank < num_runs else grid[-1]
                    for d in self.chi2.data:
                        if par in d.pars_init.keys():
                            d.pars_init[par] = val
                    local_result = np.asarray(send_one_fit())

            mpi_comm.barrier()
            results = mpi_gather_data(local_result)
            mpi_comm.barrier()
            if cpu_rank == 0 and num_runs < num_cpus:
                results = results[:num_runs]
            return results

        def mpi_run_2d_set(num_runs, grid, par1, par2):
            '''
            Runs a 2D set in parallel. Grid must be 2D array of shape (num_runs, 2).
            Number of computations must be smaller or equal to the number of CPUs
            Results are only stored on CPU #0
            '''
            assert num_runs <= num_cpus
            mpi_comm.barrier()

            for i in range(num_cpus):
                if i == cpu_rank:
                    vals = grid[i] if cpu_rank < num_runs else grid[-1]
                    for d in self.chi2.data:
                        if par1 in d.pars_init.keys():
                            d.pars_init[par1] = vals[0]
                        if par2 in d.pars_init.keys():
                            d.pars_init[par2] = vals[1]
                    local_result = np.asarray(send_one_fit())

            mpi_comm.barrier()
            results = mpi_gather_data(local_result)
            mpi_comm.barrier()
            if cpu_rank == 0 and num_runs < num_cpus:
                results = results[:num_runs]
            return results

        result = []
        # Run 1D grid
        if dim==1:
            par = list(self.chi2.dic_chi2scan.keys())[0]
            grid = self.chi2.dic_chi2scan[par]['grid']
            num_runs = len(grid)  # Number of points to be run

            if num_runs <= num_cpus:
                result = mpi_run_1d_set(num_runs, grid, par)
            else:
                j = 0
                for __ in range(num_runs // num_cpus):
                    result += [mpi_run_1d_set(num_cpus, grid[j:j+num_cpus], par)]
                    j += num_cpus
                rest = num_runs % num_cpus
                if rest != 0:
                    result += [mpi_run_1d_set(rest, grid[-rest:], par)]

        # Run 2D grid
        elif dim == 2:
            par1 = list(self.chi2.dic_chi2scan.keys())[0]
            par2 = list(self.chi2.dic_chi2scan.keys())[1]
            values1 = self.chi2.dic_chi2scan[par1]['grid']
            values2 = self.chi2.dic_chi2scan[par2]['grid']
            grid = []
            for v1 in values1:
                for v2 in values2:
                    grid += [[v1, v2]]
            grid = np.asarray(grid)
            num_runs = len(grid)  # Number of points to be run

            if num_runs <= num_cpus:
                result = mpi_run_2d_set(num_runs, grid, par1, par2)
            else:
                j = 0
                for __ in range(num_runs // num_cpus):
                    result += [mpi_run_2d_set(num_cpus, grid[j:j+num_cpus], par1, par2)]
                    j += num_cpus
                rest = num_runs % num_cpus
                if rest != 0:
                    result += [mpi_run_2d_set(rest, grid[-rest:], par1, par2)]

        # If we are on CPU #0 concatenate results to match the normal output
        if cpu_rank == 0 and num_runs > num_cpus:
            temp = result[0]
            for res in result[1:]:
                temp = np.r_[temp, res]
            result = temp

        self.chi2.dic_chi2scan_result = {}
        self.chi2.dic_chi2scan_result['params'] = np.asarray(np.append(sorted(self.chi2.best_fit.values), ['fval']))
        self.chi2.dic_chi2scan_result['values'] = np.asarray(result)

        # Set all parameters to where they were before
        for d in self.chi2.data:
            store_d_pars_init = store_data_pars[d.name]['init']
            store_d_par_error = store_data_pars[d.name]['error']
            store_d_par_fixed = store_data_pars[d.name]['fixed']
            for name in d.pars_init.keys():
                d.pars_init[name] = store_d_pars_init[name]
            for name in d.par_error.keys():
                d.par_error[name] = store_d_par_error[name]
            for name in d.par_fixed.keys():
                d.par_fixed[name] = store_d_par_fixed[name]

    def _mpi_fastMC(self):
        '''
        Run fastMC mocks in parallel using MPI.
        Each CPU will output its own .h5 file when it finishes its allotted computation.

        In some cases the model fails and we get ValueError when minimizing. For these
        runs, the output will be filled with inf and should be ignored in the analysis until
        we figure out why they happen.
        '''
        if not hasattr(self.chi2, "nfast_mc"):
            return

        mpi_comm = MPI.COMM_WORLD
        cpu_rank = mpi_comm.Get_rank()
        num_cpus = mpi_comm.Get_size()

        # Seed is incremented from the input value by the CPU #
        np.random.seed(self.chi2.seedfast_mc + cpu_rank)
        nfast_mc = self.chi2.nfast_mc // num_cpus
        # If the division is not exact we will run one more set
        # This means that we are actually computing more than requested
        # This makes the implementation simpler and makes sure no CPU is idle
        if self.chi2.nfast_mc % num_cpus != 0:
            nfast_mc += 1

        # Scale the cov and compute Cholesky
        for d, s in zip(self.chi2.data, self.chi2.scalefast_mc):
            d.co = s*d.co
            d.ico = d.ico/s
            d.cho = cholesky(d.co)

        # Initialize fiducial values
        self.chi2.fiducial_values = dict(self.chi2.best_fit.values).copy()
        for p in self.chi2.fidfast_mc:
            self.chi2.fiducial_values[p] = self.chi2.fidfast_mc[p]
            for d in self.chi2.data:
                if p in d.par_names:
                    d.pars_init[p] = self.chi2.fidfast_mc[p]
                    d.par_fixed['fix_'+p] = self.chi2.fixfast_mc['fix_'+p]

        # Compute fiducial model
        # This is copied from the chi2.py which in turn comes from data.py
        # This functionality should be standalone and in one place in the future
        self.chi2.fiducial_values['SB'] = False
        for d in self.chi2.data:
            # Compute Xi Peak
            d.fiducial_model = self.chi2.fiducial_values['bao_amp'] \
                * d.xi_model(self.chi2.k, self.chi2.pk_lin-self.chi2.pksb_lin, self.chi2.fiducial_values)

            self.chi2.fiducial_values['SB'] = True
            snl_per = self.chi2.fiducial_values['sigmaNL_per']
            snl_par = self.chi2.fiducial_values['sigmaNL_par']
            self.chi2.fiducial_values['sigmaNL_per'] = 0
            self.chi2.fiducial_values['sigmaNL_par'] = 0
            # Compute Xi Continuum
            d.fiducial_model += d.xi_model(self.chi2.k, self.chi2.pksb_lin, self.chi2.fiducial_values)
            self.chi2.fiducial_values['SB'] = False
            self.chi2.fiducial_values['sigmaNL_per'] = snl_per
            self.chi2.fiducial_values['sigmaNL_par'] = snl_par
        del self.chi2.fiducial_values['SB']

        # Run parallel fastMC
        # Each CPU writes output once it's done
        self.chi2.fast_mc = {}
        self.chi2.fast_mc['chi2'] = []
        self.chi2.fast_mc_data = {}
        for it in range(nfast_mc):
            for d in self.chi2.data:
                g = np.random.randn(len(d.da))
                d.da = d.cho.dot(g) + d.fiducial_model
                self.chi2.fast_mc_data[d.name+'_'+str(it)] = d.da
                d.da_cut = d.da[d.mask]

            try:
                best_fit = self.chi2._minimize()
                for p, v in best_fit.values.items():
                    if p not in self.chi2.fast_mc:
                        self.chi2.fast_mc[p] = []
                    self.chi2.fast_mc[p].append([v, best_fit.errors[p]])
                self.chi2.fast_mc['chi2'].append(best_fit.fval)
                sys.stderr.write("\nINFO: CPU #" + str(cpu_rank)
                                 + " finished fastMC iteration " + str(it+1)
                                 + " of " + str(nfast_mc) + " \n")
            except ValueError:
                best_fit = self.chi2.best_fit
                for p, v in best_fit.values.items():
                    if p not in self.chi2.fast_mc:
                        self.chi2.fast_mc[p] = []
                    self.chi2.fast_mc[p].append([np.inf, np.inf])
                self.chi2.fast_mc['chi2'].append(np.inf)
                sys.stderr.write("\nINFO: CPU #" + str(cpu_rank)
                                 + " finished fastMC iteration " + str(it+1)
                                 + " of " + str(nfast_mc) + " \n")

        save_output = copy.deepcopy(self.chi2.outfile)
        self.chi2.outfile = self.chi2.outfile[:-3] + '_cpu' + str(cpu_rank) + '.h5'
        self.chi2.export()
        self.chi2.outfile = save_output
