from __future__ import print_function
import os.path
import numpy as np
import scipy as sp
import iminuit
import time
import h5py
import sys
import pypolychord
import time
from numba import jit
from pypolychord.settings import PolyChordSettings
from pypolychord.priors import UniformPrior
from scipy.linalg import cholesky

from . import priors,utils

class sample:
    def __init__(self,dic_init):
        self.zeff = dic_init['data sets']['zeff']
        self.data = dic_init['data sets']['data']
        self.par_names = sp.unique([name for d in self.data for name in d.par_names])
        self.outfile = os.path.expandvars(dic_init['outfile'])
        self.polychord_setup = dic_init['Polychord']

        self.k = dic_init['fiducial']['k']
        self.pk_lin = dic_init['fiducial']['pk']
        self.pksb_lin = dic_init['fiducial']['pksb']
        self.full_shape = dic_init['fiducial']['full-shape']

        # self.verbosity = 1
        # if 'verbosity' in dic_init:
        #     self.verbosity = dic_init['verbosity']

        # What's this?
        # self.hesse = False
        # if 'hesse' in dic_init:
        #     self.hesse = dic_init['hesse']

        # par_names = [name for d in self.data for name in d.pars_init]
        # kwargs = {name:val for d in self.data for name, val in d.pars_init.items()}
        # kwargs.update({name:err for d in self.data for name, err in d.par_error.items()})
        # kwargs.update({name:lim for d in self.data for name, lim in d.par_limit.items()})
        # kwargs.update({name:fix for d in self.data for name, fix in d.par_fixed.items()})


    def chisq(self, pars):
        # dic = {p:pars[i] for i,p in enumerate(self.par_names)}
        pars['SB'] = False
        chi2 = 0
        for d in self.data:
            chi2 += d.chi2(self.k,self.pk_lin,self.pksb_lin,self.full_shape,pars)

        for prior in priors.prior_dic.values():
            chi2 += prior(pars)

        # if self.verbosity == 1:
        #     del dic['SB']
        #     for p in sorted(dic.keys()):
        #         print(p+" "+str(dic[p]))

        #     print("Chi2: "+str(chi2))
        #     print("---\n")
        return chi2

    def run_sampler(self):
        '''
        Run Polychord

        We need to pass 3 functions:
        log_lik - compute likelihood for a paramater set theta
        prior - defines the prior - for now box prior
        dumper - extracts info during runtime - empty for now
        '''
        par_names = {name:name for d in self.data for name in d.pars_init}
        val_dict = {name:val for d in self.data for name, val in d.pars_init.items()}
        lim_dict = {name:lim for d in self.data for name, lim in d.par_limit.items()}
        fix_dict = {name:fix for d in self.data for name, fix in d.par_fixed.items()}

        # fix_list = [fix for d in self.data for _, fix in d.par_fixed.items()]
        # limits_list = [lim for d in self.data for name, lim in d.par_limit.items()]


        # Select the parameters we sample
        sampled_pars_ind = np.array([i for i,val in enumerate(fix_dict.values()) if not val])
        npar = len(sampled_pars_ind)
        nder = 0

        # Get the limits for the free params 
        limits = np.array([list(lim_dict.values())[i] for i in sampled_pars_ind])
        names = np.array([list(par_names.values())[i] for i in sampled_pars_ind])

        # @utils.timeit
        # @jit
        def log_lik(theta):
            pars = val_dict.copy()
            i = 0
            for key,value in pars.items():
                if key in names:
                    pars[key] = theta[i]
                    i += 1

            log_lik = self.chisq(pars)
            return -0.5 * log_lik, []

        # print(log_lik([1.11,1.01]))
        # print(log_lik([1.12,1.02]))
        # print(log_lik([1.13,1.03]))
        
        def prior(hypercube):
            """ Uniform prior """
            prior = []
            for i, lims in enumerate(limits):
                prior.append(UniformPrior(lims[0], lims[1])(hypercube[i]))
            return prior

    
        def dumper(live, dead, logweights, logZ, logZerr):
            pass

        nlive = int(self.polychord_setup.get('nlive', int(25*npar)))
        seed = int(self.polychord_setup.get('seed', int(0)))
        num_repeats = int(self.polychord_setup.get('num_repeats', int(5*npar)))
        precision = float(self.polychord_setup.get('precision', float(0.001)))
        resume = self.polychord_setup.get('resume', True)
        path = self.polychord_setup.get('path')
        filename = self.polychord_setup.get('name')
        settings = PolyChordSettings(npar, nder,
                        base_dir = path, file_root = filename,
                        seed = seed,
                        nlive = nlive,
                        precision_criterion = precision,
                        num_repeats = num_repeats,
                        cluster_posteriors = False,
                        do_clustering = False,
                        equals = False,
                        write_resume = resume,
                        read_resume = resume,
                        write_live = False,
                        write_dead = True,
                        write_prior = False)

        pypolychord.run_polychord(log_lik, npar, nder, settings, prior, dumper)