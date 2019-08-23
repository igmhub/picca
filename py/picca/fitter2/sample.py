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

from . import priors,utils,chi2

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

        if 'fast mc' in dic_init:
            if 'seed' in dic_init['fast mc']:
                self.seedfast_mc = dic_init['fast mc']['seed']
            else:
                self.seedfast_mc = 0
            self.nfast_mc = dic_init['fast mc']['niterations']
            if 'covscaling' in dic_init['fast mc']:
                self.scalefast_mc = dic_init['fast mc']['covscaling']
            else:
                self.scalefast_mc = sp.ones(len(self.data))
            self.fidfast_mc = dic_init['fast mc']['fiducial']['values']
            self.fixfast_mc = dic_init['fast mc']['fiducial']['fix']

        run_mock = self.polychord_setup.getboolean('run_mock', False)
        if run_mock:
            filename = self.polychord_setup.get('mock_file')
            mock_da = np.loadtxt(filename)
            self.data[0].da = mock_da
            self.data[0].da_cut = mock_da[self.data[0].mask]
            print('Replaced data with mock: ' + filename)
    
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


    def log_lik(self, pars):
        # dic = {p:pars[i] for i,p in enumerate(self.par_names)}
        pars['SB'] = False
        log_lik = 0
        for d in self.data:
            log_lik += d.log_lik(self.k,self.pk_lin,self.pksb_lin,self.full_shape,pars)

        for prior in priors.prior_dic.values():
            log_lik += prior(pars)

        return log_lik

    def make_mock(self, dic_init):
        self.chi = chi2.chi2(dic_init)
        self.chi.minimize()

        sp.random.seed(self.seedfast_mc)
        nfast_mc = self.nfast_mc

        for d, s in zip(self.data, self.scalefast_mc):
            d.co = s*d.co
            d.ico = d.ico/s
            d.cho = cholesky(d.co)

        self.fiducial_values = dict(self.chi.best_fit.values).copy()
        for p in self.fidfast_mc:
            self.fiducial_values[p] = self.fidfast_mc[p]
            for d in self.data:
                if p in d.par_names:
                    d.pars_init[p] = self.fidfast_mc[p]
                    d.par_fixed['fix_'+p] = self.fixfast_mc['fix_'+p]

        self.fiducial_values['SB'] = False
        for d in self.data:
            d.fiducial_model = self.fiducial_values['bao_amp']*d.xi_model(self.k, self.pk_lin-self.pksb_lin, self.fiducial_values)

            self.fiducial_values['SB'] = True
            snl_per = self.fiducial_values['sigmaNL_per']
            snl_par = self.fiducial_values['sigmaNL_par']
            self.fiducial_values['sigmaNL_per'] = 0
            self.fiducial_values['sigmaNL_par'] = 0
            d.fiducial_model += d.xi_model(self.k, self.pksb_lin, self.fiducial_values)
            self.fiducial_values['SB'] = False
            self.fiducial_values['sigmaNL_per'] = snl_per
            self.fiducial_values['sigmaNL_par'] = snl_par
        del self.fiducial_values['SB']

        self.fast_mc_data = {}

        for i in range(self.nfast_mc):
            for d in self.data:
                g = sp.random.randn(len(d.da))
                mock = d.cho.dot(g) + d.fiducial_model
                self.fast_mc_data[d.name+'_'+str(i)] = mock

                np.savetxt(self.outfile+'/mock_'+d.name+'_'+str(i), mock)

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

            log_lik = self.log_lik(pars)
            return log_lik, []

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

        nlive = self.polychord_setup.getint('nlive', int(25*npar))
        seed = self.polychord_setup.getint('seed', int(0))
        num_repeats = self.polychord_setup.getint('num_repeats', int(5*npar))
        precision = self.polychord_setup.getfloat('precision', float(0.001))
        resume = self.polychord_setup.getboolean('resume', True)
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