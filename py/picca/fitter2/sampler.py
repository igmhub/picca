from __future__ import print_function
import os.path
import numpy as np
import pypolychord
from pypolychord.settings import PolyChordSettings
from pypolychord.priors import UniformPrior

from . import priors


class sampler:
    ''' Interface between picca and the nested sampler PolyChord '''

    def __init__(self, dic_init):
        # Setup the data we need and extract the PolyChord config
        self.zeff = dic_init['data sets']['zeff']
        self.data = dic_init['data sets']['data']
        self.par_names = np.unique([name for d in self.data for name in d.par_names])
        self.outfile = os.path.expandvars(dic_init['outfile'])
        self.polychord_setup = dic_init['Polychord']

        # Get the fiducial P(k) for the peak and continuum
        self.k = dic_init['fiducial']['k']
        self.pk_lin = dic_init['fiducial']['pk']
        self.pksb_lin = dic_init['fiducial']['pksb']

        # Flag for running fullshape. Needed by data.py, never tested
        self.full_shape = dic_init['fiducial']['full-shape']

    def log_lik(self, pars):
        ''' Compute log likelihood for all datasets and add them '''
        # data.py assumes this exists, so we need to add it to be sure
        # This should be in data.py in the future as it makes no sense
        # for it to be here
        pars['SB'] = False

        log_lik = 0
        for d in self.data:
            log_lik += d.log_lik(self.k, self.pk_lin, self.pksb_lin,
                                 self.full_shape, pars)

        for prior in priors.prior_dic.values():
            log_lik += prior(pars)

        return log_lik

    def run(self):
        '''
        Run Polychord

        We need to pass 3 functions:
        log_lik - compute likelihood for a paramater set theta
        prior - defines the box prior
        dumper - extracts info during runtime - empty for now
        '''
        par_names = {name: name for d in self.data for name in d.pars_init}
        val_dict = {name: val for d in self.data for name, val in d.pars_init.items()}
        lim_dict = {name: lim for d in self.data for name, lim in d.par_limit.items()}
        fix_dict = {name: fix for d in self.data for name, fix in d.par_fixed.items()}

        # Select the parameters we sample
        sampled_pars_ind = np.array([i for i, val in enumerate(fix_dict.values()) if not val])
        npar = len(sampled_pars_ind)
        nder = 0

        # Get the limits for the free params
        limits = np.array([list(lim_dict.values())[i] for i in sampled_pars_ind])
        names = np.array([list(par_names.values())[i] for i in sampled_pars_ind])

        def log_lik(theta):
            ''' Wrapper for likelihood function passed to Polychord '''
            pars = val_dict.copy()
            for i, name in enumerate(names):
                pars[name] = theta[i]

            log_lik = self.log_lik(pars)
            return log_lik, []

        def prior(hypercube):
            ''' Uniform prior '''
            prior = []
            for i, lims in enumerate(limits):
                prior.append(UniformPrior(lims[0], lims[1])(hypercube[i]))
            return prior

        def dumper(live, dead, logweights, logZ, logZerr):
            ''' Dumper function empty for now '''
            pass

        # Get the settings we need and add defaults
        # These are the same as PolyChord recommends
        nlive = self.polychord_setup.getint('nlive', int(25*npar))
        seed = self.polychord_setup.getint('seed', int(0))
        num_repeats = self.polychord_setup.getint('num_repeats', int(5*npar))
        precision = self.polychord_setup.getfloat('precision', float(0.001))
        boost_posterior = self.polychord_setup.getfloat('boost_posterior', float(0.0))
        resume = self.polychord_setup.getboolean('resume', True)
        cluster_posteriors = self.polychord_setup.getboolean('cluster_posteriors', False)
        do_clustering = self.polychord_setup.getboolean('do_clustering', False)
        path = self.polychord_setup.get('path')
        filename = self.polychord_setup.get('name')
        write_live = self.polychord_setup.getboolean('write_live', False)
        write_dead = self.polychord_setup.getboolean('write_dead', True)
        write_prior = self.polychord_setup.getboolean('write_prior', False)

        # Initialize and run PolyChord
        settings = PolyChordSettings(npar, nder,
                                     base_dir=path,
                                     file_root=filename,
                                     seed=seed,
                                     nlive=nlive,
                                     precision_criterion=precision,
                                     num_repeats=num_repeats,
                                     boost_posterior=boost_posterior,
                                     cluster_posteriors=cluster_posteriors,
                                     do_clustering=do_clustering,
                                     equals=False,
                                     write_resume=resume,
                                     read_resume=resume,
                                     write_live=write_live,
                                     write_dead=write_dead,
                                     write_prior=write_prior)
        pypolychord.run_polychord(log_lik, npar, nder, settings, prior, dumper)
