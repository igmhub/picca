import os.path
import numpy as np
import iminuit
import time
import h5py
import sys
from scipy.linalg import cholesky
import scipy.stats

from ..utils import userprint
from . import priors

def _wrap_chi2(d, dic=None, k=None, pk=None, pksb=None):
    return d.chi2(k, pk, pksb, dic)

class chi2:
    def __init__(self,dic_init):
        self.zeff = dic_init['data sets']['zeff']
        self.data = dic_init['data sets']['data']
        self.par_names = np.unique([name for d in self.data for name in d.par_names])
        self.outfile = os.path.expandvars(dic_init['outfile'])

        self.k = dic_init['fiducial']['k']
        self.pk_lin = dic_init['fiducial']['pk']
        self.pksb_lin = dic_init['fiducial']['pksb']
        self.full_shape = dic_init['fiducial']['full-shape']

        self.verbosity = 1
        if 'verbosity' in dic_init:
            self.verbosity = dic_init['verbosity']

        self.hesse = False
        if 'hesse' in dic_init:
            self.hesse = dic_init['hesse']

        if 'fast mc' in dic_init:
            if 'seed' in dic_init['fast mc']:
                self.seedfast_mc = dic_init['fast mc']['seed']
            else:
                self.seedfast_mc = 0
            self.nfast_mc = dic_init['fast mc']['niterations']
            if 'covscaling' in dic_init['fast mc']:
                self.scalefast_mc = dic_init['fast mc']['covscaling']
            else:
                self.scalefast_mc = np.ones(len(self.data))
            self.fidfast_mc = dic_init['fast mc']['fiducial']['values']
            self.fixfast_mc = dic_init['fast mc']['fiducial']['fix']
            # if set to true, will not add randomness to FastMC mock
            if 'forecast' in dic_init['fast mc']:
                self.forecast_mc = dic_init['fast mc']['forecast']
                if self.nfast_mc != 1:
                    sys.exit('ERROR: Why forecast more than once?')
            else:
                self.forecast_mc = False

        if 'minos' in dic_init:
            self.minos_para = dic_init['minos']

        if 'chi2 scan' in dic_init:
            self.dic_chi2scan = dic_init['chi2 scan']

    def __call__(self, *pars):
        dic = {p:pars[i] for i,p in enumerate(self.par_names)}
        dic['SB'] = False
        chi2 = 0
        for d in self.data:
            chi2 += d.chi2(self.k,self.pk_lin,self.pksb_lin,self.full_shape,dic)

        for prior in priors.prior_dic.values():
            chi2 += prior(dic)

        if self.verbosity == 1:
            del dic['SB']
            for p in sorted(dic.keys()):
                userprint(p+" "+str(dic[p]))

            userprint("Chi2: "+str(chi2))
            userprint("---\n")
        return chi2

    def _minimize(self):
        t0 = time.time()
        par_names = [name for d in self.data for name in d.pars_init]
        par_val_init = {name:val for d in self.data for name, val in d.pars_init.items()}
        par_err = {k.split('error_')[1]:err for d in self.data for k,err in d.par_error.items()}
        par_lim = {k.split('limit_')[1]:lim for d in self.data for k,lim in d.par_limit.items()}
        par_fix = {k.split('fix_')[1]:fix for d in self.data for k,fix in d.par_fixed.items()}

        ## do an initial "fast" minimization fixing everything except the biases
        mig_init = iminuit.Minuit(self, name=self.par_names, **par_val_init)
        for name in par_names:
            mig_init.errors[name] = par_err[name]
            mig_init.limits[name] = par_lim[name]
            mig_init.fixed[name]  = par_fix[name]
            if name[:4] != "bias":
                mig_init.fixed[name] = True
        mig_init.errordef = 1
        mig_init.print_level = 1
        mig_init.migrad()
        print(mig_init.fmin)
        print(mig_init.params)
        
        ## now get the best fit values for the biases and start a full minimization
        par_val={}
        for name, value in mig_init.values.to_dict().items():
            par_val[name] = value

        mig = iminuit.Minuit(self, name=self.par_names,**par_val)
        for name in par_names:
            mig.errors[name] = par_err[name]
            mig.limits[name] = par_lim[name]
            mig.fixed[name]  = par_fix[name]
        mig.errordef = 1
        mig.print_level = 1
        mig.migrad()
        print(mig.fmin)
        print(mig.params)

        userprint("INFO: minimized in {}".format(time.time()-t0))
        sys.stdout.flush()
        return mig

    def minimize(self):
        self.best_fit = self._minimize()
        if self.hesse:
            self.best_fit.hesse()
            #self.best_fit.print_fmin()

        values = self.best_fit.values.to_dict()
        values['SB'] = False
        for d in self.data:
            d.best_fit_model = values['bao_amp']*d.xi_model(self.k, self.pk_lin-self.pksb_lin, values)

            values['SB'] = True & (not self.full_shape)
            sigmaNL_par = values['sigmaNL_par']
            sigmaNL_per = values['sigmaNL_per']
            values['sigmaNL_par'] = 0.
            values['sigmaNL_per'] = 0.
            d.best_fit_model += d.xi_model(self.k, self.pksb_lin, values)
            values['SB'] = False
            values['sigmaNL_par'] = sigmaNL_par
            values['sigmaNL_per'] = sigmaNL_per

    def chi2scan(self):
        if not hasattr(self, "dic_chi2scan"): return

        dim = len(self.dic_chi2scan)

        ### Set all parameters to the minimum and store the current state
        store_data_pars = {}
        for d in self.data:
            store_d_pars_init = {}
            store_d_par_error = {}
            store_d_par_fixed = {}
            for name in d.pars_init.keys():
                store_d_pars_init[name] = d.pars_init[name]
                d.pars_init[name] = self.best_fit.values[name]
            for name in d.par_error.keys():
                store_d_par_error[name] = d.par_error[name]
                d.par_error[name] = self.best_fit.errors[name.split('error_')[1]]
            for name in d.par_fixed.keys():
                store_d_par_fixed[name] = d.par_fixed[name]
            store_data_pars[d.name] = {'init':store_d_pars_init, 'error':store_d_par_error, 'fixed':store_d_par_fixed}

        ###
        for p in self.dic_chi2scan.keys():
            for d in self.data:
                if 'error_'+p in d.par_error.keys():
                    d.par_error['error_'+p] = 0.
                if 'fix_'+p in d.par_fixed.keys():
                    d.par_fixed['fix_'+p] = True

        ###
        def send_one_fit():
            try:
                best_fit = self._minimize()
                chi2_result = best_fit.fval
            except:
                chi2_result = np.nan
            tresult = []
            for p in sorted(best_fit.values.to_dict().keys()):
                tresult += [best_fit.values[p]]
            tresult += [chi2_result]
            return tresult

        result = []
        ###
        if dim==1:
            par = list(self.dic_chi2scan.keys())[0]
            for it, step in enumerate(self.dic_chi2scan[par]['grid']):
                for d in self.data:
                    if par in d.pars_init.keys():
                        d.pars_init[par] = step
                result += [send_one_fit()]
                sys.stderr.write("\nINFO: finished chi2scan iteration {} of {}\n".format(it+1,
                    self.dic_chi2scan[par]['grid'].size))
        elif dim==2:
            par1  = list(self.dic_chi2scan.keys())[0]
            par2  = list(self.dic_chi2scan.keys())[1]
            for it1, step1 in enumerate(self.dic_chi2scan[par1]['grid']):
                for it2, step2 in enumerate(self.dic_chi2scan[par2]['grid']):
                    for d in self.data:
                        if par1 in d.pars_init.keys():
                            d.pars_init[par1] = step1
                        if par2 in d.pars_init.keys():
                            d.pars_init[par2] = step2
                    result += [send_one_fit()]
                    sys.stderr.write("\nINFO: finished chi2scan iteration {} of {}\n".format(
                        it1*self.dic_chi2scan[par2]['grid'].size+it2+1,
                        self.dic_chi2scan[par1]['grid'].size*self.dic_chi2scan[par2]['grid'].size))

        self.dic_chi2scan_result = {}
        self.dic_chi2scan_result['params'] = np.asarray(np.append(sorted(self.best_fit.values.to_dict().keys()),['fval']))
        self.dic_chi2scan_result['values'] = np.asarray(result)

        ### Set all parameters to where they were before
        for d in self.data:
            store_d_pars_init = store_data_pars[d.name]['init']
            store_d_par_error = store_data_pars[d.name]['error']
            store_d_par_fixed = store_data_pars[d.name]['fixed']
            for name in d.pars_init.keys():
                d.pars_init[name] = store_d_pars_init[name]
            for name in d.par_error.keys():
                d.par_error[name] = store_d_par_error[name]
            for name in d.par_fixed.keys():
                d.par_fixed[name] = store_d_par_fixed[name]

    def fastMC(self):
        if not hasattr(self,"nfast_mc"): return

        np.random.seed(self.seedfast_mc)
        nfast_mc = self.nfast_mc

        for d, s in zip(self.data, self.scalefast_mc):
            d.co = s*d.co
            d.ico = d.ico/s
            # no need to compute Cholesky when computing forecast
            if not self.forecast_mc:
                d.cho = cholesky(d.co)

        self.fiducial_values = self.best_fit.values.to_dict().copy()
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

        self.fast_mc = {}
        self.fast_mc['chi2'] = []
        self.fast_mc_data = {}
        for it in range(nfast_mc):
            for d in self.data:
                # if computing forecast, do not add randomness
                if self.forecast_mc:
                    d.da = d.fiducial_model
                else:
                    g = np.random.randn(len(d.da))
                    d.da = d.cho.dot(g) + d.fiducial_model
                self.fast_mc_data[d.name+'_'+str(it)] = d.da
                d.da_cut = d.da[d.mask]

            best_fit = self._minimize()
            for p, v in best_fit.values.to_dict().items():
                if not p in self.fast_mc:
                    self.fast_mc[p] = []
                self.fast_mc[p].append([v, best_fit.errors[p]])
            self.fast_mc['chi2'].append(best_fit.fval)
            sys.stderr.write("\nINFO: finished fastMC iteration {} of {}\n".format(it+1,nfast_mc))

    def minos(self):
        if not hasattr(self,"minos_para"): return
        sigma = self.minos_para['sigma']
        cl=scipy.stats.norm.cdf(sigma, loc=0, scale=1)-scipy.stats.norm.cdf(-sigma, loc=0, scale=1)    
        #TODO: it might be more complicated to select the right cl than this, as this is for the 1d-case only
        #if old minos actually did sigmas in the n-d parameter space and new minos is doing the correct cl for that
        #dimensionality this would not be giving the correct results; I think the test case only checks 1d
        if 'all' in self.minos_para['parameters']:
            self.best_fit.minos(cl=cl)
        else:
            fixed_pars=[name for (name,fix) in self.best_fit.fixed.to_dict().items() if fix]
            varied_pars=[name for (name,fix) in self.best_fit.fixed.to_dict().items() if not fix]
            for var in self.minos_para['parameters']:
                if var in varied_pars:   #testing for varied parameters
                    self.best_fit.minos(var,cl=cl)
                else:
                    if var in fixed_pars:   #testing for fixed parameters
                        userprint('WARNING: Can not run minos on a fixed parameter: {}'.format(var))
                    else:
                        userprint('WARNING: Can not run minos on a unknown parameter: {}'.format(var))

    def export(self):
        f = h5py.File(self.outfile,"w")

        g=f.create_group("best fit")

        ## write down all parameters
        fixed_pars=[name for (name,fix) in self.best_fit.fixed.to_dict().items() if fix]
        varied_pars=[name for (name,fix) in self.best_fit.fixed.to_dict().items() if not fix]

        for i, p in enumerate(self.best_fit.values.to_dict().keys()):
            v = self.best_fit.values[p]
            e = self.best_fit.errors[p]
            if p in fixed_pars:
                e = 0
            g.attrs[p] = (v, e)

        for i1, p1 in enumerate(self.best_fit.values.to_dict().keys()):
            for i2, p2 in enumerate(self.best_fit.values.to_dict().keys()):
                if (p1 in varied_pars and
                   p2 in varied_pars):
                    #only store correlations for params that have been varied
                    g.attrs["cov[{}, {}]".format(p1,p2)] = self.best_fit.covariance[i1,i2] #cov

        if len(priors.prior_dic) != 0:
            for prior in priors.prior_dic.values():
                values = [prior.func.__name__.encode('utf8')]
                for value in prior.keywords['prior_pars']:
                    values.append(value)
                g.attrs["prior[{}]".format(prior.keywords['name'])] = values

        ndata = [d.mask.sum() for d in self.data]
        ndata = sum(ndata)
        g.attrs['zeff'] = self.zeff
        g.attrs['ndata'] = ndata
        g.attrs['npar'] = len(varied_pars)                 
        g.attrs['list of free pars'] = [a.encode('utf8') for a in varied_pars]
        g.attrs['list of fixed pars'] = [a.encode('utf8') for a in fixed_pars]
        if len(priors.prior_dic) != 0:
            g.attrs['list of prior pars'] = [a.encode('utf8') for a in priors.prior_dic.keys()]

        ## write down all attributes of the minimum
        fmin_keys=[ 'edm', 'fval', 'has_accurate_covar', 'has_covariance', 'has_made_posdef_covar', 
         'has_posdef_covar', 'has_reached_call_limit', 'has_valid_parameters', 'hesse_failed', 'is_above_max_edm', 
         'is_valid', 'nfcn']
        dic_fmin = {k:getattr(self.best_fit.fmin,k) for k in fmin_keys}
        for item, value in dic_fmin.items():
            g.attrs[item] = value
        g.attrs['ncalls']=self.best_fit.nfcn #this should probably be changed to new nomenclature in the outputs instead of using the old kw; note that fmin.nfcn is the same now, but has been different before
        g.attrs['tolerance']=self.best_fit.tol
        g.attrs['up']=self.best_fit.errordef  #this should probably be changed to new nomenclature in the outputs instead of using the old kw

        values = self.best_fit.values.to_dict()
        values['SB'] = False
        for d in self.data:
            g = f.create_group(d.name)
            g.attrs['ndata'] = d.mask.sum()
            g.attrs['chi2'] = d.chi2(self.k, self.pk_lin, self.pksb_lin, self.full_shape, values)
            fit = g.create_dataset("fit", d.da.shape, dtype = "f")
            fit[...] = d.best_fit_model
            if not d.bb is None:
                gbb = g.create_group("broadband")
                for bbs in d.bb.values():
                    for bb in bbs:
                        tbb = bb(d.r, d.mu, **values)
                        bband = gbb.create_dataset(bb.name,
                                tbb.shape, dtype = "f")
                        bband[...] = tbb

        if hasattr(self, "fast_mc"):
            g = f.create_group("fast mc")
            g.attrs['niterations'] = self.nfast_mc
            g.attrs['seed'] = self.seedfast_mc
            g.attrs['covscaling'] = self.scalefast_mc
            g.attrs['forecast'] = self.forecast_mc
            if len(self.fidfast_mc) != 0:
                fid = []
                for p in self.fidfast_mc:
                    fix = "fixed"
                    if not self.fixfast_mc['fix_'+p]: fix = "free"
                    g.attrs["fiducial[{}]".format(p)] = [self.fidfast_mc[p], fix.encode('utf8')]
                    fid.append(p.encode('utf8'))
                g.attrs['list of fiducial pars'] = fid
                for d in self.data:
                    fiducial = g.create_dataset("{}_fiducial".format(d.name), d.da.shape, dtype = "f")
                    fiducial[...] = d.fiducial_model
            for p in self.fast_mc:
                vals = np.array(self.fast_mc[p])
                if p == 'chi2':
                    d = g.create_dataset("{}".format(p), vals.shape, dtype="f")
                    d[...] = vals
                else:
                    d = g.create_dataset("{}/values".format(p), vals[:,0].shape, dtype="f")
                    d[...] = vals[:,0]
                    d = g.create_dataset("{}/errors".format(p), vals[:,1].shape, dtype="f")
                    d[...] = vals[:,1]
            for p in self.fast_mc_data:
                xi = self.fast_mc_data[p]
                d = g.create_dataset(p, xi.shape, dtype="f")
                d[...] = xi

        ## write down all attributes of parameters minos was run over
        if hasattr(self, "minos_para"):
            g = f.create_group("minos")
            g.attrs['sigma'] = self.minos_para['sigma']
            minos_results = self.best_fit.merrors
            for par in list(minos_results.keys()):
                subgrp = g.create_group(par)
                minos_keys = ['at_lower_limit', 'at_lower_max_fcn', 'at_upper_limit', 'at_upper_max_fcn', 'is_valid', 'lower', 'lower_new_min', 'lower_valid', 'min', 'name', 'nfcn', 'upper', 'upper_new_min', 'upper_valid']
                dic_minos = {k:getattr(minos_results[par],k) for k in minos_keys}
                for item, value in dic_minos.items():
                    if item=='name': value = str(value) ###TODO: Fix h5py not handling numpy.str_
                    subgrp.attrs[item] = value

        if hasattr(self, "dic_chi2scan"):
            g = f.create_group("chi2 scan")
            for p, dic in self.dic_chi2scan.items():
                subgrp = g.create_group(p)
                subgrp.attrs['min']    = dic['min']
                subgrp.attrs['max']    = dic['max']
                subgrp.attrs['nb_bin'] = dic['nb_bin']
            subgrp = g.create_group('result')
            params = self.dic_chi2scan_result['params']
            for i,p in enumerate(params):
                subgrp.attrs[p] = i
            values = self.dic_chi2scan_result['values']
            vals = subgrp.create_dataset("values", values.shape, dtype = "f")
            vals[...] = values

        f.close()
