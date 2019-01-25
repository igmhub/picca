from __future__ import print_function
import os.path
import scipy as sp
import iminuit
import time
import h5py
import sys
from scipy.linalg import cholesky

from . import utils, priors

def _wrap_chi2(d, dic=None, k=None, pk=None, pksb=None):
    return d.chi2(k, pk, pksb, dic)

class chi2:
    def __init__(self,dic_init):
        self.zeff = dic_init['data sets']['zeff']
        self.data = dic_init['data sets']['data']
        self.par_names = sp.unique([name for d in self.data for name in d.par_names])
        self.outfile = os.path.expandvars(dic_init['outfile'])

        self.k = dic_init['fiducial']['k']
        self.pk_lin = dic_init['fiducial']['pk']
        self.pksb_lin = dic_init['fiducial']['pksb']

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
                self.scalefast_mc = sp.ones(len(self.data))
            self.fidfast_mc = dic_init['fast mc']['fiducial']['values']
            self.fixfast_mc = dic_init['fast mc']['fiducial']['fix']

        if 'minos' in dic_init:
            self.minos_para = dic_init['minos']

        if 'chi2 scan' in dic_init:
            self.dic_chi2scan = dic_init['chi2 scan']

    def __call__(self, *pars):
        dic = {p:pars[i] for i,p in enumerate(self.par_names)}
        dic['SB'] = False
        chi2 = 0
        for d in self.data:
            chi2 += d.chi2(self.k,self.pk_lin,self.pksb_lin,dic)

        for prior in priors.prior_dic.values():
            chi2 += prior(dic)

        if self.verbosity == 1:
            del dic['SB']
            for p in sorted(dic.keys()):
                print(p+" "+str(dic[p]))

            print("Chi2: "+str(chi2))
            print("---\n")
        return chi2

    def _minimize(self):
        t0 = time.time()
        par_names = [name for d in self.data for name in d.pars_init]
        kwargs = {name:val for d in self.data for name, val in d.pars_init.items()}
        kwargs.update({name:err for d in self.data for name, err in d.par_error.items()})
        kwargs.update({name:lim for d in self.data for name, lim in d.par_limit.items()})
        kwargs.update({name:fix for d in self.data for name, fix in d.par_fixed.items()})

        ## do an initial "fast" minimization fixing everything except the biases
        kwargs_init = {}
        for k,v in kwargs.items():
            kwargs_init[k] = v
        for name in par_names:
            if name[:4] != "bias":
                kwargs_init["fix_"+name] = True

        mig_init = iminuit.Minuit(self,forced_parameters=self.par_names,errordef=1,**kwargs_init)
        mig_init.migrad()

        ## now get the best fit values for the biases and start a full minimization
        for name, value in mig_init.values.items():
            kwargs[name] = value

        mig = iminuit.Minuit(self,forced_parameters=self.par_names,errordef=1,**kwargs)
        mig.migrad()

        print("INFO: minimized in {}".format(time.time()-t0))
        return mig

    def minimize(self):
        self.best_fit = self._minimize()
        if self.hesse:
            self.best_fit.hesse()

        values = dict(self.best_fit.values)
        values['SB'] = False
        for d in self.data:
            d.best_fit_model = values['bao_amp']*d.xi_model(self.k, self.pk_lin-self.pksb_lin, values)

            values['SB'] = True
            snl = values['sigmaNL_per']
            values['sigmaNL_per'] = 0
            d.best_fit_model += d.xi_model(self.k, self.pksb_lin, values)
            values['SB'] = False
            values['sigmaNL_per'] = snl

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
                chi2_result = sp.nan
            tresult = []
            for p in sorted(best_fit.values):
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
        self.dic_chi2scan_result['params'] = sp.asarray(sp.append(sorted(self.best_fit.values),['fval']))
        self.dic_chi2scan_result['values'] = sp.asarray(result)

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

        sp.random.seed(self.seedfast_mc)
        nfast_mc = self.nfast_mc

        for d, s in zip(self.data, self.scalefast_mc):
            d.co = s*d.co
            d.ico = d.ico/s
            d.cho = cholesky(d.co)

        self.fiducial_values = self.best_fit.values.copy()
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
            snl = self.fiducial_values['sigmaNL_per']
            self.fiducial_values['sigmaNL_per'] = 0
            d.fiducial_model += d.xi_model(self.k, self.pksb_lin, self.fiducial_values)
            self.fiducial_values['SB'] = False
            self.fiducial_values['sigmaNL_per'] = snl
        del self.fiducial_values['SB']

        self.fast_mc = {}
        self.fast_mc['chi2'] = []
        self.fast_mc_data = {}
        for it in range(nfast_mc):
            for d in self.data:
                g = sp.random.randn(len(d.da))
                d.da = d.cho.dot(g) + d.fiducial_model
                self.fast_mc_data[d.name+'_'+str(it)] = d.da
                d.da_cut = d.da[d.mask]

            best_fit = self._minimize()
            for p, v in best_fit.values.items():
                if not p in self.fast_mc:
                    self.fast_mc[p] = []
                self.fast_mc[p].append([v, best_fit.errors[p]])
            self.fast_mc['chi2'].append(best_fit.fval)
            sys.stderr.write("\nINFO: finished fastMC iteration {} of {}\n".format(it+1,nfast_mc))

    def minos(self):
        if not hasattr(self,"minos_para"): return

        sigma = self.minos_para['sigma']
        if 'all' in self.minos_para['parameters']:
            self.best_fit.minos(var=None,sigma=sigma)
        else:
            for var in self.minos_para['parameters']:
                if var in self.best_fit.list_of_vary_param():
                    self.best_fit.minos(var=var,sigma=sigma)
                else:
                    if var in self.best_fit.list_of_fixed_param():
                        print('WARNING: Can not run minos on a fixed parameter: {}'.format(var))
                    else:
                        print('WARNING: Can not run minos on a unknown parameter: {}'.format(var))

    def export(self):
        f = h5py.File(self.outfile,"w")

        g=f.create_group("best fit")

        ## write down all parameters
        for i, p in enumerate(self.best_fit.values):
            v = self.best_fit.values[p]
            e = self.best_fit.errors[p]
            if p in self.best_fit.list_of_fixed_param():
                e = 0
            g.attrs[p] = (v, e)

        for (p1, p2), cov in self.best_fit.covariance.items():
            g.attrs["cov[{}, {}]".format(p1,p2)] = cov

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
        g.attrs['npar'] = len(self.best_fit.list_of_vary_param())
        g.attrs['list of free pars'] = [a.encode('utf8') for a in self.best_fit.list_of_vary_param()]
        g.attrs['list of fixed pars'] = [a.encode('utf8') for a in self.best_fit.list_of_fixed_param()]
        if len(priors.prior_dic) != 0:
            g.attrs['list of prior pars'] = [a.encode('utf8') for a in priors.prior_dic.keys()]

        ## write down all attributes of the minimum
        dic_fmin = self.best_fit.get_fmin()
        for item, value in dic_fmin.items():
            g.attrs[item] = value

        values = dict(self.best_fit.values)
        values['SB'] = False
        for d in self.data:
            g = f.create_group(d.name)
            g.attrs['ndata'] = d.mask.sum()
            g.attrs['chi2'] = d.chi2(self.k, self.pk_lin, self.pksb_lin, values)
            fit = g.create_dataset("fit", d.da.shape, dtype = "f")
            fit[...] = d.best_fit_model
            if d.bb != None:
                gbb = g.create_group("broadband")
                for bbs in d.bb.values():
                    for bb in bbs:
                        bband = gbb.create_dataset(bb.name,
                                d.da.shape, dtype = "f")
                        bband[...] = bb(d.r, d.mu, **values)

        if hasattr(self, "fast_mc"):
            g = f.create_group("fast mc")
            g.attrs['niterations'] = self.nfast_mc
            g.attrs['seed'] = self.seedfast_mc
            g.attrs['covscaling'] = self.scalefast_mc
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
                vals = sp.array(self.fast_mc[p])
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
            minos_results = self.best_fit.get_merrors()
            for par in list(minos_results.keys()):
                subgrp = g.create_group(par)
                dic_minos = minos_results[par]
                for item, value in dic_minos.items():
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
