from __future__ import print_function
import os.path
import scipy as sp
import iminuit
import time
import h5py
from scipy.linalg import cholesky
from scipy import random

def _wrap_chi2(d, dic=None, k=None, pk=None, pksb=None):
    return d.chi2(k, pk, pksb, dic)

class chi2:
    def __init__(self,dic_init):
        self.data = dic_init['data sets']
        self.par_names = sp.unique([name for d in self.data for name in d.par_names])
        self.outfile = os.path.expandvars(dic_init['outfile'])

        self.k = dic_init['fiducial']['k']
        self.pk_lin = dic_init['fiducial']['pk']
        self.pksb_lin = dic_init['fiducial']['pksb']
        self.nfast_mc = 0
        if 'fast mc' in dic_init:
            self.nfast_mc = dic_init['fast mc']['niterations']

    def __call__(self, *pars):
        dic = {p:pars[i] for i,p in enumerate(self.par_names)}
        chi2 = 0
        for d in self.data:
            chi2 += d.chi2(self.k,self.pk_lin,self.pksb_lin,dic)

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
        fmin = mig.migrad()
        print("INFO: minimized in {}".format(time.time()-t0))
        return mig
    
    def minimize(self):
        self.best_fit = self._minimize()

        for d in self.data:
            d.best_fit_model = d.xi_model(self.k, self.pk_lin-self.pksb_lin, self.best_fit.values)

            ap = self.best_fit.values['ap']
            at = self.best_fit.values['at']
            self.best_fit.values['ap'] = 1
            self.best_fit.values['at'] = 1
            snl = self.best_fit.values['sigmaNL_per']
            self.best_fit.values['sigmaNL_per'] = 0
            d.best_fit_model += d.xi_model(self.k, self.pksb_lin, self.best_fit.values)
            self.best_fit.values['ap'] = ap
            self.best_fit.values['at'] = at
            self.best_fit.values['sigmaNL_per'] = snl

    def fastMC(self):
        nfast_mc = self.nfast_mc
        for d in self.data:
            d.cho = cholesky(d.co)
        
        self.fast_mc = {}
        for it in range(nfast_mc):
            for d in self.data:
                g = random.randn(len(d.da))
                d.da = d.cho.dot(g) + d.best_fit_model
                d.da_cut = d.da[d.mask]

            best_fit = self._minimize()
            for p, v in best_fit.values.items():
                if not p in self.fast_mc:
                    self.fast_mc[p] = []
                self.fast_mc[p].append([v, best_fit.errors[p]])

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

        g.attrs['fval'] = self.best_fit.fval
        ndata = [d.mask.sum() for d in self.data]
        ndata = sum(ndata)
        g.attrs['ndata'] = ndata
        g.attrs['npar'] = len(self.best_fit.list_of_vary_param())
        g.attrs['list of free pars'] = [a.encode('utf8') for a in self.best_fit.list_of_vary_param()]
        g.attrs['list of fixed pars'] = [a.encode('utf8') for a in self.best_fit.list_of_fixed_param()]

        for d in self.data:
            g = f.create_group(d.name)
            g.attrs['chi2'] = d.chi2(self.k, self.pk_lin, self.pksb_lin, self.best_fit.values)
            fit = g.create_dataset("fit", d.da.shape, dtype = "f")
            fit[...] = d.best_fit_model
    
        if hasattr(self, "fast_mc"):
            g = f.create_group("fast mc")
            for p in self.fast_mc:
                vals = sp.array(self.fast_mc[p])
                d = g.create_dataset("{}/values".format(p), vals[:,0].shape, dtype="f")
                d[...] = vals[:,0]
                d = g.create_dataset("{}/errors".format(p), vals[:,1].shape, dtype="f")
                d[...] = vals[:,1]


        f.close()
