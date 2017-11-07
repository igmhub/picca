from __future__ import print_function
import numpy as np
import iminuit
import time
import h5py

class chi2:
    def __init__(self,dic_init):
        self.data = dic_init['data sets']
        self.par_names = np.unique([name for d in self.data for name in d.par_names])

        self.k = dic_init['fiducial']['k']
        self.pk_lin = dic_init['fiducial']['pk']
        self.pksb_lin = dic_init['fiducial']['pksb']

    def __call__(self, *pars):
        dic = {p:pars[i] for i,p in enumerate(self.par_names)}
        chi2 = 0
        for d in self.data:
            chi2 += d.chi2(self.k,self.pk_lin,self.pksb_lin,dic)

        for p in dic:
            print(p+" "+str(dic[p]))
        
        print("Chi2: "+str(chi2))
        return chi2

    def minimize(self):
        t0 = time.time()
        kwargs = {name:val for d in self.data for name, val in d.pars_init.items()}
        kwargs.update({name:err for d in self.data for name, err in d.par_error.items()})
        kwargs.update({name:lim for d in self.data for name, lim in d.par_limit.items()})
        kwargs.update({name:fix for d in self.data for name, fix in d.par_fixed.items()})

        mig = iminuit.Minuit(self,forced_parameters=self.par_names,errordef=1,**kwargs)
        fmin = mig.migrad()
        print("INFO: minimized in {}".format(time.time()-t0))
        self.mig = mig
    
    def export(self, filename):
        f = h5py.File(filename,"w")
        g=f.create_group("best_fit")

        for i, p in enumerate(self.mig.values):
            g.attrs[p] = (self.mig.values[p], self.mig.errors[p])

        g.attrs['fval'] = self.mig.fval
        ndata = [d.mask.sum() for d in self.data]
        ndata = sum(ndata)
        g.attrs['ndata'] = ndata
        g.attrs['npar'] = len(self.mig.values)

        for d in self.data:
            g = f.create_group(d.name)
            g.attrs['chi2'] = d.chi2(self.k, self.pk_lin, self.pksb_lin, self.mig.values)
            data = g.create_dataset("data", d.da.shape, dtype = "f")
            data[...] = d.da
            err = g.create_dataset("error", d.da.shape, dtype = "f")
            err[...] = np.sqrt(d.co.diagonal())
            fit = g.create_dataset("fit", d.da.shape, dtype = "f")
            fit[...] = d.xi_model(self.k, self.pk_lin-self.pksb_lin, self.mig.values)
            ap = self.mig.values['ap']
            at = self.mig.values['at']
            self.mig.values['ap'] = 1
            self.mig.values['at'] = 1
            fit[...] += d.xi_model(self.k, self.pksb_lin, self.mig.values)
            self.mig.values['ap'] = ap
            self.mig.values['at'] = at
            

        f.close()
