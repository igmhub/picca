from __future__ import print_function
import numpy as np
import iminuit

class chi2:
    def __init__(self,dic_init):
        self.data = dic_init['data sets']
        self.par_names = np.unique([name for d in self.data for name in d.par_names])

        self.k = dic_init['fiducial']['k']
        self.pk_lin = dic_init['fiducial']['pk']
        self.pksb_lin = dic_init['fiducial']['pksb']

    def __call__(self,*pars):
        dic = {p:pars[i] for i,p in enumerate(self.par_names)}
        chi2 = 0
        for d in self.data:
            chi2 += d.chi2(self.k,self.pk_lin,self.pksb_lin,dic)

        for p in dic:
            print(p+" "+str(dic[p]))
        
        print("Chi2: "+str(chi2))
        return chi2

    def minimize(self):
        kwargs = {name:val for d in self.data for name, val in d.pars_init.items()}
        kwargs.update({name:err for d in self.data for name, err in d.par_error.items()})
        kwargs.update({name:lim for d in self.data for name, lim in d.par_limit.items()})
        kwargs.update({name:fix for d in self.data for name, fix in d.par_fixed.items()})

        mig = iminuit.Minuit(self,forced_parameters=self.par_names,errordef=1,**kwargs)
        fmin = mig.migrad()
        return mig

