import fitsio
from functools import partial
import numpy as np
from numpy import linalg
from . import utils
from . import pk
from . import xi
from scipy.sparse import csr_matrix

class data:
    def __init__(self,dic_init):

        self.name = dic_init['data']['name']
        self.tracer1 = dic_init['data']['tracer1']
        if 'tracer2' in dic_init['data']:
            self.tracer2 = dic_init['data']['tracer2']
        else:
            self.tracer2 = self.tracer1

        self.ell_max = dic_init['data']['ell-max']
        self.zref = dic_init['data']['zref']

        fdata = dic_init['data']['filename']
        h = fitsio.FITS(fdata)
        da = h[1]['DA'][:]
        co = h[1]['CO'][:]
        dm = csr_matrix(h[1]['DM'][:])
        rp = h[1]['RP'][:]
        rt = h[1]['RT'][:]
        z = h[1]['Z'][:]

        h.close()

        r = np.sqrt(rp**2+rt**2)
        mu = rp/r

        rp_min = dic_init['cuts']['rp-min']
        rp_max = dic_init['cuts']['rp-max']

        rt_min = dic_init['cuts']['rt-min']
        rt_max = dic_init['cuts']['rt-max']

        r_min = dic_init['cuts']['r-min']
        r_max = dic_init['cuts']['r-max']

        mu_min = dic_init['cuts']['mu-min']
        mu_max = dic_init['cuts']['mu-max']

        ## select data within cuts
        mask = (rp > rp_min) & (rp < rp_max)
        mask = mask & (rt > rt_min) & (rt < rt_max)
        mask = mask & (r > r_min) & (r < r_max)
        mask = mask & (mu > mu_min) & (mu < mu_max)

        nmask = mask.sum()
        self.mask = mask
        self.da = da
        self.da_cut = np.zeros(mask.sum())
        self.da_cut[:] = da[mask]
        self.co = co
        ico = co[:,mask]
        ico = ico[mask,:]
        self.ico = linalg.inv(ico)
        self.dm = dm

        self.rp = rp
        self.rt = rt
        self.z = z

        self.r = r
        self.mu = mu

        self.par_names = dic_init['parameters']['values'].keys()
        self.pars_init = dic_init['parameters']['values']
        self.par_error = dic_init['parameters']['errors']
        self.par_limit = dic_init['parameters']['limits']
        self.par_fixed = dic_init['parameters']['fix']

        self.pk = pk.pk(getattr(pk, dic_init['model']['model-pk']))
        self.pk *= partial(getattr(pk,'G2'),dataset_name=self.name)
        if 'dnl' in dic_init['model']:
            self.pk *= getattr(pk, dic_init['model']['dnl'])

        if 'velocity dispersion' in dic_init['model']:
            self.pk *= getattr(pk, dic_init['model']['velocity dispersion'])

        self.xi = getattr(xi, dic_init['model']['model-xi'])

        self.z_evol = {}
        self.z_evol[self.tracer1] = partial(getattr(xi, dic_init['model']['z evol {}'.format(self.tracer1)]), zref=self.zref)
        self.z_evol[self.tracer2] = partial(getattr(xi, dic_init['model']['z evol {}'.format(self.tracer2)]), zref = self.zref)
        self.growth_function = partial(getattr(xi, dic_init['model']['growth function']), zref = self.zref)

        self.dm_met = {}
        self.rp_met = {}
        self.rt_met = {}
        self.z_met = {}

        if 'metals' in dic_init:
            self.pk_met = pk.pk(getattr(pk, dic_init['metals']['model-pk-met']))
            self.pk_met *= partial(getattr(pk,'G2'),dataset_name=self.name)
            self.xi_met = getattr(xi, dic_init['metals']['model-xi-met'])

            hmet = fitsio.FITS(dic_init['metals']['filename'])

            if 'in tracer2' in dic_init['metals']:
                for m in dic_init['metals']['in tracer2']:
                    self.z_evol[m] = partial(getattr(xi, dic_init['metals']['z evol met']), zref = self.zref)
                    self.rp_met[(self.tracer1, m)] = hmet[2]["RP_{}_{}".format(self.tracer1,m)][:]
                    self.rt_met[(self.tracer1, m)] = hmet[2]["RT_{}_{}".format(self.tracer1,m)][:]
                    self.z_met[(self.tracer1, m)] = hmet[2]["Z_{}_{}".format(self.tracer1,m)][:]
                    self.dm_met[(self.tracer1, m)] = csr_matrix(hmet[2]["DM_{}_{}".format(self.tracer1,m)][:])

            if 'in tracer1' in dic_init['metals']:
                for m in dic_init['metals']['in tracer1']:
                    self.z_evol[m] = partial(getattr(xi, dic_init['metals']['z evol met']), zref = self.zref)
                    self.rp_met[(m, self.tracer2)] = hmet[2]["RP_{}_{}".format(self.tracer2,m)][:]
                    self.rt_met[(m, self.tracer2)] = hmet[2]["RT_{}_{}".format(self.tracer2,m)][:]
                    self.z_met[(m, self.tracer2)] = hmet[2]["Z_{}_{}".format(self.tracer2,m)][:]
                    self.dm_met[(m, self.tracer2)] = csr_matrix(hmet[2]["DM_{}_{}".format(self.tracer2,m)][:])

            ## add metal-metal cross correlations
            if 'in tracer1' in dic_init['metals'] and 'in tracer2' in dic_init['metals']:
                for i,m1 in enumerate(dic_init['metals']['in tracer1']):
                    j0=0
                    if self.tracer1 == self.tracer2:
                        j0=i
                    for m2 in dic_init['metals']['in tracer2'][j0:]:
                        self.rp_met[(m1, m2)] = hmet[2]["RP_{}_{}".format(m1,m2)][:]
                        self.rt_met[(m1, m2)] = hmet[2]["RT_{}_{}".format(m1,m2)][:]
                        self.z_met[(m1, m2)] = hmet[2]["Z_{}_{}".format(m1,m2)][:]
                        self.dm_met[(m1, m2)] = csr_matrix(hmet[2]["DM_{}_{}".format(m1,m2)][:])

    def xi_model(self, k, pk_lin, pksb_lin, pars):
        xi_peak = self.xi(self.r, self.mu, k, pk_lin-pksb_lin, self.pk, \
                    tracer1 = self.tracer1, tracer2 = self.tracer2, ell_max = self.ell_max, **pars)

        ap = pars["ap"]
        at = pars["at"]
        pars["ap"]=1.
        pars["at"]=1.
        xi_sb = self.xi(self.r, self.mu, k, pksb_lin, self.pk, \
                    tracer1 = self.tracer1, tracer2 = self.tracer2, ell_max = self.ell_max, **pars)

        pars["ap"] = ap
        pars["at"] = at
        xi_full = xi_peak + xi_sb

        xi_full *= self.z_evol[self.tracer1](self.z, self.tracer1, **pars)*self.z_evol[self.tracer2](self.z, self.tracer2, **pars)
        xi_full *= self.growth_function(self.z, **pars)**2

        for tracer1, tracer2 in self.dm_met:
            rp = self.rp_met[(tracer1, tracer2)]
            rt = self.rt_met[(tracer1, tracer2)]
            z = self.z_met[(tracer1, tracer2)]
            dm_met = self.dm_met[(tracer1, tracer2)]
            r = np.sqrt(rp**2+rt**2)
            w = r == 0
            r[w] = 1e-6
            mu = rp/r
            xi_met_peak = self.xi_met(r, mu, k, pk_lin - pksb_lin, self.pk_met, \
                tracer1 = tracer1, tracer2 = tracer2, ell_max = self.ell_max, **pars)
            ap = pars["ap"]
            at = pars["at"]
            pars["ap"]=1.
            pars["at"]=1.
            xi_met_sb = self.xi_met(r, mu, k, pksb_lin, self.pk_met, \
                tracer1 = tracer1, tracer2 = tracer2, ell_max = self.ell_max, **pars)

            pars["ap"] = ap
            pars["at"] = at
            xi_met_full = xi_met_peak + xi_met_sb
            xi_met_full *= self.z_evol[tracer1](z, tracer1, **pars)*self.z_evol[tracer2](z, tracer2, **pars)
            xi_met_full *= self.growth_function(z, **pars)**2
            xi_met_full = dm_met.dot(xi_met_full)
            xi_full += xi_met_full

        xi_full = self.dm.dot(xi_full)

        return xi_full

    def chi2(self, k, pk_lin, pksb_lin, pars):
        xi_full = self.xi_model(k, pk_lin, pksb_lin, pars)
        dxi = self.da_cut-xi_full[self.mask]

        return dxi.T.dot(self.ico.dot(dxi))
