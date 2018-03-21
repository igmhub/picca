import fitsio
from functools import partial
import scipy as sp
from scipy import linalg
from . import utils
from . import pk
from . import xi
from scipy.sparse import csr_matrix

class data:
    def __init__(self,dic_init):

        self.name = dic_init['data']['name']
        self.tracer1 = dic_init['data']['tracer1']
        self.tracer1_type = dic_init['data']['tracer1-type']
        if 'tracer2' in dic_init['data']:
            self.tracer2 = dic_init['data']['tracer2']
            self.tracer2_type = dic_init['data']['tracer2-type']
        else:
            self.tracer2 = self.tracer1
            self.tracer2_type = self.tracer1_type

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
        head = h[1].read_header()

        h.close()

        r = sp.sqrt(rp**2+rt**2)
        mu = rp/r

        rp_min = dic_init['cuts']['rp-min']
        rp_max = dic_init['cuts']['rp-max']

        rt_min = dic_init['cuts']['rt-min']
        rt_max = dic_init['cuts']['rt-max']

        r_min = dic_init['cuts']['r-min']
        r_max = dic_init['cuts']['r-max']

        mu_min = dic_init['cuts']['mu-min']
        mu_max = dic_init['cuts']['mu-max']

        bin_size_rp = (head['RPMAX']-head['RPMIN'])/head['NP']
        bin_center_rp = sp.zeros(rp.size)
        for i,trp in enumerate(rp):
            idx = ( (trp-head['RPMIN'])/bin_size_rp ).astype(int)
            bin_center_rp[i] = head['RPMIN']+(idx+0.5)*bin_size_rp

        bin_size_rt = head['RTMAX']/head['NT']
        bin_center_rt = sp.zeros(rt.size)
        for i,trt in enumerate(rt):
            idx = ( trt/bin_size_rt ).astype(int)
            bin_center_rt[i] = (idx+0.5)*bin_size_rt

        bin_center_r = sp.sqrt(bin_center_rp**2+bin_center_rt**2)
        bin_center_mu = bin_center_rp/bin_center_r

        ## select data within cuts
        mask = (bin_center_rp > rp_min) & (bin_center_rp < rp_max)
        mask &= (bin_center_rt > rt_min) & (bin_center_rt < rt_max)
        mask &= (bin_center_r > r_min) & (bin_center_r < r_max)
        mask &= (bin_center_mu > mu_min) & (bin_center_mu < mu_max)

        nmask = mask.sum()
        self.mask = mask
        self.da = da
        self.da_cut = sp.zeros(mask.sum())
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
        self.pk *= partial(getattr(pk,'G2'), dataset_name=self.name)
        if 'small scale nl' in dic_init['model']:
            self.pk *= getattr(pk, dic_init['model']['small scale nl'])

        if 'velocity dispersion' in dic_init['model']:
            self.pk *= getattr(pk, dic_init['model']['velocity dispersion'])

        ## add non linear large scales
        self.pk *= pk.pk_NL

        self.xi = partial(getattr(xi, dic_init['model']['model-xi']), name=self.name)

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
            self.pk_met *= partial(getattr(pk,'G2'), dataset_name=self.name)

            if 'velocity dispersion' in dic_init['model']:
                self.pk_met *= getattr(pk, dic_init['model']['velocity dispersion'])

            self.xi_met = partial(getattr(xi, dic_init['metals']['model-xi-met']), name=self.name)

            hmet = fitsio.FITS(dic_init['metals']['filename'])

            assert 'in tracer1' in dic_init['metals'] or 'in tracer2' in dic_init['metals']

            if self.tracer1 == self.tracer2:
                assert dic_init['metals']['in tracer1'] == dic_init['metals']['in tracer2']

                for m in dic_init['metals']['in tracer1']:
                    self.z_evol[m] = partial(getattr(xi, dic_init['metals']['z evol']), zref = self.zref)
                    self.rp_met[(self.tracer1, m)] = hmet[2]["RP_{}_{}".format(self.tracer1,m)][:]
                    self.rt_met[(self.tracer1, m)] = hmet[2]["RT_{}_{}".format(self.tracer1,m)][:]
                    self.z_met[(self.tracer1, m)] = hmet[2]["Z_{}_{}".format(self.tracer1,m)][:]
                    try:
                        self.dm_met[(self.tracer1, m)] = csr_matrix(hmet[2]["DM_{}_{}".format(self.tracer1,m)][:])
                    except:
                        self.dm_met[(self.tracer1, m)] = csr_matrix(hmet[3]["DM_{}_{}".format(self.tracer1,m)][:])

                    self.rp_met[(m, self.tracer1)] = hmet[2]["RP_{}_{}".format(self.tracer1,m)][:]
                    self.rt_met[(m, self.tracer1)] = hmet[2]["RT_{}_{}".format(self.tracer1,m)][:]
                    self.z_met[(m, self.tracer1)] = hmet[2]["Z_{}_{}".format(self.tracer1,m)][:]
                    try:
                        self.dm_met[(m, self.tracer1)] = csr_matrix(hmet[2]["DM_{}_{}".format(self.tracer1,m)][:])
                    except:
                        self.dm_met[(m, self.tracer1)] = csr_matrix(hmet[3]["DM_{}_{}".format(self.tracer1,m)][:])

            else:
                if 'in tracer2' in dic_init['metals']:
                    for m in dic_init['metals']['in tracer2']:
                        self.z_evol[m] = partial(getattr(xi, dic_init['metals']['z evol']), zref = self.zref)
                        self.rp_met[(self.tracer1, m)] = hmet[2]["RP_{}_{}".format(self.tracer1,m)][:]
                        self.rt_met[(self.tracer1, m)] = hmet[2]["RT_{}_{}".format(self.tracer1,m)][:]
                        self.z_met[(self.tracer1, m)] = hmet[2]["Z_{}_{}".format(self.tracer1,m)][:]
                        try:
                            self.dm_met[(self.tracer1, m)] = csr_matrix(hmet[2]["DM_{}_{}".format(self.tracer1,m)][:])
                        except:
                            self.dm_met[(self.tracer1, m)] = csr_matrix(hmet[3]["DM_{}_{}".format(self.tracer1,m)][:])

                if 'in tracer1' in dic_init['metals']:
                    for m in dic_init['metals']['in tracer1']:
                        self.z_evol[m] = partial(getattr(xi, dic_init['metals']['z evol']), zref = self.zref)
                        self.rp_met[(m, self.tracer2)] = hmet[2]["RP_{}_{}".format(m, self.tracer2)][:]
                        self.rt_met[(m, self.tracer2)] = hmet[2]["RT_{}_{}".format(m, self.tracer2)][:]
                        self.z_met[(m, self.tracer2)] = hmet[2]["Z_{}_{}".format(m, self.tracer2)][:]
                        try:
                            self.dm_met[(m, self.tracer2)] = csr_matrix(hmet[2]["DM_{}_{}".format(m, self.tracer2)][:])
                        except:
                            self.dm_met[(m, self.tracer2)] = csr_matrix(hmet[3]["DM_{}_{}".format(m, self.tracer2)][:])

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

    def xi_model(self, k, pk_lin, pars):
        xi = self.xi(self.r, self.mu, k, pk_lin, self.pk, \
                    tracer1 = self.tracer1, tracer2 = self.tracer2, ell_max = self.ell_max, **pars)

        xi *= self.z_evol[self.tracer1](self.z, self.tracer1, **pars)*self.z_evol[self.tracer2](self.z, self.tracer2, **pars)
        xi *= self.growth_function(self.z, **pars)**2

        for tracer1, tracer2 in self.dm_met:
            rp = self.rp_met[(tracer1, tracer2)]
            rt = self.rt_met[(tracer1, tracer2)]
            z = self.z_met[(tracer1, tracer2)]
            dm_met = self.dm_met[(tracer1, tracer2)]
            r = sp.sqrt(rp**2+rt**2)
            w = r == 0
            r[w] = 1e-6
            mu = rp/r
            xi_met = self.xi_met(r, mu, k, pk_lin, self.pk_met, \
                tracer1 = tracer1, tracer2 = tracer2, ell_max = self.ell_max, **pars)

            xi_met *= self.z_evol[tracer1](z, tracer1, **pars)*self.z_evol[tracer2](z, tracer2, **pars)
            xi_met *= self.growth_function(z, **pars)**2
            xi_met = dm_met.dot(xi_met)
            xi += xi_met

        xi = self.dm.dot(xi)

        return xi

    def chi2(self, k, pk_lin, pksb_lin, pars):
        xi_peak = self.xi_model(k, pk_lin-pksb_lin, pars)

        ap = pars['ap']
        at = pars['at']
        pars['ap']=1.
        pars['at']=1.

        sigmaNL_per = pars['sigmaNL_per']
        pars['sigmaNL_per'] = 0
        xi_sb = self.xi_model(k, pksb_lin, pars)
        pars['ap'] = ap
        pars['at'] = at
        pars['sigmaNL_per'] = sigmaNL_per

        xi_full = pars['bao_amp']*xi_peak + xi_sb
        dxi = self.da_cut-xi_full[self.mask]

        return dxi.T.dot(self.ico.dot(dxi))
