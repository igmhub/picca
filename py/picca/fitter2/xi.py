import numpy as np
from utils import Pk2Xi

def xi(r, mu, k, pk_lin, pk_func, tracer1, tracer2, pars, ell_max=None):
    pk_full = pk_func(k, pk_lin, tracer1, tracer2, **pars)
    
    ap = pars["ap"]
    at = pars["at"]
    rp = r*mu + pars["drp"]
    rt = r*np.sqrt(1-mu**2)
    arp = ap*rp
    art = at*rt
    ar = np.sqrt(arp**2+art**2)
    amu = arp/ar

    xi_full = Pk2Xi(ar, amu, k, pk_full, ell_max = ell_max)
    return xi_full


def cached_xi_kaiser(r, mu, k, pk_lin, pk_func, tracer1, tracer2, pars, ell_max=None):
    pk_full = pk_func(k, pk_lin, tracer1, tracer2, **pars)
    
    ap = pars["ap"]
    at = pars["at"]
    rp = r*mu + pars["drp"]
    rt = r*np.sqrt(1-mu**2)
    arp = ap*rp
    art = at*rt
    ar = np.sqrt(arp**2+art**2)
    amu = arp/ar

    xi_full = Pk2Xi(ar, amu, k, pk_full, ell_max = ell_max)
    return xi_full


### QSO radiation model
def xi_qso_radiation_model(r, mu, k, pk_lin, tracer1, tracer2, pars, ell_max=None):
    assert tracer1 == "QSO" and tracer2 == "LYA" or tracer1 == "LYA" and tracer2 == "QSO"

    rp_shift = rp+pars['drp']

    xi_rad  = pars['qso_rad_strength']/(r**2.)
    xi_rad *= 1.-pars['qso_rad_asymmetry']*(1.-mu**2.)
    xi_rad *= sp.exp(-r*( (1.+mur)/pars['qso_rad_lifetime'] + 1./pars['qso_rad_decrease']) )

    return xi(r, mu, k, pk_lin, tracer1, tracer2, pars, ell_max=ell_max) + xi_rad

### Growth factor evolution
def growth_function_no_de(z, zref = None, **kwargs):
    return (1+zref)/(1.+z)

### Lya bias evolution
def bias_vs_z_std(z, tracer, zref = None, **kwargs):
    p0 = kwargs['alpha_{}'.format(tracer)]
    return ((1.+z)/(1+zref))**p0

def qso_bias_vs_z_croom(z, tracer, zref = None, **kwargs):
    assert tracer=="QSO"
    p0 = kwargs["croom_par0"]
    p1 = kwargs["croom_par1"]
    return p0 + p1*(1.+z)**2/(p0 + p1*(1+zref)**2)
