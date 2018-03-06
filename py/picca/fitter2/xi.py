import scipy as sp
from .utils import Pk2Xi, bias_beta, ap_at

def xi(r, mu, k, pk_lin, pk_func, tracer1=None, tracer2=None, bao_dilation=None, ell_max=None, **pars):
    pk_full = pk_func(k, pk_lin, tracer1, tracer2, **pars)
    
    ap, at = ap_at(pars, bao_dilation)
    rp = r*mu 
    rt = r*sp.sqrt(1-mu**2)
    arp = ap*rp
    art = at*rt
    ar = sp.sqrt(arp**2+art**2)
    amu = arp/ar

    xi_full = Pk2Xi(ar, amu, k, pk_full, ell_max = ell_max)
    return xi_full


def cache_xi_drp(function):
    cache = {}
    def wrapper(*args, **kwargs):
        name = kwargs['name']
        bao_dilation = kwargs['bao_dilation']
        tracer1 = kwargs['tracer1']
        tracer2 = kwargs['tracer2']

        bias1, beta1, bias2, beta2 = bias_beta(kwargs, tracer1, tracer2)

        ap, at = ap_at(kwargs, bao_dilation)
        drp = kwargs['drp']

        ## args[3] is the pk_lin, we need to make sure we recalculate
        ## when it changes (e.g. when we pass the pksb_lin)
        t = tuple(x for x in args[3])
        pair = (name, tracer1, tracer2, hash(t))

        recalc = True
        if pair in cache and sp.allclose(cache[pair][0][2:], [beta1, beta2, ap, at, drp]):
            recalc = False
        
        if not recalc:
            ret = cache[pair][1]*bias1*bias2/cache[pair][0][0]/cache[pair][0][1]
        else:
            cache[pair] = [[bias1, bias2, beta1, beta2, ap, at, drp], xi_drp(*args, **kwargs)]
            ret = cache[pair][1]

        return ret*1.

    return wrapper

def xi_drp(r, mu, k, pk_lin, pk_func, tracer1=None, tracer2=None, bao_dilation=None, ell_max=None, **pars):
    pk_full = pk_func(k, pk_lin, tracer1, tracer2, **pars)
    
    ap, at = ap_at(pars, bao_dilation)
    rp = r*mu + pars["drp"]
    rt = r*sp.sqrt(1-mu**2)
    arp = ap*rp
    art = at*rt
    ar = sp.sqrt(arp**2+art**2)
    amu = arp/ar

    xi_full = Pk2Xi(ar, amu, k, pk_full, ell_max = ell_max)
    return xi_full

@cache_xi_drp
def cached_xi_drp(*args, **kwargs):
    return xi_drp(*args, **kwargs)

def cache_kaiser(function):
    cache = {}
    def wrapper(*args, **kwargs):
        name = kwargs['name']
        bao_dilation = kwargs['bao_dilation']
        tracer1 = kwargs['tracer1']
        tracer2 = kwargs['tracer2']

        bias1, beta1, bias2, beta2 = bias_beta(kwargs, tracer1, tracer2)

        ap, at = ap_at(kwargs, bao_dilation)

        ## args[3] is the pk_lin, we need to make sure we recalculate
        ## when it changes (e.g. when we pass the pksb_lin)
        t = tuple(x for x in args[3])
        pair = (name, tracer1, tracer2, hash(t))

        recalc = True
        if pair in cache and sp.allclose(cache[pair][0][2:], [beta1, beta2, ap, at]):
            recalc = False
        
        if not recalc:
            ret = cache[pair][1]*bias1*bias2/cache[pair][0][0]/cache[pair][0][1]
        else:
            cache[pair] = [[bias1, bias2, beta1, beta2, ap, at], xi(*args, **kwargs)]
            ret = cache[pair][1]

        return ret*1.

    return wrapper

@cache_kaiser
def cached_xi_kaiser(*args, **kwargs):
    return xi(*args, **kwargs)

### QSO radiation model
def xi_qso_radiation(r, mu, k, pk_lin, pk_func, tracer1, tracer2, bao_dilation, ell_max=None, **pars):
    assert (tracer1=="QSO" or tracer2=="QSO") and (tracer1!=tracer2)

    rp = r*mu + pars["drp"]
    rt = r*sp.sqrt(1-mu**2)
    r_shift = sp.sqrt(rp**2.+rt**2.)
    mu_shift = rp/r_shift

    xi_rad = pars["qso_rad_strength"]/(r_shift**2.)
    xi_rad *= 1.-pars["qso_rad_asymmetry"]*(1.-mu_shift**2.)
    xi_rad *= sp.exp(-r_shift*( (1.+mu_shift)/pars["qso_rad_lifetime"] + 1./pars["qso_rad_decrease"]) )

    return xi_drp(r, mu, k, pk_lin, pk_func, tracer1, tracer2, bao_dilation, ell_max, **pars) + xi_rad

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
    return (p0 + p1*(1.+z)**2)/(p0 + p1*(1+zref)**2)
