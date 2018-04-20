import scipy as sp
from . import utils
from scipy.integrate import quad
from scipy.interpolate import interp1d

def xi(r, mu, k, pk_lin, pk_func, tracer1=None, tracer2=None, ell_max=None, **pars):
    pk_full = pk_func(k, pk_lin, tracer1, tracer2, **pars)

    ap, at = utils.cosmo_fit_func(pars)
    rp = r*mu
    rt = r*sp.sqrt(1-mu**2)
    arp = ap*rp
    art = at*rt
    ar = sp.sqrt(arp**2+art**2)
    amu = arp/ar

    xi_full = utils.Pk2Xi(ar, amu, k, pk_full, ell_max = ell_max)
    return xi_full


def cache_xi_drp(function):
    cache = {}
    def wrapper(*args, **kwargs):
        name = kwargs['name']
        tracer1 = kwargs['tracer1']
        tracer2 = kwargs['tracer2']

        bias1, beta1, bias2, beta2 = utils.bias_beta(kwargs, tracer1, tracer2)
        ap, at = utils.cosmo_fit_func(kwargs)
        drp = kwargs['drp']

        ## args[3] is the pk_lin, we need to make sure we recalculate
        ## when it changes (e.g. when we pass the pksb_lin)
        t = tuple(x for x in args[3])
        pair = (name, tracer1['name'], tracer2['name'], hash(t))

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

def xi_drp(r, mu, k, pk_lin, pk_func, tracer1=None, tracer2=None, ell_max=None, **pars):
    pk_full = pk_func(k, pk_lin, tracer1, tracer2, **pars)

    ap, at = utils.cosmo_fit_func(pars)
    rp = r*mu + pars["drp"]
    rt = r*sp.sqrt(1-mu**2)
    arp = ap*rp
    art = at*rt
    ar = sp.sqrt(arp**2+art**2)
    amu = arp/ar

    xi_full = utils.Pk2Xi(ar, amu, k, pk_full, ell_max = ell_max)
    return xi_full

@cache_xi_drp
def cached_xi_drp(*args, **kwargs):
    return xi_drp(*args, **kwargs)

def cache_kaiser(function):
    cache = {}
    def wrapper(*args, **kwargs):
        name = kwargs['name']
        tracer1 = kwargs['tracer1']
        tracer2 = kwargs['tracer2']

        bias1, beta1, bias2, beta2 = utils.bias_beta(kwargs, tracer1, tracer2)
        ap, at = utils.cosmo_fit_func(kwargs)

        ## args[3] is the pk_lin, we need to make sure we recalculate
        ## when it changes (e.g. when we pass the pksb_lin)
        t = tuple(x for x in args[3])
        pair = (name, tracer1['name'], tracer2['name'], hash(t))

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
def xi_qso_radiation(r, mu, k, pk_lin, pk_func, tracer1, tracer2, ell_max=None, **pars):
    assert (tracer1['name']=="QSO" or tracer2['name']=="QSO") and (tracer1['name']!=tracer2['name'])

    rp = r*mu + pars["drp"]
    rt = r*sp.sqrt(1-mu**2)
    r_shift = sp.sqrt(rp**2.+rt**2.)
    mu_shift = rp/r_shift

    xi_rad = pars["qso_rad_strength"]/(r_shift**2.)
    xi_rad *= 1.-pars["qso_rad_asymmetry"]*(1.-mu_shift**2.)
    xi_rad *= sp.exp(-r_shift*( (1.+mu_shift)/pars["qso_rad_lifetime"] + 1./pars["qso_rad_decrease"]) )

    return xi_drp(r, mu, k, pk_lin, pk_func, tracer1, tracer2, ell_max, **pars) + xi_rad

### Growth factor evolution
def growth_factor_no_de(z, zref=None, **kwargs):
    return (1+zref)/(1.+z)

def cache_growth_factor_de(function):
    cache = {}
    def wrapper(*args, **kwargs):
        Om = kwargs['Om']
        OL = kwargs['OL']
        pair = ('Om', 'OL')
        if pair not in cache.keys() or not sp.allclose(cache[pair], (Om,OL)):
            cache[pair] = (Om, OL)
            cache[1] = cached_growth_factor_de(*args, **kwargs)

        return cache[1](args[0])/cache[1](kwargs['zref'])

    return wrapper

@cache_growth_factor_de
def growth_factor_de(*args, **kwargs):
    return cached_growth_factor_de(*args, **kwargs)

def cached_growth_factor_de(z, zref=None, Om=None, OL=None, **kwargs):
    '''
    Implements eq. 7.77 from S. Dodelson's Modern Cosmology book
    '''
    print('Calculating growth factor for Om = {}, OL = {}'.format(Om, OL))

    def hubble(z, Om, OL):
        return sp.sqrt(Om*(1+z)**3 + OL + (1-Om-OL)*(1+z)**2)

    def dD1(a, Om, OL):
        z = 1/a-1
        return 1./(a*hubble(z=z, Om=Om, OL=OL))**3

    ## Calculate D1 in 100 values of z between 0 and zmax, then interpolate
    nbins = 100
    zmax = 5.
    z = zmax*sp.arange(nbins, dtype=float)/(nbins-1)
    D1 = sp.zeros(nbins, dtype=float)
    pars = (Om, OL)
    for i in range(nbins):
        a = 1/(1+z[i])
        D1[i] = 5/2.*Om*hubble(z[i], *pars)*quad(dD1, 0, a, args=pars)[0]

    D1 = interp1d(z, D1)
    return D1

### Lya bias evolution
def bias_vs_z_std(z, tracer, zref = None, **kwargs):
    p0 = kwargs['alpha_{}'.format(tracer['name'])]
    return ((1.+z)/(1+zref))**p0

def qso_bias_vs_z_croom(z, tracer, zref = None, **kwargs):
    assert tracer['name']=="QSO"
    p0 = kwargs["croom_par0"]
    p1 = kwargs["croom_par1"]
    return (p0 + p1*(1.+z)**2)/(p0 + p1*(1+zref)**2)
