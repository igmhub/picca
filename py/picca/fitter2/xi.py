import numpy as np
from . import utils
from scipy.integrate import quad
from scipy.interpolate import interp1d

from ..utils import userprint

def xi(r, mu, k, pk_lin, pk_func, tracer1=None, tracer2=None, ell_max=None, **pars):
    pk_full = pk_func(k, pk_lin, tracer1, tracer2, **pars)

    ap, at = utils.cosmo_fit_func(pars)
    rp = r*mu
    rt = r*np.sqrt(1-mu**2)
    arp = ap*rp
    art = at*rt
    ar = np.sqrt(arp**2+art**2)
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
        if tracer1['type']=='discrete':
            drp = kwargs['drp_'+tracer1['name']]
        elif tracer2['type']=='discrete':
            drp = kwargs['drp_'+tracer2['name']]

        ## args[3] is the pk_lin, we need to make sure we recalculate
        ## when it changes (e.g. when we pass the pksb_lin)
        t = tuple(x for x in args[3])
        pair = (name, tracer1['name'], tracer2['name'], hash(t))

        recalc = True
        if pair in cache and np.allclose(cache[pair][0][2:], [beta1, beta2, ap, at, drp]):
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

    if tracer1['type']=='discrete':
        drp = pars['drp_'+tracer1['name']]
    elif tracer2['type']=='discrete':
        drp = pars['drp_'+tracer2['name']]
    ap, at = utils.cosmo_fit_func(pars)
    rp = r*mu + drp
    rt = r*np.sqrt(1-mu**2)
    arp = ap*rp
    art = at*rt
    ar = np.sqrt(arp**2+art**2)
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
        if pair in cache and np.allclose(cache[pair][0][2:], [beta1, beta2, ap, at]):
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
def xi_qso_radiation(r, mu, tracer1, tracer2, **pars):
    assert (tracer1['name']=="QSO" or tracer2['name']=="QSO") and (tracer1['name']!=tracer2['name'])

    if tracer1['type']=='discrete':
        drp = pars['drp_'+tracer1['name']]
    elif tracer2['type']=='discrete':
        drp = pars['drp_'+tracer2['name']]
    rp = r*mu + drp
    rt = r*np.sqrt(1-mu**2)
    r_shift = np.sqrt(rp**2.+rt**2.)
    mu_shift = rp/r_shift

    xi_rad = pars["qso_rad_strength"]/(r_shift**2.)
    xi_rad *= 1.-pars["qso_rad_asymmetry"]*(1.-mu_shift**2.)
    xi_rad *= np.exp(-r_shift*( (1.+mu_shift)/pars["qso_rad_lifetime"] + 1./pars["qso_rad_decrease"]) )

    return xi_rad

def xi_relativistic(r, mu, k, pk_lin, tracer1, tracer2, **pars):
    """Calculate the cross-correlation contribution from relativistic effects (Bonvin et al. 2014).

    Args:
        r (float): r coordinates
        mu (float): mu coordinates
        k (float): wavenumbers
        pk_lin (float): linear matter power spectrum
        tracer1: dictionary of tracer1
        tracer2: dictionary of tracer2
        pars: dictionary of fit parameters

    Returns:
        sum of dipole and octupole correlation terms (float)

    """
    assert (tracer1['type']=="continuous" or tracer2['type']=="continuous") and (tracer1['type']!=tracer2['type'])

    if tracer1['type']=='discrete':
        drp = pars['drp_'+tracer1['name']]
    elif tracer2['type']=='discrete':
        drp = pars['drp_'+tracer2['name']]

    ap, at = utils.cosmo_fit_func(pars)
    rp = r*mu + drp
    rt = r*np.sqrt(1-mu**2)
    arp = ap*rp
    art = at*rt
    ar = np.sqrt(arp**2+art**2)
    amu = arp/ar

    xi_rel = utils.Pk2XiRel(ar, amu, k, pk_lin, pars)
    return xi_rel

def xi_asymmetry(r, mu, k, pk_lin, tracer1, tracer2, **pars):
    """Calculate the cross-correlation contribution from standard asymmetry (Bonvin et al. 2014).

    Args:
        r (float): r coordinates
        mu (float): mu coordinates
        k (float): wavenumbers
        pk_lin (float): linear matter power spectrum
        tracer1: dictionary of tracer1
        tracer2: dictionary of tracer2
        pars: dictionary of fit parameters

    Returns:
        sum of dipole and octupole correlation terms (float)

    """
    assert (tracer1['type']=="continuous" or tracer2['type']=="continuous") and (tracer1['type']!=tracer2['type'])

    if tracer1['type']=='discrete':
        drp = pars['drp_'+tracer1['name']]
    elif tracer2['type']=='discrete':
        drp = pars['drp_'+tracer2['name']]

    ap, at = utils.cosmo_fit_func(pars)
    rp = r*mu + drp
    rt = r*np.sqrt(1-mu**2)
    arp = ap*rp
    art = at*rt
    ar = np.sqrt(arp**2+art**2)
    amu = arp/ar

    xi_asy = utils.Pk2XiAsy(ar, amu, k, pk_lin, pars)
    return xi_asy

### Growth factor evolution
def growth_factor_no_de(z, zref=None, **kwargs):
    return (1+zref)/(1.+z)

def cache_growth_factor_de(function):
    cache = {}
    def wrapper(*args, **kwargs):
        Om = kwargs['Om']
        OL = kwargs['OL']
        pair = ('Om', 'OL')
        if pair not in cache.keys() or not np.allclose(cache[pair], (Om,OL)):
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
    userprint('Calculating growth factor')

    def hubble(z, Om, OL):
        return np.sqrt(Om*(1+z)**3 + OL + (1-Om-OL)*(1+z)**2)

    def dD1(a, Om, OL):
        z = 1/a-1
        return 1./(a*hubble(z=z, Om=Om, OL=OL))**3

    ## Calculate D1 in 100 values of z between 0 and zmax, then interpolate
    nbins = 100
    zmax = 5.
    z = zmax*np.arange(nbins, dtype=float)/(nbins-1)
    D1 = np.zeros(nbins, dtype=float)
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

def broadband_sky(r, mu, name=None, bin_size_rp=None, *pars, **kwargs):
    '''
        Broadband function interface.
        Calculates a Gaussian broadband in rp,rt for the sky residuals
    Arguments:
        - r,mu (array or float): where to calcualte the broadband
        - bin_size_rp (array): Bin size of the distortion matrix along the line-of-sight
        - name: (string) name ot identify the corresponding parameters,
                    typically the dataset name and whether it's multiplicative
                    of additive
        - *pars: additional parameters that are ignored (for convenience)
        **kwargs (dict): dictionary containing all the polynomial
                    coefficients. Any extra keywords are ignored
    Returns:
        - cor (array of float): Correlation function
    '''

    rp = r*mu
    rt = r*np.sqrt(1-mu**2)
    cor = kwargs[name+'-scale-sky']/(kwargs[name+'-sigma-sky']*np.sqrt(2.*np.pi))*np.exp(-0.5*(rt/kwargs[name+'-sigma-sky'])**2)
    w = (rp>=0.) & (rp<bin_size_rp)
    cor[~w] = 0.

    return cor

def broadband(r, mu, deg_r_min=None, deg_r_max=None,
        ddeg_r=None, deg_mu_min=None, deg_mu_max=None,
        ddeg_mu=None, deg_mu=None, name=None,
        rp_rt=False, bin_size_rp=None, *pars, **kwargs):
    '''
    Broadband function interface.
    Calculates a power-law broadband in r and mu or rp,rt
    Arguments:
        - r,mu: (array or float) where to calcualte the broadband
        - deg_r_min: (int) degree of the lowest-degree monomial in r or rp
        - deg_r_max: (int) degree of the highest-degree monomual in r or rp
        - ddeg_r: (int) degree step in r or rp
        - deg_mu_min: (int) degree of the lowest-degree monomial in mu or rt
        - deg_mu_max: (int) degree of the highest-degree monmial in mu or rt
        - ddeg_mu: (int) degree step in mu or rt
        - name: (string) name ot identify the corresponding parameters,
                    typically the dataset name and whether it's multiplicative
                    of additive
        - rt_rp: (bool) use r,mu (if False) or rp,rt (if True)
        - *pars: additional parameters that are ignored (for convenience)
        **kwargs: (dict) dictionary containing all the polynomial
                    coefficients. Any extra keywords are ignored
    '''

    r1 = r/100
    r2 = mu
    if rp_rt:
        r1 = (r/100)*mu
        r2 = (r/100)*np.sqrt(1-mu**2)

    r1_pows = np.arange(deg_r_min, deg_r_max+1, ddeg_r)
    r2_pows = np.arange(deg_mu_min, deg_mu_max+1, ddeg_mu)
    BB = [kwargs['{} ({},{})'.format(name,i,j)] for i in r1_pows
            for j in r2_pows]
    BB = np.array(BB).reshape(-1,deg_r_max-deg_r_min+1)

    return (BB[:,:,None,None]*r1**r1_pows[:,None,None]*\
            r2**r2_pows[None,:,None]).sum(axis=(0,1,2))
