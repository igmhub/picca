import numpy as np
from scipy import special

### Legendre Polynomial
def L(mu,ell):
    return special.legendre(ell)(mu)

### Growth factor evolution
def evolution_growth_factor_by_hand(z):
    return 1./(1.+z)

### Lya bias evolution
def evolution_Lya_bias_0(z,param):
    return (1.+z)**param[0]

### QSO bias evolution
def evolution_QSO_bias_none(z,param):
    return 1.+0.*z
def evolution_QSO_bias_croom(z,param):
    return param[0] + param[1]*(1.+z)**2

### QSO radiation model
def qso_radiation_model(rp,rt,pars):

    ###
    rp_shift = rp+pars['drp']
    r        = np.sqrt( rp_shift**2. + rt**2.)
    mur      = rp_shift/r

    ###
    xi_rad  = pars['qso_rad_strength']/(r**2.)
    xi_rad *= 1.-pars['qso_rad_asymmetry']*(1.-mur**2.)
    xi_rad *= np.exp(-r*( (1.+mur)/pars['qso_rad_lifetime'] + 1./pars['qso_rad_decrease']) )

    return xi_rad

### Absorber names and wavelengths
absorber_IGM = {
    'MgI(2853)'   : 2852.96,
    'MgII(2804)'  : 2803.5324,
    'MgII(2796)'  : 2796.3511,
    'FeII(2600)'  : 2600.1724835,
    'FeII(2587)'  : 2586.6495659,
    'MnII(2577)'  : 2576.877,
    'FeII(2383)'  : 2382.7641781,
    'FeII(2374)'  : 2374.4603294,
    'FeII(2344)'  : 2344.2129601,
    'AlIII(1863)' : 1862.79113,
    'AlIII(1855)' : 1854.71829,
    'AlII(1671)'  : 1670.7886,
    'FeII(1609)'  : 1608.4511,
    'CIV(1551)'   : 1550.77845,
    'CIV(1548)'   : 1548.2049,
    'SiII(1527)'  : 1526.70698,
    'SiIV(1403)'  : 1402.77291,
    'SiIV(1394)'  : 1393.76018,
    'CII(1335)'   : 1334.5323,
    'SiII(1304)'  : 1304.3702,
    'OI(1302)'    : 1302.1685,
    'SiII(1260)'  : 1260.4221,
    'NV(1243)'    : 1242.804,
    'NV(1239)'    : 1238.821,
    'LYA'         : 1215.67,
    'SiIII(1207)' : 1206.500,
    'NI(1200)'    : 1200.,
    'SiII(1193)'  : 1193.2897,
    'SiII(1190)'  : 1190.4158,
    'LYB'         : 1025.7223,
}
