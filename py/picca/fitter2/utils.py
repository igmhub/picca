
import scipy as sp
from numpy import fft
from scipy import special
import scipy.interpolate

from . import myGamma

nmuk = 1000
muk=(sp.arange(nmuk)+0.5)/nmuk
dmuk = 1./nmuk
muk=muk[:,None]

def Pk2Mp(ar,k,pk,ell_max=None):
    """
    Implementation of FFTLog from A.J.S. Hamilton (2000)
    assumes log(k) are equally spaced
    """

    k0 = k[0]
    l=sp.log(k.max()/k0)
    r0=1.

    N=len(k)
    emm=N*fft.fftfreq(N)
    r=r0*sp.exp(-emm*l/N)
    dr=abs(sp.log(r[1]/r[0]))
    s=sp.argsort(r)
    r=r[s]

    xi=sp.zeros([ell_max/2+1,len(ar)])

    for ell in range(0,ell_max+1,2):
        pk_ell=sp.sum(dmuk*L(muk,ell)*pk,axis=0)*(2*ell+1)*(-1)**(ell/2)
        mu=ell+0.5
        n=2.
        q=2-n-0.5
        x=q+2*sp.pi*1j*emm/l
        lg1=myGamma.LogGammaLanczos((mu+1+x)/2)
        lg2=myGamma.LogGammaLanczos((mu+1-x)/2)

        um=(k0*r0)**(-2*sp.pi*1j*emm/l)*2**x*sp.exp(lg1-lg2)
        um[0]=sp.real(um[0])
        an=fft.fft(pk_ell*k**n/2/sp.pi**2*sp.sqrt(sp.pi/2))
        an*=um
        xi_loc=fft.ifft(an)
        xi_loc=xi_loc[s]
        xi_loc/=r**(3-n)
        xi_loc[-1]=0
        spline=sp.interpolate.splrep(sp.log(r)-dr/2,sp.real(xi_loc),k=3,s=0)
        xi[ell/2,:]=sp.interpolate.splev(sp.log(ar),spline)

    return xi

def Pk2Xi(ar,mur,k,pk,ell_max=None):
    xi=Pk2Mp(ar,k,pk,ell_max)
    for ell in range(0,ell_max+1,2):
        xi[ell/2,:]*=L(mur,ell)
    return sp.sum(xi,axis=0)

### Legendre Polynomial
def L(mu,ell):
    return special.legendre(ell)(mu)

def bias_beta(kwargs, tracer1, tracer2):

    growth_rate = kwargs["growth_rate"]

    beta1 = kwargs["beta_{}".format(tracer1)]
    bias1 = kwargs["bias_{}".format(tracer1)]
    bias1 *= growth_rate/beta1

    beta2 = kwargs["beta_{}".format(tracer2)]
    bias2 = kwargs["bias_{}".format(tracer2)]
    bias2 *= growth_rate/beta2

    return bias1, beta1, bias2, beta2

    
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
    r        = sp.sqrt( rp_shift**2. + rt**2.)
    mur      = rp_shift/r

    ###
    xi_rad  = pars['qso_rad_strength']/(r**2.)
    xi_rad *= 1.-pars['qso_rad_asymmetry']*(1.-mur**2.)
    xi_rad *= sp.exp(-r*( (1.+mur)/pars['qso_rad_lifetime'] + 1./pars['qso_rad_decrease']) )

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
