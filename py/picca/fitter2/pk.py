import numpy as np
from . import utils
from pkg_resources import resource_filename
import astropy.io.fits as fits
from . import constants
import random
from scipy.special import wofz
import scipy.integrate as integrate
muk = utils.muk
bias_beta = utils.bias_beta
Fvoigt_data = []

class pk:
    def __init__(self, func, name_model=None):
        self.func = func
        global Fvoigt_data
        if (not name_model is None) and (Fvoigt_data == []):
            #path = '{}/models/fvoigt_models/Fvoigt_{}.txt'.format(resource_filename('picca', 'fitter2'),name_model)
            #Fvoigt_data = np.loadtxt(path)
            type_pdf = str(name_model)
            Fvoigt_data=get_Fhcd(type_pdf,NHI=None)

    def __call__(self, k, pk_lin, tracer1, tracer2, **kwargs):
        return self.func(k, pk_lin, tracer1, tracer2, **kwargs)

    def __mul__(self,func2):
        func = lambda k, pk_lin, tracer1, tracer2, **kwargs: self(k, pk_lin, tracer1, tracer2, **kwargs)*func2(k, pk_lin, tracer1, tracer2, **kwargs)
        return pk(func)

    __imul__ = __mul__
    __rmul__ = __mul__

def pk_NL(k, pk_lin, tracer1, tracer2, **kwargs):
    kp = k*muk
    kt = k*np.sqrt(1-muk**2)
    st2 = kwargs['sigmaNL_per']**2
    sp2 = kwargs['sigmaNL_par']**2
    return np.exp(-(kp**2*sp2+kt**2*st2)/2)

def pk_kaiser(k, pk_lin, tracer1, tracer2, **kwargs):
    bias1, beta1, bias2, beta2 = bias_beta(kwargs, tracer1, tracer2)
    pk = bias1*bias2*pk_lin*(1+beta1*muk**2)*(1+beta2*muk**2)

    return pk

def pk_hcd(k, pk_lin, tracer1, tracer2, **kwargs):

    bias1, beta1, bias2, beta2 = bias_beta(kwargs, tracer1, tracer2)

    bias_hcd = kwargs["bias_hcd"]
    beta_hcd = kwargs["beta_hcd"]
    L0 = kwargs["L0_hcd"]

    kp = k*muk
    F_hcd = utils.sinc(kp*L0)

    bias_eff1 = bias1 + bias_hcd*F_hcd
    beta_eff1 = (bias1 * beta1 + bias_hcd*beta_hcd*F_hcd)/(bias1 + bias_hcd*F_hcd)

    bias_eff2 = bias2 + bias_hcd*F_hcd
    beta_eff2 = (bias2 * beta2 + bias_hcd*beta_hcd*F_hcd)/(bias2 + bias_hcd*F_hcd)

    pk = pk_lin*bias_eff1*bias_eff2*(1 + beta_eff1*muk**2)*(1 + beta_eff2*muk**2)

    return pk

def pk_hcd_Rogers2018(k, pk_lin, tracer1, tracer2, **kwargs):
    """Model the effect of HCD systems with the Fourier transform
       of a Lorentzian profile. Motivated by Rogers et al. (2018).

    Args:
        Same than pk_hcd

    Returns:
        Same than pk_hcd

    """

    bias1, beta1, bias2, beta2 = bias_beta(kwargs, tracer1, tracer2)

    key = "bias_hcd_{}".format(kwargs['name'])
    if key in kwargs :
        bias_hcd = kwargs[key]
    else :
        bias_hcd = kwargs["bias_hcd"]
    beta_hcd = kwargs["beta_hcd"]
    L0 = kwargs["L0_hcd"]

    kp = k*muk
    F_hcd = np.exp(-L0*kp)

    bias_eff1 = bias1 + bias_hcd*F_hcd
    beta_eff1 = (bias1 * beta1 + bias_hcd*beta_hcd*F_hcd)/(bias1 + bias_hcd*F_hcd)

    bias_eff2 = bias2 + bias_hcd*F_hcd
    beta_eff2 = (bias2 * beta2 + bias_hcd*beta_hcd*F_hcd)/(bias2 + bias_hcd*F_hcd)

    pk = pk_lin*bias_eff1*bias_eff2*(1 + beta_eff1*muk**2)*(1 + beta_eff2*muk**2)

    return pk

def get_Fhcd(type_pdf='masking',NHI=None):
    path = '{}/'.format(resource_filename('picca', 'fitter2'))
    version = '4.7'
    path_qso = path+'data/zcat_desi_drq.fits'
    if type_pdf=='masking':
        path_dla = path+'data/zcat_desi_drq_DLA_inf_203.fits'
        path_weight_lambda = path+'data/weight_lambda.txt'
    elif type_pdf=='nomasking':
        path_dla = path+'data/zcat_desi_drq_DLA.fits'
        path_weight_lambda = path+'data/weight_lambda_nomasking.txt'
    ################
    if version == '4.4':
        mockid = 'MOCKID'
        z_dla = 'Z_DLA_RSD'
    elif version == '4.7':
        mockid = 'THING_ID'
        z_dla = 'Z'
    data = fits.open(path_dla)[1].data
    qso = fits.open(path_qso)[1].data
    # keep only DLA which are front of a QSO.
    data = data[:][np.in1d(data[mockid], qso['THING_ID'])]
    nb_qso = qso['Z'].size # number of line of sight
    weight_lambda = np.loadtxt(path_weight_lambda)
    lamb_w = weight_lambda[:,0]
    weight = weight_lambda[:,1]
    zdla = np.mean(data[z_dla])

    def cddf_lbg(mu1,sigma1,number1,mu2,sigma2,number2):
        NHI2 = np.random.normal(mu2,sigma2,size=int(number2))
        if number1>100:
            NHI1 = np.random.normal(mu1,sigma1,size=int(number1))
            NHI=np.append(NHI1,NHI2)
        else:
            NHI1 = []
            NHI=NHI2
        NHI=NHI[NHI>17.15]
        count, bins = np.histogram(NHI, bins=50,density=True)
        return count, bins
    
    def multi_gauss(x,mu1,sigma1,number1,mu2,sigma2,number2):
        y=number1 * np.exp(-(x - mu1) ** 2 / (2 * sigma1 ** 2)) + number2 * np.exp(-(x - mu2) ** 2 / (2 * sigma2 ** 2))
        return x,y/np.trapz(y,x)

    def renorm_pdf(y_norm, bins):
        z = bins[:-1] + (bins[1]/2-bins[0]/2)
        return y_norm, z

    def build_pdf(data_NHI,data_Z,type_pdf,NHI=None,reshape=False):
        cddf_Z, dN_Z = np.histogram(data_Z, bins=50, density=True) 
        if not NHI:
            cddf_NHI, dN_NHI = np.histogram(data_NHI, bins=50, density=True)
        else:
            cddf_NHI, dN_NHI = np.histogram([NHI]*np.ones(len(data_NHI)), bins=50, density=True)
            cddf_NHI = cddf_NHI+0.0001
        if reshape:
            cddf_NHI, dN_NHI = renorm_pdf(cddf_NHI, dN_NHI)
            cddf_Z, dN_Z = renorm_pdf(cddf_Z, dN_Z)
            return cddf_NHI, dN_NHI, cddf_Z, dN_Z
        else:
            return cddf_NHI, dN_NHI, cddf_Z, dN_Z
        
    def build_pdf_Z(data_Z,reshape=False):
        cddf_Z, dN_Z = np.histogram(data_Z, bins=50, density=True)    
        if reshape:
            cddf_Z, dN_Z = renorm_pdf(cddf_Z, dN_Z)
        return cddf_Z, dN_Z
    
    def voigt(x, sigma=1, gamma=1):
        return np.real(wofz((x + 1j*gamma)/(sigma*np.sqrt(2))))

    def tau(lamb, z, N_hi): # lamb = lambda in A and N_HI in log10 and 10**N_hi in cm^-2
        lamb_rf = lamb/(1+z)
        e = 1.6021e-19 # C
        epsilon0 = 8.8541e-12 # C^2.s^2.kg^-1.m^-3
        f = 0.4164
        mp = 1.6726e-27 # kg
        me = 9.109e-31 # kg
        c = 2.9979e8 # m.s^-1
        k = 1.3806e-23 # m^2.kg.s^-2.K-1
        T = 1e4 # K
        gamma = 6.265e8 # s^-1
        lamb_alpha = constants.absorber_IGM["LYA"] # A
        Deltat_lamb = lamb_alpha/c*np.sqrt(2*k*T/mp) # A

        a = gamma/(4*np.pi*Deltat_lamb)*lamb_alpha**2/c*1e-10
        u = (lamb_rf - lamb_alpha)/Deltat_lamb
        H = voigt(u, np.sqrt(1/2), a)

        absorb = np.sqrt(np.pi)*e**2*f*lamb_alpha**2*1e-10/(4*np.pi*epsilon0*me*c**2*Deltat_lamb)*H
        # 10^N_hi in cm^-2 and absorb in m^2
        return 10**N_hi*1e4*absorb

    def profile_voigt_lambda(x, z, N_hi):
        t = tau(x, z, N_hi).astype(float)
        return np.exp(-t)

    def profile_lambda_to_r(lamb, profile_lambda, fidcosmo): # for lyman-alpha otherwise use an other emission line
        z = lamb/constants.absorber_IGM["LYA"] - 1
        r = fidcosmo.r_comoving(z)
        rr = np.linspace(r[0], r[-1], r.size)
        profile_r = np.interp(rr, r, profile_lambda) # to have a linear sample
        return rr, profile_r

    def fft_profile(profile, dx): # not normalized
        n = profile.size
        tmp = (profile-1)
        ft_profile = dx*np.fft.fftshift(np.fft.fft(tmp))
        k = np.fft.fftshift(np.fft.fftfreq(n, dx))*(2*np.pi)
        return ft_profile, k

    def lambda_to_r(lamb, profile_lambda, fidcosmo): # f(lambda)dlambda = f(r)dr
            z = lamb/constants.absorber_IGM["LYA"] - 1
            r = fidcosmo.r_comoving(z)
            rr = np.linspace(r[0], r[-1], r.size)
            profile_lambda = profile_lambda*fidcosmo.hubble(z)*constants.absorber_IGM["LYA"]/3e5
            profile_r = np.interp(rr,r,profile_lambda)
            return rr, profile_r

    def dla_catalog(pdf_lbg_NHI,pdf_lbg_Z,number):
        data_dla_NHI = []
        data_dla_Z = []
        for i in range(len(pdf_lbg_NHI[0])):
            num = int(pdf_lbg_NHI[0][i]*(number)/np.sum(pdf_lbg_NHI[0]))
            diff = pdf_lbg_NHI[1][1]-pdf_lbg_NHI[1][0]
            data_dla_NHI=data_dla_NHI+list(pdf_lbg_NHI[1][i]+diff*np.random.random(num))
        if len(data_dla_NHI)!=number:
            for i in range(abs(len(data_dla_NHI)-number)):
                data_dla_NHI.append(pdf_lbg_NHI[1][i])

        for i in range(len(pdf_lbg_Z[0])):
            num = int(pdf_lbg_Z[0][i]*(number)/np.sum(pdf_lbg_Z[0]))
            diff = pdf_lbg_Z[1][1]-pdf_lbg_Z[1][0]
            data_dla_Z=data_dla_Z+list(pdf_lbg_Z[1][i]+diff*np.random.random(num))
        if len(data_dla_Z)!=number:
            for i in range(abs(len(data_dla_Z)-number)):
                data_dla_Z.append(pdf_lbg_Z[1][i])

        data_dla_NHI = random.sample(data_dla_NHI,len(data_dla_NHI))
        data_dla_Z = random.sample(data_dla_Z,len(data_dla_Z))
        data_dla_NHI = np.array(data_dla_NHI)
        data_dla_Z = np.array(data_dla_Z)
        return data_dla_NHI,data_dla_Z

    def save_function(data,type_pdf,NHI=0):
        fidcosmo = constants.cosmo(Om=0.3)
        lamb = np.arange(2000, 8000, 1)
        if type_pdf=='nomasking':
            f_lambda=np.loadtxt(path+'data/f_lambda_nomasking.txt')
        elif type_pdf=='masking':
            f_lambda=np.loadtxt(path+'data/f_lambda_masking.txt')
        r, f_r = lambda_to_r(f_lambda[0], f_lambda[1], fidcosmo)
        r_w, weight_r = profile_lambda_to_r(lamb_w, weight, fidcosmo)
        weight_interp = np.interp(r, r_w, weight_r, left=0, right=0)
        mean_density = np.average(f_r, weights=weight_interp)
        cddf_NHI, dN_NHI, cddf_Z, dN_Z = build_pdf(data['NHI'],data['Z'],type_pdf,NHI,reshape=True)
        number = len(data['NHI'])
        cat_NHI, cat_Z = dla_catalog([cddf_NHI, dN_NHI],[cddf_Z, dN_Z],number)
        zdla = np.mean(cat_Z)
        for i in range(dN_NHI.size):
            profile_lambda = profile_voigt_lambda(lamb, zdla, dN_NHI[i])
            profile_lambda = profile_lambda/np.mean(profile_lambda)
            r, profile_r = profile_lambda_to_r(lamb, profile_lambda, fidcosmo) # r is in Mpc h^-1 --> k (from tf) will be in (Mpc h^-1)^-1 = h Mpc^-1 :)
            ft_profile, k = fft_profile(profile_r, np.abs(r[1]-r[0]))
            ft_profile = np.abs(ft_profile)
            if i == 0:
                df = np.array([ft_profile*mean_density*cddf_NHI[i]])
            else:
                df = np.concatenate((df, np.array([ft_profile*mean_density*cddf_NHI[i]])))
        Fvoigt = np.zeros(k.size)
        for i in range(k.size):
            Fvoigt[i] = integrate.trapz(df[:,i], dN_NHI)
        Fvoigt = -Fvoigt[k>0]
        k = k[k>0]
        save = np.transpose(np.concatenate((np.array([k]), np.array([Fvoigt]))))
        return save
    save_all = save_function(data,type_pdf,NHI)
    return save_all

def pk_hcd_voigt(k, pk_lin, tracer1, tracer2, **kwargs):
    global Fvoigt_data
    bias1, beta1, bias2, beta2 = bias_beta(kwargs, tracer1, tracer2)
    key = "bias_hcd_{}".format(kwargs['name'])
    if key in kwargs :
        bias_hcd = kwargs[key]
    else :
        bias_hcd = kwargs["bias_hcd"]
    beta_hcd = kwargs["beta_hcd"]
    L0 = kwargs["L0_hcd"]
    type_pdf = kwargs["L0_hcd"]

    kp = k*muk
    
    k_data = Fvoigt_data[:,0]
    F_data = Fvoigt_data[:,1]
    from scipy.interpolate import splev, splrep
    f_pk = splrep(k_data, F_data)
    F_hcd = splev(kp,f_pk)

    bias_eff1 = bias1 + bias_hcd*F_hcd
    beta_eff1 = (bias1 * beta1 + bias_hcd*beta_hcd*F_hcd)/(bias1 + bias_hcd*F_hcd)

    bias_eff2 = bias2 + bias_hcd*F_hcd
    beta_eff2 = (bias2 * beta2 + bias_hcd*beta_hcd*F_hcd)/(bias2 + bias_hcd*F_hcd)

    pk = pk_lin*bias_eff1*bias_eff2*(1 + beta_eff1*muk**2)*(1 + beta_eff2*muk**2)

    return pk

def pk_hcd_no_mask(k, pk_lin, tracer1, tracer2, **kwargs):
    """
    Use Fvoigt function to fit the DLA in the autocorrelation Lyman-alpha without masking them ! (L0 = 1)

    (If you want to mask them --> use Fvoigt_exp.txt and L0 = 10 as eBOOS DR14)

    """
    global Fvoigt_data
    bias1, beta1, bias2, beta2 = bias_beta(kwargs, tracer1, tracer2)

    key = "bias_hcd_{}".format(kwargs['name'])
    if key in kwargs :
        bias_hcd = kwargs[key]
    else :
        bias_hcd = kwargs["bias_hcd"]
    beta_hcd = kwargs["beta_hcd"]
    L0 = kwargs["L0_hcd"]

    kp = k*muk

    k_data = Fvoigt_data[:,0]
    F_data = Fvoigt_data[:,1]

    F_hcd = np.interp(L0*kp, k_data, F_data, left=0, right=0)

    bias_eff1 = bias1 + bias_hcd*F_hcd
    beta_eff1 = (bias1 * beta1 + bias_hcd*beta_hcd*F_hcd)/(bias1 + bias_hcd*F_hcd)

    bias_eff2 = bias2 + bias_hcd*F_hcd
    beta_eff2 = (bias2 * beta2 + bias_hcd*beta_hcd*F_hcd)/(bias2 + bias_hcd*F_hcd)

    pk = pk_lin*bias_eff1*bias_eff2*(1 + beta_eff1*muk**2)*(1 + beta_eff2*muk**2)

    return pk

def pk_uv(k, pk_lin, tracer1, tracer2, **kwargs):

    bias1, beta1, bias2, beta2 = bias_beta(kwargs, tracer1, tracer2)

    bias_gamma = kwargs["bias_gamma"]
    bias_prim = kwargs["bias_prim"]
    lambda_uv = kwargs["lambda_uv"]

    W = np.arctan(k*lambda_uv)/(k*lambda_uv)
    beta1 = beta1/(1 + bias_gamma/bias1*W/(1 + bias_prim*W))
    bias1 = bias1 + bias_gamma*W/(1+bias_prim*W)

    beta2 = beta2/(1 + bias_gamma/bias2*W/(1 + bias_prim*W))
    bias2 = bias2 + bias_gamma*W/(1+bias_prim*W)

    return pk_lin*bias1*bias2*(1+beta1*muk**2)*(1+beta2*muk**2)

def pk_hcd_uv(k, pk_lin, tracer1, tracer2, **kwargs):
    bias1, beta1, bias2, beta2 = bias_beta(kwargs, tracer1, tracer2)

    bias_gamma = kwargs["bias_gamma"]
    bias_prim = kwargs["bias_prim"]
    lambda_uv = kwargs["lambda_uv"]

    W = np.arctan(k*lambda_uv)/(k*lambda_uv)
    beta1 = beta1/(1 + bias_gamma/bias1*W/(1 + bias_prim*W))
    bias1 = bias1 + bias_gamma*W/(1+bias_prim*W)

    beta2 = beta2/(1 + bias_gamma/bias2*W/(1 + bias_prim*W))
    bias2 = bias2 + bias_gamma*W/(1+bias_prim*W)

    bias_hcd = kwargs["bias_hcd"]
    beta_hcd = kwargs["beta_hcd"]
    L0 = kwargs["L0_hcd"]

    kp = k*muk
    F_hcd = utils.sinc(kp*L0)

    bias_eff1 = bias1 + bias_hcd*F_hcd
    beta_eff1 = (bias1 * beta1 + bias_hcd*beta_hcd*F_hcd)/(bias1 + bias_hcd*F_hcd)

    bias_eff2 = bias2 + bias_hcd*F_hcd
    beta_eff2 = (bias2 * beta2 + bias_hcd*beta_hcd*F_hcd)/(bias2 + bias_hcd*F_hcd)

    pk = pk_lin*bias_eff1*bias_eff2*(1 + beta_eff1*muk**2)*(1 + beta_eff2*muk**2)

    return pk

def pk_hcd_Rogers2018_uv(k, pk_lin, tracer1, tracer2, **kwargs):
    bias1, beta1, bias2, beta2 = bias_beta(kwargs, tracer1, tracer2)

    bias_gamma = kwargs["bias_gamma"]
    bias_prim = kwargs["bias_prim"]
    lambda_uv = kwargs["lambda_uv"]

    W = np.arctan(k*lambda_uv)/(k*lambda_uv)
    beta1 = beta1/(1 + bias_gamma/bias1*W/(1 + bias_prim*W))
    bias1 = bias1 + bias_gamma*W/(1+bias_prim*W)

    beta2 = beta2/(1 + bias_gamma/bias2*W/(1 + bias_prim*W))
    bias2 = bias2 + bias_gamma*W/(1+bias_prim*W)

    key = "bias_hcd_{}".format(kwargs['name'])
    if key in kwargs :
        bias_hcd = kwargs[key]
    else :
        bias_hcd = kwargs["bias_hcd"]
    beta_hcd = kwargs["beta_hcd"]
    L0 = kwargs["L0_hcd"]

    kp = k*muk
    F_hcd = np.exp(-kp*L0)

    bias_eff1 = bias1 + bias_hcd*F_hcd
    beta_eff1 = (bias1 * beta1 + bias_hcd*beta_hcd*F_hcd)/(bias1 + bias_hcd*F_hcd)

    bias_eff2 = bias2 + bias_hcd*F_hcd
    beta_eff2 = (bias2 * beta2 + bias_hcd*beta_hcd*F_hcd)/(bias2 + bias_hcd*F_hcd)

    pk = pk_lin*bias_eff1*bias_eff2*(1 + beta_eff1*muk**2)*(1 + beta_eff2*muk**2)

    return pk

def dnl_mcdonald(k, pk_lin, tracer1, tracer2, pk_fid, **kwargs):
    assert tracer1['name']=="LYA" and tracer2['name']=="LYA"
    kvel = 1.22*(1+k/0.923)**0.451
    dnl = np.exp((k/6.4)**0.569-(k/15.3)**2.01-(k*muk/kvel)**1.5)
    return dnl

def dnl_arinyo(k, pk_lin, tracer1, tracer2, pk_fid, **kwargs):
    assert tracer1['name']=="LYA" and tracer2['name']=="LYA"
    q1 = kwargs["dnl_arinyo_q1"]
    kv = kwargs["dnl_arinyo_kv"]
    av = kwargs["dnl_arinyo_av"]
    bv = kwargs["dnl_arinyo_bv"]
    kp = kwargs["dnl_arinyo_kp"]

    growth = q1*k*k*k*pk_fid/(2*np.pi*np.pi)
    pecvelocity = np.power(k/kv,av)*np.power(np.fabs(muk),bv)
    pressure = (k/kp)*(k/kp)
    dnl = np.exp(growth*(1-pecvelocity)-pressure)
    return dnl

def cached_g2(function):
  memo = {}
  def wrapper(*args, **kwargs):

    dataset_name = kwargs['dataset_name']
    Lpar = kwargs["par binsize {}".format(dataset_name)]
    Lper = kwargs["per binsize {}".format(dataset_name)]

    if dataset_name in memo and np.allclose(memo[dataset_name][0], [Lpar, Lper]):
      return memo[dataset_name][1]
    else:
      rv = function(*args, **kwargs)
      memo[dataset_name] = [[Lpar, Lper], rv]
      return rv
  return wrapper

@cached_g2
def G2(k, pk_lin, tracer1, tracer2, dataset_name = None, **kwargs):
    Lpar = kwargs["par binsize {}".format(dataset_name)]
    Lper = kwargs["per binsize {}".format(dataset_name)]

    kp = k*muk
    kt = k*np.sqrt(1-muk**2)
    return utils.sinc(kp*Lpar/2)*utils.sinc(kt*Lper/2)

def pk_hcd_cross(k, pk_lin, tracer1, tracer2, **kwargs):
    bias1, beta1, bias2, beta2 = bias_beta(kwargs, tracer1, tracer2)
    assert (tracer1['name']=="LYA" or tracer2['name']=="LYA") and (tracer1['name']!=tracer2['name'])

    bias_hcd = kwargs["bias_hcd"]
    beta_hcd = kwargs["beta_hcd"]
    L0 = kwargs["L0_hcd"]

    kp = k*muk
    F_hcd = utils.sinc(kp*L0)

    if tracer1['name'] == "LYA":
        bias_eff1 = bias1 + bias_hcd*F_hcd
        beta_eff1 = (bias1 * beta1 + bias_hcd*beta_hcd*F_hcd)/(bias1 + bias_hcd*F_hcd)
        pk = pk_lin*bias_eff1*bias2*(1 + beta_eff1*muk**2)*(1 + beta2*muk**2)
    else:
        bias_eff2 = bias2 + bias_hcd*F_hcd
        beta_eff2 = (bias2 * beta2 + bias_hcd*beta_hcd*F_hcd)/(bias2 + bias_hcd*F_hcd)
        pk = pk_lin*bias1*bias_eff2*(1 + beta1*muk**2)*(1 + beta_eff2*muk**2)

    return pk

def pk_hcd_Rogers2018_cross(k, pk_lin, tracer1, tracer2, **kwargs):
    bias1, beta1, bias2, beta2 = bias_beta(kwargs, tracer1, tracer2)
    assert (tracer1['name']=="LYA" or tracer2['name']=="LYA") and (tracer1['name']!=tracer2['name'])

    key = "bias_hcd_{}".format(kwargs['name'])
    if key in kwargs :
        bias_hcd = kwargs[key]
    else :
        bias_hcd = kwargs["bias_hcd"]
    beta_hcd = kwargs["beta_hcd"]
    L0 = kwargs["L0_hcd"]

    kp = k*muk
    F_hcd = np.exp(-kp*L0)

    if tracer1['name'] == "LYA":
        bias_eff1 = bias1 + bias_hcd*F_hcd
        beta_eff1 = (bias1 * beta1 + bias_hcd*beta_hcd*F_hcd)/(bias1 + bias_hcd*F_hcd)
        pk = pk_lin*bias_eff1*bias2*(1 + beta_eff1*muk**2)*(1 + beta2*muk**2)
    else:
        bias_eff2 = bias2 + bias_hcd*F_hcd
        beta_eff2 = (bias2 * beta2 + bias_hcd*beta_hcd*F_hcd)/(bias2 + bias_hcd*F_hcd)
        pk = pk_lin*bias1*bias_eff2*(1 + beta1*muk**2)*(1 + beta_eff2*muk**2)

    return pk

def pk_hcd_cross_no_mask(k, pk_lin, tracer1, tracer2, **kwargs):
    bias1, beta1, bias2, beta2 = bias_beta(kwargs, tracer1, tracer2)
    assert (tracer1['name']=="LYA" or tracer2['name']=="LYA") and (tracer1['name']!=tracer2['name'])

    key = "bias_hcd_{}".format(kwargs['name'])
    if key in kwargs :
        bias_hcd = kwargs[key]
    else :
        bias_hcd = kwargs["bias_hcd"]
    beta_hcd = kwargs["beta_hcd"]
    L0 = kwargs["L0_hcd"]

    kp = k*muk
    k_data = Fvoigt_data[0]
    F_data = Fvoigt_data[1]
    from scipy.interpolate import splev, splrep
    f_pk = splrep(k_data, F_data)
    F_hcd = splev(kp,f_pk)
    
    if tracer1['name'] == "LYA":
        bias_eff1 = bias1 + bias_hcd*F_hcd
        beta_eff1 = (bias1 * beta1 + bias_hcd*beta_hcd*F_hcd)/(bias1 + bias_hcd*F_hcd)
        pk = pk_lin*bias_eff1*bias2*(1 + beta_eff1*muk**2)*(1 + beta2*muk**2)
    else:
        bias_eff2 = bias2 + bias_hcd*F_hcd
        beta_eff2 = (bias2 * beta2 + bias_hcd*beta_hcd*F_hcd)/(bias2 + bias_hcd*F_hcd)
        pk = pk_lin*bias1*bias_eff2*(1 + beta1*muk**2)*(1 + beta_eff2*muk**2)

    return pk

def pk_uv_cross(k, pk_lin, tracer1, tracer2, **kwargs):
    bias1, beta1, bias2, beta2 = bias_beta(kwargs, tracer1, tracer2)
    assert (tracer1['type']=="continuous" or tracer2['type']=="continuous") and (tracer1['type']!=tracer2['type'])

    bias_gamma = kwargs["bias_gamma"]
    bias_prim = kwargs["bias_prim"]
    lambda_uv = kwargs["lambda_uv"]

    W = np.arctan(k*lambda_uv)/(k*lambda_uv)

    if tracer1['type'] == "continuous":
        beta1 = beta1/(1 + bias_gamma/bias1*W/(1 + bias_prim*W))
        bias1 = bias1 + bias_gamma*W/(1+bias_prim*W)
    else:
        beta2 = beta2/(1 + bias_gamma/bias2*W/(1 + bias_prim*W))
        bias2 = bias2 + bias_gamma*W/(1+bias_prim*W)

    return pk_lin*bias1*bias2*(1+beta1*muk**2)*(1+beta2*muk**2)

def pk_hcd_uv_cross(k, pk_lin, tracer1, tracer2, **kwargs):
    bias1, beta1, bias2, beta2 = bias_beta(kwargs, tracer1, tracer2)
    assert (tracer1['name']=="LYA" or tracer2['name']=="LYA") and (tracer1['name']!=tracer2['name'])

    bias_gamma = kwargs["bias_gamma"]
    bias_prim = kwargs["bias_prim"]
    lambda_uv = kwargs["lambda_uv"]

    W = np.arctan(k*lambda_uv)/(k*lambda_uv)

    bias_hcd = kwargs["bias_hcd"]
    beta_hcd = kwargs["beta_hcd"]
    L0 = kwargs["L0_hcd"]

    kp = k*muk
    F_hcd = utils.sinc(kp*L0)

    if tracer1['name'] == "LYA":
        beta1 = beta1/(1 + bias_gamma/bias1*W/(1 + bias_prim*W))
        bias1 = bias1 + bias_gamma*W/(1+bias_prim*W)
        bias_eff1 = bias1 + bias_hcd*F_hcd
        beta_eff1 = (bias1 * beta1 + bias_hcd*beta_hcd*F_hcd)/(bias1 + bias_hcd*F_hcd)
        pk = pk_lin*bias_eff1*bias2*(1 + beta_eff1*muk**2)*(1 + beta2*muk**2)
    else:
        beta2 = beta2/(1 + bias_gamma/bias2*W/(1 + bias_prim*W))
        bias2 = bias2 + bias_gamma*W/(1+bias_prim*W)
        bias_eff2 = bias2 + bias_hcd*F_hcd
        beta_eff2 = (bias2 * beta2 + bias_hcd*beta_hcd*F_hcd)/(bias2 + bias_hcd*F_hcd)
        pk = pk_lin*bias1*bias_eff2*(1 + beta1*muk**2)*(1 + beta_eff2*muk**2)

    return pk

def pk_hcd_Rogers2018_uv_cross(k, pk_lin, tracer1, tracer2, **kwargs):
    bias1, beta1, bias2, beta2 = bias_beta(kwargs, tracer1, tracer2)
    assert (tracer1['name']=="LYA" or tracer2['name']=="LYA") and (tracer1['name']!=tracer2['name'])

    bias_gamma = kwargs["bias_gamma"]
    bias_prim = kwargs["bias_prim"]
    lambda_uv = kwargs["lambda_uv"]

    W = np.arctan(k*lambda_uv)/(k*lambda_uv)

    key = "bias_hcd_{}".format(kwargs['name'])
    if key in kwargs :
        bias_hcd = kwargs[key]
    else :
        bias_hcd = kwargs["bias_hcd"]
    beta_hcd = kwargs["beta_hcd"]
    L0 = kwargs["L0_hcd"]

    kp = k*muk
    F_hcd = np.exp(-kp*L0)

    if tracer1['name'] == "LYA":
        beta1 = beta1/(1 + bias_gamma/bias1*W/(1 + bias_prim*W))
        bias1 = bias1 + bias_gamma*W/(1+bias_prim*W)
        bias_eff1 = bias1 + bias_hcd*F_hcd
        beta_eff1 = (bias1 * beta1 + bias_hcd*beta_hcd*F_hcd)/(bias1 + bias_hcd*F_hcd)
        pk = pk_lin*bias_eff1*bias2*(1 + beta_eff1*muk**2)*(1 + beta2*muk**2)
    else:
        beta2 = beta2/(1 + bias_gamma/bias2*W/(1 + bias_prim*W))
        bias2 = bias2 + bias_gamma*W/(1+bias_prim*W)
        bias_eff2 = bias2 + bias_hcd*F_hcd
        beta_eff2 = (bias2 * beta2 + bias_hcd*beta_hcd*F_hcd)/(bias2 + bias_hcd*F_hcd)
        pk = pk_lin*bias1*bias_eff2*(1 + beta1*muk**2)*(1 + beta_eff2*muk**2)

    return pk

def pk_gauss_smoothing(k, pk_lin, tracer1, tracer2, **kwargs):
    """
    Apply a Gaussian smoothing to the full correlation function
    Args:
        k: array containing k (in h/Mpc)
        pk_lin: not used
        tracer1: not used
        tracer2: not used
        par_sigma_smooth (in kwargs): sigma of the smoothing in the
                                        parallel direction (in Mpc/h)
        per_sigma_smooth (in kwargs): sigma of the smoothing in the
                                        perpendicular direction (in Mpc/h)
    return : G(k)^2
    with G(k) = exp(-(kpar^2 sigma_par^2 + kperp^2 sigma_perp^2)/2)
    where G(k) is the smoothing applied to density field in mocks
    """
    kp  = k*muk
    kt  = k*np.sqrt(1.-muk**2)
    st2 = kwargs['per_sigma_smooth']**2
    sp2 = kwargs['par_sigma_smooth']**2
    return np.exp(-(kp**2*sp2+kt**2*st2)/2.)**2

def pk_gauss_exp_smoothing(k, pk_lin, tracer1, tracer2, **kwargs):
    """
    Apply a Gaussian and exp smoothing to the full correlation function (use full for london_mocks_v6.0

    """
    kp  = k*muk
    kt  = k*np.sqrt(1.-muk**2)
    st2 = kwargs['per_sigma_smooth']**2
    sp2 = kwargs['par_sigma_smooth']**2

    et2 = kwargs['per_exp_smooth']**2
    ep2 = kwargs['par_exp_smooth']**2

    return np.exp(-(kp**2*sp2+kt**2*st2)/2.)*np.exp(-(np.absolute(kp)*ep2+np.absolute(kt)*et2) )

def pk_velo_gaus(k, pk_lin, tracer1, tracer2, **kwargs):
    assert 'discrete' in [tracer1['type'],tracer2['type']]
    kp = k*muk
    smooth = np.ones(kp.shape)
    if tracer1['type']=='discrete':
        smooth *= np.exp( -0.25*(kp*kwargs['sigma_velo_gaus_'+tracer1['name']])**2)
    if tracer2['type']=='discrete':
        smooth *= np.exp( -0.25*(kp*kwargs['sigma_velo_gaus_'+tracer2['name']])**2)
    return smooth

def pk_velo_lorentz(k, pk_lin, tracer1, tracer2, **kwargs):
    assert 'discrete' in [tracer1['type'],tracer2['type']]
    kp = k*muk
    smooth = np.ones(kp.shape)
    if tracer1['type']=='discrete':
        smooth *= 1./np.sqrt(1.+(kp*kwargs['sigma_velo_lorentz_'+tracer1['name']])**2)
    if tracer2['type']=='discrete':
        smooth *= 1./np.sqrt(1.+(kp*kwargs['sigma_velo_lorentz_'+tracer2['name']])**2)
    return smooth
