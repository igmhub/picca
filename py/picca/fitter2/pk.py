import scipy as sp
import numpy as np
from . import utils

muk = utils.muk
bias_beta = utils.bias_beta

data_london = np.loadtxt('/global/homes/e/edmondc/software/picca/py/picca/fitter2/Fvoigt_london_6.0.txt')
data_london_2 = np.loadtxt('/global/homes/e/edmondc/software/picca/py/picca/fitter2/Fvoigt_london_6.0_norm.txt')
data_saclay = np.loadtxt('/global/homes/e/edmondc/software/picca/py/picca/fitter2/Fvoigt_saclay_4.4.txt')
data_saclay_bin1 = np.loadtxt('/global/homes/e/edmondc/software/picca/py/picca/fitter2/Fvoigt_saclay_4.4_bin1.txt')
data_saclay_bin2 = np.loadtxt('/global/homes/e/edmondc/software/picca/py/picca/fitter2/Fvoigt_saclay_4.4_bin2.txt')
data_saclay_2 = np.loadtxt('/global/homes/e/edmondc/software/picca/py/picca/fitter2/Fvoigt_saclay_4.4_norm.txt')
##faire attention ou est le fichier Fvoigt.txt

class pk:
    def __init__(self, func):
        self.func = func

    def __call__(self, k, pk_lin, tracer1, tracer2, **kwargs):
        return self.func(k, pk_lin, tracer1, tracer2, **kwargs)

    def __mul__(self,func2):
        func = lambda k, pk_lin, tracer1, tracer2, **kwargs: self(k, pk_lin, tracer1, tracer2, **kwargs)*func2(k, pk_lin, tracer1, tracer2, **kwargs)
        return pk(func)

    __imul__ = __mul__
    __rmul__ = __mul__

def pk_NL(k, pk_lin, tracer1, tracer2, **kwargs):
    kp = k*muk
    kt = k*sp.sqrt(1-muk**2)
    st2 = kwargs['sigmaNL_per']**2
    sp2 = kwargs['sigmaNL_par']**2
    return sp.exp(-(kp**2*sp2+kt**2*st2)/2)

def pk_gauss_smoothing(k, pk_lin, tracer1, tracer2, **kwargs):
    """
    Apply a Gaussian smoothing to the full correlation function

    """
    kp  = k*muk
    kt  = k*sp.sqrt(1.-muk**2)
    st2 = kwargs['per_sigma_smooth']**2
    sp2 = kwargs['par_sigma_smooth']**2
    return sp.exp(-(kp**2*sp2+kt**2*st2)/2.)

def pk_gauss_exp_smoothing(k, pk_lin, tracer1, tracer2, **kwargs):
    """
    Apply a Gaussian smoothing to the full correlation function

    """
    kp  = k*muk
    kt  = k*sp.sqrt(1.-muk**2)
    st2 = kwargs['per_sigma_smooth']**2
    sp2 = kwargs['par_sigma_smooth']**2

    et2 = kwargs['per_exp_smooth']**2
    ep2 = kwargs['par_exp_smooth']**2

    return sp.exp(-(kp**2*sp2+kt**2*st2)/2.)*sp.exp(-(np.abs(kp)*ep2+np.abs(kt)*et2) )

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
    F_hcd = sp.exp(-L0*kp)

    bias_eff1 = bias1 + bias_hcd*F_hcd
    beta_eff1 = (bias1 * beta1 + bias_hcd*beta_hcd*F_hcd)/(bias1 + bias_hcd*F_hcd)

    bias_eff2 = bias2 + bias_hcd*F_hcd
    beta_eff2 = (bias2 * beta2 + bias_hcd*beta_hcd*F_hcd)/(bias2 + bias_hcd*F_hcd)

    pk = pk_lin*bias_eff1*bias_eff2*(1 + beta_eff1*muk**2)*(1 + beta_eff2*muk**2)

    return pk

def pk_hcd_Voigt_london(k, pk_lin, tracer1, tracer2, **kwargs):
    """Model the effect of HCD systems with the Fourier transform
       of a Voigt profile. Motivated by Rogers et al. (2018). eq (3).
       using cddf of london

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

    k_data = data_london[:,0]
    F_data = data_london[:,1]

    F_hcd = np.interp(L0*kp, k_data, F_data)

    bias_eff1 = bias1 + bias_hcd*F_hcd
    beta_eff1 = (bias1 * beta1 + bias_hcd*beta_hcd*F_hcd)/(bias1 + bias_hcd*F_hcd)

    bias_eff2 = bias2 + bias_hcd*F_hcd
    beta_eff2 = (bias2 * beta2 + bias_hcd*beta_hcd*F_hcd)/(bias2 + bias_hcd*F_hcd)

    pk = pk_lin*bias_eff1*bias_eff2*(1 + beta_eff1*muk**2)*(1 + beta_eff2*muk**2)

    return pk

def pk_hcd_Voigt_saclay(k, pk_lin, tracer1, tracer2, **kwargs):
    """Model the effect of HCD systems with the Fourier transform
       of a Voigt profile. Motivated by Rogers et al. (2018). eq (3).
       using cddf of saclay

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

    k_data = data_saclay[:,0]
    F_data = data_saclay[:,1]

    F_hcd = np.interp(L0*kp, k_data, F_data)

    bias_eff1 = bias1 + bias_hcd*F_hcd
    beta_eff1 = (bias1 * beta1 + bias_hcd*beta_hcd*F_hcd)/(bias1 + bias_hcd*F_hcd)

    bias_eff2 = bias2 + bias_hcd*F_hcd
    beta_eff2 = (bias2 * beta2 + bias_hcd*beta_hcd*F_hcd)/(bias2 + bias_hcd*F_hcd)

    pk = pk_lin*bias_eff1*bias_eff2*(1 + beta_eff1*muk**2)*(1 + beta_eff2*muk**2)

    return pk

def pk_hcd_Voigt_saclay_bin(k, pk_lin, tracer1, tracer2, **kwargs):
    """Model the effect of HCD systems with the Fourier transform
       of a Voigt profile. Motivated by Rogers et al. (2018). eq (3).
       using cddf of saclay and binning the HCDs in two parts

    Args:
        Same than pk_hcd

    Returns:
        Same than pk_hcd

    """

    bias1, beta1, bias2, beta2 = bias_beta(kwargs, tracer1, tracer2)

    bias_hcd_1 = kwargs["bias_hcd1"]
    beta_hcd_1 = kwargs["beta_hcd1"]
    L_1 = kwargs["L0_hcd1"]

    bias_hcd_2 = kwargs["bias_hcd2"]
    beta_hcd_2 = kwargs["beta_hcd2"]
    L_2 = kwargs["L0_hcd2"]

    kp = k*muk

    k_data1 = data_saclay_bin1[:,0]
    F_data1 = data_saclay_bin1[:,1]
    k_data2 = data_saclay_bin2[:,0]
    F_data2 = data_saclay_bin2[:,1]

    F_hcd1 = np.interp(L_1*kp, k_data1, F_data1)
    F_hcd2 = np.interp(L_2*kp, k_data2, F_data2)

    bf = bias1*(1 + beta1*muk**2)
    bh1 = bias_hcd_1*(1 + beta_hcd_1*muk**2)*F_hcd1
    bh2 = bias_hcd_2*(1 + beta_hcd_2*muk**2)*F_hcd2

    pk = pk_lin*(bf**2 + 2*bf*bh1 + 2*bf*bh2 + 2*bh1*bh2 + bh1**2 + bh2**2)

    return pk

def pk_hcd_Voigt_saclay_norm(k, pk_lin, tracer1, tracer2, **kwargs):
    """
        we do not normalized the voigt profile but only at the end
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

    k_data = data_saclay_2[:,0]
    F_data = data_saclay_2[:,1]

    F_hcd = np.interp(L0*kp, k_data, F_data)

    bias_eff1 = bias1 + bias_hcd*F_hcd
    beta_eff1 = (bias1 * beta1 + bias_hcd*beta_hcd*F_hcd)/(bias1 + bias_hcd*F_hcd)

    bias_eff2 = bias2 + bias_hcd*F_hcd
    beta_eff2 = (bias2 * beta2 + bias_hcd*beta_hcd*F_hcd)/(bias2 + bias_hcd*F_hcd)

    pk = pk_lin*bias_eff1*bias_eff2*(1 + beta_eff1*muk**2)*(1 + beta_eff2*muk**2)

    return pk


def pk_hcd_Voigt_london_norm(k, pk_lin, tracer1, tracer2, **kwargs):
    """
        we do not normalized the voigt profile but only at the end
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

    k_data = data_london_2[:,0]
    F_data = data_london_2[:,1]

    F_hcd = np.interp(L0*kp, k_data, F_data)

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

    W = sp.arctan(k*lambda_uv)/(k*lambda_uv)
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

    W = sp.arctan(k*lambda_uv)/(k*lambda_uv)
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

    W = sp.arctan(k*lambda_uv)/(k*lambda_uv)
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
    F_hcd = sp.exp(-kp*L0)

    bias_eff1 = bias1 + bias_hcd*F_hcd
    beta_eff1 = (bias1 * beta1 + bias_hcd*beta_hcd*F_hcd)/(bias1 + bias_hcd*F_hcd)

    bias_eff2 = bias2 + bias_hcd*F_hcd
    beta_eff2 = (bias2 * beta2 + bias_hcd*beta_hcd*F_hcd)/(bias2 + bias_hcd*F_hcd)

    pk = pk_lin*bias_eff1*bias_eff2*(1 + beta_eff1*muk**2)*(1 + beta_eff2*muk**2)

    return pk

def dnl_mcdonald(k, pk_lin, tracer1, tracer2, pk_fid, **kwargs):
    assert tracer1['name']=="LYA" and tracer2['name']=="LYA"
    kvel = 1.22*(1+k/0.923)**0.451
    dnl = sp.exp((k/6.4)**0.569-(k/15.3)**2.01-(k*muk/kvel)**1.5)
    return dnl

def dnl_arinyo(k, pk_lin, tracer1, tracer2, pk_fid, **kwargs):
    assert tracer1['name']=="LYA" and tracer2['name']=="LYA"
    q1 = kwargs["dnl_arinyo_q1"]
    kv = kwargs["dnl_arinyo_kv"]
    av = kwargs["dnl_arinyo_av"]
    bv = kwargs["dnl_arinyo_bv"]
    kp = kwargs["dnl_arinyo_kp"]

    growth = q1*k*k*k*pk_fid/(2*sp.pi*sp.pi)
    pecvelocity = sp.power(k/kv,av)*sp.power(sp.fabs(muk),bv)
    pressure = (k/kp)*(k/kp)
    dnl = sp.exp(growth*(1-pecvelocity)-pressure)
    return dnl

def cached_g2(function):
  memo = {}
  def wrapper(*args, **kwargs):

    dataset_name = kwargs['dataset_name']
    Lpar = kwargs["par binsize {}".format(dataset_name)]
    Lper = kwargs["per binsize {}".format(dataset_name)]

    if dataset_name in memo and sp.allclose(memo[dataset_name][0], [Lpar, Lper]):
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
    kt = k*sp.sqrt(1-muk**2)
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
    F_hcd = sp.exp(-kp*L0)

    if tracer1['name'] == "LYA":
        bias_eff1 = bias1 + bias_hcd*F_hcd
        beta_eff1 = (bias1 * beta1 + bias_hcd*beta_hcd*F_hcd)/(bias1 + bias_hcd*F_hcd)
        pk = pk_lin*bias_eff1*bias2*(1 + beta_eff1*muk**2)*(1 + beta2*muk**2)
    else:
        bias_eff2 = bias2 + bias_hcd*F_hcd
        beta_eff2 = (bias2 * beta2 + bias_hcd*beta_hcd*F_hcd)/(bias2 + bias_hcd*F_hcd)
        pk = pk_lin*bias1*bias_eff2*(1 + beta1*muk**2)*(1 + beta_eff2*muk**2)

    return pk

def pk_hcd_Voigt_cross_london(k, pk_lin, tracer1, tracer2, **kwargs):
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
    k_data = data_london_2[:,0]
    F_data = data_london_2[:,1]
    F_hcd = np.interp(L0*kp, k_data, F_data)

    if tracer1['name'] == "LYA":
        bias_eff1 = bias1 + bias_hcd*F_hcd
        beta_eff1 = (bias1 * beta1 + bias_hcd*beta_hcd*F_hcd)/(bias1 + bias_hcd*F_hcd)
        pk = pk_lin*bias_eff1*bias2*(1 + beta_eff1*muk**2)*(1 + beta2*muk**2)
    else:
        bias_eff2 = bias2 + bias_hcd*F_hcd
        beta_eff2 = (bias2 * beta2 + bias_hcd*beta_hcd*F_hcd)/(bias2 + bias_hcd*F_hcd)
        pk = pk_lin*bias1*bias_eff2*(1 + beta1*muk**2)*(1 + beta_eff2*muk**2)

    return pk

def pk_hcd_Voigt_cross_saclay(k, pk_lin, tracer1, tracer2, **kwargs):
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
    k_data = data_saclay_2[:,0]
    F_data = data_saclay_2[:,1]
    F_hcd = np.interp(L0*kp, k_data, F_data)

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

    W = sp.arctan(k*lambda_uv)/(k*lambda_uv)

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

    W = sp.arctan(k*lambda_uv)/(k*lambda_uv)

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

    W = sp.arctan(k*lambda_uv)/(k*lambda_uv)

    key = "bias_hcd_{}".format(kwargs['name'])
    if key in kwargs :
        bias_hcd = kwargs[key]
    else :
        bias_hcd = kwargs["bias_hcd"]
    beta_hcd = kwargs["beta_hcd"]
    L0 = kwargs["L0_hcd"]

    kp = k*muk
    F_hcd = sp.exp(-kp*L0)

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

def pk_velo_gaus(k, pk_lin, tracer1, tracer2, **kwargs):
    assert 'discrete' in [tracer1['type'],tracer2['type']]
    kp = k*muk
    smooth = sp.ones(kp.shape)
    if tracer1['type']=='discrete':
        smooth *= sp.exp( -0.25*(kp*kwargs['sigma_velo_gaus_'+tracer1['name']])**2)
    if tracer2['type']=='discrete':
        smooth *= sp.exp( -0.25*(kp*kwargs['sigma_velo_gaus_'+tracer2['name']])**2)
    return smooth

def pk_velo_lorentz(k, pk_lin, tracer1, tracer2, **kwargs):
    assert 'discrete' in [tracer1['type'],tracer2['type']]
    kp = k*muk
    smooth = sp.ones(kp.shape)
    if tracer1['type']=='discrete':
        smooth *= 1./sp.sqrt(1.+(kp*kwargs['sigma_velo_lorentz_'+tracer1['name']])**2)
    if tracer2['type']=='discrete':
        smooth *= 1./sp.sqrt(1.+(kp*kwargs['sigma_velo_lorentz_'+tracer2['name']])**2)
    return smooth

