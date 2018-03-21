import scipy as sp
from . import utils

muk = utils.muk
bias_beta = utils.bias_beta

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
    st2 = kwargs["sigmaNL_per"]**2
    sp2 = st2*kwargs['1+f']**2
    return sp.exp(-(kp**2*sp2+kt**2*st2)/2)

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

    bias_eff1 = (bias1 + bias_hcd*F_hcd)
    beta_eff1 = (bias1 * beta1 + bias_hcd*beta_hcd*F_hcd)/(bias1 + bias_hcd*F_hcd)

    bias_eff2 = (bias2 + bias_hcd*F_hcd)
    beta_eff2 = (bias2 * beta2 + bias_hcd*beta_hcd*F_hcd)/(bias2 + bias_hcd*F_hcd)

    pk = pk_lin*bias_eff1*bias_eff2*(1 + beta_eff1*muk**2)*(1 + beta_eff2*muk**2)

    return pk

def pk_uv(k, pk_lin, tracer1, tracer2, **kwargs):

    bias1, beta1, bias2, beta2 = bias_beta(kwargs, tracer1, tracer2)

    bias_gamma = kwargs["bias_gamma"]
    bias_prim = kwargs["bias_prim"]
    lambda_uv = kwargs["lambda_uv"]

    W = sp.arctan(k*lambda_uv)/(k*lambda_uv)
    beta1 = beta1/(1 + bias_gamma*W/bias1/(1 + bias1*W))
    bias1 = bias1 + bias_gamma*W/(1+bias_prim*W)

    beta2 = beta2/(1 + bias_gamma*W/bias2/(1 + bias1*W))
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

    bias_eff1 = (bias1 + bias_hcd*F_hcd)
    beta_eff1 = (bias1 * beta1 + bias_hcd*beta_hcd*F_hcd)/(bias1 + bias_hcd*F_hcd)

    bias_eff2 = (bias2 + bias_hcd*F_hcd)
    beta_eff2 = (bias2 * beta2 + bias_hcd*beta_hcd*F_hcd)/(bias2 + bias_hcd*F_hcd)

    pk = pk_lin*bias_eff1*bias_eff2*(1 + beta_eff1*muk**2)*(1 + beta_eff2*muk**2)

    return pk

def dnl_mcdonald(k, pk_lin, tracer1, tracer2, **kwargs):
    assert tracer1=="LYA" and tracer2 == "LYA"
    kvel = 1.22*(1+k/0.923)**0.451
    dnl = sp.exp((k/6.4)**0.569-(k/15.3)**2.01-(k*muk/kvel)**1.5)
    return dnl

def dnl_arinyo(k, pk_lin, tracer1, tracer2, **kwargs):
    assert tracer1=="LYA" and tracer2 == "LYA"
    q1 = kwargs["dnl_arinyo_q1"]
    kv = kwargs["dnl_arinyo_kv"]
    av = kwargs["dnl_arinyo_av"]
    bv = kwargs["dnl_arinyo_bv"]
    kp = kwargs["dnl_arinyo_kp"]

    growth = q1*k*k*k*pk/(2*sp.pi*sp.pi)
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
    return utils.sinc(kp*Lpar/2)**2*utils.sinc(kt*Lper/2)**2

def pk_hcd_cross(k, pk_lin, tracer1, tracer2, **kwargs):
    bias1, beta1, bias2, beta2 = bias_beta(kwargs, tracer1, tracer2)
    assert (tracer1=="LYA" or tracer2=="LYA") and (tracer1!=tracer2)

    bias_hcd = kwargs["bias_hcd"]
    beta_hcd = kwargs["beta_hcd"]
    L0 = kwargs["L0_hcd"]

    kp = k*muk
    F_hcd = utils.sinc(kp*L0)

    if tracer1 == "LYA":
        bias_eff1 = (bias1 + bias_hcd*F_hcd)
        beta_eff1 = (bias1 * beta1 + bias_hcd*beta_hcd*F_hcd)/(bias1 + bias_hcd*F_hcd)
        pk = pk_lin*bias_eff1*bias2*(1 + beta_eff1*muk**2)*(1 + beta2*muk**2)
    else:
        bias_eff2 = (bias2 + bias_hcd*F_hcd)
        beta_eff2 = (bias2 * beta2 + bias_hcd*beta_hcd*F_hcd)/(bias2 + bias_hcd*F_hcd)
        pk = pk_lin*bias1*bias_eff2*(1 + beta1*muk**2)*(1 + beta_eff2*muk**2)

    return pk

def pk_velo_gaus(k, pk_lin, tracer1, tracer2, **kwargs):
    assert tracer1 == "QSO" or tracer2 == "QSO"
    kp = k*muk
    return sp.exp( -0.25*(kp*kwargs['sigma_velo_gauss'])**2)

def pk_velo_lorentz(k, pk_lin, tracer1, tracer2, **kwargs):
    assert tracer1 == "QSO" or tracer2 == "QSO"
    kp = k*muk
    return 1/sp.sqrt(1.+(kp*kwargs['sigma_velo_lorentz'])**2)
