import numpy as np
import utils

muk = utils.muk
bias_beta = utils.bias_beta

class pk:
    def __init__(self, func):
        self.func = func

    def __call__(self, k, pk_lin, tracer1, tracer2, **kwargs):
        return self.func(k, pk_lin, tracer1, tracer2, **kwargs)

    def __mul__(self,func2):
        return lambda k, pk_lin, tracer1, tracer2, **kwargs: self(k, pk_lin, tracer1, tracer2, **kwargs)*func2(k, pk_lin, tracer1, tracer2, **kwargs)

    __imul__ = __mul__
    __rmul__ = __mul__


def cached(function):
  memo = {}
  def wrapper(*args, **kwargs):
    if 1 in memo:
      return memo[1]
    else:
      rv = function(*args, **kwargs)
      memo[1] = rv
      return rv
  return wrapper

def pk_kaiser(k, pk_lin, tracer1, tracer2, **kwargs):
    bias1, beta1, bias2, beta2 = bias_beta(kwargs, tracer1, tracer2)
    pk = bias1*bias2*pk_lin*(1+beta1*muk**2)*(1+beta2*muk**2)

    return pk

def pk_lls(k, pk_lin, tracer1, tracer2, **kwargs):
    
    bias1, beta1, bias2, beta2 = bias_beta(kwargs, tracer1, tracer2)

    bias_lls = kwargs["bias_lls"]
    beta_lls = kwargs["beta_lls"]
    L0 = kwargs["L0_lls"]

    kp = k*muk
    kt = k*(1-muk**2)

    F_lls = utils.sinc(kp*L0)

    bias_eff1 = (bias1 + bias_lls*F_lls)
    bias_eff2 = (bias2 + bias_lls*F_lls)
    beta_eff1 = (bias1 * beta1 + bias_lls*beta_lls*F_lls)/(bias1 + bias_lls*F_lls)
    beta_eff2 = (bias2 * beta2 + bias_lls*beta_lls*F_lls)/(bias2 + bias_lls*F_lls)

    pk = pk_lin*bias_eff1*bias_eff2*(1 + beta_eff1*muk**2)*(1 + beta_eff2*muk**2)

    return pk

def pk_uv(k, pk_lin, tracer1, tracer2, **kwargs):

    bias1, beta1, bias2, beta2 = bias_beta(kwargs, tracer1, tracer2)

    bias_gamma = kwargs["bias_gamma"]
    bias_prim = kwargs["bias_prim"]
    lambda_uv = kwargs["lambda_uv"]

    W = np.arctan(k*lambda_uv)/(k*lambda_uv)
    beta1 = beta1/(1 + bias_gamma*W/bias1/(1 + bias1*W))
    bias1 = bias1 + bias_gamma*W/(1+bias_prim*W)

    beta2 = beta2/(1 + bias_gamma*W/bias2/(1 + bias1*W))
    bias2 = bias2 + bias_gamma*W/(1+bias_prim*W)

    return pk_lin*bias1*bias2*(1+beta1*muk**2)*(1+beta2*muk**2)

def pk_lls_uv(k, pk_lin, tracer1, tracer2, **kwargs):
    bias1, beta1, bias2, beta2 = bias_beta(kwargs, tracer1, tracer2)

    bias_gamma = kwargs["bias_gamma"]
    bias_prim = kwargs["bias_prim"]
    lambda_uv = kwargs["lambda_uv"]

    W = np.arctan(k*lambda_uv)/(k*lambda_uv)
    beta1 = beta1/(1 + bias_gamma*W/bias1/(1 + bias1*W))
    bias1 = bias1 + bias_gamma*W/(1+bias_prim*W)

    beta2 = beta2/(1 + bias_gamma*W/bias2/(1 + bias1*W))
    bias2 = bias2 + bias_gamma*W/(1+bias_prim*W)

    bias_lls = kwargs["bias_lls"]
    beta_lls = kwargs["beta_lls"]
    L0 = kwargs["L0_lls"]

    kp = k*muk
    kt = k*(1-muk**2)

    F_lls = utils.sinc(kp*L0)

    bias_eff1 = (bias1 + bias_lls*F_lls)
    bias_eff2 = (bias2 + bias_lls*F_lls)
    beta_eff1 = (bias1 * beta1 + bias_lls*beta_lls*F_lls)/(bias1 + bias_lls*F_lls)
    beta_eff2 = (bias2 * beta2 + bias_lls*beta_lls*F_lls)/(bias2 + bias_lls*F_lls)

    pk = pk_lin*bias_eff1*bias_eff2*(1 + beta_eff1*muk**2)*(1 + beta_eff2*muk**2)

    return pk

def dnl_mcdonald(k, pk_lin, tracer1, tracer2, **kwargs):
    assert tracer1=="LYA" and tracer2 == "LYA"
    kvel = 1.22*(1+k/0.923)**0.451
    dnl = np.exp((k/6.4)**0.569-(k/15.3)**2.01-(k*muk/kvel)**1.5)
    return dnl

def dnl_arinyo(k, pk_lin, tracer1, tracer2, **kwargs):
    assert tracer1=="LYA" and tracer2 == "LYA"
    q1 = kwargs["dnl_arinyo_q1"]
    kv = kwargs["dnl_arinyo_kv"]
    av = kwargs["dnl_arinyo_av"]
    bv = kwargs["dnl_arinyo_bv"]
    kp = kwargs["dnl_arinyo_kp"]

    growth = q1*k*k*k*pk/(2*np.pi*np.pi)
    pecvelocity = np.power(k/kv,av)*np.power(np.fabs(muk),bv)
    pressure = (k/kp)*(k/kp)
    dnl = np.exp(growth*(1-pecvelocity)-pressure)
    return dnl

@cached
def G2(k, pk_lin, tracer1, tracer2, dataset_name = None, **kwargs):
    Lpar = kwargs["par binsize {}".format(dataset_name)]
    Lper = kwargs["per binsize {}".format(dataset_name)]

    kp = k*muk
    kt = k*np.sqrt(1-muk**2)
    return utils.sinc(kp*Lpar/2)**2*utils.sinc(kt*Lper/2)**2
