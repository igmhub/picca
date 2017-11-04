import numpy as np
import utils

muk = utils.muk
bias_beta = utils.bias_beta

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

    F_lls = np.sinc(kp*L0/np.pi)

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

    F_lls = np.sinc(kp*L0/np.pi)

    bias_eff1 = (bias1 + bias_lls*F_lls)
    bias_eff2 = (bias2 + bias_lls*F_lls)
    beta_eff1 = (bias1 * beta1 + bias_lls*beta_lls*F_lls)/(bias1 + bias_lls*F_lls)
    beta_eff2 = (bias2 * beta2 + bias_lls*beta_lls*F_lls)/(bias2 + bias_lls*F_lls)

    pk = pk_lin*bias_eff1*bias_eff2*(1 + beta_eff1*muk**2)*(1 + beta_eff2*muk**2)

    return pk
