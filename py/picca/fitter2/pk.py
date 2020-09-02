import numpy as np
from . import utils
from pkg_resources import resource_filename

muk = utils.muk
bias_beta = utils.bias_beta
Fvoigt_data = []

class pk:
    def __init__(self, func, name_model=None):
        self.func = func
        global Fvoigt_data
        if (not name_model is None) and (Fvoigt_data == []):
            path = '{}/models/fvoigt_models/Fvoigt_{}.txt'.format(resource_filename('picca', 'fitter2'),name_model)
            Fvoigt_data = np.loadtxt(path)

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
    k_data = Fvoigt_data[:,0]
    F_data = Fvoigt_data[:,1]
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
