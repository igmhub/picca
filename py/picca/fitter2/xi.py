import numpy as np
from utils import Pk2Xi

def xi(r, mu, k, pk_lin, pk_func, tracer1, tracer2, pars, ell_max=None):
    pk_full = pk_func(k, pk_lin, tracer1, tracer2, **pars)
    
    ap = pars["ap"]
    at = pars["at"]
    rp = r*mu + pars["drp"]
    rt = r*np.sqrt(1-mu**2)
    arp = ap*rp
    art = at*rt
    ar = np.sqrt(arp**2+art**2)
    amu = arp/ar

    xi_full = Pk2Xi(ar, amu, k, pk_full, ell_max = ell_max)
    return xi_full
