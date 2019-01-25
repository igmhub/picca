
import scipy as sp
from picca import constants

class dla:
    def __init__(self,data,zabs,nhi):
        self.thid = data.thid
        self.zabs=zabs
        self.nhi=nhi

        self.t = self.p_voigt_a(10**data.ll,zabs,nhi)
        self.t*= self.p_voigt_b(10**data.ll,zabs,nhi)

    @staticmethod
    def p_voigt_a(la,zabs,nhi):
        return sp.exp(-dla.tau_a(la,zabs,nhi))

    @staticmethod
    def tau_a(la,zabs,nhi):
        lam_lya = constants.absorber_IGM["LYA"]
        gamma = 6.625e8
        f = 0.4164
        c = 3e8 ## speed of light m/s
        b = 30000.
        nn = 10**nhi
        lrf = la/(1+zabs)

        u = (c/b)*(lam_lya/lrf-1)
        a = lam_lya*1e-10*gamma/(4*sp.pi*b)
        h = dla.voigt(a,u)
        b/=1000.
        tau = 1.497e-15*nn*f*lrf*h/b
        return tau

    @staticmethod
    def p_voigt_b(la,zabs,nhi):
        return sp.exp(-dla.tau_b(la,zabs,nhi))

    @staticmethod
    def tau_b(la,zabs,nhi):
        lam_lyb = constants.absorber_IGM["LYB"]
        gamma = 0.079120
        f = 1.897e8
        c = 3e8 ## speed of light m/s
        b = 30000.
        nn = 10**nhi
        lrf = la/(1+zabs)

        u = (c/b)*(lam_lyb/lrf-1)
        a = lam_lyb*1e-10*gamma/(4*sp.pi*b)
        h = dla.voigt(a,u)
        b/=1000.
        tau = 1.497e-15*nn*f*lrf*h/b
        return tau

    @staticmethod
    def voigt(a,u):
        nmc = 1000
        y = sp.random.normal(size=nmc)*sp.sqrt(2)
        m = sp.mean(1/(a**2+(y[:,None]-u)**2),axis=0)
        return m*a/sp.sqrt(sp.pi)

