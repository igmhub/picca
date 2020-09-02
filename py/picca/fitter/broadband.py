
from picca.fitter.utils import L
import numpy as np
from scipy import linalg

class model:
    def __init__(self,data,imin,imax,istep,ellmin,ellmax,ellstep,distort):

        ico = data.ico
        rt = data.rt
        rp = data.rp

        self.imin = imin
        self.imax = imax
        self.istep= istep
        ni = (imax-imin)//istep+1
        self.ni = ni

        self.ellmin = ellmin
        self.ellmax = ellmax
        self.ellstep = ellstep
        nell = (ellmax-ellmin)//ellstep+1
        self.nell = nell

        npar = ni*nell
        self.npar = npar
        self.pars = None

        r0 = 100.
        self.r0 = r0

        r = np.sqrt(rt**2+rp**2)
        mu = rp/r

        A = np.zeros([npar,len(r)])

        for ipar in range(npar):
            i = ipar%ni
            ell = (ipar-i)//ni

            i = imin + i * istep
            ell = ellmin+ell * ellstep

            A[ipar,:]=(r0/r)**i*L(mu,ell)

            if distort:
                A[ipar,:]=np.dot(data.dm,A[ipar,:])

        A = A[:,data.cuts]

        self.A = A
        M = np.dot(A,np.dot(ico,A.T))
        IM =linalg.inv(M)
        self.IM = IM
        self.IMA = np.dot(IM,A)

        self.ico = ico

        ### Get parameter names
        self.par_name = []
        for ipar in range(self.npar):
            i   = ipar%self.ni
            ell = (ipar-i)//self.ni
            i   = self.imin + i*self.istep
            ell = self.ellmin + ell*self.ellstep
            self.par_name += ['a_auto_'+str(i)+'_'+str(ell)]
        self.par_name = np.array(self.par_name)

    def value(self,data):
        tmp = np.dot(data,np.dot(self.ico,self.A.T))
        p = np.dot(self.IM,tmp)
        d = np.dot(p,self.A)
        return p,d

    def __call__(self,rt,rp,pars):
        r = np.sqrt(rt**2+rp**2)
        mu = rp/r
        bb = np.zeros(len(r))


        for ipar in range(len(pars)):
            i = ipar%self.ni
            ell = (ipar-i)//self.ni
            i = self.imin + i * self.istep
            ell = self.ellmin + ell * self.ellstep

            bb += pars[ipar]*(self.r0/r)**i*L(mu,ell)

            self.pars = pars

        return bb
