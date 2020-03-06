
from picca.fitter.utils import L
import numpy as np
import scipy as sp
from scipy import linalg
import sys

class model:
    def __init__(self,data,imin,imax,istep,ellmin,ellmax,ellstep,distort,bb_rPerp_rParal):

        ### Constants
        self.r0 = 100.

        ### data atribut
        self.ico = data.ico
        self.rp  = data.rp
        self.rt  = data.rt
        self.dm  = data.dm
        self.cuts = data.cuts

        ### Power of 'r'
        self.imin  = imin
        self.imax  = imax
        self.istep = istep
        self.ni    = 1 + (self.imax-self.imin)//self.istep
        if (self.ni<=0):
            print('  fit/py/broadband_cross.py:: negative number of parameters.')
            sys.exit(0)

        ### Legendre Polynomial
        self.ellmin  = ellmin
        self.ellmax  = ellmax
        self.ellstep = ellstep
        self.nell    = 1 + (self.ellmax-self.ellmin)//self.ellstep
        if (self.nell<=0):
            print('  fit/py/broadband_cross.py:: negative number of parameters.')
            sys.exit(0)

        ###
        self.distort = distort
        self.bb_rPerp_rParal = bb_rPerp_rParal
        self.npar = self.nell*self.ni
        self.pars = None

        ### Get parameter names
        self.par_name = []
        for ipar in range(self.npar):
            i   = ipar%self.ni
            ell = (ipar-i)//self.ni
            i   = self.imin + i*self.istep
            ell = self.ellmin + ell*self.ellstep
            self.par_name += ['a_cross_'+str(i)+'_'+str(ell)]
        self.par_name = sp.array(self.par_name)

        return
    def value(self,data_rest,drp):

        rt       = self.rt
        rp_shift = self.rp+drp
        r        = sp.sqrt(rt**2 + rp_shift**2)
        mu       = rp_shift/r

        A = np.zeros([self.npar,len(r)])

        for ipar in range(self.npar):
            i   = ipar%self.ni
            ell = (ipar-i)//self.ni
            i   = self.imin + i*self.istep
            ell = self.ellmin + ell*self.ellstep
            if self.bb_rPerp_rParal:
                A[ipar,:] = (rt/self.r0)**i *(rp_shift/self.r0)**ell
            else :
                A[ipar,:] = (self.r0/r)**i*L(mu,ell)



        if self.distort:
            A = sp.dot(A,self.dm.T)
        A = A[:,self.cuts]

        M   = sp.dot(A,sp.dot(self.ico,A.T))
        IM  = linalg.inv(M)
        tmp = sp.dot(data_rest,sp.dot(self.ico,A.T))
        p   = sp.dot(IM,tmp)
        d   = sp.dot(p,A)

        return p,d
    def __call__(self,rt,rp,drp,pars):

        self.pars = pars
        rp_shift = rp+drp
        r        = sp.sqrt(rt**2 + rp_shift**2)
        mu       = rp_shift/r
        bb = np.zeros(len(r))

        for ipar in range(self.npar):
            i   = ipar%self.ni
            ell = (ipar-i)//self.ni
            i   = self.imin + i*self.istep
            ell = self.ellmin + ell*self.ellstep
            if self.bb_rPerp_rParal:
                bb += self.pars[ipar] *(rt/self.r0)**i *(rp_shift/self.r0)**ell
            else :
                bb += self.pars[ipar] *(self.r0/r)**i*L(mu,ell)

        return bb
