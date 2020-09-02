import astropy.io.fits as pyfits
import numpy as np
import scipy as sp
import scipy.interpolate
import sys

from picca.utils import userprint
from picca.fitter import myGamma
from picca.fitter import utils
from . import fftlog

from picca.fitter.utils import L

class model:

    '''
    parameters for FFT
    '''
    nmuk = 1000
    muk=(np.arange(nmuk)+0.5)/nmuk
    dmuk = 1./nmuk
    muk=muk[:,None]

    def __init__(self,dic_init):
        self.ell_max=dic_init['ell_max']
        self.lls=dic_init['hcds']
        self.uv_fluct = dic_init['uv']
        h=pyfits.open(dic_init['model'])
        self.zref = h[1].header['ZREF']
        self.k = h[1].data.K
        self.pk = h[1].data.PK
        self.pkSB = h[1].data.PKSB


        if dic_init['fit_aiso']:
            self.pglob = ['bias_lya*(1+beta_lya)','beta_lya','aiso','1+epsilon','alpha_lya','SigmaNL_perp','1+f','bao_amp']
            self.fit_aiso=True
        else:
            self.pglob = ['bias_lya*(1+beta_lya)','beta_lya','ap','at','alpha_lya','SigmaNL_perp','1+f','bao_amp']
            self.fit_aiso=False

        self.pinit = [ dic_init[el] for el in self.pglob ]
        if self.lls:
            self.pglob.extend(['bias_lls','beta_lls','L0_lls'])
            self.pinit.extend([dic_init[el] for el in ['bias_lls','beta_lls','L0_lls'] ])

        if self.uv_fluct:
            self.pglob.extend(['bias_gamma','bias_prim','lambda_uv'])
            self.pinit.extend([dic_init[el] for el in ['bias_gamma','bias_prim','lambda_uv'] ])

        self.fix_bias_beta_peak = dic_init['fix_bias_beta_peak']
        if self.fix_bias_beta_peak:
            self.bias_lya_peak=dic_init['bias_lya_peak']
            self.beta_lya_peak=dic_init['beta_lya_peak']
            userprint("Fixing for BAO peak bias=",self.bias_lya_peak)
            userprint("Fixing for BAO peak beta=",self.beta_lya_peak)

        self.pall = self.pglob[:]

        self.fix=['alpha_lya','SigmaNL_perp','1+f','bao_amp']

        self.xi_auto_prev = None
        self.xi_cross_prev = None
        self.xi_autoQSO_prev = None

        self.evolution_growth_factor    = utils.evolution_growth_factor_by_hand
        self.evolution_Lya_bias         = utils.evolution_Lya_bias_0
        if dic_init['QSO_evolution'] is None:
            self.evolution_QSO_bias = utils.evolution_QSO_bias_none
        if dic_init['QSO_evolution']=='croom':
            self.evolution_QSO_bias = utils.evolution_QSO_bias_croom

        self.dnl_model = None
        self.q1_dnl = None
        self.kv_dnl = None
        self.av_dnl = None
        self.bv_dnl = None
        self.kp_dnl = None
        if dic_init['dnl_model'] == "mcdonald":
            userprint("with DNL (McDonald 2003)")
            self.dnl_model = "mcdonald"
        elif dic_init['dnl_model'] == "arinyo":
            userprint("with DNL (Arinyo et al. 2015)")
            self.dnl_model = "arinyo"
            z_dnl = [2.2000, 2.4000, 2.6000, 2.8000, 3.0000]
            q1_dnl = [0.8670, 0.8510, 0.7810, 0.7730, 0.7920]
            kv_dnl = [1.1200, 1.1122, 1.2570, 1.2765, 1.2928]
            av_dnl = [0.5140, 0.5480, 0.6110, 0.6080, 0.5780]
            bv_dnl = [1.6000, 1.6100, 1.6400, 1.6500, 1.6300]
            kp_dnl = [19.400, 19.500, 21.100, 19.200, 17.100]
            q1_dnl_interp = sp.interpolate.interp1d(z_dnl, q1_dnl, kind='linear', fill_value=(q1_dnl[0],q1_dnl[-1]), bounds_error=False)
            kv_dnl_interp = sp.interpolate.interp1d(z_dnl, kv_dnl, kind='linear', fill_value=(kv_dnl[0],kv_dnl[-1]), bounds_error=False)
            av_dnl_interp = sp.interpolate.interp1d(z_dnl, av_dnl, kind='linear', fill_value=(av_dnl[0],av_dnl[-1]), bounds_error=False)
            bv_dnl_interp = sp.interpolate.interp1d(z_dnl, bv_dnl, kind='linear', fill_value=(bv_dnl[0],bv_dnl[-1]), bounds_error=False)
            kp_dnl_interp = sp.interpolate.interp1d(z_dnl, kp_dnl, kind='linear', fill_value=(kp_dnl[0],kp_dnl[-1]), bounds_error=False)
            self.q1_dnl = q1_dnl_interp(self.zref)
            self.kv_dnl = kv_dnl_interp(self.zref)
            self.av_dnl = av_dnl_interp(self.zref)
            self.bv_dnl = bv_dnl_interp(self.zref)
            self.kp_dnl = kp_dnl_interp(self.zref)
            userprint("q1 =", self.q1_dnl)
            userprint("kv =", self.kv_dnl)
            userprint("av =", self.av_dnl)
            userprint("bv =", self.bv_dnl)
            userprint("kp =", self.kp_dnl)
        elif (not dic_init['dnl_model'] is None) & (not dic_init['dnl_model'] == "mcdonald") & (not dic_init['dnl_model'] == "arinyo"):
            userprint('  Unknown dnl model: ', dic_init['dnl_model'])
            userprint('  Exit')
            sys.exit(0)
        else :
            userprint("without DNL")

        self.twod = dic_init['2d']
        if self.twod :
            userprint("initalize pk2D array for 2D transfo ...")
            kmin=1.e-7
            kmax=100.
            nk  = 1024
            self.k1d  = np.exp(np.linspace(np.log(kmin),np.log(kmax),nk))
            # compute Pk
            self.kp=np.tile(self.k1d,(nk,1)).T
            self.kt=np.tile(self.k1d,(nk,1))
            kk=np.sqrt(self.kp**2+self.kt**2)
            self.muk=self.kp/kk
            self.pk_2d=fftlog.extrapolate_pk_logspace(kk.ravel(),self.k,self.pk).reshape(kk.shape)
            self.pkSB_2d=fftlog.extrapolate_pk_logspace(kk.ravel(),self.k,self.pkSB).reshape(kk.shape)
            self.k=kk
            userprint("done")

    def add_cross(self,dic_init):

        self.pcross = ['bias_qso','growth_rate','drp','Lpar_cross','Lper_cross','qso_evol_0','qso_evol_1']
        self.p0_cross = [ dic_init[el] for el in self.pcross ]
        self.fix.extend(['qso_evol_0','qso_evol_1'])

        ### Velocity dispersion
        self.velo_gauss = dic_init['velo_gauss']
        if (self.velo_gauss):
            self.pcross.extend(['sigma_velo_gauss'])
            self.p0_cross.extend([dic_init['sigma_velo_gauss']])
        self.velo_lorentz = dic_init['velo_lorentz']
        if (self.velo_lorentz):
            self.pcross.extend(['sigma_velo_lorentz'])
            self.p0_cross.extend([dic_init['sigma_velo_lorentz']])

        ### Radiation model
        self.fit_qso_rad_model = dic_init['fit_qso_radiation_model']
        if self.fit_qso_rad_model:
            listParam = ['qso_rad_strength','qso_rad_asymmetry','qso_rad_lifetime','qso_rad_decrease']
            self.pcross.extend(listParam)
            self.p0_cross.extend([dic_init[el] for el in listParam ])

        self.pall.extend(self.pcross)
        self.pinit.extend(self.p0_cross)
        self.pars_cross_prev=None

    def add_auto(self,dic_init):
        self.pauto = ['Lpar_auto','Lper_auto']
        self.p0_auto = [ dic_init[el] for el in self.pauto ]
        self.fix.extend(['Lpar_auto'])

        self.pall.extend(self.pauto)
        self.pinit.extend(self.p0_auto)
        self.pars_auto_prev=None

    def add_autoQSO(self,dic_init):

        self.pautoQSO = ['bias_qso','growth_rate','Lpar_autoQSO','Lper_autoQSO','qso_evol_0','qso_evol_1']
        self.p0_autoQSO = [ dic_init[el] for el in self.pautoQSO ]
        self.fix.extend(['qso_evol_0','qso_evol_1'])

        ### Velocity dispersion
        self.velo_gauss = dic_init['velo_gauss']
        if (self.velo_gauss):
            self.pautoQSO.extend(['sigma_velo_gauss'])
            self.p0_autoQSO.extend([dic_init['sigma_velo_gauss']])
        self.velo_lorentz = dic_init['velo_lorentz']
        if (self.velo_lorentz):
            self.pautoQSO.extend(['sigma_velo_lorentz'])
            self.p0_autoQSO.extend([dic_init['sigma_velo_lorentz']])

        tmp_pautoQSO   = [ i for i in self.pautoQSO if not any(i in el for el in self.pall) ]
        tmp_p0_autoQSO = [ dic_init[i] for i in tmp_pautoQSO ]

        self.pall.extend(tmp_pautoQSO)
        self.pinit.extend(tmp_p0_autoQSO)
        self.pars_autoQSO_prev = None

    @staticmethod
    def DNL(k,muk,pk,q1,kv,av,bv,kp,model):
        dnl = 1
        if model == "mcdonald":
            kvel = 1.22*(1+k/0.923)**0.451
            dnl = np.exp((k/6.4)**0.569-(k/15.3)**2.01-(k*muk/kvel)**1.5)
        elif model == "arinyo":
            growth = q1*k*k*k*pk/(2*np.pi*np.pi)
            pecvelocity = np.power(k/kv,av)*np.power(np.fabs(muk),bv)
            pressure = (k/kp)*(k/kp)
            dnl = np.exp(growth*(1-pecvelocity)-pressure)
        return dnl

    def valueAuto(self,rp,rt,z,pars):
        if self.xi_auto_prev is None or not np.allclose(list(pars.values()),self.pars_auto_prev):
            parsSB = pars.copy()
            if not self.fit_aiso:
                parsSB["at"]=1.
                parsSB["ap"]=1.
            else:
                parsSB["aiso"]=1.
                parsSB["1+epsilon"]=1.
            parsSB["SigmaNL_perp"]=0.
            if self.fix_bias_beta_peak :
                pars["bias_lya*(1+beta_lya)"]=self.bias_lya_peak*(1+self.beta_lya_peak)
                pars["beta_lya"]=self.beta_lya_peak

            if self.twod :
                xiSB = self.getXiAuto2D(rp,rt,z,self.pkSB_2d,parsSB)
                xi = self.getXiAuto2D(rp,rt,z,self.pk_2d-self.pkSB_2d,pars)
            else :
                xiSB = self.getXiAuto(rp,rt,z,self.pkSB,parsSB)
                xi = self.getXiAuto(rp,rt,z,self.pk-self.pkSB,pars)

            self.pars_auto_prev = list(pars.values())
            self.xi_auto_prev = xiSB + pars["bao_amp"]*xi

        return self.xi_auto_prev.copy()

    def getXiAuto(self,rp,rt,z,pk_lin,pars):
        k = self.k
        if not self.fit_aiso:
            ap=pars["ap"]
            at=pars["at"]
        else:
            ap=pars["aiso"]*pars["1+epsilon"]*pars["1+epsilon"]
            at=pars["aiso"]/pars["1+epsilon"]

        ar=np.sqrt(rt**2*at**2+rp**2*ap**2)
        mur=rp*ap/ar

        muk = model.muk
        kp = k * muk
        kt = k * np.sqrt(1-muk**2)

        bias_lya = pars["bias_lya*(1+beta_lya)"]/(1.+pars["beta_lya"])
        beta_lya = pars["beta_lya"]

        if self.uv_fluct:
            bias_gamma = pars["bias_gamma"]
            bias_prim = pars["bias_prim"]
            lambda_uv = pars["lambda_uv"]
            W = np.arctan(k*lambda_uv)/(k*lambda_uv)
            bias_lya_prim = bias_lya + bias_gamma*W/(1+bias_prim*W)
            beta_lya = bias_lya*beta_lya/bias_lya_prim
            bias_lya = bias_lya_prim

        if self.lls:
            bias_lls = pars["bias_lls"]
            beta_lls = pars["beta_lls"]
            L0_lls = pars["L0_lls"]
            F_lls = np.sinc(kp*L0_lls/np.pi)
            beta_lya = (bias_lya*beta_lya + bias_lls*beta_lls*F_lls)/(bias_lya+bias_lls*F_lls)
            bias_lya = bias_lya + bias_lls*F_lls

        pk_full = pk_lin * (1+beta_lya*muk**2)**2*bias_lya**2

        Lpar=pars["Lpar_auto"]
        Lper=pars["Lper_auto"]
        Gpar = np.sinc(kp*Lpar/2/np.pi)
        Gper = np.sinc(kt*Lper/2/np.pi)
        pk_full*=Gpar**2
        pk_full*=Gper**2

        sigmaNLper = pars["SigmaNL_perp"]
        sigmaNLpar = sigmaNLper*pars["1+f"]
        pk_nl = np.exp(-(kp*sigmaNLpar)**2/2-(kt*sigmaNLper)**2/2)
        pk_full *= pk_nl
        pk_full *= self.DNL(k,muk,self.pk,self.q1_dnl,self.kv_dnl,self.av_dnl,self.bv_dnl,self.kp_dnl,self.dnl_model)

        evol  = self.evolution_Lya_bias(z,[pars["alpha_lya"]])*self.evolution_growth_factor(z)
        evol /= self.evolution_Lya_bias(self.zref,[pars["alpha_lya"]])*self.evolution_growth_factor(self.zref)
        evol  = evol**2.

        return self.Pk2Xi(ar,mur,k,pk_full,ell_max=self.ell_max)*evol

    def getXiAuto2D(self,rp,rt,z,pk2d,pars):

        if not self.fit_aiso:
            ap=pars["ap"]
            at=pars["at"]
        else:
            ap=pars["aiso"]*pars["1+epsilon"]*pars["1+epsilon"]
            at=pars["aiso"]/pars["1+epsilon"]

        art=at*rt
        arp=ap*rp

        bias_lya = pars["bias_lya*(1+beta_lya)"]/(1.+pars["beta_lya"])
        beta_lya = pars["beta_lya"]

        if self.uv_fluct:
            bias_gamma = pars["bias_gamma"]
            bias_prim = pars["bias_prim"]
            lambda_uv = pars["lambda_uv"]
            W = np.arctan(self.k*lambda_uv)/(self.k*lambda_uv)
            bias_lya_prim = bias_lya + bias_gamma*W/(1+bias_prim*W)
            beta_lya = bias_lya*beta_lya/bias_lya_prim
            bias_lya = bias_lya_prim

        if self.lls:
            bias_lls = pars["bias_lls"]
            beta_lls = pars["beta_lls"]
            L0_lls = pars["L0_lls"]
            F_lls = np.sinc(self.kp*L0_lls/np.pi)
            beta_lya = (bias_lya*beta_lya + bias_lls*beta_lls*F_lls)/(bias_lya+bias_lls*F_lls)
            bias_lya = bias_lya + bias_lls*F_lls

        sigmaNLper = pars["SigmaNL_perp"]
        sigmaNLpar = sigmaNLper*pars["1+f"]

        pk_full = pk2d * np.exp(-(sigmaNLper**2*self.kt**2 + sigmaNLpar**2*self.kp**2)/2)
        pk_full =pk_full * (1+beta_lya*self.muk**2)**2*bias_lya**2

        Lpar=pars["Lpar_auto"]
        Lper=pars["Lper_auto"]
        pk_full *= np.sinc(self.kp*Lpar/2/np.pi)**2
        pk_full *= np.sinc(self.kt*Lper/2/np.pi)**2
        pk_full *= self.DNL(self.k,self.muk,self.pk_2d,self.q1_dnl,self.kv_dnl,self.av_dnl,self.bv_dnl,self.kp_dnl,self.dnl_model)

        evol  = self.evolution_Lya_bias(z,[pars["alpha_lya"]])*self.evolution_growth_factor(z)
        evol /= self.evolution_Lya_bias(self.zref,[pars["alpha_lya"]])*self.evolution_growth_factor(self.zref)
        evol  = evol**2.

        return fftlog.Pk2XiA(self.k1d,pk_full,arp,art)*evol

    def valueCross(self,rp,rt,z,pars):
        if self.xi_cross_prev is None or not np.allclose(list(pars.values()),self.pars_cross_prev):
            parsSB = pars.copy()
            if not self.fit_aiso:
                parsSB["at"]=1.
                parsSB["ap"]=1.
            else:
                parsSB['aiso']=1.
                parsSB['1+epsilon']=1.
            parsSB["SigmaNL_perp"]=0.
            parsSB["1+f"]=0.
            xiSB = self.getXiCross(rp,rt,z,self.pkSB,parsSB)

            if self.fix_bias_beta_peak:
                pars["bias_lya*(1+beta_lya)"] = self.bias_lya_peak*(1+self.beta_lya_peak)
                pars["beta_lya"]              = self.beta_lya_peak
            xi = self.getXiCross(rp,rt,z,self.pk-self.pkSB,pars)

            self.pars_cross_prev = list(pars.values())
            self.xi_cross_prev = xiSB + pars["bao_amp"]*xi

            if self.fit_qso_rad_model:
                self.xi_cross_prev += utils.qso_radiation_model(rp,rt,pars)

        return self.xi_cross_prev.copy()

    def getXiCross(self,rp,rt,z,pk_lin,pars):
        k = self.k
        if not self.fit_aiso:
            ap=pars["ap"]
            at=pars["at"]
        else:
            ap=pars["aiso"]*pars["1+epsilon"]*pars["1+epsilon"]
            at=pars["aiso"]/pars["1+epsilon"]

        drp=pars["drp"]
        Lpar=pars["Lpar_cross"]
        Lper=pars["Lper_cross"]
        qso_evol = [pars['qso_evol_0'],pars['qso_evol_1']]
        rp_shift=rp+drp
        ar=np.sqrt(rt**2*at**2+rp_shift**2*ap**2)
        mur=rp_shift*ap/ar

        muk = model.muk
        kp = k * muk
        kt = k * np.sqrt(1-muk**2)

        bias_lya = pars["bias_lya*(1+beta_lya)"]/(1.+pars["beta_lya"])
        beta_lya = pars["beta_lya"]

        ### UV fluctuation
        if self.uv_fluct:
            bias_gamma    = pars["bias_gamma"]
            bias_prim     = pars["bias_prim"]
            lambda_uv     = pars["lambda_uv"]
            W             = np.arctan(k*lambda_uv)/(k*lambda_uv)
            bias_lya_prim = bias_lya + bias_gamma*W/(1+bias_prim*W)
            beta_lya      = bias_lya*beta_lya/bias_lya_prim
            bias_lya      = bias_lya_prim

        ### LYA-QSO cross correlation
        bias_qso = pars["bias_qso"]
        beta_qso = pars["growth_rate"]/bias_qso
        pk_full  = bias_lya*bias_qso*(1+beta_lya*muk**2)*(1+beta_qso*muk**2)*pk_lin

        ### HCDS-QSO cross correlation
        if self.lls:
            bias_lls = pars["bias_lls"]
            beta_lls = pars["beta_lls"]
            L0_lls = pars["L0_lls"]
            F_lls = np.sinc(kp*L0_lls/np.pi)
            pk_full+=bias_lls*F_lls*bias_qso*(1+beta_lls*muk**2)*(1+beta_qso*muk**2)*pk_lin

        ### Velocity dispersion
        if (self.velo_gauss):
            pk_full *= np.exp( -0.25*(kp*pars['sigma_velo_gauss'])**2 )
        if (self.velo_lorentz):
            pk_full /= np.sqrt(1.+(kp*pars['sigma_velo_lorentz'])**2)

        ### Peak broadening
        sigmaNLper = pars["SigmaNL_perp"]
        sigmaNLpar = sigmaNLper*pars["1+f"]
        pk_full   *= np.exp( -0.5*( (sigmaNLper*kt)**2 + (sigmaNLpar*kp)**2 ) )

        ### Pixel size
        pk_full *= np.sinc(kp*Lpar/2./np.pi)**2
        pk_full *= np.sinc(kt*Lper/2./np.pi)**2

        ### Non-linear correction
        pk_full *= np.sqrt(self.DNL(self.k,self.muk,self.pk,self.q1_dnl,self.kv_dnl,self.av_dnl,self.bv_dnl,self.kp_dnl,self.dnl_model))

        ### Redshift evolution
        evol  = np.power( self.evolution_growth_factor(z)/self.evolution_growth_factor(self.zref),2. )
        evol *= self.evolution_Lya_bias(z,[pars["alpha_lya"]])/self.evolution_Lya_bias(self.zref,[pars["alpha_lya"]])
        evol *= self.evolution_QSO_bias(z,qso_evol)/self.evolution_QSO_bias(self.zref,qso_evol)

        return self.Pk2Xi(ar,mur,k,pk_full,ell_max=self.ell_max)*evol

    def valueAutoQSO(self,rp,rt,z,pars):
        if self.xi_autoQSO_prev is None or not np.allclose(list(pars.values()),self.pars_autoQSO_prev):
            parsSB = pars.copy()
            if not self.fit_aiso:
                parsSB["at"]=1.
                parsSB["ap"]=1.
            else:
                parsSB['aiso']=1.
                parsSB['1+epsilon']=1.
            parsSB["SigmaNL_perp"]=0.
            parsSB["1+f"]=0.
            xiSB = self.getXiAutoQSO(rp,rt,z,self.pkSB,parsSB)

            xi = self.getXiAutoQSO(rp,rt,z,self.pk-self.pkSB,pars)
            self.pars_autoQSO_prev = list(pars.values())
            self.xi_autoQSO_prev = xiSB + pars["bao_amp"]*xi

        return self.xi_autoQSO_prev.copy()

    def getXiAutoQSO(self,rp,rt,z,pk_lin,pars):
        k = self.k

        if not self.fit_aiso:
            ap = pars["ap"]
            at = pars["at"]
        else:
            ap = pars["aiso"]*pars["1+epsilon"]*pars["1+epsilon"]
            at = pars["aiso"]/pars["1+epsilon"]
        ar  = np.sqrt(rt**2*at**2+rp**2*ap**2)
        mur = rp*ap/ar

        muk = model.muk
        kp = k * muk
        kt = k * np.sqrt(1-muk**2)

        ### QSO-QSO auto correlation
        bias_qso = pars["bias_qso"]
        beta_qso = pars["growth_rate"]/bias_qso
        pk_full  = pk_lin*(bias_qso*(1.+beta_qso*muk**2))**2

        ### Velocity dispersion
        if (self.velo_gauss):
            pk_full *= np.exp( -0.5*(kp*pars['sigma_velo_gauss'])**2 )
        if (self.velo_lorentz):
            pk_full /= 1.+(kp*pars['sigma_velo_lorentz'])**2

        ### Peak broadening
        sigmaNLper = pars["SigmaNL_perp"]
        sigmaNLpar = sigmaNLper*pars["1+f"]
        pk_full   *= np.exp( -0.5*( (sigmaNLper*kt)**2 + (sigmaNLpar*kp)**2 ) )

        ### Pixel size
        Lpar     = pars["Lpar_autoQSO"]
        Lper     = pars["Lper_autoQSO"]
        pk_full *= np.sinc(kp*Lpar/2./np.pi)**2
        pk_full *= np.sinc(kt*Lper/2./np.pi)**2

        ### Redshift evolution
        qso_evol = [pars['qso_evol_0'],pars['qso_evol_1']]
        evol  = np.power( self.evolution_growth_factor(z)/self.evolution_growth_factor(self.zref),2. )
        evol *= np.power( self.evolution_QSO_bias(z,qso_evol)/self.evolution_QSO_bias(self.zref,qso_evol),2. )

        return self.Pk2Xi(ar,mur,k,pk_full,ell_max=self.ell_max)*evol


    @staticmethod
    def Pk2Mp(ar,k,pk,ell_max=None):
        """
        Implementation of FFTLog from A.J.S. Hamilton (2000)
        assumes log(k) are equally spaced
        """

        muk = model.muk
        dmuk = model.dmuk

        k0 = k[0]
        l=np.log(k.max()/k0)
        r0=1.

        N=len(k)
        emm=N*np.fft.fftfreq(N)
        r=r0*np.exp(-emm*l/N)
        dr=abs(np.log(r[1]/r[0]))
        s=np.argsort(r)
        r=r[s]

        xi=np.zeros([ell_max//2+1,len(ar)])

        for ell in range(0,ell_max+1,2):
            pk_ell=np.sum(dmuk*L(muk,ell)*pk,axis=0)*(2*ell+1)*(-1)**(ell//2)
            mu=ell+0.5
            n=2.
            q=2-n-0.5
            x=q+2*np.pi*1j*emm/l
            lg1=myGamma.LogGammaLanczos((mu+1+x)/2)
            lg2=myGamma.LogGammaLanczos((mu+1-x)/2)

            um=(k0*r0)**(-2*np.pi*1j*emm/l)*2**x*np.exp(lg1-lg2)
            um[0]=np.real(um[0])
            an=np.fft.fft(pk_ell*k**n/2/np.pi**2*np.sqrt(np.pi/2))
            an*=um
            xi_loc=np.fft.ifft(an)
            xi_loc=xi_loc[s]
            xi_loc/=r**(3-n)
            xi_loc[-1]=0
            spline=np.interpolate.splrep(np.log(r)-dr/2,np.real(xi_loc),k=3,s=0)
            xi[ell//2,:]=np.interpolate.splev(np.log(ar),spline)

        return xi

    @staticmethod
    def Pk2Xi(ar,mur,k,pk,ell_max=None):
        xi=model.Pk2Mp(ar,k,pk,ell_max)
        for ell in range(0,ell_max+1,2):
            xi[ell//2,:]*=L(mur,ell)
        return np.sum(xi,axis=0)
