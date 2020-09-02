import numpy as np
import fitsio
import sys

from picca.utils import userprint
from picca.fitter import utils
from picca.fitter.cosmo import model as cosmo_model


class model:

    def __init__(self,dic_init):
        self.met_prefix = dic_init['metal_prefix']
        self.met_names = dic_init['metals']
        nmet = len(self.met_names)
        self.nmet = nmet
        self.pname = []
        self.pinit = []
        self.fix = []
        self.hcds_mets = dic_init['hcds_mets']
        self.verbose = dic_init['verbose']

        self.templates = True

        for name in self.met_names:
            self.pname.append("bias_"+name)
            self.pname.append("beta_"+name)
            self.pname.append("alpha_"+name)

            self.pinit.append(dic_init["bias_"+name])
            self.pinit.append(dic_init["beta_"+name])
            self.pinit.append(dic_init["alpha_"+name])

        ### Load the power spectrum
        self.ell_max = dic_init['ell_max']
        h            = fitsio.FITS(dic_init['model'])
        self.zref    = h[1].read_header()['ZREF']
        self.k       = h[1]["K"][:]
        self.pk      = h[1]["PK"][:]
        self.pkSB    = h[1]["PKSB"][:]

        ### Redshift evolution of correlation
        self.evolution_growth_factor    = utils.evolution_growth_factor_by_hand
        self.evolution_Lya_bias         = utils.evolution_Lya_bias_0
        if dic_init['QSO_evolution'] is None:
            self.evolution_QSO_bias = utils.evolution_QSO_bias_none
        if dic_init['QSO_evolution']=='croom':
            self.evolution_QSO_bias = utils.evolution_QSO_bias_croom

    def add_auto(self):

        met_prefix=self.met_prefix
        nmet = self.nmet
        met_names = self.met_names

        if self.templates:
            to = np.loadtxt(met_prefix+"_Lya_"+met_names[0]+".0.dat")
            nd = len(to[:,0])

            self.temp_lya_met=np.zeros([nd,nmet,3])
            self.temp_met_met=np.zeros([nd,nmet,nmet,3])

            for i in range(nmet):
                for mp in range(3):
                    fmet=met_prefix+"_Lya_"+met_names[i]+"."+str(2*mp)+".dat"
                    userprint("reading "+fmet)
                    to=np.loadtxt(fmet)
                    self.temp_lya_met[to[:,0].astype(int),i,mp]=to[:,1]

                    for j in range(i,nmet):
                        fmet=met_prefix+"_"+met_names[i]+"_"+met_names[j]+"."+str(2*mp)+".dat"
                        userprint("reading "+fmet)
                        to=np.loadtxt(fmet)
                        self.temp_met_met[to[:,0].astype(int),i,j,mp]=to[:,1]
        else:
            h = fitsio.FITS(met_prefix)
            self.dmat={}
            self.auto_rt = {}
            self.auto_rp = {}
            self.auto_zeff = {}

            self.prev_pmet = {"beta_lya":0.,"alpha_lya":0}
            if self.hcds_mets:
                self.prev_pmet["beta_lls"]=0
                self.prev_pmet["bias_lls"]=0
                self.prev_pmet["L0_lls"]=1e-3
            self.prev_xi_lya_met = {}
            self.prev_xi_dla_met = {}
            self.prev_xi_met_met = {}
            for i,m1 in enumerate(met_names):
                sys.stdout.write("reading LYA {}\n".format(m1))
                self.dmat["LYA_"+m1] = h[2]["DM_LYA_"+m1][:]
                self.prev_pmet["beta_"+m1]=0.
                self.prev_pmet["alpha_"+m1]=0.
                self.prev_xi_lya_met["LYA_"+m1] = np.zeros(self.dmat["LYA_"+m1].shape[0])
                self.prev_xi_dla_met[m1] = np.zeros(self.dmat["LYA_"+m1].shape[0])

                self.auto_rt["LYA_"+m1] = h[2]["RT_LYA_"+m1][:]
                self.auto_rp["LYA_"+m1] = h[2]["RP_LYA_"+m1][:]
                self.auto_zeff["LYA_"+m1] = h[2]["Z_LYA_"+m1][:]

                for m2 in met_names[i:]:
                    sys.stdout.write("reading {} {}\n".format(m1,m2))
                    self.dmat[m1+"_"+m2] = h[2]["DM_"+m1+"_"+m2][:]
                    self.prev_xi_met_met[m1+"_"+m2] = np.zeros(self.dmat[m1+"_"+m2].shape[0])

                    self.auto_rt[m1+"_"+m2] = h[2]["RT_"+m1+"_"+m2][:]
                    self.auto_rp[m1+"_"+m2] = h[2]["RP_"+m1+"_"+m2][:]
                    self.auto_zeff[m1+"_"+m2] = h[2]["Z_"+m1+"_"+m2][:]

    def add_cross(self,dic_init):

        self.pname.append("qso_metal_boost")
        self.pinit.append(dic_init["qso_metal_boost"])

        self.different_drp = dic_init['different_drp']
        if (self.different_drp):
            if not self.grid:
                userprint("different drp and metal matrix not implemented")
                sys.exit(1)
            for name in self.met_names:
                self.pname.append("drp_"+name)
                self.pinit.append(dic_init["drp_"+name])

        met_prefix=self.met_prefix
        nmet = self.nmet
        met_names = self.met_names

        if self.grid:

            to = np.loadtxt(met_prefix + '_QSO_' + met_names[0] + '.grid')
            self.nd_cross = to[:,0].size

            ### Get the grid of the metals
            self.grid_qso_met=np.zeros([self.nd_cross,nmet,3])
            for i in range(nmet):
                fmet = met_prefix + '_QSO_' + met_names[i] + '.grid'
                userprint('  Reading cross correlation metal grid : ')
                userprint('  ', fmet)
                to = np.loadtxt(fmet)
                idx = to[:,0].astype(int)
                self.grid_qso_met[idx,i,0] = to[:,1]
                self.grid_qso_met[idx,i,1] = to[:,2]
                self.grid_qso_met[idx,i,2] = to[:,3]
            userprint()
        else:
            h = fitsio.FITS(self.met_prefix)
            self.abs_igm_cross = [i.strip() for i in h[1]["ABS_IGM"][:]]

            self.xdmat = {}
            self.xrp = {}
            self.xrt = {}
            self.xzeff = {}
            self.prev_pmet = {'growth_rate':0.,'drp':0.,'qso_evol':[0.,0.]}
            self.prev_xi_qso_met = {}
            for i in self.abs_igm_cross:
                self.xdmat[i] = h[2]["DM_"+i][:]
                self.xrp[i] = h[2]["RP_"+i][:]
                self.xrt[i] = h[2]["RT_"+i][:]
                self.xzeff[i] = h[2]["Z_"+i][:]

                self.prev_pmet['beta_'+i]=0.
                self.prev_pmet['alpha_'+i]=0.
                self.prev_xi_qso_met[i] = np.zeros(self.xdmat[i].shape[0])

    def valueAuto(self,pars):


        bias_lya=pars["bias_lya*(1+beta_lya)"]
        beta_lya=pars["beta_lya"]
        bias_lya /= 1+beta_lya


        if self.templates:
            bias_met=np.array([pars['bias_'+met] for met in self.met_names])
            beta_met=np.array([pars['beta_'+met] for met in self.met_names])
            amp=np.zeros([self.nmet,3])
            amp[:,0] = bias_met*(1 + (beta_lya+beta_met)/3 + beta_lya*beta_met/5)
            amp[:,1] = bias_met*(2*(beta_lya+beta_met)/3 + 4*beta_lya*beta_met/7)
            amp[:,2] = bias_met*8*beta_met*beta_lya/35

            amp*=bias_lya

            xi_lya_met=amp*self.temp_lya_met
            xi_lya_met=np.sum(xi_lya_met,axis=(1,2))

            amp=np.zeros([self.nmet,self.nmet,3])

            bias_met2 = bias_met*bias_met[None,:]

            amp[:,:,0] = bias_met2*(1+(beta_met+beta_met[None,:])/3+beta_met*beta_met[None,:]/5)
            amp[:,:,1] = bias_met2*(2*(beta_met+beta_met[None,:])/3+4*beta_met*beta_met[None,:]/7)
            amp[:,:,2] = bias_met2*8*beta_met*beta_met[None,:]/35

            xi_met_met=amp*self.temp_met_met
            xi_met_met=np.sum(xi_met_met,axis=(1,2,3))

        else:
            muk = cosmo_model.muk
            k = self.k
            kp = k*muk
            kt = k*np.sqrt(1-muk**2)
            nbins = self.dmat["LYA_"+self.met_names[0]].shape[0]

            if self.hcds_mets:
                bias_lls = pars["bias_lls"]
                beta_lls = pars["beta_lls"]
                L0_lls = pars["L0_lls"]
                Flls = np.sin(kp*L0_lls)/(kp*L0_lls)

            Lpar_auto = pars["Lpar_auto"]
            Lper_auto = pars["Lper_auto"]
            alpha_lya = pars["alpha_lya"]

            Gpar = np.sinc(kp*Lpar_auto/2/np.pi)**2
            Gper = np.sinc(kt*Lper_auto/2/np.pi)**2

            xi_lya_met = np.zeros(nbins)
            for met in self.met_names:
                bias_met = pars['bias_'+met]
                beta_met = pars['beta_'+met]
                alpha_met = pars["alpha_"+met]
                dm = self.dmat["LYA_"+met]
                recalc = beta_met != self.prev_pmet["beta_"+met]\
                        or beta_lya != self.prev_pmet["beta_lya"]\
                        or alpha_lya != self.prev_pmet["alpha_lya"]\
                        or alpha_met != self.prev_pmet["alpha_"+met]

                rt = self.auto_rt["LYA_"+met]
                rp = self.auto_rp["LYA_"+met]
                zeff  = self.auto_zeff["LYA_"+met]
                r = np.sqrt(rt**2+rp**2)
                w = (r==0)
                r[w] = 1e-6
                mur = rp/r

                if recalc:
                    if self.verbose:
                        userprint("recalculating ",met)
                    pk  = (1+beta_lya*muk**2)*(1+beta_met*muk**2)*self.pk
                    pk *= Gpar*Gper
                    xi = cosmo_model.Pk2Xi(r,mur,self.k,pk,ell_max=self.ell_max)
                    xi *= ((1+zeff)/(1+self.zref))**((alpha_lya-1)*(alpha_met-1))
                    self.prev_xi_lya_met["LYA_"+met] = self.dmat["LYA_"+met].dot(xi)

                if self.hcds_mets:
                    recalc = self.prev_pmet["beta_lls"] != beta_lls\
                        or self.prev_pmet["L0_lls"] != L0_lls
                    if recalc:
                        pk = (1+beta_lls*muk**2)*(1+beta_met*muk**2)*self.pk*Flls
                        pk *= Gpar*Gper
                        xi = cosmo_model.Pk2Xi(r,mur,self.k,pk,ell_max=self.ell_max)
                        xi *= ((1+zeff)/(1+self.zref))**((alpha_lya-1)*(alpha_met-1))
                        self.prev_xi_dla_met[met] = xi

                xi_lya_met += bias_lya*bias_met*self.prev_xi_lya_met["LYA_"+met]
                if self.hcds_mets:
                    xi_lya_met += bias_lls*bias_met*self.prev_xi_dla_met[met]

            xi_met_met = np.zeros(nbins)
            for i,met1 in enumerate(self.met_names):
                bias_met1 = pars['bias_'+met1]
                beta_met1 = pars['beta_'+met1]
                alpha_met1 = pars["alpha_"+met1]
                for met2 in self.met_names[i:]:
                    rt = self.auto_rt[met1+"_"+met2]
                    rp = self.auto_rp[met1+"_"+met2]
                    zeff  = self.auto_zeff[met1+"_"+met2]
                    bias_met2 = pars['bias_'+met2]
                    beta_met2 = pars['beta_'+met2]
                    alpha_met2 = pars["alpha_"+met2]
                    dm = self.dmat[met1+"_"+met2]
                    recalc = beta_met1 != self.prev_pmet["beta_"+met1]\
                            or beta_met2 != self.prev_pmet["beta_"+met2]

                    if recalc:
                        if self.verbose:
                            print("recalculating ",met1,met2)
                        r = np.sqrt(rt**2+rp**2)
                        w=r==0
                        r[w]=1e-6
                        mur = rp/r
                        pk  = (1+beta_met1*muk**2)*(1+beta_met2*muk**2)*self.pk
                        pk *= Gpar*Gper
                        xi = cosmo_model.Pk2Xi(r,mur,self.k,pk,ell_max=self.ell_max)
                        xi *= ((1+zeff)/(1+self.zref))**((alpha_met1-1)*(alpha_met2-1))
                        self.prev_xi_met_met[met1+"_"+met2] = self.dmat[met1+"_"+met2].dot(xi)

                    xi_met_met += bias_met1*bias_met2*self.prev_xi_met_met[met1+"_"+met2]
            for i in self.prev_pmet:
                self.prev_pmet[i] = pars[i]
        return xi_lya_met + xi_met_met

    def valueCross(self,pars):

        qso_boost         = pars["qso_metal_boost"]
        qso_evol          = [pars['qso_evol_0'],pars['qso_evol_1']]
        bias_qso          = pars["bias_qso"]
        growth_rate = pars["growth_rate"]
        beta_qso          = growth_rate/bias_qso
        bias_met = np.array([pars['bias_'+met] for met in self.met_names])
        beta_met = np.array([pars['beta_'+met] for met in self.met_names])
        Lpar = pars["Lpar_cross"]
        Lper = pars["Lper_cross"]

        ### Scales
        if (self.different_drp):
            drp_met = np.array([pars['drp_'+met]  for met in self.met_names])
            drp     = np.outer(np.ones(self.nd_cross),drp_met)
        else:
            drp = pars["drp"]

        if self.grid:

            ### Redshift evolution
            z     = self.grid_qso_met[:,:,2]
            evol  = np.power( self.evolution_growth_factor(z)/self.evolution_growth_factor(self.zref),2. )
            evol *= self.evolution_Lya_bias(z,[pars["alpha_lya"]])/self.evolution_Lya_bias(self.zref,[pars["alpha_lya"]])
            evol *= self.evolution_QSO_bias(z,qso_evol)/self.evolution_QSO_bias(self.zref,qso_evol)


            rp_shift = self.grid_qso_met[:,:,0]+drp
            rt       = self.grid_qso_met[:,:,1]
            r        = np.sqrt(rp_shift**2 + rt**2)
            mur      = rp_shift/r

        muk      = cosmo_model.muk
        kp       = self.k * muk
        kt       = self.k * np.sqrt(1.-muk**2)

        ### Correction to linear power-spectrum
        pk_corr = (1.+0.*muk)*self.pk
        pk_corr *= np.sinc(kp*Lpar/2./np.pi)**2
        pk_corr *= np.sinc(kt*Lper/2./np.pi)**2

        ### Biases
        b1b2 = qso_boost*bias_qso*bias_met

        if self.grid:
            xi_qso_met = np.zeros(self.grid_qso_met[:,0,0].size)
            for i in range(self.nmet):
                pk_full  = b1b2[i]*(1. + beta_met[i]*muk**2)*(1. + beta_qso*muk**2)*pk_corr
                xi_qso_met += cosmo_model.Pk2Xi(r[:,i],mur[:,i],self.k,pk_full,ell_max=self.ell_max)*evol[:,i]

        else:
            nbins = list(self.xdmat.values())[0].shape[0]
            xi_qso_met = np.zeros(nbins)
            for i in self.met_names:
                bias_met = pars["bias_"+i]
                beta_met = pars["beta_"+i]

                recalc = beta_met != self.prev_pmet['beta_'+i] or\
                    growth_rate != self.prev_pmet['growth_rate'] or\
                    not np.allclose(qso_evol,self.prev_pmet['qso_evol']) or\
                    self.prev_pmet['drp'] != drp
                if recalc:
                    if self.verbose:
                        userprint("recalculating metal {}".format(i))
                    self.prev_pmet['beta_'+i] = beta_met
                    self.prev_pmet['growth_rate'] = growth_rate
                    self.prev_pmet['qso_evol'] = qso_evol
                    self.prev_pmet['drp'] = drp

                    z = self.xzeff[i]
                    evol  = np.power( self.evolution_growth_factor(z)/self.evolution_growth_factor(self.zref),2. )
                    evol *= self.evolution_Lya_bias(z,[pars["alpha_"+i]])/self.evolution_Lya_bias(self.zref,[pars["alpha_"+i]])
                    evol *= self.evolution_QSO_bias(z,qso_evol)/self.evolution_QSO_bias(self.zref,qso_evol)

                    rp = self.xrp[i] + drp
                    rt = self.xrt[i]
                    r = np.sqrt(rp**2+rt**2)
                    w=r==0
                    r[w]=1e-6
                    mur = rp/r
                    pk_full  = (1. + beta_met*muk**2)*(1. + beta_qso*muk**2)*pk_corr
                    self.prev_xi_qso_met[i]  = cosmo_model.Pk2Xi(r,mur,self.k,pk_full,ell_max=self.ell_max)
                    self.prev_xi_qso_met[i] = self.xdmat[i].dot(self.prev_xi_qso_met[i]*evol)

                xi_qso_met += qso_boost*bias_qso*bias_met*self.prev_xi_qso_met[i]

        return xi_qso_met
