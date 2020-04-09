
import numpy as np
import scipy as sp
from picca import constants
from picca.utils import userprint, unred
import iminuit
from .dla import dla
import fitsio

def variance(var,eta,var_lss,fudge):
    return eta*var + var_lss + fudge/var


class Qso:
    def __init__(self,thingid,ra,dec,z_qso,plate,mjd,fiberid):
        self.ra = ra
        self.dec = dec

        self.plate=plate
        self.mjd=mjd
        self.fiberid=fiberid

        ## cartesian coordinates
        self.x_cart = sp.cos(ra)*sp.cos(dec)
        self.y_cart = sp.sin(ra)*sp.cos(dec)
        self.z_cart = sp.sin(dec)
        self.cos_dec = sp.cos(dec)

        self.z_qso = z_qso
        self.thingid = thingid

    def __xor__(self,data):
        try:
            x = sp.array([d.x_cart for d in data])
            y = sp.array([d.y_cart for d in data])
            z = sp.array([d.z_cart for d in data])
            ra = sp.array([d.ra for d in data])
            dec = sp.array([d.dec for d in data])

            cos = x*self.x_cart+y*self.y_cart+z*self.z_cart
            w = cos>=1.
            if w.sum()!=0:
                userprint('WARNING: {} pairs have cos>=1.'.format(w.sum()))
                cos[w] = 1.
            w = cos<=-1.
            if w.sum()!=0:
                userprint('WARNING: {} pairs have cos<=-1.'.format(w.sum()))
                cos[w] = -1.
            angl = sp.arccos(cos)

            w = (np.absolute(ra-self.ra)<constants.small_angle_cut_off) & (np.absolute(dec-self.dec)<constants.small_angle_cut_off)
            if w.sum()!=0:
                angl[w] = sp.sqrt( (dec[w]-self.dec)**2 + (self.cos_dec*(ra[w]-self.ra))**2 )
        except:
            x = data.x_cart
            y = data.y_cart
            z = data.z_cart
            ra = data.ra
            dec = data.dec

            cos = x*self.x_cart+y*self.y_cart+z*self.z_cart
            if cos>=1.:
                userprint('WARNING: 1 pair has cosinus>=1.')
                cos = 1.
            elif cos<=-1.:
                userprint('WARNING: 1 pair has cosinus<=-1.')
                cos = -1.
            angl = sp.arccos(cos)
            if (np.absolute(ra-self.ra)<constants.small_angle_cut_off) & (np.absolute(dec-self.dec)<constants.small_angle_cut_off):
                angl = sp.sqrt( (dec-self.dec)**2 + (self.cos_dec*(ra-self.ra))**2 )
        return angl

class forest(Qso):

    lmin = None
    lmax = None
    lmin_rest = None
    lmax_rest = None
    rebin = None
    dll = None

    ### Correction function for multiplicative errors in pipeline flux calibration
    correc_flux = None
    ### Correction function for multiplicative errors in inverse pipeline variance calibration
    correc_ivar = None

    ### map of g-band extinction to thingids for dust correction
    ebv_map = None

    ## absorber pixel mask limit
    absorber_mask = None

    ## minumum dla transmission
    dla_mask = None

    var_lss = None
    eta = None
    mean_cont = None

    ## quality variables
    mean_SNR = None
    mean_reso = None
    mean_z = None


    def __init__(self,log_lambda,fl,iv,thingid,ra,dec,z_qso,plate,mjd,fiberid,order, diff=None,reso=None, mmef = None):
        Qso.__init__(self,thingid,ra,dec,z_qso,plate,mjd,fiberid)

        if not self.ebv_map is None:
            corr = unred(10**log_lambda,self.ebv_map[thingid])
            fl /= corr
            iv *= corr**2
            if not diff is None:
                diff /= corr

        ## cut to specified range
        bins = sp.floor((log_lambda-forest.lmin)/forest.dlog_lambda+0.5).astype(int)
        log_lambda = forest.lmin + bins*forest.dlog_lambda
        w = (log_lambda>=forest.lmin)
        w = w & (log_lambda<forest.lmax)
        w = w & (log_lambda-sp.log10(1.+self.z_qso)>forest.lmin_rest)
        w = w & (log_lambda-sp.log10(1.+self.z_qso)<forest.lmax_rest)
        w = w & (iv>0.)
        if w.sum()==0:
            return
        bins = bins[w]
        log_lambda = log_lambda[w]
        fl = fl[w]
        iv = iv[w]
        ## mmef is the mean expected flux fraction using the mock continuum
        if mmef is not None:
            mmef = mmef[w]
        if diff is not None:
            diff=diff[w]
        if reso is not None:
            reso=reso[w]

        ## rebin
        rebin_log_lambda = forest.lmin + np.arange(bins.max()+1)*forest.dlog_lambda
        cfl = np.zeros(bins.max()+1)
        civ = np.zeros(bins.max()+1)
        if mmef is not None:
            cmmef = np.zeros(bins.max()+1)
        ccfl = sp.bincount(bins,weights=iv*fl)
        cciv = sp.bincount(bins,weights=iv)
        if mmef is not None:
            ccmmef = sp.bincount(bins, weights=iv*mmef)
        if diff is not None:
            cdiff = sp.bincount(bins,weights=iv*diff)
        if reso is not None:
            creso = sp.bincount(bins,weights=iv*reso)

        cfl[:len(ccfl)] += ccfl
        civ[:len(cciv)] += cciv
        if mmef is not None:
            cmmef[:len(ccmmef)] += ccmmef
        w = (civ>0.)
        if w.sum()==0:
            return
        log_lambda = rebin_log_lambda[w]
        fl = cfl[w]/civ[w]
        iv = civ[w]
        if mmef is not None:
            mmef = cmmef[w]/civ[w]
        if diff is not None:
            diff = cdiff[w]/civ[w]
        if reso is not None:
            reso = creso[w]/civ[w]

        ## Flux calibration correction
        if not self.correc_flux is None:
            correction = self.correc_flux(log_lambda)
            fl /= correction
            iv *= correction**2
        if not self.correc_ivar is None:
            correction = self.correc_ivar(log_lambda)
            iv /= correction

        self.Fbar = None
        self.T_dla = None
        self.log_lambda = log_lambda
        self.fl = fl
        self.iv = iv
        self.mmef = mmef
        self.order = order
        #if diff is not None :
        self.diff = diff
        self.reso = reso
        #else :
        #   self.diff = np.zeros(len(log_lambda))
        #   self.reso = sp.ones(len(log_lambda))

        # compute means
        if reso is not None : self.mean_reso = sum(reso)/float(len(reso))

        err = 1.0/sp.sqrt(iv)
        SNR = fl/err
        self.mean_SNR = sum(SNR)/float(len(SNR))
        lam_lya = constants.absorber_IGM["LYA"]
        self.mean_z = (sp.power(10.,log_lambda[len(log_lambda)-1])+sp.power(10.,log_lambda[0]))/2./lam_lya -1.0


    def __add__(self,d):

        if not hasattr(self,'log_lambda') or not hasattr(d,'log_lambda'):
            return self

        dic = {}  # this should contain all quantities that are to be coadded with ivar weighting

        log_lambda = sp.append(self.log_lambda,d.log_lambda)
        dic['fl'] = sp.append(self.fl, d.fl)
        iv = sp.append(self.iv,d.iv)

        if self.mmef is not None:
            dic['mmef'] = sp.append(self.mmef, d.mmef)
        if self.diff is not None:
            dic['diff'] = sp.append(self.diff, d.diff)
        if self.reso is not None:
            dic['reso'] = sp.append(self.reso, d.reso)

        bins = sp.floor((log_lambda-forest.lmin)/forest.dlog_lambda+0.5).astype(int)
        rebin_log_lambda = forest.lmin + np.arange(bins.max()+1)*forest.dlog_lambda
        civ = np.zeros(bins.max()+1)
        cciv = sp.bincount(bins,weights=iv)
        civ[:len(cciv)] += cciv
        w = (civ>0.)
        self.log_lambda = rebin_log_lambda[w]
        self.iv = civ[w]

        for k, v in dic.items():
            cnew = np.zeros(bins.max() + 1)
            ccnew = sp.bincount(bins, weights=iv * v)
            cnew[:len(ccnew)] += ccnew
            setattr(self, k, cnew[w] / civ[w])

        # recompute means of quality variables
        if self.reso is not None:
            self.mean_reso = self.reso.mean()
        err = 1./sp.sqrt(self.iv)
        SNR = self.fl/err
        self.mean_SNR = SNR.mean()
        lam_lya = constants.absorber_IGM["LYA"]
        self.mean_z = (sp.power(10.,log_lambda[len(log_lambda)-1])+sp.power(10.,log_lambda[0]))/2./lam_lya -1.0

        return self

    def mask(self,mask_obs,mask_RF):
        if not hasattr(self,'log_lambda'):
            return

        w = sp.ones(self.log_lambda.size,dtype=bool)
        for l in mask_obs:
            w &= (self.log_lambda<l[0]) | (self.log_lambda>l[1])
        for l in mask_RF:
            w &= (self.log_lambda-sp.log10(1.+self.z_qso)<l[0]) | (self.log_lambda-sp.log10(1.+self.z_qso)>l[1])

        ps = ['iv','log_lambda','fl','T_dla','Fbar','mmef','diff','reso']
        for p in ps:
            if hasattr(self,p) and (getattr(self,p) is not None):
                setattr(self,p,getattr(self,p)[w])

        return

    def add_optical_depth(self,tau,gamma,waveRF):
        """Add mean optical depth
        """
        if not hasattr(self,'log_lambda'):
            return

        if self.Fbar is None:
            self.Fbar = sp.ones(self.log_lambda.size)

        w = 10.**self.log_lambda/(1.+self.z_qso)<=waveRF
        z = 10.**self.log_lambda/waveRF-1.
        self.Fbar[w] *= sp.exp(-tau*(1.+z[w])**gamma)

        return

    def add_dla(self,zabs,nhi,mask=None):
        if not hasattr(self,'log_lambda'):
            return
        if self.T_dla is None:
            self.T_dla = sp.ones(len(self.log_lambda))

        self.T_dla *= dla(self,zabs,nhi).t

        w = self.T_dla>forest.dla_mask
        if not mask is None:
            for l in mask:
                w &= (self.log_lambda-sp.log10(1.+zabs)<l[0]) | (self.log_lambda-sp.log10(1.+zabs)>l[1])

        ps = ['iv','log_lambda','fl','T_dla','Fbar','mmef','diff','reso']
        for p in ps:
            if hasattr(self,p) and (getattr(self,p) is not None):
                setattr(self,p,getattr(self,p)[w])

        return

    def add_absorber(self,lambda_absorber):
        if not hasattr(self,'log_lambda'):
            return

        w = sp.ones(self.log_lambda.size, dtype=bool)
        w &= sp.fabs(1.e4*(self.log_lambda-sp.log10(lambda_absorber)))>forest.absorber_mask

        ps = ['iv','log_lambda','fl','T_dla','Fbar','mmef','diff','reso']
        for p in ps:
            if hasattr(self,p) and (getattr(self,p) is not None):
                setattr(self,p,getattr(self,p)[w])

        return

    def cont_fit(self):
        lmax = forest.lmax_rest+sp.log10(1+self.z_qso)
        lmin = forest.lmin_rest+sp.log10(1+self.z_qso)
        try:
            mc = forest.mean_cont(self.log_lambda-sp.log10(1+self.z_qso))
        except ValueError:
            raise Exception

        if not self.Fbar is None:
            mc *= self.Fbar
        if not self.T_dla is None:
            mc*=self.T_dla

        var_lss = forest.var_lss(self.log_lambda)
        eta = forest.eta(self.log_lambda)
        fudge = forest.fudge(self.log_lambda)

        def model(p0,p1):
            line = p1*(self.log_lambda-lmin)/(lmax-lmin)+p0
            return line*mc

        def chi2(p0,p1):
            m = model(p0,p1)
            var_pipe = 1./self.iv/m**2
            ## prep_del.variance is the variance of delta
            ## we want here the we = ivar(flux)

            var_tot = variance(var_pipe,eta,var_lss,fudge)
            we = 1/m**2/var_tot

            # force we=1 when use-constant-weight
            # TODO: make this condition clearer, maybe pass an option
            # use_constant_weights?
            if (eta==0).all() :
                we=sp.ones(len(we))
            v = (self.fl-m)**2*we
            return v.sum()-sp.log(we).sum()

        p0 = (self.fl*self.iv).sum()/self.iv.sum()
        p1 = 0

        mig = iminuit.Minuit(chi2,p0=p0,p1=p1,error_p0=p0/2.,error_p1=p0/2.,errordef=1.,print_level=0,fix_p1=(self.order==0))
        fmin,_ = mig.migrad()

        self.co=model(mig.values["p0"],mig.values["p1"])
        self.p0 = mig.values["p0"]
        self.p1 = mig.values["p1"]

        self.bad_cont = None
        if not fmin.is_valid:
            self.bad_cont = "minuit didn't converge"
        if sp.any(self.co <= 0):
            self.bad_cont = "negative continuum"


        ## if the continuum is negative, then set it to a very small number
        ## so that this forest is ignored
        if self.bad_cont is not None:
            self.co = self.co*0+1e-10
            self.p0 = 0.
            self.p1 = 0.


class delta(Qso):

    def __init__(self,thingid,ra,dec,z_qso,plate,mjd,fiberid,log_lambda,we,co,de,order,iv,diff,m_SNR,m_reso,m_z,dll):

        Qso.__init__(self,thingid,ra,dec,z_qso,plate,mjd,fiberid)
        self.log_lambda = log_lambda
        self.we = we
        self.co = co
        self.de = de
        self.order = order
        self.iv = iv
        self.diff = diff
        self.mean_SNR = m_SNR
        self.mean_reso = m_reso
        self.mean_z = m_z
        self.dll = dll

    @classmethod
    def from_forest(cls,f,st,var_lss,eta,fudge,mc=False):

        log_lambda = f.log_lambda
        mst = st(log_lambda)
        var_lss = var_lss(log_lambda)
        eta = eta(log_lambda)
        fudge = fudge(log_lambda)

        #if mc is True use the mock continuum to compute the mean expected flux fraction
        if mc : mef = f.mmef
        else : mef = f.co * mst
        de = f.fl/ mef -1.
        var = 1./f.iv/mef**2
        we = 1./variance(var,eta,var_lss,fudge)
        diff = f.diff
        if f.diff is not None:
            diff /= mef
        iv = f.iv/(eta+(eta==0))*(mef**2)

        return cls(f.thingid,f.ra,f.dec,f.z_qso,f.plate,f.mjd,f.fiberid,log_lambda,we,f.co,de,f.order,
                   iv,diff,f.mean_SNR,f.mean_reso,f.mean_z,f.dll)


    @classmethod
    def from_fitsio(cls,h,Pk1D_type=False):


        head = h.read_header()

        de = h['DELTA'][:]
        log_lambda = h['LOGLAM'][:]


        if  Pk1D_type :
            iv = h['IVAR'][:]
            diff = h['DIFF'][:]
            m_SNR = head['MEANSNR']
            m_reso = head['MEANRESO']
            m_z = head['MEANZ']
            dll =  head['DLL']
            we = None
            co = None
        else :
            iv = None
            diff = None
            m_SNR = None
            m_reso = None
            dll = None
            m_z = None
            we = h['WEIGHT'][:]
            co = h['CONT'][:]


        thingid = head['THING_ID']
        ra = head['RA']
        dec = head['DEC']
        z_qso = head['Z']
        plate = head['PLATE']
        mjd = head['MJD']
        fiberid = head['FIBERID']

        try:
            order = head['ORDER']
        except KeyError:
            order = 1
        return cls(thingid,ra,dec,z_qso,plate,mjd,fiberid,log_lambda,we,co,de,order,
                   iv,diff,m_SNR,m_reso,m_z,dll)


    @classmethod
    def from_ascii(cls,line):

        a = line.split()
        plate = int(a[0])
        mjd = int(a[1])
        fiberid = int(a[2])
        ra = float(a[3])
        dec = float(a[4])
        z_qso = float(a[5])
        m_z = float(a[6])
        m_SNR = float(a[7])
        m_reso = float(a[8])
        dll = float(a[9])

        nbpixel = int(a[10])
        de = sp.array(a[11:11+nbpixel]).astype(float)
        log_lambda = sp.array(a[11+nbpixel:11+2*nbpixel]).astype(float)
        iv = sp.array(a[11+2*nbpixel:11+3*nbpixel]).astype(float)
        diff = sp.array(a[11+3*nbpixel:11+4*nbpixel]).astype(float)


        thingid = 0
        order = 0
        we = None
        co = None

        return cls(thingid,ra,dec,z_qso,plate,mjd,fiberid,log_lambda,we,co,de,order,
                   iv,diff,m_SNR,m_reso,m_z,dll)

    @staticmethod
    def from_image(f):
        h=fitsio.FITS(f)
        de = h[0].read()
        iv = h[1].read()
        log_lambda = h[2].read()
        ra = h[3]["RA"][:].astype(sp.float64)*sp.pi/180.
        dec = h[3]["DEC"][:].astype(sp.float64)*sp.pi/180.
        z = h[3]["Z"][:].astype(sp.float64)
        plate = h[3]["PLATE"][:]
        mjd = h[3]["MJD"][:]
        fiberid = h[3]["FIBER"]
        thingid = h[3]["THING_ID"][:]

        nspec = h[0].read().shape[1]
        deltas=[]
        for i in range(nspec):
            if i%100==0:
                userprint("\rreading deltas {} of {}".format(i,nspec),end="")

            delt = de[:,i]
            ivar = iv[:,i]
            w = ivar>0
            delt = delt[w]
            ivar = ivar[w]
            lam = log_lambda[w]

            order = 1
            diff = None
            m_SNR = None
            m_reso = None
            dll = None
            m_z = None

            deltas.append(delta(thingid[i],ra[i],dec[i],z[i],plate[i],mjd[i],fiberid[i],lam,ivar,None,delt,order,iv,diff,m_SNR,m_reso,m_z,dll))

        h.close()
        return deltas


    def project(self):
        mde = sp.average(self.de,weights=self.we)
        res=0
        if (self.order==1) and self.de.shape[0] > 1:
            mll = sp.average(self.log_lambda,weights=self.we)
            mld = sp.sum(self.we*self.de*(self.log_lambda-mll))/sp.sum(self.we*(self.log_lambda-mll)**2)
            res = mld * (self.log_lambda-mll)
        elif self.order==1:
            res = self.de

        self.de -= mde + res
