import scipy as sp
from picca import constants
import iminuit
from .dla import dla
import fitsio
import sys

def variance(var,eta,var_lss,fudge):
    return eta*var + var_lss + fudge/var


class qso:
    def __init__(self,thid,ra,dec,zqso,plate,mjd,fiberid):
        self.ra = ra
        self.dec = dec

        self.plate=plate
        self.mjd=mjd
        self.fid=fiberid

        ## cartesian coordinates
        self.xcart = sp.cos(ra)*sp.cos(dec)
        self.ycart = sp.sin(ra)*sp.cos(dec)
        self.zcart = sp.sin(dec)
        self.cosdec = sp.cos(dec)

        self.zqso = zqso
        self.thid = thid

    def __xor__(self,data):
        try:
            x = sp.array([d.xcart for d in data])
            y = sp.array([d.ycart for d in data])
            z = sp.array([d.zcart for d in data])
            ra = sp.array([d.ra for d in data])
            dec = sp.array([d.dec for d in data])

            cos = x*self.xcart+y*self.ycart+z*self.zcart
            w = cos>=1.
            if w.sum()!=0:
                print('WARNING: {} pairs have cosinus>=1.'.format(w.sum()))
                cos[w] = 1.
            w = cos<=-1.
            if w.sum()!=0:
                print('WARNING: {} pairs have cosinus<=-1.'.format(w.sum()))
                cos[w] = -1.
            angl = sp.arccos(cos)

            w = (sp.absolute(ra-self.ra)<constants.small_angle_cut_off) & (sp.absolute(dec-self.dec)<constants.small_angle_cut_off)
            if w.sum()!=0:
                angl[w] = sp.sqrt( (dec[w]-self.dec)**2 + (self.cosdec*(ra[w]-self.ra))**2 )
        except:
            x = data.xcart
            y = data.ycart
            z = data.zcart
            ra = data.ra
            dec = data.dec

            cos = x*self.xcart+y*self.ycart+z*self.zcart
            if cos>=1.:
                print('WARNING: 1 pair has cosinus>=1.')
                cos = 1.
            elif cos<=-1.:
                print('WARNING: 1 pair has cosinus<=-1.')
                cos = -1.
            angl = sp.arccos(cos)
            if (sp.absolute(ra-self.ra)<constants.small_angle_cut_off) & (sp.absolute(dec-self.dec)<constants.small_angle_cut_off):
                angl = sp.sqrt( (dec-self.dec)**2 + (self.cosdec*(ra-self.ra))**2 )
        return angl

class forest(qso):

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

    ## minumum dla transmission
    dla_mask = None

    var_lss = None
    eta = None
    mean_cont = None

    ## quality variables
    mean_SNR = None
    mean_reso = None
    mean_z = None


    def __init__(self,ll,fl,iv,thid,ra,dec,zqso,plate,mjd,fid,order,diff=None,reso=None):
        qso.__init__(self,thid,ra,dec,zqso,plate,mjd,fid)

        ## cut to specified range
        bins = sp.floor((ll-forest.lmin)/forest.dll+0.5).astype(int)
        ll = forest.lmin + bins*forest.dll
        w = (ll>=forest.lmin)
        w = w & (ll<forest.lmax)
        w = w & (ll-sp.log10(1.+self.zqso)>forest.lmin_rest)
        w = w & (ll-sp.log10(1.+self.zqso)<forest.lmax_rest)
        w = w & (iv>0.)
        if w.sum()==0:
            return
        bins = bins[w]
        ll = ll[w]
        fl = fl[w]
        iv = iv[w]
        if diff is not None :
            diff=diff[w]
            reso=reso[w]

        ## rebin
        cll = forest.lmin + sp.arange(bins.max()+1)*forest.dll
        cfl = sp.zeros(bins.max()+1)
        civ = sp.zeros(bins.max()+1)
        ccfl = sp.bincount(bins,weights=iv*fl)
        cciv = sp.bincount(bins,weights=iv)
        if diff is not None :
            cdiff = sp.bincount(bins,weights=iv*diff)
            creso = sp.bincount(bins,weights=iv*reso)

        cfl[:len(ccfl)] += ccfl
        civ[:len(cciv)] += cciv
        w = (civ>0.)
        if w.sum()==0:
            return
        ll = cll[w]
        fl = cfl[w]/civ[w]
        iv = civ[w]
        if diff is not None :
            diff = cdiff[w]/civ[w]
            reso = creso[w]/civ[w]

        ## Flux calibration correction
        if not self.correc_flux is None:
            correction = self.correc_flux(ll)
            fl /= correction
            iv *= correction**2
        if not self.correc_ivar is None:
            correction = self.correc_ivar(ll)
            iv /= correction

        self.T_dla = None
        self.ll = ll
        self.fl = fl
        self.iv = iv
        self.order = order
        #if diff is not None :
        self.diff = diff
        self.reso = reso
#        else :
#           self.diff = sp.zeros(len(ll))
#           self.reso = sp.ones(len(ll))

        # compute means
        if reso is not None : self.mean_reso = sum(reso)/float(len(reso))

        err = 1.0/sp.sqrt(iv)
        SNR = fl/err
        self.mean_SNR = sum(SNR)/float(len(SNR))
        lam_lya = constants.absorber_IGM["LYA"]
        self.mean_z = (sp.power(10.,ll[len(ll)-1])+sp.power(10.,ll[0]))/2./lam_lya -1.0


    def __add__(self,d):

        if not hasattr(self,'ll') or not hasattr(d,'ll'):
            return self

        ll = sp.append(self.ll,d.ll)
        fl = sp.append(self.fl,d.fl)
        iv = sp.append(self.iv,d.iv)

        bins = sp.floor((ll-forest.lmin)/forest.dll+0.5).astype(int)
        cll = forest.lmin + sp.arange(bins.max()+1)*forest.dll
        cfl = sp.zeros(bins.max()+1)
        civ = sp.zeros(bins.max()+1)
        ccfl = sp.bincount(bins,weights=iv*fl)
        cciv = sp.bincount(bins,weights=iv)
        cfl[:len(ccfl)] += ccfl
        civ[:len(cciv)] += cciv
        w = (civ>0.)

        self.ll = cll[w]
        self.fl = cfl[w]/civ[w]
        self.iv = civ[w]

        return self

    def mask(self,mask_obs,mask_RF):
        if not hasattr(self,'ll'):
            return

        w = sp.ones(self.ll.size).astype(bool)
        for l in mask_obs:
            w = w & ( (self.ll<l[0]) | (self.ll>l[1]) )
        for l in mask_RF:
            w = w & ( (self.ll-sp.log10(1.+self.zqso)<l[0]) | (self.ll-sp.log10(1.+self.zqso)>l[1]) )

        self.ll = self.ll[w]
        self.fl = self.fl[w]
        self.iv = self.iv[w]
        if self.diff is not None :
             self.diff = self.diff[w]
             self.reso = self.reso[w]

    def add_dla(self,zabs,nhi,mask=None):
        if not hasattr(self,'ll'):
            return
        if self.T_dla is None:
            self.T_dla = sp.ones(len(self.ll))

        self.T_dla *= dla(self,zabs,nhi).t

        w = (self.T_dla>forest.dla_mask)
        if not mask is None:
            for l in mask:
                w = w & ( (self.ll-sp.log10(1.+zabs)<l[0]) | (self.ll-sp.log10(1.+zabs)>l[1]) )

        self.iv = self.iv[w]
        self.ll = self.ll[w]
        self.fl = self.fl[w]
        self.T_dla = self.T_dla[w]
        if self.diff is not None :
            self.diff = self.diff[w]
            self.reso = self.reso[w]


    def cont_fit(self):
        lmax = forest.lmax_rest+sp.log10(1+self.zqso)
        lmin = forest.lmin_rest+sp.log10(1+self.zqso)
        try:
            mc = forest.mean_cont(self.ll-sp.log10(1+self.zqso))
        except ValueError:
            raise Exception

        if not self.T_dla is None:
            mc*=self.T_dla

        var_lss = forest.var_lss(self.ll)
        eta = forest.eta(self.ll)
        fudge = forest.fudge(self.ll)

        def model(p0,p1):
            line = p1*(self.ll-lmin)/(lmax-lmin)+p0
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


class delta(qso):

    def __init__(self,thid,ra,dec,zqso,plate,mjd,fid,ll,we,co,de,order,iv,diff,m_SNR,m_reso,m_z,dll):

        qso.__init__(self,thid,ra,dec,zqso,plate,mjd,fid)
        self.ll = ll
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
    def from_forest(cls,f,st,var_lss,eta,fudge):

        ll = f.ll
        mst = st(ll)
        var_lss = var_lss(ll)
        eta = eta(ll)
        fudge = fudge(ll)
        co = f.co
        de = f.fl/(co*mst)-1.
        var = 1./f.iv/(co*mst)**2
        we = 1./variance(var,eta,var_lss,fudge)
        diff = f.diff
        if f.diff is not None:
            diff /= co*mst
        iv = f.iv/(eta+(eta==0))*(co**2)*(mst**2)

        return cls(f.thid,f.ra,f.dec,f.zqso,f.plate,f.mjd,f.fid,ll,we,co,de,f.order,
                   iv,diff,f.mean_SNR,f.mean_reso,f.mean_z,f.dll)


    @classmethod
    def from_fitsio(cls,h,Pk1D_type=False):


        head = h.read_header()

        de = h['DELTA'][:]
        ll = h['LOGLAM'][:]


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


        thid = head['THING_ID']
        ra = head['RA']
        dec = head['DEC']
        zqso = head['Z']
        plate = head['PLATE']
        mjd = head['MJD']
        fid = head['FIBERID']

        try:
            order = head['ORDER']
        except KeyError:
            order = 1
        return cls(thid,ra,dec,zqso,plate,mjd,fid,ll,we,co,de,order,
                   iv,diff,m_SNR,m_reso,m_z,dll)


    @classmethod
    def from_ascii(cls,line):

        a = line.split()
        plate = int(a[0])
        mjd = int(a[1])
        fid = int(a[2])
        ra = float(a[3])
        dec = float(a[4])
        zqso = float(a[5])
        m_z = float(a[6])
        m_SNR = float(a[7])
        m_reso = float(a[8])
        dll = float(a[9])

        nbpixel = int(a[10])
        de = sp.array(a[11:11+nbpixel]).astype(float)
        ll = sp.array(a[11+nbpixel:11+2*nbpixel]).astype(float)
        iv = sp.array(a[11+2*nbpixel:11+3*nbpixel]).astype(float)
        diff = sp.array(a[11+3*nbpixel:11+4*nbpixel]).astype(float)


        thid = 0
        order = 0
        we = None
        co = None

        return cls(thid,ra,dec,zqso,plate,mjd,fid,ll,we,co,de,order,
                   iv,diff,m_SNR,m_reso,m_z,dll)

    @staticmethod
    def from_image(f):
        h=fitsio.FITS(f)
        de = h[0].read()
        iv = h[1].read()
        ll = h[2].read()
        ra = h[3]["RA"][:]*sp.pi/180.
        dec = h[3]["DEC"][:]*sp.pi/180.
        z = h[3]["Z"][:]
        plate = h[3]["PLATE"][:]
        mjd = h[3]["MJD"][:]
        fid = h[3]["FIBER"]
        thid = h[3]["THING_ID"][:]

        nspec = h[0].read().shape[1]
        deltas=[]
        for i in range(nspec):
            if i%100==0:
                sys.stderr.write("\rreading deltas {} of {}".format(i,nspec))

            delt = de[:,i]
            ivar = iv[:,i]
            w = ivar>0
            delt = delt[w]
            ivar = ivar[w]
            lam = ll[w]

            order = 1
            diff = None
            m_SNR = None
            m_reso = None
            dll = None
            m_z = None

            deltas.append(delta(thid[i],ra[i],dec[i],z[i],plate[i],mjd[i],fid[i],lam,ivar,None,delt,order,iv,diff,m_SNR,m_reso,m_z,dll))

        h.close()
        return deltas


    def project(self):
        mde = sp.average(self.de,weights=self.we)
        res=0
        if (self.order==1) and self.de.shape[0] > 1:
            mll = sp.average(self.ll,weights=self.we)
            mld = sp.sum(self.we*self.de*(self.ll-mll))/sp.sum(self.we*(self.ll-mll)**2)
            res = mld * (self.ll-mll)
        elif self.order==1:
            res = self.de

        self.de -= mde + res
