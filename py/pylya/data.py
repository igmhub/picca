import scipy as sp
from astropy.io import fits
from pylya import constants
import iminuit
import dla
from scipy import random

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

        self.zqso = zqso
        self.thid = thid

    def __xor__(self,data):
        try:
            x = sp.array([d.xcart for d in data])
            y = sp.array([d.ycart for d in data])
            z = sp.array([d.zcart for d in data])

            cos = x*self.xcart+y*self.ycart+z*self.zcart
            w = cos>=1.
            cos[w]=1.
        except:
            x = data.xcart
            y = data.ycart
            z = data.zcart
            cos = x*self.xcart+y*self.ycart+z*self.zcart
        
        return sp.arccos(cos)

class forest(qso):

    lmin = None
    lmax = None
    lmin_rest = None
    lmax_rest = None
    rebin = None
    dll = None

    ### Correction function for multiplicative errors in calibration
    correc_flux = None

    ## minumum dla transmission
    dla_mask = None

    var_lss = None
    eta = None
    mean_cont = None


    def __init__(self,ll,fl,iv,thid,ra,dec,zqso,plate,mjd,fid):
        qso.__init__(self,thid,ra,dec,zqso,plate,mjd,fid)
        ## rebin

        bins = ((ll-forest.lmin)/forest.dll+0.5).astype(int)
        w = bins>=0
        fl=fl[w]
        iv =iv[w]
        bins=bins[w]

        ll = forest.lmin + sp.unique(bins)*forest.dll
        civ = sp.bincount(bins,weights=iv)
        w=civ>0
        civ=civ[bins.min():]
        cfl = sp.bincount(bins,weights=iv*fl)
        cfl = cfl[bins.min():]
        w=civ>0
        cfl[w]/=civ[w]
        iv = civ
        fl = cfl


        ## cut to specified range
        w= (ll<forest.lmax) & (ll-sp.log10(1+self.zqso)>forest.lmin_rest) & (ll-sp.log10(1+self.zqso)<forest.lmax_rest)
        w = w & (iv>0)
        if w.sum()==0:return
        
        ll=ll[w]
        fl=fl[w]
        iv=iv[w]

        if not self.correc_flux is None:
            correction = self.correc_flux(ll)
            fl /= correction
            iv *= correction**2

        self.T_dla = None
        self.ll = ll
        self.fl = fl
        self.iv = iv

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

    def add_dla(self,zabs,nhi,mask=None):
        if not hasattr(self,'ll'):
            return
        if self.T_dla is None:
            self.T_dla = sp.ones(len(self.ll))

        self.T_dla *= dla.p_voigt(10**self.ll,zabs,nhi).t

        w = (self.T_dla>forest.dla_mask)
        if not mask is None:
            for l in mask:
                w = w & ( (self.ll-sp.log10(1.+zabs)<l[0]) | (self.ll-sp.log10(1.+zabs)>l[1]) )

        self.iv = self.iv[w]
        self.ll = self.ll[w]
        self.fl = self.fl[w]
        self.T_dla = self.T_dla[w]

    def cont_fit(self, fit_dlas=False):
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

        def model(p0,p1):
            line = p1*(self.ll-lmin)/(lmax-lmin)+p0*(lmax-self.ll)/(lmax-lmin)
            return line*mc

        def chi2(p0,p1):
            m = model(p0,p1)
            iv = self.iv/eta
            we = iv/(iv*var_lss*m**2+1)
            v = (self.fl-m)**2*we
            return v.sum()-sp.log(we).sum()

        def chi2_dlas(p0,p1,zabs,nhi):
            m = model(p0,p1)
            for z,n in zip(self.zabs,self.nhi):
                m*=dla.p_voigt(10**self.ll,z,n)
            m*=dla.p_voigt(10**self.ll,zabs,nhi)
            iv = self.iv/eta
            we = iv/(iv*var_lss*m**2+1)
            v = (self.fl-m)**2*we
            return v.sum()-sp.log(we).sum()

        p0 = p1 = (self.fl*self.iv).sum()/self.iv.sum()

        if not fit_dlas:
            mig = iminuit.Minuit(chi2,p0=p0,p1=p1,error_p0=p0/2.,error_p1=p1/2.,errordef=1.,print_level=0)
        else:
            if not hasattr(self,"zabs"):
                self.cos=[]
                self.zabs=[]
                self.nhi=[]
                self.dla_abs=[]
                self.chi2s=[]
            zmin = 10**self.ll.min()/constants.absorber_IGM["LYA"]-1
            zmax = 10**self.ll.max()/constants.absorber_IGM["LYA"]-1
            zabs = (zmin+zmax)/2.

            nhi_min = 19.
            nhi_max = 22.
            nhi = (nhi_max+nhi_min)/2

            chi2 = []
            zabss= []
            nhis = []
            dz = 0.05
            z0 = zmin
            while z0 < zmax:
                mig = iminuit.Minuit(chi2_dlas,p0=p0,p1=p1,zabs=z0,nhi=nhi_min+0.5,
                            error_p0=p0/2.,error_p1=p1/2.,
                            error_zabs=dz,error_nhi=(nhi_max-nhi_min)/2,
                            limit_nhi=(nhi_min,nhi_max),limit_zabs=(z0-dz,z0+dz),
                            errordef=1.,print_level=3)
                mig.migrad()
                chi2.append(mig.fval)
                zabss.append(mig.values["zabs"])
                nhis.append(mig.values["nhi"])
                z0 += dz
            
            imin = sp.argmin(chi2)
            zabs = zabss[imin]
            nhi = nhis[imin]
            mig = iminuit.Minuit(chi2_dlas,p0=p0,p1=p1,zabs=zabs,nhi=nhi,
                                 error_p0=p0/2.,error_p1=p1/2.,error_zabs=zabs/2.,error_nhi=3.,
                                 limit_nhi=(nhi_min,nhi_max),limit_zabs=(zabs-2*dz,zabs+2*dz),
                                 errordef=1.,print_level=0)
            
        mig.migrad()
        self.co=model(mig.values["p0"],mig.values["p1"])
        self.p0 = mig.values["p0"]
        self.p1 = mig.values["p1"]
        if fit_dlas:
            self.cos.append(self.co)
            self.zabs.append(mig.values["zabs"])
            self.nhi.append(mig.values["nhi"])
            self.dla_abs.append(dla.p_voigt(10**self.ll,mig.values["zabs"],mig.values["nhi"]))
            self.chi2s.append(mig.fval)
        else:
            self.chi2_no_dla = mig.fval
        self.chi2 = mig.fval


class delta(qso):
    
    def __init__(self,thid,ra,dec,zqso,plate,mjd,fid,ll,we,co,de):
        qso.__init__(self,thid,ra,dec,zqso,plate,mjd,fid)
        self.ll = ll
        self.we = we
        self.co = co
        self.de = de

    @classmethod
    def from_forest(cls,f,st,var_lss,eta):

        de = f.fl/f.co/st(f.ll)-1
        ll = f.ll
        iv = f.iv/eta(f.ll)
        we = iv*f.co**2/(iv*f.co**2*var_lss(f.ll)+1)
        co = f.co
        return cls(f.thid,f.ra,f.dec,f.zqso,f.plate,f.mjd,f.fid,ll,we,co,de)

    @classmethod
    def from_fitsio(cls,h):
        de = h['DELTA'][:]
        we = h['WEIGHT'][:]
        ll = h['LOGLAM'][:]
        co = h['CONT'][:]

        head = h.read_header()
        thid = head['THING_ID']
        ra = head['RA']
        dec = head['DEC']
        zqso = head['Z']
        plate = head['PLATE']
        mjd = head['MJD']
        fid = head['FIBERID']
        return cls(thid,ra,dec,zqso,plate,mjd,fid,ll,we,co,de)

    @staticmethod
    def from_image(f):
        h=fitsio.FITS(f)
        de = h[0].read()
        iv = h[1].read()
        ll = h[2].read()
        ra = h[3]["RA"][:]*sp.pi/180.
        dec = h[3]["DEC"][:]*sp/180.
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
            w = iv>0
            delt=delt[w]
            ivar=ivar[w]
            lam = ll[w]
            
            deltas.append(delta(thid[i],ra[i],dec[i],z[i],plate[i],mjd[i],fid[i],lam,ivar,None,delt))

        h.close()
        return deltas


    def project(self):
        mde = sp.average(self.de,weights=self.we)
        mll = sp.average(self.ll,weights=self.we)
        mld = sp.sum(self.we*self.de*(self.ll-mll))/sp.sum(self.we*(self.ll-mll)**2)

        self.de -= mde + mld * (self.ll-mll)

