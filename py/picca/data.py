import scipy as sp
from astropy.io import fits
from picca import constants
import iminuit
from dla import dla
import sys

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


    def __init__(self,ll,fl,iv,thid,ra,dec,zqso,plate,mjd,fid,order):
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
        self.order=order

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

        self.T_dla *= dla(self,zabs,nhi).t

        w = (self.T_dla>forest.dla_mask)
        if not mask is None:
            for l in mask:
                w = w & ( (self.ll-sp.log10(1.+zabs)<l[0]) | (self.ll-sp.log10(1.+zabs)>l[1]) )

        self.iv = self.iv[w]
        self.ll = self.ll[w]
        self.fl = self.fl[w]
        self.T_dla = self.T_dla[w]

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

        def model(p0,p1):
            line = p1*(self.ll-lmin)/(lmax-lmin)+p0
            return line*mc

        def chi2(p0,p1):
            m = model(p0,p1)
            iv = self.iv/eta
            we = iv/(iv*var_lss*m**2+1)
            v = (self.fl-m)**2*we
            return v.sum()-sp.log(we).sum()

        p0 = (self.fl*self.iv).sum()/self.iv.sum()
        p1 = 0

        mig = iminuit.Minuit(chi2,p0=p0,p1=p1,error_p0=p0/2.,error_p1=p1/2.,errordef=1.,print_level=0,fix_p1=(self.order==0))
        mig.migrad()
        self.co=model(mig.values["p0"],mig.values["p1"])
        self.p0 = mig.values["p0"]
        self.p1 = mig.values["p1"]
        sys.stderr.write("CONT FIT {} {} {}\n".format(self.thid,p0,p1))


class delta(qso):
    
    def __init__(self,thid,ra,dec,zqso,plate,mjd,fid,ll,we,co,de,order):
        qso.__init__(self,thid,ra,dec,zqso,plate,mjd,fid)
        self.ll = ll
        self.we = we
        self.co = co
        self.de = de
        self.order=order

    @classmethod
    def from_forest(cls,f,st,var_lss,eta):

        de = f.fl/f.co/st(f.ll)-1
        ll = f.ll
        iv = f.iv/eta(f.ll)
        we = iv*f.co**2/(iv*f.co**2*var_lss(f.ll)+1)
        co = f.co
        return cls(f.thid,f.ra,f.dec,f.zqso,f.plate,f.mjd,f.fid,ll,we,co,de,f.order)

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
        try: 
            order = head['ORDER']
        except ValueError:
            order = 1
        return cls(thid,ra,dec,zqso,plate,mjd,fid,ll,we,co,de,order)

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
        res=0
        if (self.order==1): 
            mll = sp.average(self.ll,weights=self.we)
            mld = sp.sum(self.we*self.de*(self.ll-mll))/sp.sum(self.we*(self.ll-mll)**2)
            res = mld * (self.ll-mll) 

        self.de -= mde + res
