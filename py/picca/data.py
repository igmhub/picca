"""This module defines data structure to deal with line of sight data.

This module provides with three classes (Qso, Forest, Delta) and one
function (variance) to manage the line-of-sight data. See the respective
docstrings for more details
"""
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
    """Class to represent quasar objects.

    Attributes:
        ra: float
            Right-ascension of the quasar (in radians).
        dec: float
            Declination of the quasar (in radians).
        z_qso: float
            Redshift of the quasar.
        plate: integer
            Plate number of the observation.
        fiberid: integer
            Fiberid of the observation.
        mjd: integer
            Modified Julian Date of the observation.
        thingid: integer
            Thingid of the observation.
        x_cart: float
            The x coordinate when representing ra, dec in a cartesian
            coordinate system.
        y_cart: float
            The y coordinate when representing ra, dec in a cartesian
            coordinate system.
        z_cart: float
            The z coordinate when representing ra, dec in a cartesian
            coordinate system.
        cos_dec: float
            Cosine of the declination angle.

    Note that plate-fiberid-mjd is a unique identifier
    for the quasar.

    Methods:
        __init__: Initialize class instance.
        __xor__: Computes the angular separation between two quasars.
    """
    def __init__(self,thingid,ra,dec,z_qso,plate,mjd,fiberid):
        """Initialize class instance."""
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

    # TODO: rename method, update class docstring
    def __xor__(self,data):
        """Computes the angular separation between two quasars.

        Args:
            data: Qso or list of Qso
                Objects with which the angular separation will
                be computed.

        Returns
            A float or an array (depending on input data) with the angular
            separation between this quasar and the object(s) in data.
        """
        # case 1: data is list-like
        try:
            x_cart = sp.array([d.x_cart for d in data])
            y_cart= sp.array([d.y_cart for d in data])
            z_cart = sp.array([d.z_cart for d in data])
            ra = sp.array([d.ra for d in data])
            dec = sp.array([d.dec for d in data])

            cos = x_cart*self.x_cart+y_cart*self.y_cart+z_cart*self.z_cart
            w = cos>=1.
            if w.sum()!=0:
                userprint('WARNING: {} pairs have cos>=1.'.format(w.sum()))
                cos[w] = 1.
            w = cos<=-1.
            if w.sum()!=0:
                userprint('WARNING: {} pairs have cos<=-1.'.format(w.sum()))
                cos[w] = -1.
            angl = sp.arccos(cos)

            w = ((np.absolute(ra-self.ra)<constants.small_angle_cut_off) &
                (np.absolute(dec-self.dec)<constants.small_angle_cut_off))
            if w.sum()!=0:
                angl[w] = sp.sqrt((dec[w] - self.dec)**2 +
                                  (self.cos_dec*(ra[w] - self.ra))**2)
        # case 2: data is a Qso
        except:
            x_cart = data.x_cart
            y_cart = data.y_cart
            z_cart = data.z_cart
            ra = data.ra
            dec = data.dec

            cos = x_cart*self.x_cart+y_cart*self.y_cart+z_cart*self.z_cart
            if cos>=1.:
                userprint('WARNING: 1 pair has cosinus>=1.')
                cos = 1.
            elif cos<=-1.:
                userprint('WARNING: 1 pair has cosinus<=-1.')
                cos = -1.
            angl = sp.arccos(cos)
            if ((np.absolute(ra-self.ra)<constants.small_angle_cut_off) &
                (np.absolute(dec-self.dec)<constants.small_angle_cut_off)):
                angl = sp.sqrt((dec - self.dec)**2 +
                               (self.cos_dec*(ra - self.ra))**2)
        return angl

class Forest(Qso):
    # TODO: revise and complete
    """Class to represent a Lyman alpha (or other absorption) forest

    This class stores the information of an absorption forest.
    This includes the information required to extract the delta
    field from it: flux correction, inverse variance corrections,
    dlas, absorbers, ...

    Attributes:
        ## Inherits from Qso ##
        log_lambda : array of floats
            Array containing the logarithm of the wavelengths (in Angs)
        flux : array of floats
            Array containing the flux associated to each wavelength
        ivar: array of floats
            Array containing the inverse variance associated to each flux
        mean_optical_depth: array of floats or None
            Mean optical depth at the redshift of each pixel in the forest
        dla_transmission: array of floats or None
            Decrease of the transmitted flux due to the presence of a Damped
            Lyman alpha absorbers

    Class attributes:
        log_lambda_max: float
            Logarithm of the maximum wavelength (in Angs) to be considered in a
            forest.
        log_lambda_min: float
            Logarithm of the minimum wavelength (in Angs) to be considered in a
            forest.
        log_lambda_max_rest_frame: float
            As log_lambda_max but for rest-frame wavelength.
        log_lambda_min_rest_frame: float
            As log_lambda_min but for rest-frame wavelength.
        rebin: integer
            Rebin wavelength grid by combining this number of adjacent pixels
            (inverse variance weighting).
        delta_log_lambda: float
            Variation of the logarithm of the wavelength (in Angs) between two
            pixels.
        extinction_bv_map: dict
            B-V extinction due to dust. Maps thingids (integers) to the dust
            correction (array).
        absorber_mask_width: float
            Mask width on each side of the absorber central observed wavelength
            in units of 1e4*dlog10(lambda/Angs).
        dla_mask_limit: float
            Lower limit on the DLA transmission. Transmissions below this
            number are masked.

    Methods:
        correct_flux: Corrects for multiplicative errors in pipeline flux
            calibration.
        correct_ivar: Corrects for multiplicative errors in pipeline inverse
            variance calibration.
        get_var_lss: Computes the pixel variance due to the Large Scale
            Strucure.
        get_eta: Computes the correction factor to the contribution of the
            pipeline estimate of the instrumental noise to the variance.
        get_mean_cont: Interpolates the mean quasar continuum over the whole
            sample on the wavelength array.
    """
    log_lambda_min = None
    log_lambda_max = None
    log_lambda_min_rest_frame = None
    log_lambda_max_rest_frame = None
    rebin = None
    delta_log_lambda = None

    @classmethod
    def correct_flux(cls, log_lambda):
        """Corrects for multiplicative errors in pipeline flux calibration.

        Empty function to be loaded at run-time.

        Args:
            log_lambda: float
                Array containing the logarithm of the wavelengths (in Angs)

        Returns:
            An array with the correction

        Raises:
            NotImplementedError: Function was not specified
        """
        raise NotImplementedError("Function should be specified at run-time")

    @classmethod
    def correct_ivar(cls, lol_lambda):
        """Corrects for multiplicative errors in pipeline inverse variance
           calibration.

        Empty function to be loaded at run-time.

        Args:
            log_lambda: float
                Array containing the logarithm of the wavelengths (in Angs)

        Returns:
            An array with the correction

        Raises:
            NotImplementedError: Function was not specified
        """
        raise NotImplementedError("Function should be specified at run-time")

    # map of g-band extinction to thingids for dust correction
    extinction_bv_map = None

    # absorber pixel mask limit
    absorber_mask_width = None

    ## minumum dla transmission
    dla_mask_limit = None

    @classmethod
    def get_var_lss(cls, lol_lambda):
        """Computes the pixel variance due to the Large Scale Strucure

        Empty function to be loaded at run-time.

        Args:
            log_lambda: float
                Array containing the logarithm of the wavelengths (in Angs)

        Returns:
            An array with the correction

        Raises:
            NotImplementedError: Function was not specified
        """
        raise NotImplementedError("Function should be specified at run-time")

    @classmethod
    def get_eta(cls, lol_lambda):
        # TODO: update reference to DR16 paper
        """Computes the correction factor to the contribution of the pipeline
        estimate of the instrumental noise to the variance.

        See equation 4 of du Mas des Bourboux et al. In prep. for details.

        Empty function to be loaded at run-time.

        Args:
            log_lambda: float
                Array containing the logarithm of the wavelengths (in Angs)

        Returns:
            An array with the correction

        Raises:
            NotImplementedError: Function was not specified
        """
        raise NotImplementedError("Function should be specified at run-time")

    @classmethod
    def get_mean_cont(cls, lol_lambda):
        # TODO: update reference to DR16 paper
        """Interpolates the mean quasar continuum over the whole sample on
        the wavelength array

        See equation 2 of du Mas des Bourboux et al. In prep. for details.

        Empty function to be loaded at run-time.

        Args:
            log_lambda: float
                Array containing the logarithm of the wavelengths (in Angs)

        Returns:
            An array with the correction

        Raises:
            NotImplementedError: Function was not specified
        """
        raise NotImplementedError("Function should be specified at run-time")

    def __init__(self, log_lambda, flux, ivar, thingid, ra, dec, z_qso, plate,
                 mjd, fiberid, order, exposures_diff=None, reso=None,
                 mean_expected_flux_frac=None, igm_absorption="LYA"):
        """ Initialize class instances.

        Args:
            log_lambda : array of floats
                Array containing the logarithm of the wavelengths (in Angs).
            flux : array of floats
                Array containing the flux associated to each wavelength.
            ivar : array of floats
                Array containing the inverse variance associated to each flux.
            thingis : float
                ThingID of the observation.
            ra: float
                Right-ascension of the quasar (in radians).
            dec: float
                Declination of the quasar (in radians).
            z_qso: float
                Redshift of the quasar.
            plate: integer
                Plate number of the observation.
            mjd: integer
                Modified Julian Date of the observation.
            fiberid: integer
                Fiberid of the observation.
            order: 1 or 0
                Renamed to make meaning more explicit.
            exposures_diff: array of floats or None - default: None
                Difference between exposures.
            reso: array of floats or None - default: None
                Resolution of the forest.
            mean_expected_flux_frac: array of floats or None - default: None
                Mean expected flux fraction using the mock continuum
            igm_absorption: string - default: "LYA"
                Name of the absorption in picca.constants defining the
                redshift of the forest pixels
        """
        Qso.__init__(self, thingid, ra, dec, z_qso, plate, mjd, fiberid)

        # apply dust extinction correction
        if not Forest.extinction_bv_map is None:
            corr = unred(10**log_lambda, Forest.extinction_bv_map[thingid])
            flux /= corr
            ivar *= corr**2
            if not exposures_diff is None:
                exposures_diff /= corr

        ## cut to specified range
        bins = (np.floor((log_lambda - Forest.log_lambda_min)
                /Forest.delta_log_lambda + 0.5).astype(int))
        log_lambda = Forest.log_lambda_min + bins*Forest.delta_log_lambda
        w = (log_lambda >= Forest.log_lambda_min)
        w = w & (log_lambda < Forest.log_lambda_max)
        w = w & (log_lambda - sp.log10(1.+self.z_qso) >
                 Forest.log_lambda_min_rest_frame)
        w = w & (log_lambda - sp.log10(1.+self.z_qso) <
                 Forest.log_lambda_max_rest_frame)
        w = w & (ivar > 0.)
        if w.sum() == 0:
            return
        bins = bins[w]
        log_lambda = log_lambda[w]
        flux = flux[w]
        ivar = ivar[w]
        if mean_expected_flux_frac is not None:
            mean_expected_flux_frac = mean_expected_flux_frac[w]
        if exposures_diff is not None:
            exposures_diff = exposures_diff[w]
        if reso is not None:
            reso = reso[w]

        # rebin arrays
        rebin_log_lambda = (Forest.log_lambda_min + np.arange(bins.max()+1)
                            *Forest.delta_log_lambda)
        #### old way rebinning
        #rebin_flux = np.zeros(bins.max()+1)
        #rebin_ivar = np.zeros(bins.max()+1)
        #if mean_expected_flux_frac is not None:
        #    rebin_mean_expected_flux_frac = np.zeros(bins.max()+1)
        #ccfl = np.bincount(bins, weights=ivar*flux)
        #cciv = np.bincount(bins, weights=ivar)
        #if mean_expected_flux_frac is not None:
        #    ccmmef = sp.bincount(bins, weights=ivar*mean_expected_flux_frac)
        #rebin_flux[:len(ccfl)] += ccfl
        #rebin_ivar[:len(cciv)] += cciv
        #if mean_expected_flux_frac is not None:
        #    rebin_mean_expected_flux_frac[:len(ccmmef)] += ccmmef
        #### end of old way rebinning
        rebin_flux = np.bincount(bins, weights=ivar*flux)
        rebin_ivar = np.bincount(bins, weights=ivar)
        if mean_expected_flux_frac is not None:
            rebin_mean_expected_flux_frac = np.bincount(bins, weights=ivar*
                                                        mean_expected_flux_frac)
        if exposures_diff is not None:
            rebin_exposures_diff = np.bincount(bins,
                                               weights=ivar*exposures_diff)
        if reso is not None:
            rebin_reso = sp.bincount(bins, weights=ivar*reso)

        w = (rebin_ivar > 0.)
        if w.sum() == 0:
            return
        log_lambda = rebin_log_lambda[w]
        flux = rebin_flux[w]/rebin_ivar[w]
        ivar = rebin_ivar[w]
        if mean_expected_flux_frac is not None:
            mean_expected_flux_frac = (rebin_mean_expected_flux_frac[w]
                                       /rebin_ivar[w])
        if exposures_diff is not None:
            exposures_diff = rebin_exposures_diff[w]/rebin_ivar[w]
        if reso is not None:
            reso = rebin_reso[w]/rebin_ivar[w]

        # Flux calibration correction
        try:
            correction = Forest.correct_flux(log_lambda)
            flux /= correction
            ivar *= correction**2
        except NotImplementedError:
            pass
        # Inverse variance correction
        try:
            correction = Forest.correct_ivar(log_lambda)
            ivar /= correction
        except NotImplementedError:
            pass

        # keep the results so far in this instance
        self.mean_optical_depth = None
        self.dla_transmission = None
        self.log_lambda = log_lambda
        self.flux = flux
        self.ivar = ivar
        self.mean_expected_flux_frac = mean_expected_flux_frac
        self.order = order
        self.exposures_diff = exposures_diff
        self.reso = reso
        self.igm_absorption = igm_absorption

        # compute mean quality variables
        if reso is not None:
            self.mean_reso = reso.mean()
        else:
            self.mean_reso = None

        err = 1.0/sp.sqrt(ivar)
        snr = flux/err
        self.mean_snr = sum(snr)/float(len(snr))
        lambda_igm_absorption = constants.absorber_IGM[self.igm_absorption]
        self.mean_z = ((np.power(10., log_lambda[len(log_lambda) - 1]) +
                        np.power(10., log_lambda[0]))/2./lambda_igm_absorption
                        - 1.0)

    def __add__(self, other):
        """Add the information of another forest.

        Forests are coadded by using inverse variance weighting.

        Args:
            other: Forest
                The forest instance to be coadded. If other does not have the
                attribute log_lambda, then the method returns without doing
                anything.

        Returns:
            The coadded forest.
        """
        if not hasattr(self,'log_lambda') or not hasattr(other,'log_lambda'):
            return self

        # this should contain all quantities that are to be coadded using
        # ivar weighting
        ivar_coadd_data = {}

        log_lambda = np.append(self.log_lambda, other.log_lambda)
        ivar_coadd_data['flux'] = np.append(self.flux, other.flux)
        ivar = np.append(self.ivar, other.ivar)

        if self.mean_expected_flux_frac is not None:
            mean_expected_flux_frac = np.append(self.mean_expected_flux_frac,
                                                other.mean_expected_flux_frac)
            ivar_coadd_data['mean_expected_flux_frac'] = mean_expected_flux_frac

        if self.exposures_diff is not None:
            ivar_coadd_data['exposures_diff'] = np.append(self.exposures_diff,
                                                          other.exposures_diff)
        if self.reso is not None:
            ivar_coadd_data['reso'] = np.append(self.reso, other.reso)

        # coadd the deltas by rebinning
        bins = np.floor((log_lambda - Forest.log_lambda_min)
                        /Forest.delta_log_lambda + 0.5).astype(int)
        rebin_log_lambda = Forest.log_lambda_min + (np.arange(bins.max() + 1)
                                                    *Forest.delta_log_lambda)
        ## old way rebinning
        #rebin_ivar = np.zeros(bins.max() + 1)
        #cciv = np.bincount(bins,weights=ivar)
        #rebin_ivar[:len(cciv)] += cciv
        ## end of old way rebinning
        rebin_ivar = np.bincount(bins,weights=ivar)
        w = (rebin_ivar>0.)
        self.log_lambda = rebin_log_lambda[w]
        self.ivar = rebin_ivar[w]

        # rebin using inverse variance weighting
        for key, value in ivar_coadd_data.items():
            ## old way rebinning
            #cnew = np.zeros(bins.max() + 1)
            #ccnew = np.bincount(bins, weights=ivar * value)
            #cnew[:len(ccnew)] += ccnew
            #setattr(self, key, cnew[w] / rebin_ivar[w])
            ## end of old way rebinning
            rebin_value = np.bincount(bins, weights=ivar * value)
            rebin_value = rebin_value[w]/rebin_ivar[w])
            setattr(self, key, rebin_value)

        # recompute means of quality variables
        if self.reso is not None:
            self.mean_reso = self.reso.mean()
        err = 1./sp.sqrt(self.ivar)
        snr = self.flux/err
        self.mean_snr = snr.mean()
        lambda_igm_absorption = constants.absorber_IGM[self.igm_absorption]
        self.mean_z = ((np.power(10., log_lambda[len(log_lambda) - 1]) +
                        bp.power(10.,log_lambda[0]))/2./lambda_igm_absorption
                        - 1.0)

        return self

    def mask(self,mask_obs,mask_RF):
        if not hasattr(self,'log_lambda'):
            return

        w = sp.ones(self.log_lambda.size,dtype=bool)
        for l in mask_obs:
            w &= (self.log_lambda<l[0]) | (self.log_lambda>l[1])
        for l in mask_RF:
            w &= (self.log_lambda-sp.log10(1.+self.z_qso)<l[0]) | (self.log_lambda-sp.log10(1.+self.z_qso)>l[1])

        ps = ['ivar','log_lambda','flux','dla_transmission','mean_optical_depth','mean_expected_flux_frac','exposures_diff','reso']
        for p in ps:
            if hasattr(self,p) and (getattr(self,p) is not None):
                setattr(self,p,getattr(self,p)[w])

        return

    def add_optical_depth(self,tau,gamma,waveRF):
        """Add mean optical depth
        """
        if not hasattr(self,'log_lambda'):
            return

        if self.mean_optical_depth is None:
            self.mean_optical_depth = sp.ones(self.log_lambda.size)

        w = 10.**self.log_lambda/(1.+self.z_qso)<=waveRF
        z = 10.**self.log_lambda/waveRF-1.
        self.mean_optical_depth[w] *= sp.exp(-tau*(1.+z[w])**gamma)

        return

    def add_dla(self,zabs,nhi,mask=None):
        if not hasattr(self,'log_lambda'):
            return
        if self.dla_transmission is None:
            self.dla_transmission = sp.ones(len(self.log_lambda))

        self.dla_transmission *= dla(self,zabs,nhi).t

        w = self.dla_transmission>Forest.dla_mask_limit
        if not mask is None:
            for l in mask:
                w &= (self.log_lambda-sp.log10(1.+zabs)<l[0]) | (self.log_lambda-sp.log10(1.+zabs)>l[1])

        ps = ['ivar','log_lambda','flux','dla_transmission','mean_optical_depth','mean_expected_flux_frac','exposures_diff','reso']
        for p in ps:
            if hasattr(self,p) and (getattr(self,p) is not None):
                setattr(self,p,getattr(self,p)[w])

        return

    def add_absorber(self,lambda_absorber):
        if not hasattr(self,'log_lambda'):
            return

        w = sp.ones(self.log_lambda.size, dtype=bool)
        w &= sp.fabs(1.e4*(self.log_lambda-sp.log10(lambda_absorber)))>Forest.absorber_mask_width

        ps = ['ivar','log_lambda','flux','dla_transmission','mean_optical_depth','mean_expected_flux_frac','exposures_diff','reso']
        for p in ps:
            if hasattr(self,p) and (getattr(self,p) is not None):
                setattr(self,p,getattr(self,p)[w])

        return

    def cont_fit(self):
        log_lambda_max = Forest.log_lambda_max_rest_frame+sp.log10(1+self.z_qso)
        log_lambda_min = Forest.log_lambda_min_rest_frame+sp.log10(1+self.z_qso)
        try:
            mean_cont = Forest.get_mean_cont(self.log_lambda-sp.log10(1+self.z_qso))
        except ValueError:
            raise Exception

        if not self.mean_optical_depth is None:
            mean_cont *= self.mean_optical_depth
        if not self.dla_transmission is None:
            mean_cont*=self.dla_transmission

        var_lss = Forest.get_var_lss(self.log_lambda)
        eta = Forest.get_eta(self.log_lambda)
        fudge = Forest.fudge(self.log_lambda)

        def model(p0,p1):
            line = p1*(self.log_lambda-log_lambda_min)/(log_lambda_max-log_lambda_min)+p0
            return line*mean_cont

        def chi2(p0,p1):
            m = model(p0,p1)
            var_pipe = 1./self.ivar/m**2
            ## prep_del.variance is the variance of delta
            ## we want here the we = ivar(flux)

            var_tot = variance(var_pipe,eta,var_lss,fudge)
            we = 1/m**2/var_tot

            # force we=1 when use-constant-weight
            # TODO: make this condition clearer, maybe pass an option
            # use_constant_weights?
            if (eta==0).all() :
                we=sp.ones(len(we))
            v = (self.flux-m)**2*we
            return v.sum()-sp.log(we).sum()

        p0 = (self.flux*self.ivar).sum()/self.ivar.sum()
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

    def __init__(self,thingid,ra,dec,z_qso,plate,mjd,fiberid,log_lambda,we,co,de,order,ivar,exposures_diff,mean_snr,m_reso,m_z,delta_log_lambda):

        Qso.__init__(self,thingid,ra,dec,z_qso,plate,mjd,fiberid)
        self.log_lambda = log_lambda
        self.we = we
        self.co = co
        self.de = de
        self.order = order
        self.ivar = ivar
        self.exposures_diff = exposures_diff
        self.mean_snr = mean_snr
        self.mean_reso = m_reso
        self.mean_z = m_z
        self.delta_log_lambda = delta_log_lambda

    @classmethod
    def from_forest(cls,f,st,get_var_lss,get_eta,fudge,mc=False):

        log_lambda = f.log_lambda
        mst = st(log_lambda)
        var_lss = get_var_lss(log_lambda)
        eta = get_eta(log_lambda)
        fudge = fudge(log_lambda)

        #if mc is True use the mock continuum to compute the mean expected flux fraction
        if mc : mef = f.mean_expected_flux_frac
        else : mef = f.co * mst
        de = f.flux/ mef -1.
        var = 1./f.ivar/mef**2
        we = 1./variance(var,eta,var_lss,fudge)
        exposures_diff = f.exposures_diff
        if f.exposures_diff is not None:
            exposures_diff /= mef
        ivar = f.ivar/(eta+(eta==0))*(mef**2)

        return cls(f.thingid,f.ra,f.dec,f.z_qso,f.plate,f.mjd,f.fiberid,log_lambda,we,f.co,de,f.order,
                   ivar,exposures_diff,f.mean_snr,f.mean_reso,f.mean_z,f.delta_log_lambda)


    @classmethod
    def from_fitsio(cls,h,Pk1D_type=False):


        head = h.read_header()

        de = h['DELTA'][:]
        log_lambda = h['LOGLAM'][:]


        if  Pk1D_type :
            ivar = h['IVAR'][:]
            exposures_diff = h['DIFF'][:]
            mean_snr = head['MEANSNR']
            m_reso = head['MEANRESO']
            m_z = head['MEANZ']
            delta_log_lambda =  head['DLL']
            we = None
            co = None
        else :
            ivar = None
            exposures_diff = None
            mean_snr = None
            m_reso = None
            delta_log_lambda = None
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
                   ivar,exposures_diff,mean_snr,m_reso,m_z,delta_log_lambda)


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
        mean_snr = float(a[7])
        m_reso = float(a[8])
        delta_log_lambda = float(a[9])

        nbpixel = int(a[10])
        de = sp.array(a[11:11+nbpixel]).astype(float)
        log_lambda = sp.array(a[11+nbpixel:11+2*nbpixel]).astype(float)
        ivar = sp.array(a[11+2*nbpixel:11+3*nbpixel]).astype(float)
        exposures_diff = sp.array(a[11+3*nbpixel:11+4*nbpixel]).astype(float)


        thingid = 0
        order = 0
        we = None
        co = None

        return cls(thingid,ra,dec,z_qso,plate,mjd,fiberid,log_lambda,we,co,de,order,
                   ivar,exposures_diff,mean_snr,m_reso,m_z,delta_log_lambda)

    @staticmethod
    def from_image(f):
        h=fitsio.FITS(f)
        de = h[0].read()
        ivar = h[1].read()
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
            aux_ivar = flux[:,i]
            w = aux_ivar>0
            delt = delt[w]
            aux_ivar = aux_ivar[w]
            lam = log_lambda[w]

            order = 1
            exposures_diff = None
            mean_snr = None
            m_reso = None
            delta_log_lambda = None
            m_z = None

            deltas.append(delta(thingid[i],ra[i],dec[i],z[i],plate[i],mjd[i],fiberid[i],lam,aux_ivar,None,delt,order,ivar,exposures_diff,mean_snr,m_reso,m_z,delta_log_lambda))

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
