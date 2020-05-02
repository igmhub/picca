"""This module defines data structure to deal with line of sight data.

This module provides with three classes (QSO, Forest, Delta) and one
function (get_variance) to manage the line-of-sight data. See the respective
docstrings for more details
"""
import numpy as np
import scipy as sp
from picca import constants
from picca.utils import userprint, unred
import iminuit
from picca.dla import DLA
import fitsio


def get_variance(var_pipe, eta, var_lss, fudge):
    """Computes the total variance.

    This includes contributions from pipeline noise, Large Scale Structure
    variance, and a fudge contribution.

    Args:
        var_pipe: array
            Pipeline variance
        eta: array
            Correction factor to the contribution of the pipeline estimate of
            the instrumental noise to the variance.
        var_lss: array
            Pixel variance due to the Large Scale Strucure
        fudge: array
            Fudge contribution to the pixel variance

    Returns:
        The total variance
    """
    return eta*var_pipe + var_lss + fudge/var_pipe


class QSO:
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
    def __init__(self, thingid, ra, dec, z_qso, plate, mjd, fiberid):
        """Initializes class instance.

        Args:
            thingid: integer
                Thingid of the observation.
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
        """
        self.ra = ra
        self.dec = dec

        self.plate = plate
        self.mjd = mjd
        self.fiberid = fiberid

        ## cartesian coordinates
        self.x_cart = sp.cos(ra)*sp.cos(dec)
        self.y_cart = sp.sin(ra)*sp.cos(dec)
        self.z_cart = sp.sin(dec)
        self.cos_dec = sp.cos(dec)

        self.z_qso = z_qso
        self.thingid = thingid

    # TODO: rename method, update class docstring
    def __xor__(self, data):
        """Computes the angular separation between two quasars.

        Args:
            data: QSO or list of QSO
                Objects with which the angular separation will
                be computed.

        Returns
            A float or an array (depending on input data) with the angular
            separation between this quasar and the object(s) in data.
        """
        # case 1: data is list-like
        try:
            x_cart = sp.array([d.x_cart for d in data])
            y_cart = sp.array([d.y_cart for d in data])
            z_cart = sp.array([d.z_cart for d in data])
            ra = sp.array([d.ra for d in data])
            dec = sp.array([d.dec for d in data])

            cos = x_cart*self.x_cart + y_cart*self.y_cart + z_cart*self.z_cart
            w = cos >= 1.
            if w.sum() != 0:
                userprint('WARNING: {} pairs have cos>=1.'.format(w.sum()))
                cos[w] = 1.
            w = cos <= -1.
            if w.sum() != 0:
                userprint('WARNING: {} pairs have cos<=-1.'.format(w.sum()))
                cos[w] = -1.
            angl = sp.arccos(cos)

            w = ((np.absolute(ra - self.ra) < constants.small_angle_cut_off) &
                 (np.absolute(dec - self.dec) < constants.small_angle_cut_off))
            if w.sum() != 0:
                angl[w] = sp.sqrt((dec[w] - self.dec)**2 +
                                  (self.cos_dec*(ra[w] - self.ra))**2)
        # case 2: data is a QSO
        except:
            x_cart = data.x_cart
            y_cart = data.y_cart
            z_cart = data.z_cart
            ra = data.ra
            dec = data.dec

            cos = x_cart*self.x_cart + y_cart*self.y_cart + z_cart*self.z_cart
            if cos >= 1.:
                userprint('WARNING: 1 pair has cosinus>=1.')
                cos = 1.
            elif cos <= -1.:
                userprint('WARNING: 1 pair has cosinus<=-1.')
                cos = -1.
            angl = sp.arccos(cos)
            if ((np.absolute(ra - self.ra) < constants.small_angle_cut_off) &
                    (np.absolute(dec - self.dec) <
                     constants.small_angle_cut_off)):
                angl = sp.sqrt((dec - self.dec)**2 +
                               (self.cos_dec*(ra - self.ra))**2)
        return angl


class Forest(QSO):
    """Class to represent a Lyman alpha (or other absorption) forest

    This class stores the information of an absorption forest.
    This includes the information required to extract the delta
    field from it: flux correction, inverse variance corrections,
    dlas, absorbers, ...

    Attributes:
        ## Inherits from QSO ##
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
        mean_expected_flux_frac: array of floats or None
            Mean expected flux fraction using the mock continuum
        order: 0 or 1
            Order of the log10(lambda) polynomial for the continuum fit
        exposures_diff: array of floats or None
            Difference between exposures
        reso: array of floats or None
            Resolution of the forest
        mean_snr: float or None
            Mean signal-to-noise ratio in the forest
        mean_reso: float or None
            Mean resolution of the forest
        mean_z: float or None
            Mean redshift of the forest
        continuum: array of floats or None
            Quasar continuum
        p0: float or None
            Zero point of the linear function (flux mean)
        p1: float or None
            Slope of the linear function (evolution of the flux)
        bad_cont: string or None
            Reason as to why the continuum fit is not acceptable
        igm_absorption: string
            Name of the absorption line in picca.constants defining the
            redshift of the forest pixels

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
        __init__: Initializes class instances.
        __add__: Adds the information of another forest.
        correct_flux: Corrects for multiplicative errors in pipeline flux
            calibration.
        correct_ivar: Corrects for multiplicative errors in pipeline inverse
            variance calibration.
        get_var_lss: Interpolates the pixel variance due to the Large Scale
            Strucure on the wavelength array.
        get_eta: Interpolates the correction factor to the contribution of the
            pipeline estimate of the instrumental noise to the variance on the
            wavelength array.
        get_fudge: Interpolates the fudge contribution to the variance on the
            wavelength array.
        get_mean_cont: Interpolates the mean quasar continuum over the whole
            sample on the wavelength array.
        mask: Applies wavelength masking.
        add_optical_depth: Adds the contribution of a given species to the mean
            optical depth.
        add_dla: Adds DLA to forest. Masks it by removing the afffected pixels.
        add_absorber: Adds absorber to forest. Masks it by removing the
            afffected pixels.
        cont_fit: Computes the forest continuum.
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
        """Interpolates the pixel variance due to the Large Scale Strucure on
        the wavelength array.

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
        """Interpolates the correction factor to the contribution of the
        pipeline estimate of the instrumental noise to the variance on the
        wavelength array.

        See equation 4 of du Mas des Bourboux et al. 2020 for details.

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
        """Interpolates the mean quasar continuum over the whole
        sample on the wavelength array.

        See equation 2 of du Mas des Bourboux et al. 2020 for details.

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
    def get_fudge(cls, lol_lambda):
        """Interpolates the fudge contribution to the variance on the
        wavelength array.

        See function epsilon in equation 4 of du Mas des Bourboux et al.
        2020 for details.

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
        """Initializes class instances.

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
        QSO.__init__(self, thingid, ra, dec, z_qso, plate, mjd, fiberid)

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
        """Adds the information of another forest.

        Forests are coadded by using inverse variance weighting.

        Args:
            other: Forest
                The forest instance to be coadded. If other does not have the
                attribute log_lambda, then the method returns without doing
                anything.

        Returns:
            The coadded forest.
        """
        if not hasattr(self, 'log_lambda') or not hasattr(other, 'log_lambda'):
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
        rebin_ivar = np.bincount(bins, weights=ivar)
        w = (rebin_ivar > 0.)
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
            rebin_value = rebin_value[w]/rebin_ivar[w]
            setattr(self, key, rebin_value)

        # recompute means of quality variables
        if self.reso is not None:
            self.mean_reso = self.reso.mean()
        err = 1./sp.sqrt(self.ivar)
        snr = self.flux/err
        self.mean_snr = snr.mean()
        lambda_igm_absorption = constants.absorber_IGM[self.igm_absorption]
        self.mean_z = ((np.power(10., log_lambda[len(log_lambda) - 1]) +
                        bp.power(10., log_lambda[0]))/2./lambda_igm_absorption
                       - 1.0)

        return self

    def mask(self, mask_obs_frame, mask_rest_frame):
        """Applies wavelength masking.

        Pixels are masked according to a set of lines both in observed frame
        and in the rest-frame. Masking is done by simply removing the pixels
        from the arrays. Does nothing if the forest doesn't have the attribute
        log_lambda set.

        Args:
            mask_obs_frame: array of arrays
                Each element of the array must contain an array of two floats
                that specify the range of wavelength to mask. Values given are
                the logarithm of the wavelength in Angstroms, and both values
                are included in the masking. These wavelengths are given at the
                obseved frame.
            mask_rest_frame: array of arrays
                Same as mask_obs_frame but for rest-frame wavelengths.
        """
        if not hasattr(self, 'log_lambda'):
            return

        w = np.ones(self.log_lambda.size, dtype=bool)
        for mask_range in mask_obs_frame:
            w &= ((self.log_lambda < mask_range[0]) |
                  (self.log_lambda > mask_range[1]))
        for mask_range in mask_rest_frame:
            rest_frame_log_lambda = self.log_lambda - np.log10(1. + self.z_qso)
            w &= ((rest_frame_log_lambda < mask_range[0]) |
                  (rest_frame_log_lambda > mask_range[1]))

        parameters = ['ivar', 'log_lambda', 'flux', 'dla_transmission',
                      'mean_optical_depth', 'mean_expected_flux_frac',
                      'exposures_diff', 'reso']
        for param in parameters:
            if hasattr(self, param) and (getattr(self, param) is not None):
                setattr(self, param, getattr(self, param)[w])

        return

    def add_optical_depth(self, tau, gamma, lambda_rest_frame):
        """Adds the contribution of a given species to the mean optical depth.

        Flux will be corrected by the mean optical depth. This correction is
        governed by the optical depth-flux relation:
            `F = exp(tau(1+z)^gamma)`

        Args:
            tau: float
            Mean optical depth

            gamma: float
            Optical depth redshift evolution. Optical depth evolves as
            `(1+z)^gamma`

            lambda_rest_frame: float
            Restframe wavelength of the element responsible for the absorption.
            In Angstroms
        """
        if not hasattr(self, 'log_lambda'):
            return

        if self.mean_optical_depth is None:
            self.mean_optical_depth = np.ones(self.log_lambda.size)

        w = 10.**self.log_lambda/(1. + self.z_qso) <= lambda_rest_frame
        z = 10.**self.log_lambda/lambda_rest_frame - 1.
        self.mean_optical_depth[w] *= sp.exp(-tau*(1. + z[w])**gamma)

        return

    def add_dla(self, z_abs, nhi, mask=None):
        """Adds DLA to forest. Masks it by removing the afffected pixels.

        Args:
            z_abs: float
            Redshift of the DLA absorption

            nhi : float
            DLA column density in log10(cm^-2)

            mask : list or None - Default None
            Wavelengths to be masked in DLA rest-frame wavelength
        """
        if not hasattr(self, 'log_lambda'):
            return
        if self.dla_transmission is None:
            self.dla_transmission = np.ones(len(self.log_lambda))

        self.dla_transmission *= DLA(self, z_abs, nhi).transmission

        w = self.dla_transmission > Forest.dla_mask_limit
        if not mask is None:
            for mask_range in mask:
                w &= ((self.log_lambda - np.log10(1. + z_abs) < mask_range[0]) |
                      (self.log_lambda - np.log10(1. + z_abs) > mask_range[1]))

        # do the actual masking
        parameters = ['ivar', 'log_lambda', 'flux', 'dla_transmission',
                      'mean_optical_depth', 'mean_expected_flux_frac',
                      'exposures_diff', 'reso']
        for param in parameters:
            if hasattr(self, param) and (getattr(self, param) is not None):
                setattr(self, param, getattr(self, param)[w])

        return

    def add_absorber(self, lambda_absorber):
        """Adds absorber to forest. Masks it by removing the afffected pixels.

        Args:
            lambda_absorber: float
                Wavelength of the absorber
        """
        if not hasattr(self, 'log_lambda'):
            return

        w = np.ones(self.log_lambda.size, dtype=bool)
        w &= (np.fabs(1.e4*(self.log_lambda - np.log10(lambda_absorber)))
              > Forest.absorber_mask_width)

        parameters = ['ivar', 'log_lambda', 'flux', 'dla_transmission',
                      'mean_optical_depth', 'mean_expected_flux_frac',
                      'exposures_diff', 'reso']
        for param in parameters:
            if hasattr(self, param) and (getattr(self, param) is not None):
                setattr(self, param, getattr(self, param)[w])

        return

    def cont_fit(self):
        """Computes the forest continuum.

        Fits a model based on the mean quasar continuum and linear function
        (see equation 2 of du Mas des Bourboux et al. 2020)
        Flags the forest with bad_cont if the computation fails.
        """
        log_lambda_max = (Forest.log_lambda_max_rest_frame +
                          np.log10(1 + self.z_qso))
        log_lambda_min = (Forest.log_lambda_min_rest_frame +
                          np.log10(1 + self.z_qso))
        # get mean continuum
        try:
            mean_cont = Forest.get_mean_cont(self.log_lambda -
                                             np.log10(1+self.z_qso))
        except ValueError:
            raise Exception("Problem found when loading get_mean_cont")

        # add the optical depth correction
        # (previously computed using method add_optical_depth)
        if not self.mean_optical_depth is None:
            mean_cont *= self.mean_optical_depth
        # add the dla transmission correction
        # (previously computed using method add_dla)
        if not self.dla_transmission is None:
            mean_cont *= self.dla_transmission

        # pixel variance due to the Large Scale Strucure
        var_lss = Forest.get_var_lss(self.log_lambda)
        # correction factor to the contribution of the pipeline
        # estimate of the instrumental noise to the variance.
        eta = Forest.get_eta(self.log_lambda)
        # fudge contribution to the variance
        fudge = Forest.get_fudge(self.log_lambda)

        def get_continuum_model(p0, p1):
            """Models the flux continuum by multiplying the mean_continuum
            by a linear function

            Args:
                p0: float
                    Zero point of the linear function (flux mean)
                p1: float
                    Slope of the linear function (evolution of the flux)

            Global args (defined only in the scope of function cont_fit)
                log_lambda_min: float
                    Minimum logarithm of the wavelength (in Angs)
                log_lambda_max: float
                    Minimum logarithm of the wavelength (in Angs)
                mean_cont: array
                    Mean continuum
            """
            line = (p1*(self.log_lambda - log_lambda_min)/
                    (log_lambda_max - log_lambda_min) + p0)
            return line*mean_cont

        def chi2(p0, p1):
            """Copmputes the chi2 of a given model (see function model above).

            Args:
                p1: float
                    Slope of the linear function (see function model above)

                p1: float
                    Zero point of the linear function (see function model above)

            Global args (defined only in the scope of function cont_fit)
                eta: array
                    Correction factor to the contribution of the pipeline
                    estimate of the instrumental noise to the variance.

            """
            continuum_model = get_continuum_model(p0, p1)
            var_pipe = 1./self.ivar/continuum_model**2
            ## prep_del.variance is the variance of delta
            ## we want here the weights = ivar(flux)

            variance = get_variance(var_pipe, eta, var_lss, fudge)
            weights = 1.0/continuum_model**2/variance

            # force weights=1 when use-constant-weight
            # TODO: make this condition clearer, maybe pass an option
            # use_constant_weights?
            if (eta == 0).all():
                weights = np.ones(len(weights))
            chi2_contribution = (self.flux - continuum_model)**2*weights
            return chi2_contribution.sum() - np.log(weights).sum()

        p0 = (self.flux*self.ivar).sum()/self.ivar.sum()
        p1 = 0.0

        minimizer = iminuit.Minuit(chi2, p0=p0, p1=p1, error_p0=p0/2.,
                                   error_p1=p0/2., errordef=1., print_level=0,
                                   fix_p1=(self.order == 0))
        minimizer_result, _ = minimizer.migrad()

        self.continuum = get_continuum_model(minimizer.values["p0"],
                                             minimizer.values["p1"])
        self.p0 = minimizer.values["p0"]
        self.p1 = minimizer.values["p1"]

        self.bad_cont = None
        if not minimizer_result.is_valid:
            self.bad_cont = "minuit didn't converge"
        if np.any(self.continuum <= 0):
            self.bad_cont = "negative continuum"


        ## if the continuum is negative, then set it to a very small number
        ## so that this forest is ignored
        if self.bad_cont is not None:
            self.continuum = self.continuum*0 + 1e-10
            self.p0 = 0.
            self.p1 = 0.


class Delta(QSO):
    #TODO: revise and update docstring
    """Class to represent the mean transimission fluctuation field (delta)

    This class stores the information for the deltas for a given line of sight

    Attributes:
        ## Inherits from QSO ##
        log_lambda : array of floats
            Array containing the logarithm of the wavelengths (in Angs)
        continuum: array of floats
            Quasar continuum
        delta: array of floats
            Mean transmission fluctuation (delta field)
        order: 0 or 1
            Order of the log10(lambda) polynomial for the continuum fit
        ivar: array of floats
            Inverse variance associated to each flux
        exposures_diff: array of floats
            Difference between exposures
        mean_snr: float
            Mean signal-to-noise ratio in the forest
        mean_reso: float
            Mean resolution of the forest
        mean_z: float
            Mean redshift of the forest
        delta_log_lambda: float
            Variation of the logarithm of the wavelength between two pixels

    Methods:
        __init__: Initializes class instances.
    """
    def __init__(self, thingid, ra, dec, z_qso, plate, mjd, fiberid, log_lambda,
                 weights, continuum, delta, order, ivar, exposures_diff, mean_snr,
                 mean_reso, mean_z, delta_log_lambda):
        """Initializes class instances.

        Args:
            thingid: integer
                Thingid of the observation.
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
            log_lambda: array of floats
                Logarithm of the wavelengths (in Angs)
            weights: array of floats
                Pixel weights
            continuum: array of floats
                Quasar continuum
            delta: array of floats
                Mean transmission fluctuation (delta field)
            order: 0 or 1
                Order of the log10(lambda) polynomial for the continuum fit
            ivar: array of floats
                Inverse variance associated to each flux
            exposures_diff: array of floats
                Difference between exposures
            mean_snr: float
                Mean signal-to-noise ratio in the forest
            mean_reso: float
                Mean resolution of the forest
            mean_z: float
                Mean redshift of the forest
            delta_log_lambda: float
                Variation of the logarithm of the wavelength between two pixels
        """
        QSO.__init__(self, thingid, ra, dec, z_qso, plate, mjd, fiberid)
        self.log_lambda = log_lambda
        self.weights = weights
        self.continuum = continuum
        self.delta = delta
        self.order = order
        self.ivar = ivar
        self.exposures_diff = exposures_diff
        self.mean_snr = mean_snr
        self.mean_reso = mean_reso
        self.mean_z = mean_z
        self.delta_log_lambda = delta_log_lambda

    @classmethod
    def from_forest(cls, forest, get_mean_delta, get_var_lss, get_eta,
                    get_fudge, use_mock_continuum=False):
        """Initialize instance from Forest data.

        Args:
            forest: Forest
                A forest instance from which to initialize the deltas
            get_mean_delta: function
                Interpolates the mean value of the delta field on the wavelength
                array.
            get_var_lss: Interpolates the pixel variance due to the Large Scale
                Strucure on the wavelength array.
            get_eta: Interpolates the correction factor to the contribution of the
                pipeline estimate of the instrumental noise to the variance on the
                wavelength array.
            get_fudge: Interpolates the fudge contribution to the variance on the
                wavelength array.
            use_mock_continuum: bool - default: False
                Flag to use the mock continuum to compute the mean expected
                flux fraction

        Returns:
            a Delta instance
        """
        log_lambda = forest.log_lambda
        mean_delta = get_mean_delta(log_lambda)
        var_lss = get_var_lss(log_lambda)
        eta = get_eta(log_lambda)
        fudge = get_fudge(log_lambda)

        #if mc is True use the mock continuum to compute the mean
        # expected flux fraction
        if use_mock_continuum:
            mean_expected_flux_frac = forest.mean_expected_flux_frac
        else:
            mean_expected_flux_frac = forest.continuum*mean_delta
        delta = forest.flux/mean_expected_flux_frac - 1.
        var_pipe = 1./forest.ivar/mean_expected_flux_frac**2
        weights = 1./get_variance(var_pipe, eta, var_lss, fudge)
        exposures_diff = forest.exposures_diff
        if forest.exposures_diff is not None:
            exposures_diff /= mean_expected_flux_frac
        ivar = forest.ivar/(eta + (eta == 0))*(mean_expected_flux_frac**2)

        return cls(forest.thingid, forest.ra, forest.dec, forest.z_qso,
                   forest.plate, forest.mjd, forest.fiberid, log_lambda,
                   weights, forest.continuum, delta, forest.order, ivar,
                   exposures_diff, forest.mean_snr, forest.mean_reso,
                   forest.mean_z, forest.delta_log_lambda)


    @classmethod
    def from_fitsio(cls,h, pk1d_type=False):
        """Initialize instance from a fits file

        Args:
            hdu: fitsio.hdu.table.TableHDU
                A Header Data Unit opened with fitsio
            pk1d_type: bool - default: False
                Specifies if the fits file is formatted for the 1D Power
                Spectrum analysis
        Returns:
            a Delta instance
        """
        header = hdu.read_header()

        delta = hdu['DELTA'][:]
        log_lambda = hdu['LOGLAM'][:]


        if pk1d_type:
            ivar = hdu['IVAR'][:]
            exposures_diff = hdu['DIFF'][:]
            mean_snr = header['MEANSNR']
            mean_reso = header['MEANRESO']
            mean_z = header['MEANZ']
            delta_log_lambda = header['DLL']
            weights = None
            continuum = None
        else:
            ivar = None
            exposures_diff = None
            mean_snr = None
            mean_reso = None
            delta_log_lambda = None
            mean_z = None
            weights = hdu['WEIGHT'][:]
            continuum = hdu['CONT'][:]

        thingid = header['THING_ID']
        ra = header['RA']
        dec = header['DEC']
        z_qso = header['Z']
        plate = header['PLATE']
        mjd = header['MJD']
        fiberid = header['FIBERID']
        try:
            order = header['ORDER']
        except KeyError:
            order = 1

        return cls(thingid, ra, dec, z_qso, plate, mjd, fiberid, log_lambda,
                   weights,continuum,delta,order, ivar, exposures_diff,
                   mean_snr, mean_reso, mean_z, delta_log_lambda)


    @classmethod
    def from_ascii(cls,line):

        a = line.split()
        plate = int(a[0])
        mjd = int(a[1])
        fiberid = int(a[2])
        ra = float(a[3])
        dec = float(a[4])
        z_qso = float(a[5])
        mean_z = float(a[6])
        mean_snr = float(a[7])
        mean_reso = float(a[8])
        delta_log_lambda = float(a[9])

        nbpixel = int(a[10])
        delta = sp.array(a[11:11+nbpixel]).astype(float)
        log_lambda = sp.array(a[11+nbpixel:11+2*nbpixel]).astype(float)
        ivar = sp.array(a[11+2*nbpixel:11+3*nbpixel]).astype(float)
        exposures_diff = sp.array(a[11+3*nbpixel:11+4*nbpixel]).astype(float)


        thingid = 0
        order = 0
        weights = None
        continuum = None

        return cls(thingid,ra,dec,z_qso,plate,mjd,fiberid,log_lambda,weights,continuum,delta,order,
                   ivar,exposures_diff,mean_snr,mean_reso,mean_z,delta_log_lambda)

    @staticmethod
    def from_image(f):
        h=fitsio.FITS(f)
        deltas_image = h[0].read()
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

            delta = deltas_image[:,i]
            aux_ivar = flux[:,i]
            w = aux_ivar>0
            delta = delta[w]
            aux_ivar = aux_ivar[w]
            lam = log_lambda[w]

            order = 1
            exposures_diff = None
            mean_snr = None
            mean_reso = None
            delta_log_lambda = None
            mean_z = None

            deltas.append(Delta(thingid[i],ra[i],dec[i],z[i],plate[i],mjd[i],fiberid[i],lam,aux_ivar,None,delta,order,ivar,exposures_diff,mean_snr,mean_reso,mean_z,delta_log_lambda))

        h.close()
        return deltas


    def project(self):
        mde = sp.average(self.delta,weights=self.weights)
        res=0
        if (self.order==1) and self.delta.shape[0] > 1:
            mll = sp.average(self.log_lambda,weights=self.weights)
            mld = sp.sum(self.weights*self.delta*(self.log_lambda-mll))/sp.sum(self.weights*(self.log_lambda-mll)**2)
            res = mld * (self.log_lambda-mll)
        elif self.order==1:
            res = self.delta

        self.delta -= mde + res
