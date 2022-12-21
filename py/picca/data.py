"""This module defines data structure to deal with line of sight data.

This module provides with three classes (QSO, Forest, Delta)
to manage the line-of-sight data.
See the respective docstrings for more details
"""
import numpy as np
import iminuit
from itertools import repeat
import fitsio
import warnings

from . import constants
from .utils import userprint, unred
from .dla import DLA

class QSO(object):
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
        weights: float
            Weight assigned to object
        r_comov: float or None
            Comoving distance to the object
        dist_m: float or None
            Angular diameter distance to object
        log_lambda: float or None
            Wavelength associated with the quasar redshift

    Note that plate-fiberid-mjd is a unique identifier
    for the quasar.

    Methods:
        __init__: Initialize class instance.
        get_angle_between: Computes the angular separation between two quasars.
    """

    def __init__(self, los_id, ra, dec, z_qso, plate, mjd, fiberid):
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
        self.x_cart = np.cos(ra) * np.cos(dec)
        self.y_cart = np.sin(ra) * np.cos(dec)
        self.z_cart = np.sin(dec)
        self.cos_dec = np.cos(dec)

        self.z_qso = z_qso
        self.los_id = los_id
        #this is for legacy purposes only
        self.thingid = los_id
        warnings.warn("currently a thingid entry is created in QSO.__init__, this feature will be removed", DeprecationWarning)

        # variables computed in function io.read_objects
        self.weight = None
        self.r_comov = None
        self.dist_m = None

        # variables computed in modules bin.picca_xcf_angl and bin.picca_xcf1d
        self.log_lambda = None

    def get_angle_between(self, data):
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
            x_cart = np.array([d.x_cart for d in data])
            y_cart = np.array([d.y_cart for d in data])
            z_cart = np.array([d.z_cart for d in data])
            ra = np.array([d.ra for d in data])
            dec = np.array([d.dec for d in data])

            cos = x_cart * self.x_cart + y_cart * self.y_cart + z_cart * self.z_cart
            w = cos >= 1.
            if w.sum() != 0:
                userprint('WARNING: {} pairs have cos>=1.'.format(w.sum()))
                cos[w] = 1.
            w = cos <= -1.
            if w.sum() != 0:
                userprint('WARNING: {} pairs have cos<=-1.'.format(w.sum()))
                cos[w] = -1.
            angl = np.arccos(cos)

            w = ((np.absolute(ra - self.ra) < constants.SMALL_ANGLE_CUT_OFF) &
                 (np.absolute(dec - self.dec) < constants.SMALL_ANGLE_CUT_OFF))
            if w.sum() != 0:
                angl[w] = np.sqrt((dec[w] - self.dec)**2 +
                                  (self.cos_dec * (ra[w] - self.ra))**2)
        # case 2: data is a QSO
        except TypeError:
            x_cart = data.x_cart
            y_cart = data.y_cart
            z_cart = data.z_cart
            ra = data.ra
            dec = data.dec

            cos = x_cart * self.x_cart + y_cart * self.y_cart + z_cart * self.z_cart
            if cos >= 1.:
                userprint('WARNING: 1 pair has cosinus>=1.')
                cos = 1.
            elif cos <= -1.:
                userprint('WARNING: 1 pair has cosinus<=-1.')
                cos = -1.
            angl = np.arccos(cos)
            if ((np.absolute(ra - self.ra) < constants.SMALL_ANGLE_CUT_OFF) &
                    (np.absolute(dec - self.dec) < constants.SMALL_ANGLE_CUT_OFF)):
                angl = np.sqrt((dec - self.dec)**2 + (self.cos_dec *
                                                      (ra - self.ra))**2)
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
        cont: array of floats or None
            Quasar continuum
        p0: float or None
            Zero point of the linear function (flux mean)
        p1: float or None
            Slope of the linear function (evolution of the flux)
        bad_cont: string or None
            Reason as to why the continuum fit is not acceptable
        abs_igm: string
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
            log_lambda: array of float
                Array containing the logarithm of the wavelengths (in Angs)

        Returns:
            An array with the correction

        Raises:
            NotImplementedError: Function was not specified
        """
        raise NotImplementedError("Function should be specified at run-time")

    @classmethod
    def correct_ivar(cls, log_lambda):
        """Corrects for multiplicative errors in pipeline inverse variance
           calibration.

        Empty function to be loaded at run-time.

        Args:
            log_lambda: array of float
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
    def get_var_lss(cls, log_lambda):
        """Interpolates the pixel variance due to the Large Scale Strucure on
        the wavelength array.

        Empty function to be loaded at run-time.

        Args:
            log_lambda: array of float
                Array containing the logarithm of the wavelengths (in Angs)

        Returns:
            An array with the correction

        Raises:
            NotImplementedError: Function was not specified
        """
        raise NotImplementedError("Function should be specified at run-time")

    @classmethod
    def get_eta(cls, log_lambda):
        """Interpolates the correction factor to the contribution of the
        pipeline estimate of the instrumental noise to the variance on the
        wavelength array.

        See equation 4 of du Mas des Bourboux et al. 2020 for details.

        Empty function to be loaded at run-time.

        Args:
            log_lambda: array of float
                Array containing the logarithm of the wavelengths (in Angs)

        Returns:
            An array with the correction

        Raises:
            NotImplementedError: Function was not specified
        """
        raise NotImplementedError("Function should be specified at run-time")

    @classmethod
    def get_mean_cont(cls, log_lambda):
        """Interpolates the mean quasar continuum over the whole
        sample on the wavelength array.

        See equation 2 of du Mas des Bourboux et al. 2020 for details.

        Empty function to be loaded at run-time.

        Args:
            log_lambda: array of float
                Array containing the logarithm of the wavelengths (in Angs)

        Returns:
            An array with the correction

        Raises:
            NotImplementedError: Function was not specified
        """
        raise NotImplementedError("Function should be specified at run-time")

    @classmethod
    def get_fudge(cls, log_lambda):
        """Interpolates the fudge contribution to the variance on the
        wavelength array.

        See function epsilon in equation 4 of du Mas des Bourboux et al.
        2020 for details.

        Args:
            log_lambda: array of float
                Array containing the logarithm of the wavelengths (in Angs)

        Returns:
            An array with the correction

        Raises:
            NotImplementedError: Function was not specified
        """
        raise NotImplementedError("Function should be specified at run-time")

    def __init__(self,
                 log_lambda,
                 flux,
                 ivar,
                 thingid,
                 ra,
                 dec,
                 z_qso,
                 plate,
                 mjd,
                 fiberid,
                 exposures_diff=None,
                 reso=None,
                 mean_expected_flux_frac=None,
                 abs_igm="LYA"):
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
            exposures_diff: array of floats or None - default: None
                Difference between exposures.
            reso: array of floats or None - default: None
                Resolution of the forest.
            mean_expected_flux_frac: array of floats or None - default: None
                Mean expected flux fraction using the mock continuum
            abs_igm: string - default: "LYA"
                Name of the absorption in picca.constants defining the
                redshift of the forest pixels
        """
        QSO.__init__(self, thingid, ra, dec, z_qso, plate, mjd, fiberid)


        ## cut to specified range
        bins = (np.floor((log_lambda - Forest.log_lambda_min) /
                         Forest.delta_log_lambda + 0.5).astype(int))
        log_lambda = Forest.log_lambda_min + bins * Forest.delta_log_lambda
        w = (log_lambda >= Forest.log_lambda_min)
        w = w & (log_lambda < Forest.log_lambda_max)
        w = w & (log_lambda - np.log10(1. + self.z_qso) >
                 Forest.log_lambda_min_rest_frame)
        w = w & (log_lambda - np.log10(1. + self.z_qso) <
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
        rebin_log_lambda = (Forest.log_lambda_min +
                            np.arange(bins.max() + 1) * Forest.delta_log_lambda)
        rebin_flux = np.zeros(bins.max() + 1)
        rebin_ivar = np.zeros(bins.max() + 1)
        if mean_expected_flux_frac is not None:
            rebin_mean_expected_flux_frac = np.zeros(bins.max() + 1)
        rebin_flux_aux = np.bincount(bins, weights=ivar * flux)
        rebin_ivar_aux = np.bincount(bins, weights=ivar)
        if mean_expected_flux_frac is not None:
            rebin_mean_expected_flux_frac_aux = np.bincount(
                bins, weights=ivar * mean_expected_flux_frac)
        if exposures_diff is not None:
            rebin_exposures_diff = np.bincount(bins,
                                               weights=ivar * exposures_diff)
        if reso is not None:
            rebin_reso = np.bincount(bins, weights=ivar * reso)
        rebin_flux[:len(rebin_flux_aux)] += rebin_flux_aux
        rebin_ivar[:len(rebin_ivar_aux)] += rebin_ivar_aux
        if mean_expected_flux_frac is not None:
            rebin_mean_expected_flux_frac[:len(
                rebin_mean_expected_flux_frac_aux
            )] += rebin_mean_expected_flux_frac_aux
        w = (rebin_ivar > 0.)
        if w.sum() == 0:
            return
        log_lambda = rebin_log_lambda[w]
        flux = rebin_flux[w] / rebin_ivar[w]
        ivar = rebin_ivar[w]
        if mean_expected_flux_frac is not None:
            mean_expected_flux_frac = (rebin_mean_expected_flux_frac[w] /
                                       rebin_ivar[w])
        if exposures_diff is not None:
            exposures_diff = rebin_exposures_diff[w] / rebin_ivar[w]
        if reso is not None:
            reso = rebin_reso[w] / rebin_ivar[w]

        # apply dust extinction correction
        if Forest.extinction_bv_map is not None:
            corr = unred(10**log_lambda, Forest.extinction_bv_map[thingid])
            flux /= corr
            ivar *= corr**2
            if not exposures_diff is None:
                exposures_diff /= corr

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
        self.exposures_diff = exposures_diff
        self.reso = reso
        self.abs_igm = abs_igm

        # compute mean quality variables
        if reso is not None:
            self.mean_reso = reso.mean()
        else:
            self.mean_reso = None

        error = 1.0 / np.sqrt(ivar)
        snr = flux / error
        # TODO: change mean_snr_save to mean_snr.
        # Begore that, check implications on the different computation of mean_snr
        # a 'more correct' way of computed is stored in mean_snr_save and
        # saved in the metadata file, but we need to check how changes
        # are propagated through the analysis.
        self.mean_snr_save = np.average(snr, weights=self.ivar)
        self.mean_snr = snr.mean()
        lambda_abs_igm = constants.ABSORBER_IGM[self.abs_igm]
        self.mean_z = ((np.power(10., log_lambda[len(log_lambda) - 1]) +
                        np.power(10., log_lambda[0])) / 2. / lambda_abs_igm -
                       1.0)

        # continuum-related variables
        self.cont = None
        self.p0 = None
        self.p1 = None
        self.bad_cont = None
        self.order = None

    def coadd(self, other):
        """Coadds the information of another forest.

        Forests are coadded by using inverse variance weighting.

        Args:
            other: Forest
                The forest instance to be coadded. If other does not have the
                attribute log_lambda, then the method returns without doing
                anything.

        Returns:
            The coadded forest.
        """
        if self.log_lambda is None or other.log_lambda is None:
            if other.log_lambda is None:
                return self
            else:
                return other

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
            ivar_coadd_data['exposures_diff'] = np.append(
                self.exposures_diff, other.exposures_diff)
        if self.reso is not None:
            ivar_coadd_data['reso'] = np.append(self.reso, other.reso)

        # coadd the deltas by rebinning
        bins = np.floor((log_lambda - Forest.log_lambda_min) /
                        Forest.delta_log_lambda + 0.5).astype(int)
        rebin_log_lambda = Forest.log_lambda_min + (np.arange(bins.max() + 1) *
                                                    Forest.delta_log_lambda)
        rebin_ivar = np.zeros(bins.max() + 1)
        rebin_ivar_aux = np.bincount(bins, weights=ivar)
        rebin_ivar[:len(rebin_ivar_aux)] += rebin_ivar_aux
        w = (rebin_ivar > 0.)
        self.log_lambda = rebin_log_lambda[w]
        self.ivar = rebin_ivar[w]

        # rebin using inverse variance weighting
        for key, value in ivar_coadd_data.items():
            rebin_value = np.zeros(bins.max() + 1)
            rebin_value_aux = np.bincount(bins, weights=ivar * value)
            rebin_value[:len(rebin_value_aux)] += rebin_value_aux
            setattr(self, key, rebin_value[w] / rebin_ivar[w])

        # recompute means of quality variables
        if self.reso is not None:
            self.mean_reso = self.reso.mean()
        error = 1. / np.sqrt(self.ivar)
        snr = self.flux / error
        # TODO: change mean_snr_save to mean_snr.
        self.mean_snr_save = np.average(snr, weights=self.ivar)
        self.mean_snr = snr.mean()
        lambda_abs_igm = constants.ABSORBER_IGM[self.abs_igm]
        self.mean_z = ((np.power(10., log_lambda[len(log_lambda) - 1]) +
                        np.power(10., log_lambda[0])) / 2. / lambda_abs_igm -
                       1.0)

        return self

    def mask(self, mask_table):
        """Applies wavelength masking.

        Pixels are masked according to a set of lines both in observed frame
        and in the rest-frame. Masking is done by simply removing the pixels
        from the arrays. Does nothing if the forest doesn't have the attribute
        log_lambda set.

        Args:
            mask_table: astropy table
                Table containing minimum and maximum wavelenths of absorption
                lines to mask (in both rest frame and observed frame)
        """
        if len(mask_table)==0:
            return

        select_rest_frame_mask = mask_table['frame'] == 'RF'
        select_obs_mask = mask_table['frame'] == 'OBS'

        mask_rest_frame = mask_table[select_rest_frame_mask]
        mask_obs_frame = mask_table[select_obs_mask]

        if len(mask_rest_frame)+len(mask_obs_frame)==0:
            return

        if self.log_lambda is None:
            return

        w = np.ones(self.log_lambda.size, dtype=bool)
        for mask_range in mask_obs_frame:
            w &= ((self.log_lambda < mask_range['log_wave_min']) |
                  (self.log_lambda > mask_range['log_wave_max']))
        for mask_range in mask_rest_frame:
            rest_frame_log_lambda = self.log_lambda - np.log10(1. + self.z_qso)
            w &= ((rest_frame_log_lambda < mask_range['log_wave_min']) |
                  (rest_frame_log_lambda > mask_range['log_wave_max']))

        parameters = [
            'ivar', 'log_lambda', 'flux', 'dla_transmission',
            'mean_optical_depth', 'mean_expected_flux_frac', 'exposures_diff',
            'reso'
        ]
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
        if self.log_lambda is None:
            return

        if self.mean_optical_depth is None:
            self.mean_optical_depth = np.ones(self.log_lambda.size)

        w = 10.**self.log_lambda / (1. + self.z_qso) <= lambda_rest_frame
        z = 10.**self.log_lambda / lambda_rest_frame - 1.
        self.mean_optical_depth[w] *= np.exp(-tau * (1. + z[w])**gamma)

        return

    def add_dla(self, z_abs, nhi, mask_table=None):
        """Adds DLA to forest. Masks it by removing the afffected pixels.

        Args:
            z_abs: float
            Redshift of the DLA absorption

            nhi : float
            DLA column density in log10(cm^-2)

            mask_table : astropy table for masking
            Wavelengths to be masked in DLA rest-frame wavelength
        """


        if self.log_lambda is None:
            return
        if self.dla_transmission is None:
            self.dla_transmission = np.ones(len(self.log_lambda))

        self.dla_transmission *= DLA(self, z_abs, nhi).transmission

        w = self.dla_transmission > Forest.dla_mask_limit
        if len(mask_table)>0:
            select_dla_mask = mask_table['frame'] == 'RF_DLA'
            mask = mask_table[select_dla_mask]
            if len(mask)>0:
                for mask_range in mask:
                    w &= ((self.log_lambda - np.log10(1. + z_abs) < mask_range['log_wave_min']) |
                          (self.log_lambda - np.log10(1. + z_abs) > mask_range['log_wave_max']))

        # do the actual masking
        parameters = [
            'ivar', 'log_lambda', 'flux', 'dla_transmission',
            'mean_optical_depth', 'mean_expected_flux_frac', 'exposures_diff',
            'reso'
        ]
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
        if self.log_lambda is None:
            return

        w = np.ones(self.log_lambda.size, dtype=bool)
        w &= (np.fabs(1.e4 * (self.log_lambda - np.log10(lambda_absorber))) >
              Forest.absorber_mask_width)

        parameters = [
            'ivar', 'log_lambda', 'flux', 'dla_transmission',
            'mean_optical_depth', 'mean_expected_flux_frac', 'exposures_diff',
            'reso'
        ]
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
                                             np.log10(1 + self.z_qso))
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

        def get_cont_model(p0, p1):
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
                mean_cont: array of floats
                    Mean continuum
            """
            line = (p1 * (self.log_lambda - log_lambda_min) /
                    (log_lambda_max - log_lambda_min) + p0)
            return line * mean_cont

        def chi2(p0, p1):
            """Computes the chi2 of a given model (see function model above).

            Args:
                p0: float
                    Zero point of the linear function (see function model above)
                p1: float
                    Slope of the linear function (see function model above)

            Global args (defined only in the scope of function cont_fit)
                eta: array of floats
                    Correction factor to the contribution of the pipeline
                    estimate of the instrumental noise to the variance.

            Returns:
                The obtained chi2
            """
            cont_model = get_cont_model(p0, p1)
            var_pipe = 1. / self.ivar / cont_model**2
            ## prep_del.variance is the variance of delta
            ## we want here the weights = ivar(flux)

            variance = eta * var_pipe + var_lss + fudge / var_pipe
            weights = 1.0 / cont_model**2 / variance

            # force weights=1 when use-constant-weight
            # TODO: make this condition clearer, maybe pass an option
            # use_constant_weights?
            if (eta == 0).all():
                weights = np.ones(len(weights))
            chi2_contribution = (self.flux - cont_model)**2 * weights
            return chi2_contribution.sum() - np.log(weights).sum()

        p0 = (self.flux * self.ivar).sum() / self.ivar.sum()
        p1 = 0.0

        minimizer = iminuit.Minuit(chi2,
                                   p0=p0,
                                   p1=p1)
        minimizer.errors["p0"] = p0 / 2.
        minimizer.errors["p1"] = p0 / 2.
        minimizer.errordef = 1.
        minimizer.print_level = 0
        minimizer.fixed["p1"] = self.order == 0
        minimizer.migrad()

        self.cont = get_cont_model(minimizer.values["p0"],
                                   minimizer.values["p1"])
        self.p0 = minimizer.values["p0"]
        self.p1 = minimizer.values["p1"]

        self.bad_cont = None
        if not minimizer.valid:
            self.bad_cont = "minuit didn't converge"
        if np.any(self.cont <= 0):
            self.bad_cont = "negative continuum"

        ## if the continuum is negative, then set it to a very small number
        ## so that this forest is ignored
        if self.bad_cont is not None:
            self.cont = self.cont * 0 + 1e-10
            self.p0 = 0.
            self.p1 = 0.


class Delta(QSO):
    """Class to represent the mean transimission fluctuation field (delta)

    This class stores the information for the deltas for a given line of sight

    Attributes:
        ## Inherits from QSO ##
        log_lambda : array of floats
            Array containing the logarithm of the wavelengths (in Angs)
        weights : array of floats
            Weights associated to pixel. Overloaded from parent class
        cont: array of floats
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
            Mean resolution of the forest in units of velocity (FWHM)
        mean_z: float
            Mean redshift of the forest
        mean_reso_pix: float
            Mean resolution of the forest in units of pixels (FWHM)
        mean_resolution_matrix: array of floats or None
            Mean (over wavelength) resolution matrix for that forest
        resolution_matrix: 2d array of floats or None
            Wavelength dependent resolution matrix for that forest
        delta_log_lambda: float
            Variation of the logarithm of the wavelength between two pixels
        z: array of floats or None
            Redshift of the abosrption
        r_comov: array of floats or None
            Comoving distance to the object. Overloaded from parent class
        dist_m: array of floats or None
            Angular diameter distance to object. Overloaded from parent
            class
        neighbours: list of Delta or QSO or None
            Neighbouring deltas/quasars
        fname: string or None
            String identifying Delta as part of a group

    Methods:
        __init__: Initializes class instances.
        from_fitsio: Initialize instance from a fits file.
        from_ascii: Initialize instance from an ascii file.
        from_image: Initialize instance from an ascii file.
        project: Project the delta field.

    """

    def __init__(self, los_id, ra, dec, z_qso, plate, mjd, fiberid, log_lambda,
                 weights, cont, delta, order, ivar, exposures_diff, mean_snr,
                 mean_reso, mean_z, resolution_matrix=None,
                 mean_resolution_matrix=None, mean_reso_pix=None):
        """Initializes class instances.

        Args:
            los_id: integer
                Thingid or Targetid of the observation.
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
            cont: array of floats
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
            mean_reso_pix: float
                Mean resolution of the forest in units of pixels (FWHM)
            mean_resolution_matrix: array of floats or None
                Mean (over wavelength) resolution matrix for that forest
            resolution_matrix: 2d array of floats or None
                Wavelength dependent resolution matrix for that forest
            delta_log_lambda: float
                Variation of the logarithm of the wavelength between two pixels
        """
        QSO.__init__(self, los_id, ra, dec, z_qso, plate, mjd, fiberid)
        self.log_lambda = log_lambda
        self.weights = weights
        self.cont = cont
        self.delta = delta
        self.order = order
        self.ivar = ivar
        self.exposures_diff = exposures_diff
        self.mean_snr = mean_snr
        self.mean_reso = mean_reso
        self.mean_z = mean_z
        self.resolution_matrix = resolution_matrix
        self.mean_resolution_matrix = mean_resolution_matrix
        self.mean_reso_pix = mean_reso_pix

        # variables computed in function io.read_deltas
        self.z = None
        self.r_comov = None
        self.dist_m = None

        # variables computed in function cf.fill_neighs or xcf.fill_neighs
        self.neighbours = None

        # variables used in function cf.compute_wick_terms and
        # main from bin.picca_wick
        self.fname = None

    @classmethod
    def from_fitsio(cls, hdu, pk1d_type=False):
        """Initialize instance from a fits file.

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

        # new runs of picca_deltas should have a blinding keyword
        if "BLINDING" in header:
            blinding = header["BLINDING"]
        # older runs are not from DESI main survey and should not be blinded
        else:
            blinding = "none"

        if blinding != "none":
            delta_name = "DELTA_BLIND"
        else:
            delta_name = "DELTA"

        delta = hdu[delta_name][:].astype(float)

        if 'LOGLAM' in hdu.get_colnames():
            log_lambda = hdu['LOGLAM'][:].astype(float)
        elif 'LAMBDA' in hdu.get_colnames():
            log_lambda = np.log10(hdu['LAMBDA'][:].astype(float))
        else:
            raise KeyError("Did not find LOGLAM or LAMBDA in delta file")

        if pk1d_type:
            ivar = hdu['IVAR'][:].astype(float)
            try:
                exposures_diff = hdu['DIFF'][:].astype(float)
            except (KeyError, ValueError):
                userprint('WARNING: no DIFF in hdu while pk1d_type=True, filling with zeros.')
                exposures_diff = np.zeros(delta.shape)
            mean_snr = header['MEANSNR']
            mean_reso = header['MEANRESO']
            try:
                mean_reso_pix = header['MEANRESO_PIX']
            except (KeyError, ValueError):
                mean_reso_pix = None

            mean_z = header['MEANZ']
            try:
                #transposing here gives back the actual reso matrix which has been stored transposed
                resolution_matrix = hdu['RESOMAT'][:].T.astype(float)
                if resolution_matrix is not None:
                    mean_resolution_matrix = np.mean(resolution_matrix, axis=1)
                else:
                    mean_resolution_matrix = None
            except (KeyError, ValueError):
                resolution_matrix = None
                mean_resolution_matrix = None
            weights = None
            cont = None
        else:
            ivar = None
            exposures_diff = None
            mean_snr = None
            mean_reso = None
            mean_z = None
            resolution_matrix = None
            mean_resolution_matrix = None
            mean_reso_pix = None
            weights = hdu['WEIGHT'][:].astype(float)
            cont = hdu['CONT'][:].astype(float)

        if 'THING_ID' in header:
            los_id = header['THING_ID']
            plate = header['PLATE']
            mjd = header['MJD']
            fiberid = header['FIBERID']
        elif 'LOS_ID' in header:
            los_id = header['LOS_ID']
            plate=los_id
            mjd=los_id
            fiberid=los_id
        else:
            raise Exception("Could not find THING_ID or LOS_ID")

        ra = header['RA']
        dec = header['DEC']
        z_qso = header['Z']
        try:
            order = header['ORDER']
        except KeyError:
            order = 1

        return cls(los_id, ra, dec, z_qso, plate, mjd, fiberid, log_lambda,
                   weights, cont, delta, order, ivar, exposures_diff, mean_snr,
                   mean_reso, mean_z, resolution_matrix,
                   mean_resolution_matrix, mean_reso_pix)

    @classmethod
    def from_ascii(cls, line):
        """Initialize instance from an ascii file.

        Args:
            line: string
                A line of the ascii file containing information from a line
                of sight

        Returns:
            a Delta instance
        """

        cols = line.split()
        plate = int(cols[0])
        mjd = int(cols[1])
        fiberid = int(cols[2])
        ra = float(cols[3])
        dec = float(cols[4])
        z_qso = float(cols[5])
        mean_z = float(cols[6])
        mean_snr = float(cols[7])
        mean_reso = float(cols[8])
        delta_log_lambda = float(cols[9])

        num_pixels = int(cols[10])
        delta = np.array(cols[11:11 + num_pixels]).astype(float)
        log_lambda = np.array(cols[11 + num_pixels:11 +
                                   2 * num_pixels]).astype(float)
        ivar = np.array(cols[11 + 2 * num_pixels:11 +
                             3 * num_pixels]).astype(float)
        exposures_diff = np.array(cols[11 + 3 * num_pixels:11 +
                                       4 * num_pixels]).astype(float)

        thingid = 0
        order = 0
        weights = None
        cont = None

        return cls(thingid, ra, dec, z_qso, plate, mjd, fiberid, log_lambda,
                   weights, cont, delta, order, ivar, exposures_diff, mean_snr,
                   mean_reso, mean_z, delta_log_lambda)

    @classmethod
    def from_image(cls, hdul, pk1d_type=False):
        """Initialize instance from an ascii file.

        Args:
            hdu: fitsio.hdu.table.TableHDU
                A Header Data Unit opened with fitsio
            pk1d_type: bool - default: False
                Specifies if the fits file is formatted for the 1D Power
                Spectrum analysis
        Returns:
            a Delta instance
        """
        if pk1d_type:
            raise ValueError("ImageHDU format not implemented for Pk1D forests.")

        header = hdul["METADATA"].read_header()
        N_forests = hdul["METADATA"].get_nrows()
        Nones = np.full(N_forests, None)

        # new runs of picca_deltas should have a blinding keyword
        if "BLINDING" in header:
            blinding = header["BLINDING"]
        else:
            blinding = "none"

        if blinding != "none":
            delta_name = "DELTA_BLIND"
        else:
            delta_name = "DELTA"

        delta = hdul[delta_name].read().astype(float)

        if "LOGLAM" in hdul:
            log_lambda = hdul["LOGLAM"][:].astype(float)
        elif "LAMBDA" in hdul:
            log_lambda = np.log10(hdul["LAMBDA"][:].astype(float))
        else:
            raise KeyError("Did not find LOGLAM or LAMBDA in delta file")

        ivar = Nones
        exposures_diff = Nones
        mean_snr = Nones
        mean_reso = Nones
        mean_z = Nones
        resolution_matrix = Nones
        mean_resolution_matrix = Nones
        mean_reso_pix = Nones
        weights = hdul["WEIGHT"].read().astype(float)
        w = weights > 0
        cont = hdul["CONT"].read().astype(float)

        if "THING_ID" in hdul["METADATA"].get_colnames():
            los_id = hdul["METADATA"]["THING_ID"][:]
            plate = hdul["METADATA"]["PLATE"][:]
            mjd = hdul["METADATA"]["MJD"][:]
            fiberid=hdul["METADATA"]["FIBERID"][:]
        elif "LOS_ID" in hdul["METADATA"].get_colnames():
            los_id = hdul["METADATA"]["LOS_ID"][:]
            plate=los_id
            mjd=los_id
            fiberid=los_id
        else:
            raise Exception("Could not find THING_ID or LOS_ID")

        ra = hdul["METADATA"]["RA"][:]
        dec = hdul["METADATA"]["DEC"][:]
        z_qso = hdul["METADATA"]["Z"][:]
        try:
            order = hdul["METADATA"]["ORDER"][:]
        except (KeyError, ValueError):
            order = np.full(N_forests, 1)

        deltas = []
        for (los_id_i, ra_i, dec_i, z_qso_i, plate_i, mjd_i, fiberid_i, log_lambda,
            weights_i, cont_i, delta_i, order_i, ivar_i, exposures_diff_i, mean_snr_i,
            mean_reso_i, mean_z_i, resolution_matrix_i,
            mean_resolution_matrix_i, mean_reso_pix_i, w_i
        ) in zip(los_id, ra, dec, z_qso, plate, mjd, fiberid, repeat(log_lambda),
                   weights, cont, delta, order, ivar, exposures_diff, mean_snr,
                   mean_reso, mean_z, resolution_matrix,
                   mean_resolution_matrix, mean_reso_pix, w):
            deltas.append(cls(
                los_id_i, ra_i, dec_i, z_qso_i, plate_i, mjd_i, fiberid_i, log_lambda[w_i],
                weights_i[w_i] if weights_i is not None else None, 
                cont_i[w_i], 
                delta_i[w_i],
                order_i, 
                ivar_i[w_i] if ivar_i is not None else None,
                exposures_diff_i[w_i] if exposures_diff_i is not None else None, 
                mean_snr_i, mean_reso_i, mean_z_i,
                resolution_matrix_i if resolution_matrix_i is not None else None,
                mean_resolution_matrix_i if mean_resolution_matrix_i is not None else None,
                mean_reso_pix_i,
            ))

        return deltas

    def project(self):
        """Project the delta field.

        The projection gets rid of the distortion caused by the continuum
        fitiing. See equations 5 and 6 of du Mas des Bourboux et al. 2020
        """
        # 2nd term in equation 6
        sum_weights = np.sum(self.weights)
        if sum_weights > 0.0:
            mean_delta = np.average(self.delta, weights=self.weights)
        else:
            # should probably write a warning
            return

        # 3rd term in equation 6
        res = 0
        if (self.order == 1) and self.delta.shape[0] > 1:
            mean_log_lambda = np.average(self.log_lambda, weights=self.weights)
            meanless_log_lambda = self.log_lambda - mean_log_lambda
            mean_delta_log_lambda = (
                np.sum(self.weights * self.delta * meanless_log_lambda) /
                np.sum(self.weights * meanless_log_lambda**2))
            res = mean_delta_log_lambda * meanless_log_lambda
        elif self.order == 1:
            res = self.delta

        self.delta -= mean_delta + res

    def rebin(self, factor):
        """Rebin deltas by an integer factor

        Args:
            factor: int
                Factor to rebin deltas (new_bin_size = factor * old_bin_size)
        """
        wave = 10**np.array(self.log_lambda)
        dwave = wave[1] - wave[0]
        if not np.isclose(dwave, wave[-1] - wave[-2]):
            raise ValueError('Delta rebinning only implemented for linear lambda bins')

        start = wave.min() - dwave / 2
        num_bins = np.ceil(((wave[-1] - wave[0]) / dwave + 1) / factor)

        edges = np.arange(num_bins) * dwave * factor + start

        new_indx = np.searchsorted(edges, wave)

        binned_delta = np.bincount(new_indx, weights=self.delta*self.weights,
                                   minlength=edges.size+1)[1:-1]
        binned_weight = np.bincount(new_indx, weights=self.weights, minlength=edges.size+1)[1:-1]

        mask = binned_weight != 0
        binned_delta[mask] /= binned_weight[mask]

        new_wave = (edges[1:] + edges[:-1]) / 2

        self.log_lambda = np.log10(new_wave[mask])
        self.delta = binned_delta[mask]
        self.weights = binned_weight[mask]
