"""This module defines the abstract class Forest from which all
objects representing a forest must inherit from
"""
import numpy as np

from picca.delta_extraction.errors import AstronomicalObjectError

from picca.delta_extraction.astronomical_object import AstronomicalObject

defaults = {
    "mask fields log": ["flux", "ivar", "transmission_correction", "log_lambda"],
    "mask fields lin": ["flux", "ivar", "transmission_correction", "lambda_"]
}
class Forest(AstronomicalObject):
    """Forest Object

    Methods
    -------
    __gt__ (from AstronomicalObject)
    __eq__ (from AstronomicalObject)
    __init__
    __class_variable_check
    __consistency_check
    rebin

    Class Attributes
    ----------------
    delta_lambda: float or None
    Variation of the wavelength (in Angs) between two pixels. This should not
    be None if wave_solution is "lin". Ignored if wave_solution is "log".

    delta_log_lambda: float or None
    Variation of the logarithm of the wavelength (in Angs) between two pixels.
    This should not be None if wave_solution is "log". Ignored if wave_solution
    is "lin".

    lambda_max: float or None
    Maximum wavelength (in Angs) to be considered in a forest. This should not
    be None if wave_solution is "lin". Ignored if wave_solution is "log".

    lambda_min: float or None
    Minimum wavelength (in Angs) to be considered in a forest. This should not
    be None if wave_solution is "lin". Ignored if wave_solution is "log".

    lambda_max_rest_frame: float or None
    As wavelength_max but for rest-frame wavelength. This should not
    be None if wave_solution is "lin". Ignored if wave_solution is "log".

    lambda_min_rest_frame: float or None
    As wavelength_min but for rest-frame wavelength. This should not
    be None if wave_solution is "lin". Ignored if wave_solution is "log".

    log_lambda_max: float or None
    Logarithm of the maximum wavelength (in Angs) to be considered in a forest.
    This should not be None if wave_solution is "log". Ignored if wave_solution
    is "lin".

    log_lambda_min: float or None
    Logarithm of the minimum wavelength (in Angs) to be considered in a forest.
    This should not be None if wave_solution is "log". Ignored if wave_solution
    is "lin".

    log_lambda_max_rest_frame: float or None
    As log_lambda_max but for rest-frame wavelength. This should not be None if
    wave_solution is "log". Ignored if wave_solution is "lin".

    log_lambda_min_rest_frame: float or None
    As log_lambda_min but for rest-frame wavelength. This should not be None if
    wave_solution is "log". Ignored if wave_solution is "lin".

    wave_solution: "lin" or "log"
    Determines whether the wavelength solution has linear spacing ("lin") or
    logarithmic spacing ("log").

    Attributes
    ----------
    bad_continuum_reason: str or None
    Reason as to why the continuum fit is not acceptable. None for acceptable
    contiuum.

    dec: float (from AstronomicalObject)
    Declination (in rad)

    healpix: int (from AstronomicalObject)
    Healpix number associated with (ra, dec)

    lambda_: array of float (from Forest)
    Wavelength (in Angstroms)

    los_id: longint (from AstronomicalObject)
    Line-of-sight id. Same as thingid

    ra: float (from AstronomicalObject)
    Right ascention (in rad)

    z: float (from AstronomicalObject)
    Redshift

    continuum: array of float or None
    Quasar continuum. None for no information

    deltas: array of float or None
    Flux-transmission field (delta field). None for no information

    flux: array of float
    Flux

    ivar: array of float
    Inverse variance

    mask_fields: list of str
    Names of the fields that are affected by masking. In general it will
    be "flux", "ivar" and "transmission_correction" but some child classes might
    add more.

    mean_snf: float
    Mean signal-to-noise of the forest

    transmission_correction: array of float
    Transmission correction.
    """

    delta_lambda = None
    delta_lambda = None
    lambda_max = None
    lambda_max_rest_frame = None
    lambda_min = None
    lambda_min_rest_frame = None
    log_lambda_max = None
    log_lambda_max_rest_frame = None
    log_lambda_min = None
    log_lambda_min_rest_frame = None
    wave_solution = None

    def __init__(self, **kwargs):
        """Initialize instance

        Arguments
        ---------
        **kwargs: dict
        Dictionary contiaing the information
        """
        self.__class_variable_check()

        if Forest.wave_solution == "log":
            self.lambda_ = None
            self.log_lambda = kwargs.get("log_lambda")
            if self.log_lambda is None:
                raise AstronomicalObjectError("Error constructing Forest. "
                                              "Missing variable 'log_lambda'")
            del kwargs["log_lambda"]
        elif Forest.wave_solution == "lin":
            self.log_lambda = None
            self.lambda_ = kwargs.get("lambda")
            if self.lambda_ is None:
                raise AstronomicalObjectError("Error constructing Forest. "
                                              "Missing variable 'wavelength'")
            del kwargs["lambda"]

        self.bad_continuum_reason = None
        self.continuum = kwargs.get("continuum")
        if kwargs.get("continuum") is not None:
            del kwargs["continuum"]

        self.deltas = kwargs.get("deltas")
        if kwargs.get("deltas") is not None:
            del kwargs["deltas"]

        self.flux = kwargs.get("flux")
        if self.flux is None:
            raise AstronomicalObjectError("Error constructing Forest. "
                                          "Missing variable 'flux'")
        del kwargs["flux"]

        self.ivar = kwargs.get("ivar")
        if self.ivar is None:
            raise AstronomicalObjectError("Error constructing Forest. "
                                          "Missing variable 'ivar'")
        del kwargs["ivar"]

        self.mask_fields = kwargs.get("mask_fields")
        if self.mask_fields is None:
            if Forest.wave_solution == "log":
                self.mask_fields = defaults.get("mask fields log")
            elif Forest.wave_solution == "lin":
                self.mask_fields = defaults.get("mask fields lin")
            else:
                raise AstronomicalObjectError("Error constructing Forest. "
                                              "Class variable 'wave_solution' "
                                              "must be either 'lin' or 'log'. "
                                              f"Found: {Forest.wave_solution}")
        else:
            del kwargs["mask_fields"]
        if not isinstance(self.mask_fields, list):
            raise AstronomicalObjectError(
                "Error constructing Forest. "
                "Expected list in variable 'mask fields'. "
                f"Found {self.mask_fields}.")

        self.transmission_correction = np.ones_like(self.flux)

        # compute mean quality variables
        snr = self.flux * np.sqrt(self.ivar)
        self.mean_snr = sum(snr) / float(len(snr))

        # call parent constructor
        super().__init__(**kwargs)

        self.__consistency_check()

    def __class_variable_check(self):
        """Check that class variables have been correctly initialized"""
        if Forest.wave_solution is None:
            raise AstronomicalObjectError("Error constructing Forest. "
                                          "Class variable 'wave_solution' "
                                          "must be set prior to initialize "
                                          "instances of this type")
        elif Forest.wave_solution == "log":
            if Forest.delta_log_lambda is None:
                raise AstronomicalObjectError("Error constructing Forest. "
                                              "Class variable 'delta_log_lambda' "
                                              "must be set prior to initialize "
                                              "instances of this type")
            if Forest.log_lambda_max is None:
                raise AstronomicalObjectError("Error constructing Forest. "
                                              "Class variable 'log_lambda_max' "
                                              "must be set prior to initialize "
                                              "instances of this type")
            if Forest.log_lambda_max_rest_frame is None:
                raise AstronomicalObjectError("Error constructing Forest. "
                                              "Class variable 'log_lambda_max_rest_frame' "
                                              "must be set prior to initialize "
                                              "instances of this type")
            if Forest.log_lambda_min is None:
                raise AstronomicalObjectError("Error constructing Forest. "
                                              "Class variable 'log_lambda_min' "
                                              "must be set prior to initialize "
                                              "instances of this type")
            if Forest.log_lambda_min_rest_frame is None:
                raise AstronomicalObjectError("Error constructing Forest. "
                                              "Class variable 'log_lambda_min_rest_frame' "
                                              "must be set prior to initialize "
                                              "instances of this type")
        elif Forest.wave_solution == "lin":
            if Forest.delta_lambda is None:
                raise AstronomicalObjectError("Error constructing Forest. "
                                              "Class variable 'delta_lambda' "
                                              "must be set prior to initialize "
                                              "instances of this type")
            if Forest.lambda_max is None:
                raise AstronomicalObjectError("Error constructing Forest. "
                                              "Class variable 'lambda_max' "
                                              "must be set prior to initialize "
                                              "instances of this type")
            if Forest.lambda_max_rest_frame is None:
                raise AstronomicalObjectError(
                    "Error constructing Forest. "
                    "Class variable 'lambda_max_rest_frame' "
                    "must be set prior to initialize "
                    "instances of this type")
            if Forest.lambda_min is None:
                raise AstronomicalObjectError("Error constructing Forest. "
                                              "Class variable 'lambda_min' "
                                              "must be set prior to initialize "
                                              "instances of this type")
            if Forest.lambda_min_rest_frame is None:
                raise AstronomicalObjectError(
                    "Error constructing Forest. "
                    "Class variable 'lambda_min_rest_frame' "
                    "must be set prior to initialize "
                    "instances of this type")
        else:
            raise AstronomicalObjectError("Error constructing Forest. "
                                          "Class variable 'wave_solution' "
                                          "must be either 'lin' or 'log'. "
                                          f"Found: {Forest.wave_solution}")

    def __consistency_check(self):
        """Consistency checks after __init__"""
        if self.flux.size != self.ivar.size:
            raise AstronomicalObjectError("Error constructing Forest. 'flux', "
                                          "and 'ivar' don't have the same size")
        if Forest.wave_solution == "log":
            if self.log_lambda.size != self.flux.size:
                raise AstronomicalObjectError("Error constructing Forest. "
                                              "'flux'  and 'log_lambda' don't "
                                              "have the same  size")
        elif Forest.wave_solution == "lin":
            if self.lambda_.size != self.flux.size:
                raise AstronomicalObjectError("Error constructing Forest. "
                                              "'flux' and 'lambda' don't have "
                                              "the same size")
        else:
            raise AstronomicalObjectError("Error constructing Forest. "
                                          "Class variable 'wave_solution' "
                                          "must be either 'lin' or 'log'. "
                                          f"Found: {Forest.wave_solution}")

    def coadd(self, other):
        """Coadds the information of another forest.

        Forests are coadded by using inverse variance weighting

        Arguments
        ---------
        other: Forest
        The forest instance to be coadded.
        """
        if Forest.wave_solution == "log":
            self.log_lambda = np.append(self.log_lambda, other.log_lambda)
        elif Forest.wave_solution == "lin":
            self.lambda_ = np.append(self.lambda_, other.lambda_)
        else:
            raise AstronomicalObjectError("Error in coadding Forest. "
                                          "Class variable 'wave_solution' "
                                          "must be either 'lin' or 'log'. "
                                          f"Found: {Forest.wave_solution}")
        self.flux = np.append(self.flux, other.flux)
        self.ivar = np.append(self.ivar, other.ivar)
        self.transmission_correction = np.append(self.transmission_correction,
                                                 other.transmission_correction)

        # coadd the deltas by rebinning
        self.rebin()

    def rebin(self):
        """Rebin the flux and ivar arrays."""
        # compute bins
        if Forest.wave_solution == "log":
            bins = (np.floor((self.log_lambda - Forest.log_lambda_min) /
                             Forest.delta_log_lambda + 0.5).astype(int))
            self.log_lambda = Forest.log_lambda_min + bins * Forest.delta_log_lambda
            w = (self.log_lambda >= Forest.log_lambda_min)
            w = w & (self.log_lambda < Forest.log_lambda_max)
            w = w & (self.log_lambda - np.log10(1. + self.z) >
                     Forest.log_lambda_min_rest_frame)
            w = w & (self.log_lambda - np.log10(1. + self.z) <
                     Forest.log_lambda_max_rest_frame)
            w = w & (self.ivar > 0.)
            if w.sum() == 0:
                return
            bins = bins[w]
            self.log_lambda = self.log_lambda[w]
            self.flux = self.flux[w]
            self.ivar = self.ivar[w]
            self.transmission_correction = self.transmission_correction[w]

        elif Forest.wave_solution == "lin":
            bins = (np.floor((self.lambda_ - Forest.lambda_min) /
                             Forest.delta_lambda + 0.5).astype(int))
            self.lambda_ = Forest.lambda_min + bins * Forest.delta_lambda
            w = (self.lambda_ >= Forest.lambda_min)
            w = w & (self.lambda_ < Forest.lambda_max)
            w = w & (self.lambda_ / (1. + self.z) > Forest.lambda_min_rest_frame)
            w = w & (self.lambda_ / (1. + self.z) < Forest.lambda_max_rest_frame)
            w = w & (self.ivar > 0.)
            if w.sum() == 0:
                return
            bins = bins[w]
            self.lambda_ = self.lambda_[w]
            self.flux = self.flux[w]
            self.ivar = self.ivar[w]
            self.transmission_correction = self.transmission_correction[w]
        else:
            raise AstronomicalObjectError("Error in rebinning Forest. "
                                          "Class variable 'wave_solution' "
                                          "must be either 'lin' or 'log'. "
                                          f"Found: {Forest.wave_solution}")

        # rebin flux, ivar and transmission_correction
        rebin_flux = np.zeros(bins.max() + 1)
        rebin_transmission_correction = np.zeros(bins.max() + 1)
        rebin_ivar = np.zeros(bins.max() + 1)
        rebin_flux_aux = np.bincount(bins, weights=self.ivar * self.flux)
        rebin_transmission_correction_aux = np.bincount(
            bins, weights=(self.ivar * self.transmission_correction))
        rebin_ivar_aux = np.bincount(bins, weights=self.ivar)
        rebin_flux[:len(rebin_flux_aux)] += rebin_flux_aux
        rebin_transmission_correction[:
                                      len(rebin_transmission_correction_aux
                                         )] += rebin_transmission_correction_aux
        rebin_ivar[:len(rebin_ivar_aux)] += rebin_ivar_aux

        w = (rebin_ivar > 0.)
        if w.sum() == 0:
            raise AstronomicalObjectError(
                "Attempting to rebin arrays flux and "
                "ivar in class Forest, but ivar seems "
                "to contain only zeros")
        self.flux = rebin_flux[w] / rebin_ivar[w]
        self.transmission_correction = rebin_transmission_correction[
            w] / rebin_ivar[w]
        self.ivar = rebin_ivar[w]

        # then rebin wavelength
        if self.wave_solution == "log":
            rebin_lambda = (Forest.log_lambda_min +
                            np.arange(bins.max() + 1) * Forest.delta_log_lambda)
            self.log_lambda = rebin_lambda[w]
        elif self.wave_solution == "lin":
            rebin_lambda = (Forest.lambda_min +
                            np.arange(bins.max() + 1) * Forest.delta_lambda)
            self.lambda_ = rebin_lambda[w]
        else:
            raise AstronomicalObjectError("wavelength solution must be either "
                                          "'log' or 'linear'")

        # finally update control variables
        snr = self.flux * np.sqrt(self.ivar)
        self.mean_snr = sum(snr) / float(len(snr))

        return w
