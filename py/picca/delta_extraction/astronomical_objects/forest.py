"""This module defines the abstract class Forest from which all
objects representing a forest must inherit from
"""
import numpy as np

from picca.delta_extraction.astronomical_object import AstronomicalObject
from picca.delta_extraction.errors import AstronomicalObjectError

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
    class_variable_check
    consistency_check
    coadd
    get_data
    get_header
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

    lambda_max_rest_frame: float or None
    As wavelength_max but for rest-frame wavelength. This should not
    be None if wave_solution is "lin". Ignored if wave_solution is "log".

    lambda_min: float or None
    Minimum wavelength (in Angs) to be considered in a forest. This should not
    be None if wave_solution is "lin". Ignored if wave_solution is "log".

    lambda_min_rest_frame: float or None
    As wavelength_min but for rest-frame wavelength. This should not
    be None if wave_solution is "lin". Ignored if wave_solution is "log".

    log_lambda_max: float or None
    Logarithm of the maximum wavelength (in Angs) to be considered in a forest.
    This should not be None if wave_solution is "log". Ignored if wave_solution
    is "lin".

    log_lambda_max_rest_frame: float or None
    As log_lambda_max but for rest-frame wavelength. This should not be None if
    wave_solution is "log". Ignored if wave_solution is "lin".

    log_lambda_min: float or None
    Logarithm of the minimum wavelength (in Angs) to be considered in a forest.
    This should not be None if wave_solution is "log". Ignored if wave_solution
    is "lin".

    log_lambda_min_rest_frame: float or None
    As log_lambda_min but for rest-frame wavelength. This should not be None if
    wave_solution is "log". Ignored if wave_solution is "lin".

    mask_fields: list of str
    Names of the fields that are affected by masking. In general it will
    be "flux", "ivar", "transmission_correction" and either "log_lambda" if
    Forest.wave_solution is "log" or "lambda_" if Forests.wave_solution is "lin",
    but some child classes might add more.

    wave_solution: "lin" or "log"
    Determines whether the wavelength solution has linear spacing ("lin") or
    logarithmic spacing ("log").

    Attributes
    ----------
    dec: float (from AstronomicalObject)
    Declination (in rad)

    healpix: int (from AstronomicalObject)
    Healpix number associated with (ra, dec)

    los_id: longint (from AstronomicalObject)
    Line-of-sight id. Same as thingid

    ra: float (from AstronomicalObject)
    Right ascention (in rad)

    z: float (from AstronomicalObject)
    Redshift

    bad_continuum_reason: str or None
    Reason as to why the continuum fit is not acceptable. None for acceptable
    contiuum.

    continuum: array of float or None
    Quasar continuum. None for no information

    deltas: array of float or None
    Flux-transmission field (delta field). None for no information

    flux: array of float
    Flux

    ivar: array of float
    Inverse variance

    lambda_: array of float
    Wavelength (in Angstroms)

    log_lambda: array of float or None
    Logarithm of the wavelength (in Angstroms)

    mean_snr: float
    Mean signal-to-noise of the forest

    transmission_correction: array of float
    Transmission correction.

    weights: array of float or None
    Weights associated to the delta field. None for no information
    """

    delta_lambda = None
    delta_log_lambda = None
    lambda_max = None
    lambda_max_rest_frame = None
    lambda_min = None
    lambda_min_rest_frame = None
    log_lambda_max = None
    log_lambda_max_rest_frame = None
    log_lambda_min = None
    log_lambda_min_rest_frame = None
    mask_fields = []
    wave_solution = None

    def __init__(self, **kwargs):
        """Initialize instance

        Arguments
        ---------
        **kwargs: dict
        Dictionary contiaing the information

        Raise
        -----
        AstronomicalObjectError if there are missing variables
        """
        Forest.class_variable_check()

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
                                              "Missing variable 'lambda'")
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

        self.transmission_correction = np.ones_like(self.flux)

        self.weights = kwargs.get("weights")
        if kwargs.get("weights") is not None:
            del kwargs["weights"]

        # compute mean quality variables
        snr = self.flux * np.sqrt(self.ivar)
        self.mean_snr = sum(snr) / float(len(snr))

        # call parent constructor
        super().__init__(**kwargs)

        self.consistency_check()

    @classmethod
    def class_variable_check(cls):
        """Check that class variables have been correctly initialized"""
        if cls.wave_solution is None:
            raise AstronomicalObjectError("Error constructing Forest. "
                                          "Class variable 'wave_solution' "
                                          "must be set prior to initialize "
                                          "instances of this type")
        if not isinstance(cls.mask_fields, list):
            raise AstronomicalObjectError(
                "Error constructing Forest. "
                "Expected list in class variable 'mask fields'. "
                f"Found {cls.mask_fields}.")
        if cls.wave_solution == "log":
            if cls.delta_log_lambda is None:
                raise AstronomicalObjectError("Error constructing Forest. "
                                              "Class variable 'delta_log_lambda' "
                                              "must be set prior to initialize "
                                              "instances of this type")
            if cls.log_lambda_max is None:
                raise AstronomicalObjectError("Error constructing Forest. "
                                              "Class variable 'log_lambda_max' "
                                              "must be set prior to initialize "
                                              "instances of this type")
            if cls.log_lambda_max_rest_frame is None:
                raise AstronomicalObjectError("Error constructing Forest. "
                                              "Class variable 'log_lambda_max_rest_frame' "
                                              "must be set prior to initialize "
                                              "instances of this type")
            if cls.log_lambda_min is None:
                raise AstronomicalObjectError("Error constructing Forest. "
                                              "Class variable 'log_lambda_min' "
                                              "must be set prior to initialize "
                                              "instances of this type")
            if cls.log_lambda_min_rest_frame is None:
                raise AstronomicalObjectError("Error constructing Forest. "
                                              "Class variable 'log_lambda_min_rest_frame' "
                                              "must be set prior to initialize "
                                              "instances of this type")
            if len(cls.mask_fields) == 0:
                Forest.mask_fields = defaults.get("mask fields log").copy()

        elif cls.wave_solution == "lin":
            if cls.delta_lambda is None:
                raise AstronomicalObjectError("Error constructing Forest. "
                                              "Class variable 'delta_lambda' "
                                              "must be set prior to initialize "
                                              "instances of this type")
            if cls.lambda_max is None:
                raise AstronomicalObjectError("Error constructing Forest. "
                                              "Class variable 'lambda_max' "
                                              "must be set prior to initialize "
                                              "instances of this type")
            if cls.lambda_max_rest_frame is None:
                raise AstronomicalObjectError(
                    "Error constructing Forest. "
                    "Class variable 'lambda_max_rest_frame' "
                    "must be set prior to initialize "
                    "instances of this type")
            if cls.lambda_min is None:
                raise AstronomicalObjectError("Error constructing Forest. "
                                              "Class variable 'lambda_min' "
                                              "must be set prior to initialize "
                                              "instances of this type")
            if cls.lambda_min_rest_frame is None:
                raise AstronomicalObjectError(
                    "Error constructing Forest. "
                    "Class variable 'lambda_min_rest_frame' "
                    "must be set prior to initialize "
                    "instances of this type")

            if len(cls.mask_fields) == 0:
                Forest.mask_fields = defaults.get("mask fields lin").copy()
        else:
            raise AstronomicalObjectError("Error constructing Forest. "
                                          "Class variable 'wave_solution' "
                                          "must be either 'lin' or 'log'. "
                                          f"Found: {cls.wave_solution}")

    def consistency_check(self):
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
        """Coadd the information of another forest.

        Forests are coadded by rebinning

        Arguments
        ---------
        other: Forest
        The forest instance to be coadded.

        Raise
        -----
        AstronomicalObjectError if other is not an instance of Forest
        AstronomicalObjectError if other has a different los_id
        AstronomicalObjectError if Forest.wave_solution is not 'lin' or 'log'
        """
        if not isinstance(other, Forest):
            raise AstronomicalObjectError("Error coadding Forest. Expected "
                                          "Forest instance in other. Found: "
                                          f"{type(other)}")

        if self.los_id != other.los_id:
            raise AstronomicalObjectError("Attempting to coadd two Forests "
                                          "with different los_id. This should "
                                          f"not happen. this.los_id={self.los_id}, "
                                          f"other.los_id={other.los_id}.")

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

    def get_data(self):
        """Get the data to be saved in a fits file.

        Data contains lambda_ or log_lambda depending on whether
        wave_solution is "lin" or "log"
        Data also contains the delta field, the weights and the quasar
        continuum.

        Return
        ------
        cols: list of arrays
        Data of the different variables

        names: list of str
        Names of the different variables

        units: list of str
        Units of the different variables

        comments: list of str
        Comments attached to the different variables

        Raise
        -----
        AstronomicalObjectError if Forest.wave_solution is not 'lin' or 'log'
        """
        cols = []
        names = []
        comments = []
        units = []
        if Forest.wave_solution == "log":
            cols += [self.log_lambda]
            names += ["LOGLAM"]
            comments += ["Log lambda"]
            units += ["log Angstrom"]
            array_size = self.log_lambda.size
        elif Forest.wave_solution == "lin":
            cols += [self.lambda_]
            names += ["LAMBDA"]
            comments += ["Lambda"]
            units += ["Angstrom"]
            array_size = self.lambda_.size
        else:
            raise AstronomicalObjectError("Error in getting data from Forest. "
                                          "Class variable 'wave_solution' "
                                          "must be either 'lin' or 'log'. "
                                          f"Found: {Forest.wave_solution}")

        if self.continuum is None:
            cols += [np.zeros(array_size, dtype=float)]
        else:
            cols += [self.continuum]
        names += ["CONT"]
        comments += ["Quasar continuum if BAD_CONT is 'None'. Check input "
                     "spectra for units"]
        units += ["Flux units"]

        if self.deltas is None:
            cols += [np.zeros(array_size, dtype=float)]
        else:
            cols += [self.deltas]
        names += ["DELTA"]
        comments += ["Delta field"]
        units += [""]

        if self.weights is None:
            cols += [np.zeros(array_size, dtype=float)]
        else:
            cols += [self.weights]
        names += ["WEIGHT"]
        comments += ["Pixel weights"]
        units += [""]

        return cols, names, units, comments

    def get_header(self):
        """Return line-of-sight data to be saved as a fits file header

        Adds to specific Forest keys to general header (defined in class
        AstronomicalObject)

        Return
        ------
        header : list of dict
        A list of dictionaries containing 'name', 'value' and 'comment' fields
        """
        header = super().get_header()
        header += [
            {
                'name': 'BAD_CONT',
                'value': str(self.bad_continuum_reason),
                'comment': 'Reason as to why the continuum is bad'
            },
            {
                'name': 'MEANSNR',
                'value': self.mean_snr,
                'comment': 'Mean SNR'
            },
        ]

        return header

    def rebin(self):
        """Rebin the arrays and update control variables
        Rebinned arrays are flux, ivar, lambda_ or log_lambda, and
        transmission_correction. Control variables are mean_snr

        Return
        ------
        bins: array of float
        Binning solution to be used for the rebinning

        rebin_ivar: array of float
        Rebinned version of ivar

        orig_ivar: array of float
        Original version of ivar (before applying the function)

        w1: array of bool
        Masking array for the bins solution

        w2: array of bool
        Masking array for the rebinned ivar solution

        Raise
        -----
        AstronomicalObjectError if Forest.wave_solution is not 'lin' or 'log'
        AstronomicalObjectError if ivar only has zeros
        """
        orig_ivar = self.ivar.copy()
        # compute bins
        if Forest.wave_solution == "log":
            bins = (np.floor((self.log_lambda - Forest.log_lambda_min) /
                             Forest.delta_log_lambda + 0.5).astype(int))
            self.log_lambda = Forest.log_lambda_min + bins * Forest.delta_log_lambda
            w1 = (self.log_lambda >= Forest.log_lambda_min)
            w1 = w1 & (self.log_lambda < Forest.log_lambda_max)
            w1 = w1 & (self.log_lambda - np.log10(1. + self.z) >
                       Forest.log_lambda_min_rest_frame)
            w1 = w1 & (self.log_lambda - np.log10(1. + self.z) <
                       Forest.log_lambda_max_rest_frame)
            w1 = w1 & (self.ivar > 0.)
            if w1.sum() == 0:
                self.log_lambda = np.array([])
                self.flux = np.array([])
                self.ivar = np.array([])
                self.transmission_correction = np.array([])
                return [], [], [], [], []
            bins = bins[w1]
            self.log_lambda = self.log_lambda[w1]
            self.flux = self.flux[w1]
            self.ivar = self.ivar[w1]
            self.transmission_correction = self.transmission_correction[w1]

        elif Forest.wave_solution == "lin":
            bins = (np.floor((self.lambda_ - Forest.lambda_min) /
                             Forest.delta_lambda + 0.5).astype(int))
            self.lambda_ = Forest.lambda_min + bins * Forest.delta_lambda
            w1 = (self.lambda_ >= Forest.lambda_min)
            w1 = w1 & (self.lambda_ < Forest.lambda_max)
            w1 = w1 & (self.lambda_ / (1. + self.z) > Forest.lambda_min_rest_frame)
            w1 = w1 & (self.lambda_ / (1. + self.z) < Forest.lambda_max_rest_frame)
            w1 = w1 & (self.ivar > 0.)
            if w1.sum() == 0:
                return [], [], [], [], []
            bins = bins[w1]
            self.lambda_ = self.lambda_[w1]
            self.flux = self.flux[w1]
            self.ivar = self.ivar[w1]
            self.transmission_correction = self.transmission_correction[w1]
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

        w2 = (rebin_ivar > 0.)
        if w2.sum() == 0:
            raise AstronomicalObjectError(
                "Attempting to rebin arrays flux and "
                "ivar in class Forest, but ivar seems "
                "to contain only zeros")
        self.flux = rebin_flux[w2] / rebin_ivar[w2]
        self.transmission_correction = rebin_transmission_correction[
            w2] / rebin_ivar[w2]
        self.ivar = rebin_ivar[w2]

        # then rebin wavelength
        if self.wave_solution == "log":
            rebin_lambda = (Forest.log_lambda_min +
                            np.arange(bins.max() + 1) * Forest.delta_log_lambda)
            self.log_lambda = rebin_lambda[w2]
        elif self.wave_solution == "lin":
            rebin_lambda = (Forest.lambda_min +
                            np.arange(bins.max() + 1) * Forest.delta_lambda)
            self.lambda_ = rebin_lambda[w2]
        else:
            raise AstronomicalObjectError("wavelength solution must be either "
                                          "'log' or 'linear'")

        # finally update control variables
        snr = self.flux * np.sqrt(self.ivar)
        self.mean_snr = sum(snr) / float(len(snr))

        # return weights and binning solution to be used by child classes if
        # required
        return bins, rebin_ivar, orig_ivar, w1, w2
