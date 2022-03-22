"""This module defines the abstract class Forest from which all
objects representing a forest must inherit from
"""
import numpy as np

from picca.delta_extraction.astronomical_object import AstronomicalObject
from picca.delta_extraction.errors import AstronomicalObjectError
from picca.delta_extraction.utils import find_bins

defaults = {
    "mask fields": ["flux", "ivar", "transmission_correction", "log_lambda"],
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
    set_class_variables

    Class Attributes
    ----------------
    blinding: str
    Name of the blinding strategy used

    log_lambda_grid: array of float or None
    Common grid in log_lambda based on the specified minimum and maximum
    wavelengths, and pixelisation.

    log_lambda_rest_frame_grid: array of float or None
    Same as log_lambda_grid but for rest-frame wavelengths.

    mask_fields: list of str
    Names of the fields that are affected by masking. In general it will
    be "flux", "ivar", "transmission_correction", and "log_lambda",
    but some child classes might add more.

    wave_solution: "lin" or "log"
    Determines whether the wavelength solution has linear spacing ("lin") or
    logarithmic spacing ("log").

    Attributes
    ----------
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

    log_lambda: array of float or None
    Logarithm of the wavelength (in Angstroms). Differs from log_lambda_grid
    as the particular instance might not have full wavelength coverage or
    might have some missing pixels (because they are masked)

    mean_snr: float
    Mean signal-to-noise of the forest

    transmission_correction: array of float
    Transmission correction.

    weights: array of float or None
    Weights associated to the delta field. None for no information
    """
    blinding = "none"
    log_lambda_grid = None
    log_lambda_rest_frame_grid = None
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

        self.log_lambda = kwargs.get("log_lambda")
        if self.log_lambda is None:
            raise AstronomicalObjectError("Error constructing Forest. "
                                          "Missing variable 'log_lambda'")
        del kwargs["log_lambda"]

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
        if cls.log_lambda_grid is None:
            raise AstronomicalObjectError("Error constructing Forest. "
                                          "Class variable 'log_lambda_grid' "
                                          "must be set prior to initialize "
                                          "instances of this type")
        if cls.log_lambda_rest_frame_grid is None:
            raise AstronomicalObjectError("Error constructing Forest. "
                                          "Class variable 'log_lambda_rest_frame_grid' "
                                          "must be set prior to initialize "
                                          "instances of this type")
        if cls.mask_fields is None:
            raise AstronomicalObjectError(
                "Error constructing Forest. "
                "Expected list in class variable 'mask fields'. "
                f"Found {cls.mask_fields}.")

        if cls.wave_solution is None:
            raise AstronomicalObjectError("Error constructing Forest. "
                                          "Class variable 'wave_solution' "
                                          "must be set prior to initialize "
                                          "instances of this type")

    def consistency_check(self):
        """Consistency checks after __init__"""
        if self.flux.size != self.ivar.size:
            raise AstronomicalObjectError("Error constructing Forest. 'flux', "
                                          "and 'ivar' don't have the same size")
        if self.log_lambda.size != self.flux.size:
            raise AstronomicalObjectError("Error constructing Forest. "
                                          "'flux'  and 'log_lambda' don't "
                                          "have the same  size")

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

        self.log_lambda = np.append(self.log_lambda, other.log_lambda)
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
            cols += [10**self.log_lambda]
            names += ["LAMBDA"]
            comments += ["Lambda"]
            units += ["Angstrom"]
            array_size = self.log_lambda.size
        else:
            raise AstronomicalObjectError("Error in getting data from Forest. "
                                          "Class variable 'wave_solution' "
                                          "must be either 'lin' or 'log'. "
                                          f"Found: {Forest.wave_solution}")

        if self.deltas is None:
            cols += [np.zeros(array_size, dtype=float)]
        else:
            cols += [self.deltas]
        if Forest.blinding == "none":
            names += ["DELTA"]
        else:
            names += ["DELTA_BLIND"]
        comments += ["Delta field"]
        units += [""]

        if self.weights is None:
            cols += [np.zeros(array_size, dtype=float)]
        else:
            cols += [self.weights]
        names += ["WEIGHT"]
        comments += ["Pixel weights"]
        units += [""]

        if self.continuum is None:
            cols += [np.zeros(array_size, dtype=float)]
        else:
            cols += [self.continuum]
        names += ["CONT"]
        comments += ["Quasar continuum. Check input "
                     "spectra for units"]
        units += ["Flux units"]

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
                'name': 'MEANSNR',
                'value': self.mean_snr,
                'comment': 'Mean SNR'
            },
            {
                'name': 'BLINDING',
                'value': Forest.blinding,
                'comment': "String specifying the blinding strategy"
            }
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
        delta_log_lambda = Forest.log_lambda_grid[1] - Forest.log_lambda_grid[0]
        half_delta_log_lambda = delta_log_lambda / 2.

        half_delta_log_lambda_rest_frame = (Forest.log_lambda_rest_frame_grid[1] -
                                            Forest.log_lambda_rest_frame_grid[0]) / 2.

        w1 = (self.log_lambda >= Forest.log_lambda_grid[0] - half_delta_log_lambda)
        w1 = w1 & (self.log_lambda < Forest.log_lambda_grid[-1] + half_delta_log_lambda)
        w1 = w1 & (self.log_lambda - np.log10(1. + self.z) >=
                   Forest.log_lambda_rest_frame_grid[0] - half_delta_log_lambda_rest_frame)
        w1 = w1 & (self.log_lambda - np.log10(1. + self.z) <
                   Forest.log_lambda_rest_frame_grid[-1] + half_delta_log_lambda_rest_frame)
        w1 = w1 & (self.ivar > 0.)
        if w1.sum() == 0:
            self.log_lambda = np.array([])
            self.flux = np.array([])
            self.ivar = np.array([])
            self.transmission_correction = np.array([])
            return [], [], [], [], []
        self.log_lambda = self.log_lambda[w1]
        self.flux = self.flux[w1]
        self.ivar = self.ivar[w1]
        self.transmission_correction = self.transmission_correction[w1]

        bins = find_bins(self.log_lambda, Forest.log_lambda_grid)
        self.log_lambda = Forest.log_lambda_grid[0] + bins * delta_log_lambda

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
            pixel_step = Forest.log_lambda_grid[1] - Forest.log_lambda_grid[0]
            rebin_log_lambda = (Forest.log_lambda_grid[0] +
                                np.arange(bins.max() + 1) * pixel_step)
            self.log_lambda = rebin_log_lambda[w2]
        elif self.wave_solution == "lin":
            pixel_step = 10**Forest.log_lambda_grid[1] - 10**Forest.log_lambda_grid[0]
            rebin_lambda = (10**Forest.log_lambda_grid[0] +
                            np.arange(bins.max() + 1) * pixel_step)
            self.log_lambda = np.log10(rebin_lambda[w2])
        else:
            raise AstronomicalObjectError("wavelength solution must be either "
                                          "'log' or 'linear'")

        # finally update control variables
        snr = self.flux * np.sqrt(self.ivar)
        self.mean_snr = sum(snr) / float(len(snr))

        # return weights and binning solution to be used by child classes if
        # required
        return bins, rebin_ivar, orig_ivar, w1, w2

    @classmethod
    def set_class_variables(cls, lambda_min, lambda_max,
                            lambda_min_rest_frame,
                            lambda_max_rest_frame,
                            pixel_step, wave_solution):
        """Set class variables

        Arguments
        ---------
        lambda_min: float
        Logarithm of the minimum wavelength (in Angs) to be considered in a forest.

        lambda_max: float
        Logarithm of the maximum wavelength (in Angs) to be considered in a forest.

        lambda_min_rest_frame: float or None
        As lambda_min but for rest-frame wavelength.

        lambda_max_rest_frame: float
        As lambda_max but for rest-frame wavelength.

        pixel_step: float
        Wavelength cahnge between two pixels. If pixel_step is "log" this is in
        units of the logarithm of the wavelength (in Angs). If pixel_step is "lin"
        this is in units of the wavelength (in Angs).

        wave_solution: "log" or "lin"
        Specifies whether we want to construct a wavelength grid that is evenly
        spaced on wavelength (lin) or on the logarithm of the wavelength (log)
        """
        if wave_solution == "log":
            cls.log_lambda_grid = np.arange(
                np.log10(lambda_min),
                np.log10(lambda_max) + pixel_step/2,
                pixel_step)
            cls.log_lambda_rest_frame_grid = np.arange(
                np.log10(lambda_min_rest_frame) + pixel_step/2,
                np.log10(lambda_max_rest_frame),
                pixel_step)
        elif wave_solution == "lin":
            cls.log_lambda_grid = np.log10(np.arange(
                lambda_min,
                lambda_max + pixel_step/2,
                pixel_step))
            cls.log_lambda_rest_frame_grid = np.log10(np.arange(
                lambda_min_rest_frame + pixel_step/2,
                lambda_max_rest_frame,
                pixel_step))
        else:
            raise AstronomicalObjectError("Error in setting Forest class "
                                          "variables. 'wave_solution' "
                                          "must be either 'lin' or 'log'. "
                                          f"Found: {wave_solution}")

        cls.wave_solution = wave_solution

        cls.mask_fields = defaults.get("mask fields").copy()
