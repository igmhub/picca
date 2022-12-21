"""This module defines the abstract class Forest from which all
objects representing a forest must inherit from
"""
import logging
import numpy as np
from numba import njit
from numba.types import bool_

from picca.delta_extraction.astronomical_object import AstronomicalObject
from picca.delta_extraction.errors import AstronomicalObjectError
from picca.delta_extraction.utils import find_bins

defaults = {
    "mask fields": ["flux", "ivar", "transmission_correction", "log_lambda"],
}

@njit
def get_inner_region_slice(bincounts):
    """Return the mask that removes zeros at the both
    edges. Zero inside remains

    Arguments
    ---------
    bincounts: Array of int
    Obtained from np.bincounts(bins)

    Returns
    -------
    wslice_inner: Array of bool
    False for zeros at the both edges. True for inside
    """
    size = bincounts.size
    first_nonzero_idx = size
    last_nonzero_idx = 0

    # Find first non-zero index from left
    for idx in range(bincounts.size):
        if bincounts[idx] != 0:
            first_nonzero_idx = idx
            break

    # Find last non-zero index from right
    for idx in range(bincounts.size, 0, -1):
        if bincounts[idx-1] != 0:
            last_nonzero_idx = idx
            break

    wslice_inner = np.zeros(bincounts.size, dtype=bool_)
    wslice_inner[first_nonzero_idx:last_nonzero_idx] = 1

    return wslice_inner

@njit()
def rebin(log_lambda, flux, ivar, transmission_correction, z, wave_solution,
          log_lambda_grid, log_lambda_rest_frame_grid):
    """Rebin the arrays and update control variables
    Rebinned arrays are flux, ivar, lambda_ or log_lambda, and
    transmission_correction. Control variables are mean_snr

    Arguments
    ---------
    log_lambda: array of float
    Logarithm of the wavelength (in Angstroms). Differs from log_lambda_grid
    as the particular instance might not have full wavelength coverage or
    might have some missing pixels (because they are masked)

    flux: array of float
    Flux

    ivar: array of float
    Inverse variance

    transmission_correction: array of float
    Transmission correction.

    z: float
    Quasar redshift

    wave_solution: "lin" or "log"
    Determines whether the wavelength solution has linear spacing ("lin") or
    logarithmic spacing ("log").

    log_lambda_grid: array of float or None
    Common grid in log_lambda based on the specified minimum and maximum
    wavelengths, and pixelisation.

    log_lambda_rest_frame_grid: array of float or None
    Same as log_lambda_grid but for rest-frame wavelengths.

    Return
    ------
    log_lambda: array of float
    Rebinned version of input log_lambda

    flux: array of float
    Rebinned version of input flux

    ivar: array of float
    Rebinned version of input ivar

    transmission_correction: array of float
    Rebinned version of input transmission_correction

    mean_snr: float
    Mean signal-to-noise of the forest

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
    orig_ivar = ivar.copy()
    w1 = np.ones(log_lambda.size, dtype=bool_)
    pixel_step = np.nan

    # compute bins
    # Remove only out-of-bounds pixels
    if wave_solution == "log":
        pixel_step = log_lambda_grid[1] - log_lambda_grid[0]
        half_pixel_step = pixel_step / 2.

        half_pixel_step_rest_frame = (log_lambda_rest_frame_grid[1] -
                                      log_lambda_rest_frame_grid[0]) / 2.

        w1 &= log_lambda >= log_lambda_grid[0] - half_pixel_step
        w1 &= log_lambda < log_lambda_grid[-1] + half_pixel_step
        w1 &= (log_lambda - np.log10(1. + z) >=
               log_lambda_rest_frame_grid[0] - half_pixel_step_rest_frame)
        w1 &= (log_lambda - np.log10(1. + z) <
               log_lambda_rest_frame_grid[-1] + half_pixel_step_rest_frame)

    elif wave_solution == "lin":
        pixel_step = 10**log_lambda_grid[1] - 10**log_lambda_grid[0]
        half_pixel_step = pixel_step / 2.

        half_pixel_step_rest_frame = (10**log_lambda_rest_frame_grid[1] -
                                      10**log_lambda_rest_frame_grid[0]) / 2.
        lambda_ = 10**log_lambda
        w1 &= (lambda_ >= 10**log_lambda_grid[0] - half_pixel_step)
        w1 &= (lambda_ < 10**log_lambda_grid[-1] + half_pixel_step)
        w1 &= (lambda_ / (1. + z) >=
               10**log_lambda_rest_frame_grid[0] - half_pixel_step_rest_frame)
        w1 &= (lambda_ / (1. + z) <
               10**log_lambda_rest_frame_grid[-1] + half_pixel_step_rest_frame)
    else:
        raise AstronomicalObjectError("Error in Forest.rebin(). "
                                      "Class variable 'wave_solution' "
                                      "must be either 'lin' or 'log'.")

    if (w1 & (ivar > 0.)).sum() == 0:
        log_lambda = np.zeros(0)
        flux = np.zeros(0)
        ivar = np.zeros(0)
        transmission_correction = np.zeros(0)
        mean_snr = 0.0
        bins = np.zeros(0, dtype=np.int64)
        rebin_ivar = np.zeros(0)
        w1 = np.zeros(0, dtype=bool_)
        w2 = np.zeros(0, dtype=bool_)
        return (log_lambda, flux, ivar, transmission_correction, mean_snr, bins,
                rebin_ivar, orig_ivar, w1, w2, w2)

    log_lambda = log_lambda[w1]
    flux = flux[w1]
    ivar = ivar[w1]
    transmission_correction = transmission_correction[w1]
    # Out-of-bounds pixels are removed. IVAR=0 pixels are kept

    bins = find_bins(log_lambda, log_lambda_grid, wave_solution)
    binned_arr_size = bins.max() + 1

    # Find non-empty bins. There will be empty bins
    # at the lower end by construction.
    bincounts = np.bincount(bins, minlength=binned_arr_size)
    wslice_inner = get_inner_region_slice(bincounts)
    final_arr_size = np.sum(wslice_inner)

    # rebin flux, ivar and transmission_correction
    rebin_flux = np.bincount(bins, weights=ivar * flux, minlength=binned_arr_size)
    rebin_transmission_correction = np.bincount(
            bins, weights=(ivar * transmission_correction), minlength=binned_arr_size)
    rebin_ivar = np.bincount(bins, weights=ivar, minlength=binned_arr_size)

    # this condition should always be non-zero for at least one pixel
    # this does not mean that all rebin_ivar pixels will be non-zero,
    # as we could have a masked region of the spectra
    w2_ = (rebin_ivar > 0.) & wslice_inner
    w2  = w2_[wslice_inner]
    flux = np.zeros(final_arr_size)
    transmission_correction = np.zeros(final_arr_size)
    ivar = np.zeros(final_arr_size)

    # Remove the empty pixels at the lower end
    flux[w2] = rebin_flux[w2_] / rebin_ivar[w2_]
    transmission_correction[w2] = rebin_transmission_correction[
        w2_] / rebin_ivar[w2_]
    ivar[w2] = rebin_ivar[w2_]

    # then rebin wavelength
    if wave_solution == "log":
        log_lambda = (log_lambda_grid[0] + pixel_step *
                    np.arange(binned_arr_size)[wslice_inner])
    else:  # we have already checked that it will always be "lin" at this point
        log_lambda = np.log10(10**log_lambda_grid[0] + pixel_step *
                    np.arange(binned_arr_size)[wslice_inner])

    # finally update control variables
    snr = flux * np.sqrt(ivar)
    mean_snr = np.mean(snr[w2])

    # return weights and binning solution to be used by child classes if
    # required
    return (log_lambda, flux, ivar, transmission_correction, mean_snr, bins,
            rebin_ivar, orig_ivar, w1, w2, wslice_inner)

class Forest(AstronomicalObject):
    """Forest Object

    Class Methods
    -------------
    (see AstronomicalObject in py/picca/delta_extraction/astronomical_objects/forest.py)
    class_variable_check
    get_metadata_dtype
    get_metadata_units
    set_class_variables

    Methods
    -------
    (see AstronomicalObject in py/picca/delta_extraction/astronomical_object.py)
    __init__
    consistency_check
    coadd
    get_data
    get_header
    get_metadata
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
    (see AstronomicalObject in py/picca/delta_extraction/astronomical_object.py)

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

    log_lambda_index: array of int or None
    Index of each log_lambda array element in Forest.log_lambda_grid

    logger: logging.Logger
    Logger object

    mean_snr: float
    Mean signal-to-noise of the forest

    transmission_correction: array of float
    Transmission correction.

    weights: array of float or None
    Weights associated to the delta field. None for no information
    """
    blinding = "none"
    log_lambda_grid = np.array([])  #None
    log_lambda_rest_frame_grid = np.array([])  #None
    log_lambda_index = np.array([]) #None
    mask_fields = []  #None
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
        self.logger = logging.getLogger(__name__)

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

        # call parent constructor
        super().__init__(**kwargs)

        self.consistency_check()

        # compute mean quality variables
        snr = self.flux * np.sqrt(self.ivar)
        self.mean_snr = np.mean(snr)

    @classmethod
    def class_variable_check(cls):
        """Check that class variables have been correctly initialized"""
        if cls.log_lambda_grid.size == 0:  # is None:
            raise AstronomicalObjectError(
                "Error constructing Forest. Class variable 'log_lambda_grid' "
                "must be set prior to initialize instances of this type. This "
                "probably means you did not run Forest.set_class_variables")
        if cls.log_lambda_rest_frame_grid.size == 0:  # is None:
            raise AstronomicalObjectError(
                "Error constructing Forest. Class variable "
                "'log_lambda_rest_frame_grid' must be set prior to initialize "
                "instances of this type. This probably means you did not run "
                "Forest.set_class_variables")
        if len(cls.mask_fields) == 0:  #cls.mask_fields is None:
            raise AstronomicalObjectError(
                "Error constructing Forest. Class variable "
                "'mask_fields' must be set prior to initialize "
                "instances of this type. This probably means you did not run "
                "Forest.set_class_variables")
        if not isinstance(cls.mask_fields, list):
            raise AstronomicalObjectError(
                "Error constructing Forest. "
                "Expected list in class variable 'mask fields'. "
                f"Found '{cls.mask_fields}'.")
        if cls.wave_solution is None:
            raise AstronomicalObjectError(
                "Error constructing Forest. Class variable 'wave_solution' "
                "must be set prior to initialize instances of this type. This "
                "probably means you did not run Forest.set_class_variables")

    def consistency_check(self):
        """Consistency checks after __init__"""
        if self.flux.size != self.ivar.size:
            raise AstronomicalObjectError("Error constructing Forest. 'flux' "
                                          "and 'ivar' don't have the same size")
        if self.log_lambda.size != self.flux.size:
            raise AstronomicalObjectError("Error constructing Forest. "
                                          "'flux' and 'log_lambda' don't "
                                          "have the same size")

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
                                          f"{type(other).__name__}")

        if self.los_id != other.los_id:
            raise AstronomicalObjectError(
                "Attempting to coadd two Forests "
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
            raise AstronomicalObjectError("Error in Forest.get_data(). "
                                          "Class variable 'wave_solution' "
                                          "must be either 'lin' or 'log'. "
                                          f"Found: '{Forest.wave_solution}'")

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
            },
            {
                'name': 'WAVE_SOLUTION',
                'value': Forest.wave_solution,
                'comment': "Chosen wavelength solution (linnear or logarithmic)"
            },
        ]

        if Forest.wave_solution == "log":
            header += [
                {
                    'name':
                        'DELTA_LOG_LAMBDA',
                    'value':
                        Forest.log_lambda_grid[1] - Forest.log_lambda_grid[0],
                    'comment':
                        "Pixel step in log lambda [log(Angstrom)]"
                },
            ]
        elif Forest.wave_solution == "lin":
            header += [
                {
                    'name':
                        'DELTA_LAMBDA',
                    'value':
                        10**Forest.log_lambda_grid[1] -
                        10**Forest.log_lambda_grid[0],
                    'comment':
                        "Pixel step in lambda [Angstrom]"
                },
            ]
        else:
            raise AstronomicalObjectError("Error in Forest.get_header(). "
                                          "Class variable 'wave_solution' "
                                          "must be either 'lin' or 'log'. "
                                          f"Found: '{Forest.wave_solution}'")

        return header

    def get_metadata(self):
        """Return line-of-sight data as a list. Names and types of the variables
        are given by Forest.get_metadata_dtype. Units are given by
        Forest.get_metadata_units

        Return
        ------
        metadata: list
        A list containing the line-of-sight data
        """
        metadata = super().get_metadata()
        metadata += [self.mean_snr,]
        return metadata

    @classmethod
    def get_metadata_dtype(cls):
        """Return the types and names of the line-of-sight data returned by
        method self.get_metadata

        Return
        ------
        metadata_dtype: list
        A list with tuples containing the name and data type of the line-of-sight
        data
        """
        dtype = super().get_metadata_dtype()
        dtype += [('MEANSNR', float),]
        return dtype

    @classmethod
    def get_metadata_units(cls):
        """Return the units of the line-of-sight data returned by
        method self.get_metadata

        Return
        ------
        metadata_units: list
        A list with the units of the line-of-sight data
        """
        units = super().get_metadata_units()
        units += [""]
        return units

    def rebin(self):
        """Rebin the arrays and update control variables
        Rebinned arrays are flux, ivar, lambda_ or log_lambda, and
        transmission_correction. Control variables are mean_snr

        Return
        ------
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
        (self.log_lambda, self.flux, self.ivar, self.transmission_correction,
         self.mean_snr, self.log_lambda_index, rebin_ivar, orig_ivar, w1,
         w2, wslice_inner) = rebin(self.log_lambda, self.flux, self.ivar,
                     self.transmission_correction, self.z, Forest.wave_solution,
                     Forest.log_lambda_grid, Forest.log_lambda_rest_frame_grid)

        # return weights and binning solution to be used by child classes if
        # required
        return rebin_ivar, orig_ivar, w1, w2, wslice_inner

    @classmethod
    def set_class_variables(cls, lambda_min, lambda_max, lambda_min_rest_frame,
                            lambda_max_rest_frame, pixel_step,
                            pixel_step_rest_frame, wave_solution):
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
                np.log10(lambda_max) + pixel_step / 2, pixel_step)
            cls.log_lambda_rest_frame_grid = np.arange(
                np.log10(lambda_min_rest_frame) + pixel_step_rest_frame / 2,
                np.log10(lambda_max_rest_frame), pixel_step_rest_frame)
        elif wave_solution == "lin":
            cls.log_lambda_grid = np.log10(
                np.arange(lambda_min, lambda_max + pixel_step / 2, pixel_step))
            cls.log_lambda_rest_frame_grid = np.log10(
                np.arange(lambda_min_rest_frame + pixel_step_rest_frame / 2,
                          lambda_max_rest_frame, pixel_step_rest_frame))
        else:
            raise AstronomicalObjectError("Error in setting Forest class "
                                          "variables. 'wave_solution' "
                                          "must be either 'lin' or 'log'. "
                                          f"Found: {wave_solution}")

        cls.wave_solution = wave_solution

        cls.mask_fields = defaults.get("mask fields").copy()
