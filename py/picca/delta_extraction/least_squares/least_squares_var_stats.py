"""This module defines the class LeastsSquaresVarStats"""
import numpy as np

from picca.delta_extraction.astronomical_objects.forest import Forest
from picca.delta_extraction.utils import find_bins

VAR_PIPE_MIN = np.log10(1e-5)
VAR_PIPE_MAX = np.log10(2.)
NUM_VAR_BINS = 100

FUDGE_REF = 1e-7


class LeastsSquaresVarStats:
    """This class deals with the continuum fitting.

    It is passed to iminuit and when called it will return the chi2 for a given
    set of parameters

    Methods
    -------
    __init__
    __call__
    initialize_delta_arrays
    get_num_pixels
    set_fit_bins

    Attributes
    ----------
    log_lambda_var_func_grid: array of float
    Wavelength array where variance functions will be computed.

    num_bins_variance: int
    Number of bins to be used to compute variance functions and statistics
    as a function of wavelength.

    num_pixels: array of int
    Number of pixels participating in each bin of var_delta and var2_delta

    num_qso: array of int
    Number of quasars participating in each bin of var_delta and var2_delta

    running_indexs: (int, int)
    Tuple indicating the selected bins for the fits (fits are
    wavelength-independent)

    var2_delta: array of float
    Square of the total variance of deltas for a bin in wavelength and pipeline
    variance. Dimension is (num_bins_variance, num_var_bins)

    var_delta: array of float
    Total variance of deltas for a bin in wavelength and pipeline variance.
    Dimension is (num_bins_variance, num_var_bins)

    var_pipe_values: array of float
    Array where the variance functions will be computed
    """

    def __init__(
        self,
        num_bins_variance,
        forests,
        log_lambda_var_func_grid,
        min_num_qso_in_fit,
    ):
        """Initialize class instances

        Arguments
        ---------
        num_bins_variance: int
        Number of bins to be used to compute variance functions and statistics
        as a function of wavelength.

        forests: list of Forest
        A list of Forest from which to compute the deltas.

        log_lambda_var_func_grid: array of float
        Wavelength array where variance functions will be computed.

        min_num_qso_in_fit: int
        Minimum number of quasars contributing to a bin of wavelength and pipeline
        variance in order to consider it in the fit
        """
        self.num_bins_variance = num_bins_variance
        self.log_lambda_var_func_grid = log_lambda_var_func_grid
        self.min_num_qso_in_fit = min_num_qso_in_fit

        # define an array to contain the possible values of pipeline variances
        # the measured pipeline variance of the deltas will be averaged using the
        # same binning, and the two arrays will be compared to fit the functions
        # eta, var_lss, and fudge
        self.var_pipe_values = 10**(
            VAR_PIPE_MIN + ((np.arange(NUM_VAR_BINS) + .5) *
                            (VAR_PIPE_MAX - VAR_PIPE_MIN) / NUM_VAR_BINS))

        # initialize arrays to compute the statistics of deltas
        self.var_delta = None
        self.var2_delta = None
        self.num_pixels = None
        self.num_qso = None
        self.initialize_delta_arrays(forests)

        self.running_indexs = (np.nan, np.nan)

    def __call__(self, eta, var_lss, fudge):
        """Compute the chi2 of the fit of eta, var_lss, and fudge for a
        wavelength bin

        Arguments
        ---------
        eta: float
        Correction factor to the contribution of the pipeline
        estimate of the instrumental noise to the variance.

        var_lss: float
        Pixel variance due to the Large Scale Strucure

        fudge: float
        Fudge contribution to the pixel variance

        Returns
        -------
        chi2: float
        The chi2 for this run
        """
        variance = eta * self.var_pipe_values + var_lss + fudge * FUDGE_REF / self.var_pipe_values
        chi2_contribution = (
            self.var_delta[self.running_indexs[0]:self.running_indexs[1]] -
            variance)
        weights = self.var2_delta[self.running_indexs[0]:self.running_indexs[1]]
        w = self.num_qso[self.running_indexs[0]:
                         self.running_indexs[1]] > self.min_num_qso_in_fit
        return np.sum(chi2_contribution[w]**2 / weights[w])

    def initialize_delta_arrays(self, forests):
        """Initialize arrays to compute the statistics of deltas

        Arguments
        ---------
        forests: list of Forest
        A list of Forest from which to compute the deltas.
        """
        var_delta = np.zeros(self.num_bins_variance * NUM_VAR_BINS)
        mean_delta = np.zeros(self.num_bins_variance * NUM_VAR_BINS)
        var2_delta = np.zeros(self.num_bins_variance * NUM_VAR_BINS)
        num_pixels = np.zeros(self.num_bins_variance * NUM_VAR_BINS)
        num_qso = np.zeros(self.num_bins_variance * NUM_VAR_BINS)

        # compute delta statistics, binning the variance according to 'ivar'
        for forest in forests:
            # ignore forest if continuum could not be computed
            if forest.continuum is None:
                continue

            w =  forest.ivar > 0
            var_pipe = np.empty_like(forest.log_lambda)
            var_pipe[w]  = 1 / forest.ivar[w] / forest.continuum[w]**2
            var_pipe[~w] = np.inf

            w &= ((np.log10(var_pipe) > VAR_PIPE_MIN) &
                 (np.log10(var_pipe) < VAR_PIPE_MAX))

            # select the pipeline variance bins
            var_pipe_bins = np.floor(
                (np.log10(var_pipe[w]) - VAR_PIPE_MIN) /
                (VAR_PIPE_MAX - VAR_PIPE_MIN) * NUM_VAR_BINS).astype(int)

            # select the wavelength bins
            log_lambda_bins = find_bins(forest.log_lambda[w],
                                        self.log_lambda_var_func_grid,
                                        Forest.wave_solution)

            # compute overall bin
            bins = var_pipe_bins + NUM_VAR_BINS * log_lambda_bins

            # compute deltas
            delta = forest.flux[w] / forest.continuum[w] - 1

            # add contributions to delta statistics
            mean_delta += np.bincount(bins, weights=delta, minlength=mean_delta.size)
            var_delta  += np.bincount(bins, weights=delta**2, minlength=mean_delta.size)
            var2_delta += np.bincount(bins, weights=delta**4, minlength=mean_delta.size)

            num_pixels += np.bincount(bins, minlength=mean_delta.size)
            num_qso[np.unique(bins)] += 1

        # normalise and finish the computation of delta statistics
        w = num_pixels > 0
        var_delta[w] /= num_pixels[w]
        mean_delta[w] /= num_pixels[w]
        var_delta -= mean_delta**2
        var2_delta[w] /= num_pixels[w]
        var2_delta -= var_delta**2
        var2_delta[w] /= num_pixels[w]

        self.var_delta = var_delta
        self.var2_delta = var2_delta
        self.num_qso = num_qso
        self.num_pixels = num_pixels

    def get_num_pixels(self):
        """Return the number of pixels participating in the fit"""
        return self.num_pixels[self.running_indexs[0]:self.
                               running_indexs[1]].sum()

    def set_fit_bins(self, index):
        """Set the  selected bins for the fits
        (fits are wavelength-independent)

        Arguments
        ---------
        index: int
        Index of the wavelength grid activated to fit
        """
        self.running_indexs = (index * NUM_VAR_BINS, (index + 1) * NUM_VAR_BINS)
