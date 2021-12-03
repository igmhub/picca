"""This module defines the class Dr16ExpectedFlux"""
import logging
import multiprocessing

import fitsio
import iminuit
import numpy as np
from scipy.interpolate import interp1d

from picca.delta_extraction.astronomical_objects.forest import Forest
from picca.delta_extraction.astronomical_objects.pk1d_forest import Pk1dForest
from picca.delta_extraction.errors import ExpectedFluxError
from picca.delta_extraction.expected_flux import ExpectedFlux, defaults

defaults.update({
    "iter out prefix": "delta_attributes",
    "limit eta": (0.5, 1.5),
    "limit var lss": (0., 0.3),
    "num bins variance": 20,
    "num iterations": 5,
    "order": 1,
    "use_constant_weight": False,
    "use_ivar_as_weight": False,
})


class TrueContinuum(ExpectedFlux):
    """Class to the expected flux as done in the DR16 SDSS analysys
    The mean expected flux is calculated iteratively as explained in
    du Mas des Bourboux et al. (2020)

    Methods
    -------
    extract_deltas (from ExpectedFlux)
    __init__
    _initialize_arrays_lin
    _initialize_arrays_log
    _parse_config
    compute_continuum
        get_cont_model
        chi2
    compute_delta_stack
    compute_mean_cont_lin
    compute_mean_cont_log
    compute_expected_flux
    compute_var_stats
        chi2

    Attributes
    ----------
    los_ids: dict (from ExpectedFlux)
    A dictionary to store the mean expected flux fraction, the weights, and
    the inverse variance for each line of sight. Keys are the identifier for the
    line of sight and values are dictionaries with the keys "mean expected flux",
    and "weights" pointing to the respective arrays. If the given Forests are
    also Pk1dForests, then the key "ivar" must be available. Arrays have the same
    size as the flux array for the corresponding line of sight forest instance.

    continuum_fit_parameters: dict
    A dictionary containing the continuum fit parameters for each line of sight.
    Keys are the identifier for the line of sight and values are tuples with
    the best-fit zero point and slope of the linear part of the fit.

    get_eta: scipy.interpolate.interp1d
    Interpolation function to compute mapping function eta. See equation 4 of
    du Mas des Bourboux et al. 2020 for details.

    get_fudge: scipy.interpolate.interp1d
    Interpolation function to compute mapping function fudge. See equation 4 of
    du Mas des Bourboux et al. 2020 for details.

    get_mean_cont: scipy.interpolate.interp1d
    Interpolation function to compute the unabsorbed mean quasar continua

    get_stack_delta: scipy.interpolate.interp1d or None
    Interpolation function to compute the mean delta (from stacking all lines of
    sight). None for no info.

    get_var_lss: scipy.interpolate.interp1d
    Interpolation function to compute mapping functions var_lss. See equation 4 of
    du Mas des Bourboux et al. 2020 for details.

    iter_out_prefix: str
    Prefix of the iteration files. These files contain the statistical properties
    of deltas at a given iteration step. Intermediate files will add
    '_iteration{num}.fits.gz' to the prefix for intermediate steps and '.fits.gz'
    for the final results.

    lambda_: array of float or None
    Wavelengths where the variance functions and statistics are
    computed. None (and unused) for a logarithmic wavelength solution.

    lambda_rest_frame: array of float or None
    Rest-frame wavelengths where the unabsorbed mean quasar continua is are
    computed. None (and unused) for a logarithmic wavelength solution.

    limit_eta: tuple of floats
    Limits on the correction factor to the contribution of the pipeline estimate
    of the instrumental noise to the variance.

    limit_var_lss: tuple of floats
    Limits on the pixel variance due to Large Scale Structure

    log_lambda: array of float or None
    Logarithm of the rest frame wavelengths where the variance functions and
    statistics are computed. None (and unused) for a linear wavelength solution.

    log_lambda_rest_frame: array of float or None
    Logarithm of the rest-frame wavelengths where the unabsorbed mean quasar
    continua is are computed. None (and unused) for a linear wavelength solution.

    num_bins_variance: int
    Number of bins to be used to compute variance functions and statistics as
    a function of wavelength.

    num_iterations: int
    Number of iterations to determine the mean continuum shape, LSS variances, etc.

    num_processors: int or None
    Number of processors to be used to compute the mean continua. None for no
    specified number (subprocess will take its default value).

    use_constant_weight: boolean
    If "True", set all the delta weights to one (implemented as eta = 0,
    sigma_lss = 1, fudge = 0)

    use_ivar_as_weight: boolean
    If "True", use ivar as weights (implemented as eta = 1, sigma_lss = fudge = 0)
    """

    def __init__(self, config):
        """Initialize class instance.

        Arguments
        ---------
        config: configparser.SectionProxy
        Parsed options to initialize class

        Raise
        -----
        ExpectedFluxError if Forest.wave_solution is not 'lin' or 'log'
        """
        self.logger = logging.getLogger(__name__)
        super().__init__()

        # load variables from config
        self._parse_config(config)

        # initialize variables

        self.continuum_fit_parameters = None

        self.get_stack_delta = None
        self.get_stack_delta_weights = None

    def _parse_config(self, config):
        """Parse the configuration options

        Arguments
        ---------
        config: configparser.SectionProxy
        Parsed options to initialize class

        Raises
        ------
        ExpectedFluxError if iter out prefix is not valid
        """
        raise ExpectedFluxError("Needs filling")

    def read_true_continuum(self, forest):
        """Read the forest continuum and insert it into

        Arguments
        ---------
        forest: Forest
        A forest instance where the continuum will be computed

        Return
        ------
        forest: Forest
        The modified forest instance

        Raise
        -----
        ExpectedFluxError if Forest.wave_solution is not 'lin' or 'log'
        """
        forest.continuum = None
        return forest

    def read_var_lss(self):
        """

        Arguments
        ---------

        Raise
        -----

        """

    def compute_expected_flux(self, forests):
        """

        Arguments
        ---------
        forests: List of Forest
        A list of Forest from which to compute the deltas.

        out_dir: str
        Directory where iteration statistics will be saved

        Raise
        -----
        ExpectedFluxError if Forest.wave_solution is not 'lin' or 'log'
        """

        # now loop over forests to populate los_ids
        self.populate_los_ids(forests)


    def populate_los_ids(self, forests):
        """Populate the dictionary los_ids with the mean expected flux, weights,
        and inverse variance arrays for each line-of-sight.

        Arguments
        ---------
        forests: List of Forest
        A list of Forest from which to compute the deltas.
        """
        for forest in forests:
            if forest.bad_continuum_reason is not None:
                self.logger.info(f"Rejected forest with los_id {forest.los_id} "
                                 f"due to {forest.bad_continuum_reason}")
                continue
            # get the variance functions and statistics
            log_lambda = forest.log_lambda
            stack_delta = self.get_stack_delta(log_lambda)
            var_lss = self.get_var_lss(log_lambda)
            eta = self.get_eta(log_lambda)
            fudge = self.get_fudge(log_lambda)

            mean_expected_flux = forest.continuum * stack_delta
            var_pipe = 1. / forest.ivar / mean_expected_flux**2
            variance = eta * var_pipe + var_lss + fudge / var_pipe
            weights = 1. / variance

            if isinstance(forest, Pk1dForest):
                ivar = forest.ivar / (eta +
                                      (eta == 0)) * (mean_expected_flux**2)

                self.los_ids[forest.los_id] = {
                    "mean expected flux": mean_expected_flux,
                    "weights": weights,
                    "ivar": ivar,
                    "continuum": forest.continuum,
                }
            else:
                self.los_ids[forest.los_id] = {
                    "mean expected flux": mean_expected_flux,
                    "weights": weights,
                    "continuum": forest.continuum,
                }
