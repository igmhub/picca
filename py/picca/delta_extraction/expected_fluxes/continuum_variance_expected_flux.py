"""This module defines the class Dr16ExpectedFlux"""
import logging
import multiprocessing

import numpy as np
from scipy.interpolate import interp1d

from picca.delta_extraction.astronomical_objects.forest import Forest
from picca.delta_extraction.expected_fluxes.dr16_fixed_eta_varlss_expected_flux import (
    Dr16FixedEtaVarlssExpectedFlux)
from picca.delta_extraction.expected_fluxes.dr16_fixed_eta_varlss_expected_flux import (  # pylint: disable=unused-import
    accepted_options, defaults)
from picca.delta_extraction.utils import find_bins

VAR_PIPE_MIN = np.log10(1e-5)
VAR_PIPE_MAX = np.log10(2.)


class ContinuumVarianceExpectedFlux(Dr16FixedEtaVarlssExpectedFlux):
    """Class to the expected flux with a variance term dependent
    on the rest frame wavelength.

    Methods
    -------
    (see Dr16FixedEtaVarlssExpectedFlux in
        py/picca/delta_extraction/expected_fluxes/dr16_fixed_eta_varlss_expected_flux.py)
    __init__
    _initialize_variance_functions
    __parse_config
    compute_expected_flux
    compute_forest_variance
    compute_var_stats
    hdu_cont
    hdu_var_func

    Attributes
    ----------
    (see Dr16FixedEtaVarlssExpectedFlux in
        py/picca/delta_extraction/expected_fluxes/dr16_fixed_eta_varlss_expected_flux.py)

    self.get_var_cont: scipy.interpolate.interp1d
    Variance contribution due to continuum (as a function of rest-frame wavelength)
    """

    def __init__(self, config):
        """Initialize class instance.

        Arguments
        ---------
        config: configparser.SectionProxy
        Parsed options to initialize class
        """
        self.logger = logging.getLogger(__name__)

        self.get_var_cont = None
        super().__init__(config)

    def _initialize_variance_functions(self):
        """Initialize variance functions
        The initialized arrays are:
        - self.get_eta
        - self.get_var_cont
        - self.get_num_pixels
        - self.get_valid_fit
        - self.get_var_lss
        """
        var_cont = np.zeros_like(Forest.log_lambda_rest_frame_grid)
        num_pixels = np.zeros(self.num_bins_variance)
        valid_fit = np.zeros(self.num_bins_variance, dtype=bool)

        self._initialize_get_eta()
        self._initialize_get_var_lss()
        self.get_var_cont = interp1d(Forest.log_lambda_rest_frame_grid,
                                     var_cont,
                                     fill_value='extrapolate',
                                     kind='nearest')
        self.get_num_pixels = interp1d(self.log_lambda_var_func_grid,
                                       num_pixels,
                                       fill_value="extrapolate",
                                       kind='nearest')
        self.get_valid_fit = interp1d(self.log_lambda_var_func_grid,
                                      valid_fit,
                                      fill_value="extrapolate",
                                      kind='nearest')

    def compute_expected_flux(self, forests):
        """Compute the mean expected flux of the forests.

        Arguments
        ---------
        forests: List of Forest
        A list of Forest from which to compute the deltas.
        """
        context = multiprocessing.get_context('fork')
        if self.num_processors > 1:
            with context.Pool(processes=self.num_processors) as pool:
                forests = pool.map(self.compute_continuum, forests)
        else:
            forests = [self.compute_continuum(f) for f in forests]

        # Compute mean continuum (stack in rest-frame)
        self.compute_mean_cont(forests)
        # Compute observer-frame mean quantities (var_lss, eta, fudge)
        if not (self.use_ivar_as_weight or self.use_constant_weight):
            self.compute_var_stats(forests)

        # compute the mean deltas
        self.compute_delta_stack(forests)

        # Save the iteration step
        self.save_iteration_step(-1)

        # now loop over forests to populate los_ids
        self.populate_los_ids(forests)

    def compute_forest_variance(self, forest, continuum):
        """Compute the forest variance following Du Mas 2020

        Arguments
        ---------
        forest: Forest
        A forest instance where the variance will be computed

        continuum: array of float
        Quasar continuum associated with the forest
        """
        var_pipe = 1. / forest.ivar / continuum**2
        var_lss = self.get_var_lss(forest.log_lambda)
        eta = self.get_eta(forest.log_lambda)
        sigma_c = self.get_var_cont(forest.log_lambda - np.log10(1 + forest.z))

        return eta * var_pipe + var_lss + sigma_c

    def compute_var_stats(self, forests):
        """Compute variance functions and statistics

        ## @ add description.

        Arguments
        ---------
        forests: List of Forest
        A list of Forest from which to compute the deltas.

        Raise
        -----
        ExpectedFluxError if wavelength solution is not valid
        """
        # initialize arrays to compute the statistics of deltas
        var_delta = np.zeros_like(self.log_lambda_var_func_grid)
        mean_delta = np.zeros_like(self.log_lambda_var_func_grid)
        var2_delta = np.zeros_like(self.log_lambda_var_func_grid)
        var_cont_mean = np.zeros_like(Forest.log_lambda_rest_frame_grid)
        var_cont_count = np.zeros_like(Forest.log_lambda_rest_frame_grid)
        count = np.zeros_like(self.log_lambda_var_func_grid)
        num_qso = np.zeros_like(self.log_lambda_var_func_grid)

        # compute delta statistics, binning the variance according to 'ivar'
        self.logger.progress("Computing delta statistics")
        for forest in forests:
            # ignore forest if continuum could not be computed
            if forest.continuum is None:
                continue
            var_pipe = 1 / forest.ivar / forest.continuum**2
            w = ((np.log10(var_pipe) > VAR_PIPE_MIN) &
                 (np.log10(var_pipe) < VAR_PIPE_MAX))

            # select the wavelength bins
            log_lambda_bins = find_bins(
                forest.log_lambda,
                self.log_lambda_var_func_grid,
                Forest.wave_solution,
            )[w]

            log_lambda_rest_bins = find_bins(
                forest.log_lambda - np.log10(1 + forest.z),
                Forest.log_lambda_rest_frame_grid,
                Forest.wave_solution,
            )[w]

            # compute deltas
            delta = (forest.flux / forest.continuum - 1)
            delta = delta[w]

            # add contributions to delta statistics
            rebin = np.bincount(log_lambda_bins, weights=delta)
            mean_delta[:len(rebin)] += rebin

            rebin = np.bincount(log_lambda_bins, weights=delta**2)
            var_delta[:len(rebin)] += rebin

            # compute continuum dependent variance factor
            eta = self.get_eta(forest.log_lambda[w])
            var_lss = self.get_var_lss(forest.log_lambda[w])
            var_cont = delta**2 - var_lss - eta * var_pipe[w]

            rebin = np.bincount(log_lambda_rest_bins, weights=var_cont)
            var_cont_mean[:len(rebin)] += rebin

            rebin = np.bincount(log_lambda_rest_bins)
            var_cont_count[:len(rebin)] += rebin

            rebin = np.bincount(log_lambda_bins, weights=delta**4)
            var2_delta[:len(rebin)] += rebin

            rebin = np.bincount(log_lambda_bins)
            count[:len(rebin)] += rebin
            num_qso[np.unique(log_lambda_bins)] += 1

        # normalise and finish the computation of delta statistics
        w = count > 0
        var_delta[w] /= count[w]
        mean_delta[w] /= count[w]
        var_delta -= mean_delta**2
        var2_delta[w] /= count[w]
        var2_delta -= var_delta**2
        var2_delta[w] /= count[w]

        w = var_cont_count > 0
        var_cont_mean[w] /= var_cont_count[w]

        self.get_var_cont = interp1d(Forest.log_lambda_rest_frame_grid[w],
                                     var_cont_mean[w],
                                     fill_value='extrapolate',
                                     kind='nearest')

    def hdu_cont(self, results):
        """Add to the results file an HDU with the continuum information

        Arguments
        ---------
        results: fitsio.FITS
        The open fits file
        """
        results.write([
            Forest.log_lambda_rest_frame_grid,
            self.get_mean_cont(Forest.log_lambda_rest_frame_grid),
            self.get_mean_cont_weight(Forest.log_lambda_rest_frame_grid),
            self.get_var_cont(Forest.log_lambda_rest_frame_grid),
        ],
                      names=['loglam_rest', 'mean_cont', 'weight', 'cont_var'],
                      extname='CONT')

    def hdu_var_func(self, results):
        """Add to the results file an HDU with the variance functions

        Arguments
        ---------
        results: fitsio.FITS
        The open fits file
        """
        results.write(
            [
                self.log_lambda_var_func_grid,
                self.get_eta(self.log_lambda_var_func_grid),
                self.get_var_lss(self.log_lambda_var_func_grid),
                self.get_num_pixels(self.log_lambda_var_func_grid),
                self.get_valid_fit(self.log_lambda_var_func_grid)
            ],
            names=['loglam', 'eta', 'var_lss', 'num_pixels', 'valid_fit'],
            extname='VAR_FUNC')
