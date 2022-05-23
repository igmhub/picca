
import logging
import multiprocessing

import fitsio
import iminuit
import numpy as np
from scipy.interpolate import interp1d

from picca.delta_extraction.astronomical_objects.forest import Forest
from picca.delta_extraction.astronomical_objects.pk1d_forest import Pk1dForest
from picca.delta_extraction.errors import ExpectedFluxError, AstronomicalObjectError
from picca.delta_extraction.expected_flux import ExpectedFlux
from picca.delta_extraction.expected_fluxes.dr16_expected_flux import Dr16ExpectedFlux
from picca.delta_extraction.utils import find_bins

accepted_options = [
    "fudge_value",
]

defaults = {
    "fudge_value": 0.0,
}

class Dr16FixedFudgeExpectedFlux(Dr16ExpectedFlux):
    """Class to the expected flux similar to Dr16ExpectedFlux but fixing the
    fudge factor

    Methods
    -------
    (see Dr16ExpectedFlux in py/picca/delta_extraction/expected_fluxes/dr16_expected_flux.py)
    __init__
    __initialize_variance_functions
    __parse_config
    compute_var_stats
        chi2

    Attributes
    ----------
    (see Dr16ExpectedFlux in py/picca/delta_extraction/expected_fluxes/dr16_expected_flux.py)

    fudge_value: float
    The applied fudge value
    """
    def __init__(self, config):
        """Initialize class instance.

        Arguments
        ---------
        config: configparser.SectionProxy
        Parsed options to initialize class

        Raise
        -----
        ExpectedFluxError if Forest class variables are not set
        """
        # load variables from config
        self.fudge_value = None
        self.__parse_config(config)

        super().__init__(config)

    def __initialize_variance_functions(self):
        """Initialize variance functions
        The initialized arrays are:
        - self.get_eta
        - self.get_num_pixels
        - self.get_valid_fit
        - self.get_var_lss
        """
        eta = np.ones(self.num_bins_variance)
        var_lss = np.zeros(self.num_bins_variance) + 0.2
        num_pixels = np.zeros(self.num_bins_variance)
        valid_fit = np.zeros(self.num_bins_variance, dtype=bool)

        self.get_eta = interp1d(self.log_lambda_var_func_grid,
                                eta,
                                fill_value='extrapolate',
                                kind='nearest')
        self.get_var_lss = interp1d(self.log_lambda_var_func_grid,
                                    var_lss,
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

    def __parse_config(self, config):
        """Parse the configuration options

        Arguments
        ---------
        config: configparser.SectionProxy
        Parsed options to initialize class

        Raises
        ------
        ExpectedFluxError if iter out prefix is not valid
        """
        self.fudge_value = config.get("fudge value")
        if self.fudge_value is None:
            raise ExpectedFluxError(
                "Missing argument 'fudge value' required "
                "by Dr16FixFudgeExpectedFlux")

    def compute_var_stats(self, forests):
        """Compute variance functions and statistics

        This function computes the statistics required to fit the mapping functions
        eta, var_lss. It also computes the functions themselves. See
        equation 4 of du Mas des Bourboux et al. 2020 for details.

        Arguments
        ---------
        forests: List of Forest
        A list of Forest from which to compute the deltas.

        Raise
        -----
        ExpectedFluxError if wavelength solution is not valid
        """
        # initialize arrays
        eta = np.zeros(self.num_bins_variance)
        var_lss = np.zeros(self.num_bins_variance)
        error_eta = np.zeros(self.num_bins_variance)
        error_var_lss = np.zeros(self.num_bins_variance)
        num_pixels = np.zeros(self.num_bins_variance)
        valid_fit = np.zeros(self.num_bins_variance)

        # define an array to contain the possible values of pipeline variances
        # the measured pipeline variance of the deltas will be averaged using the
        # same binning, and the two arrays will be compared to fit the functions
        # eta, var_lss
        num_var_bins = 100  # TODO: update this to self.num_bins_variance
        var_pipe_min = np.log10(1e-5)
        var_pipe_max = np.log10(2.)
        var_pipe_values = 10**(var_pipe_min +
                               ((np.arange(num_var_bins) + .5) *
                                (var_pipe_max - var_pipe_min) / num_var_bins))

        # initialize arrays to compute the statistics of deltas
        var_delta = np.zeros(self.num_bins_variance * num_var_bins)
        mean_delta = np.zeros(self.num_bins_variance * num_var_bins)
        var2_delta = np.zeros(self.num_bins_variance * num_var_bins)
        count = np.zeros(self.num_bins_variance * num_var_bins)
        num_qso = np.zeros(self.num_bins_variance * num_var_bins)

        # compute delta statistics, binning the variance according to 'ivar'
        for forest in forests:
            # ignore forest if continuum could not be computed
            if forest.continuum is None:
                continue
            var_pipe = 1 / forest.ivar / forest.continuum**2
            w = ((np.log10(var_pipe) > var_pipe_min) &
                 (np.log10(var_pipe) < var_pipe_max))

            # select the pipeline variance bins
            var_pipe_bins = np.floor(
                (np.log10(var_pipe[w]) - var_pipe_min) /
                (var_pipe_max - var_pipe_min) * num_var_bins).astype(int)

            # select the wavelength bins
            log_lambda_bins = find_bins(forest.log_lambda[w],
                                        self.log_lambda_var_func_grid,
                                        Forest.wave_solution)

            # compute overall bin
            bins = var_pipe_bins + num_var_bins * log_lambda_bins

            # compute deltas
            delta = (forest.flux / forest.continuum - 1)
            delta = delta[w]

            # add contributions to delta statistics
            rebin = np.bincount(bins, weights=delta)
            mean_delta[:len(rebin)] += rebin

            rebin = np.bincount(bins, weights=delta**2)
            var_delta[:len(rebin)] += rebin

            rebin = np.bincount(bins, weights=delta**4)
            var2_delta[:len(rebin)] += rebin

            rebin = np.bincount(bins)
            count[:len(rebin)] += rebin
            num_qso[np.unique(bins)] += 1

        # normalise and finish the computation of delta statistics
        w = count > 0
        var_delta[w] /= count[w]
        mean_delta[w] /= count[w]
        var_delta -= mean_delta**2
        var2_delta[w] /= count[w]
        var2_delta -= var_delta**2
        var2_delta[w] /= count[w]

        # fit the functions eta, var_lss
        chi2_in_bin = np.zeros(self.num_bins_variance)

        self.logger.progress(" Mean quantities in observer-frame")
        self.logger.progress(
            " loglam    eta      var_lss    chi2     num_pix valid_fit")
        for index in range(self.num_bins_variance):
            # pylint: disable-msg=cell-var-from-loop
            # this function is defined differntly at each step of the loop
            def chi2(eta, var_lss):
                """Compute the chi2 of the fit of eta, var_lss for a
                wavelength bin

                Arguments
                ---------
                eta: float
                Correction factor to the contribution of the pipeline
                estimate of the instrumental noise to the variance.

                var_lss: float
                Pixel variance due to the Large Scale Strucure

                Global arguments
                ----------------
                (defined only in the scope of function compute_var_stats):

                var_delta: array of floats
                Variance of the delta field

                var2_delta: array of floats
                Square of the variance of the delta field

                index: int
                Index with the selected wavelength bin

                num_var_bins: int
                Number of bins in which the pipeline variance values are split

                var_pipe_values: array of floats
                Value of the pipeline variance in pipeline variance bins

                num_qso: array of ints
                Number of quasars in each pipeline variance bin

                Return
                ------
                chi2: float
                The obtained chi2
                """
                variance = eta * var_pipe_values + var_lss + self.fudge_value / var_pipe_values
                chi2_contribution = (
                    var_delta[index * num_var_bins:(index + 1) * num_var_bins] -
                    variance)
                weights = var2_delta[index * num_var_bins:(index + 1) *
                                     num_var_bins]
                w = num_qso[index * num_var_bins:(index + 1) *
                            num_var_bins] > 100
                return np.sum(chi2_contribution[w]**2 / weights[w])

            minimizer = iminuit.Minuit(chi2,
                                       name=("eta", "var_lss"),
                                       eta=1.,
                                       var_lss=0.1)
            minimizer.errors["eta"] = 0.05
            minimizer.errors["var_lss"] = 0.05
            minimizer.errordef = 1.
            minimizer.print_level = 0
            minimizer.limits["eta"] = self.limit_eta
            minimizer.limits["var_lss"] = self.limit_var_lss
            minimizer.migrad()

            if minimizer.valid:
                minimizer.hesse()
                eta[index] = minimizer.values["eta"]
                var_lss[index] = minimizer.values["var_lss"]
                error_eta[index] = minimizer.errors["eta"]
                error_var_lss[index] = minimizer.errors["var_lss"]
                valid_fit[index] = True
            else:
                eta[index] = 1.
                var_lss[index] = 0.1
                error_eta[index] = 0.
                error_var_lss[index] = 0.
                valid_fit[index] = False
            num_pixels[index] = count[index * num_var_bins:(index + 1) *
                                      num_var_bins].sum()
            chi2_in_bin[index] = minimizer.fval

            self.logger.progress(
                f" {self.log_lambda_var_func_grid[index]:.3e} "
                f"{eta[index]:.2e} {var_lss[index]:.2e} " +
                f"{chi2_in_bin[index]:.2e} {num_pixels[index]:.2e} {valid_fit[index]}"
            )

        w = num_pixels > 0

        self.get_eta = interp1d(self.log_lambda_var_func_grid[w],
                                eta[w],
                                fill_value="extrapolate",
                                kind="nearest")
        self.get_var_lss = interp1d(self.log_lambda_var_func_grid[w],
                                    var_lss[w],
                                    fill_value="extrapolate",
                                    kind="nearest")
        self.get_num_pixels = interp1d(self.log_lambda_var_func_grid[w],
                                       num_pixels[w],
                                       fill_value="extrapolate",
                                       kind="nearest")
        self.get_valid_fit = interp1d(self.log_lambda_var_func_grid[w],
                                      valid_fit[w],
                                      fill_value="extrapolate",
                                      kind="nearest")
