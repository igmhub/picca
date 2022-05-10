
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

fudge_fixed_value = 0

accepted_options = [
    "iter out prefix", "limit eta", "limit var lss", "num bins variance",
    "num iterations", "num processors", "order", "out dir",
    "use constant weight", "use ivar as weight"
]

defaults = {
    "iter out prefix": "delta_attributes",
    "limit eta": (0.5, 1.5),
    "limit var lss": (0., 0.3),
    "num bins variance": 20,
    "num iterations": 5,
    "order": 1,
    "use constant weight": False,
    "use ivar as weight": False,
}
class Dr16NoFudgeExpectedFlux(Dr16ExpectedFlux):
    def _initialize_variables(self):
        """Initialize useful variables
        The initialized arrays are:
        - self.get_eta
        - self.get_fudge
        - self.get_mean_cont
        - self.get_mean_cont_weight
        - self.get_num_pixels
        - self.get_valid_fit
        - self.get_var_lss
        - self.log_lambda_var_func_grid

        Raise
        -----
        ExpectedFluxError if Forest class variables are not set
        """
        # check that Forest class variables are set
        try:
            Forest.class_variable_check()
        except AstronomicalObjectError as error:
            raise ExpectedFluxError(
                "Forest class variables need to be set "
                "before initializing variables here.") from error

        # initialize the mean quasar continuum
        # TODO: maybe we can drop this and compute first the mean quasar
        # continuum on compute_expected_flux
        self.get_mean_cont = interp1d(Forest.log_lambda_rest_frame_grid,
                                      np.ones_like(
                                          Forest.log_lambda_rest_frame_grid),
                                      fill_value="extrapolate")
        self.get_mean_cont_weight = interp1d(
            Forest.log_lambda_rest_frame_grid,
            np.zeros_like(Forest.log_lambda_rest_frame_grid),
            fill_value="extrapolate")

        # initialize the variance-related variables (see equation 4 of
        # du Mas des Bourboux et al. 2020 for details on these variables)
        if Forest.wave_solution == "log":
            self.log_lambda_var_func_grid = (
                Forest.log_lambda_grid[0] +
                (np.arange(self.num_bins_variance) + .5) *
                (Forest.log_lambda_grid[-1] - Forest.log_lambda_grid[0]) /
                self.num_bins_variance)
        # TODO: this is related with the todo in check the effect of finding
        # the nearest bin in log_lambda space versus lambda space infunction
        # find_bins in utils.py. Once we understand that we can remove
        # the dependence from Forest from here too.
        elif Forest.wave_solution == "lin":
            self.log_lambda_var_func_grid = np.log10(
                10**Forest.log_lambda_grid[0] +
                (np.arange(self.num_bins_variance) + .5) *
                (10**Forest.log_lambda_grid[-1] -
                 10**Forest.log_lambda_grid[0]) / self.num_bins_variance)

        # TODO: Replace the if/else block above by something like the commented
        # block below. We need to check the impact of doing this on the final
        # deltas first (eta, var_lss and fudge will be differently sampled).
        #start of commented block
        #resize = len(Forest.log_lambda_grid)/self.num_bins_variance
        #print(resize)
        #self.log_lambda_var_func_grid = Forest.log_lambda_grid[::int(resize)]
        #end of commented block

        # if use_ivar_as_weight is set, eta, var_lss and fudge will be ignored
        # print a message to inform the user
        if self.use_ivar_as_weight:
            self.logger.info(("using ivar as weights, ignoring eta, "
                              "var_lss, fudge fits"))
            eta = np.ones(self.num_bins_variance)
            var_lss = np.zeros(self.num_bins_variance)
            fudge = np.ones(self.num_bins_variance)*fudge_fixed_value
            num_pixels = np.zeros(self.num_bins_variance)
            valid_fit = np.ones(self.num_bins_variance)
        # if use_constant_weight is set then initialize eta, var_lss, and fudge
        # with values to have constant weights
        elif self.use_constant_weight:
            self.logger.info(("using constant weights, ignoring eta, "
                              "var_lss, fudge fits"))
            eta = np.zeros(self.num_bins_variance)
            var_lss = np.ones(self.num_bins_variance)
            fudge = np.ones(self.num_bins_variance)*fudge_fixed_value
            num_pixels = np.zeros(self.num_bins_variance)
            valid_fit = np.ones(self.num_bins_variance, dtype=bool)
        # normal initialization: eta, var_lss, and fudge are ignored in the
        # first iteration
        else:
            eta = np.ones(self.num_bins_variance)
            var_lss = np.zeros(self.num_bins_variance) + 0.2
            fudge = np.ones(self.num_bins_variance)*fudge_fixed_value
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
        self.get_fudge = interp1d(self.log_lambda_var_func_grid,
                                  fudge,
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
        fudge = np.zeros(self.num_bins_variance)
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
            " loglam    eta      var_lss  fudge    chi2     num_pix valid_fit")
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
                variance = eta * var_pipe_values + var_lss + fudge_fixed_value / var_pipe_values
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
                fudge[index] = fudge_fixed_value
                error_eta[index] = minimizer.errors["eta"]
                error_var_lss[index] = minimizer.errors["var_lss"]
                valid_fit[index] = True
            else:
                eta[index] = 1.
                var_lss[index] = 0.1
                fudge[index] = fudge_fixed_value
                error_eta[index] = 0.
                error_var_lss[index] = 0.
                valid_fit[index] = False
            num_pixels[index] = count[index * num_var_bins:(index + 1) *
                                      num_var_bins].sum()
            chi2_in_bin[index] = minimizer.fval

            self.logger.progress(
                f" {self.log_lambda_var_func_grid[index]:.3e} "
                f"{eta[index]:.2e} {var_lss[index]:.2e} {fudge[index]:.2e} " +
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
        self.get_fudge = interp1d(self.log_lambda_var_func_grid[w],
                                  fudge[w],
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