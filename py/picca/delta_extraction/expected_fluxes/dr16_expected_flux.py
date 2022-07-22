"""This module defines the class Dr16ExpectedFlux"""
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
from picca.delta_extraction.utils import find_bins

accepted_options = [
    "iter out prefix", "limit eta", "limit var lss", "min num qso in fit",
    "num bins variance", "num iterations", "num processors", "order", "out dir",
    "use constant weight", "use ivar as weight", "force stack delta to zero"
]

defaults = {
    "iter out prefix": "delta_attributes",
    "limit eta": (0.5, 1.5),
    "limit var lss": (0., 0.3),
    "num bins variance": 20,
    "num iterations": 5,
    "min num qso in fit": 100,
    "order": 1,
    "use constant weight": False,
    "use ivar as weight": False,
    "force stack delta to zero": True
}


class Dr16ExpectedFlux(ExpectedFlux):
    """Class to the expected flux as done in the DR16 SDSS analysys
    The mean expected flux is calculated iteratively as explained in
    du Mas des Bourboux et al. (2020)

    Methods
    -------
    (see ExpectedFlux in py/picca/delta_extraction/expected_flux.py)
    __init__
    _initialize_variables
    __parse_config
    compute_continuum
    compute_delta_stack
    compute_mean_cont
    compute_expected_flux
    compute_var_stats
        chi2
    get_continuum_model
    get_continuum_weights
    populate_los_ids
    save_iteration_step

    Attributes
    ----------
    (see ExpectedFlux in py/picca/delta_extraction/expected_flux.py)

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
    Interpolation function to compute the unabsorbed mean quasar continua.

    get_mean_cont_weight: scipy.interpolate.interp1d
    Interpolation function to compute the weights associated with the unabsorbed
    mean quasar continua.

    get_num_pixels: scipy.interpolate.interp1d
    Number of pixels used to fit for eta, var_lss and fudge.

    get_stack_delta: scipy.interpolate.interp1d
    Interpolation function to compute the mean delta (from stacking all lines of
    sight).

    get_stack_delta_weights: scipy.interpolate.interp1d
    Weights associated with get_stack_delta

    get_valid_fit: scipy.interpolate.interp1d
    True if the fit for eta, var_lss and fudge is converged, false otherwise.
    Since the fit is performed independently for eah observed wavelength,
    this is also given as a function of the observed wavelength.

    get_var_lss: scipy.interpolate.interp1d
    Interpolation function to compute mapping functions var_lss. See equation 4 of
    du Mas des Bourboux et al. 2020 for details.

    iter_out_prefix: str
    Prefix of the iteration files. These files contain the statistical properties
    of deltas at a given iteration step. Intermediate files will add
    '_iteration{num}.fits.gz' to the prefix for intermediate steps and '.fits.gz'
    for the final results.

    limit_eta: tuple of floats
    Limits on the correction factor to the contribution of the pipeline estimate
    of the instrumental noise to the variance.

    limit_var_lss: tuple of floats
    Limits on the pixel variance due to Large Scale Structure

    log_lambda_var_func_grid: array of float
    Logarithm of the wavelengths where the variance functions and
    statistics are computed.

    logger: logging.Logger
    Logger object

    num_bins_variance: int
    Number of bins to be used to compute variance functions and statistics as
    a function of wavelength.

    num_iterations: int
    Number of iterations to determine the mean continuum shape, LSS variances, etc.

    order: int
    Order of the polynomial for the continuum fit.

    use_constant_weight: boolean
    If "True", set all the delta weights to one (implemented as eta = 0,
    sigma_lss = 1, fudge = 0).

    use_ivar_as_weight: boolean
    If "True", use ivar as weights (implemented as eta = 1, sigma_lss = fudge = 0).

    force_stack_delta_to_zero: boolean
    If "True", continuum is corrected by stack_delta.
    """

    def __init__(self, config):
        """Initialize class instance.

        Arguments
        ---------
        config: configparser.SectionProxy
        Parsed options to initialize class
        """
        self.logger = logging.getLogger(__name__)
        super().__init__(config)

        # load variables from config
        self.iter_out_prefix = None
        self.limit_eta = None
        self.limit_var_lss = None
        self.min_num_qso_in_fit = None
        self.num_bins_variance = None
        self.num_iterations = None
        self.order = None
        self.use_constant_weight = None
        self.use_ivar_as_weight = None
        self.force_stack_delta_to_zero = None
        self.__parse_config(config)

        # initialize variables
        self.get_eta = None
        self.get_fudge = None
        self.get_mean_cont = None
        self.get_mean_cont_weight = None
        self.get_num_pixels = None
        self.get_valid_fit = None
        self.get_var_lss = None
        self.log_lambda_var_func_grid = None
        self._initialize_variables()

        self.continuum_fit_parameters = None

        self.get_stack_delta = None
        self.get_stack_delta_weights = None

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
            fudge = np.zeros(self.num_bins_variance)
            num_pixels = np.zeros(self.num_bins_variance)
            valid_fit = np.ones(self.num_bins_variance)
        # if use_constant_weight is set then initialize eta, var_lss, and fudge
        # with values to have constant weights
        elif self.use_constant_weight:
            self.logger.info(("using constant weights, ignoring eta, "
                              "var_lss, fudge fits"))
            eta = np.zeros(self.num_bins_variance)
            var_lss = np.ones(self.num_bins_variance)
            fudge = np.zeros(self.num_bins_variance)
            num_pixels = np.zeros(self.num_bins_variance)
            valid_fit = np.ones(self.num_bins_variance, dtype=bool)
        # normal initialization: eta, var_lss, and fudge are ignored in the
        # first iteration
        else:
            eta = np.ones(self.num_bins_variance)
            var_lss = np.zeros(self.num_bins_variance) + 0.2
            fudge = np.zeros(self.num_bins_variance)
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
        self.iter_out_prefix = config.get("iter out prefix")
        if self.iter_out_prefix is None:
            raise ExpectedFluxError(
                "Missing argument 'iter out prefix' required "
                "by Dr16ExpectedFlux")
        if "/" in self.iter_out_prefix:
            raise ExpectedFluxError(
                "Error constructing Dr16ExpectedFlux. "
                "'iter out prefix' should not incude folders. "
                f"Found: {self.iter_out_prefix}")

        limit_eta_string = config.get("limit eta")
        if limit_eta_string is None:
            raise ExpectedFluxError(
                "Missing argument 'limit eta' required by Dr16ExpectedFlux")
        limit_eta = limit_eta_string.split(",")
        if limit_eta[0].startswith("(") or limit_eta[0].startswith("["):
            eta_min = float(limit_eta[0][1:])
        else:
            eta_min = float(limit_eta[0])
        if limit_eta[1].endswith(")") or limit_eta[1].endswith("]"):
            eta_max = float(limit_eta[1][:-1])
        else:
            eta_max = float(limit_eta[1])
        self.limit_eta = (eta_min, eta_max)

        limit_var_lss_string = config.get("limit var lss")
        if limit_var_lss_string is None:
            raise ExpectedFluxError(
                "Missing argument 'limit var lss' required by Dr16ExpectedFlux")
        limit_var_lss = limit_var_lss_string.split(",")
        if limit_var_lss[0].startswith("(") or limit_var_lss[0].startswith("["):
            var_lss_min = float(limit_var_lss[0][1:])
        else:
            var_lss_min = float(limit_var_lss[0])
        if limit_var_lss[1].endswith(")") or limit_var_lss[1].endswith("]"):
            var_lss_max = float(limit_var_lss[1][:-1])
        else:
            var_lss_max = float(limit_var_lss[1])
        self.limit_var_lss = (var_lss_min, var_lss_max)

        self.min_num_qso_in_fit = config.getint("min num qso in fit")
        if self.min_num_qso_in_fit is None:
            raise ExpectedFluxError(
                "Missing argument 'min qso in fit' required by Dr16ExpectedFlux"
            )

        self.num_bins_variance = config.getint("num bins variance")
        if self.num_bins_variance is None:
            raise ExpectedFluxError(
                "Missing argument 'num bins variance' required by Dr16ExpectedFlux"
            )

        self.num_iterations = config.getint("num iterations")
        if self.num_iterations is None:
            raise ExpectedFluxError(
                "Missing argument 'num iterations' required by Dr16ExpectedFlux"
            )

        self.order = config.getint("order")
        if self.order is None:
            raise ExpectedFluxError(
                "Missing argument 'order' required by Dr16ExpectedFlux")

        self.use_constant_weight = config.getboolean("use constant weight")
        if self.use_constant_weight is None:
            raise ExpectedFluxError(
                "Missing argument 'use constant weight' required by Dr16ExpectedFlux"
            )

        self.use_ivar_as_weight = config.getboolean("use ivar as weight")
        if self.use_ivar_as_weight is None:
            raise ExpectedFluxError(
                "Missing argument 'use ivar as weight' required by Dr16ExpectedFlux"
            )

        self.force_stack_delta_to_zero = config.getboolean("force stack delta to zero")
        if self.force_stack_delta_to_zero is None:
            raise ExpectedFluxError(
                "Missing argument 'force stack delta to zero' required by Dr16ExpectedFlux"
            )

    # this should be a read-only function as it is called in a parallelized way
    # TODO: consider making this not a function to minimize future bugs
    def compute_continuum(self, forest):
        """Compute the forest continuum.

        Fits a model based on the mean quasar continuum and linear function
        (see equation 2 of du Mas des Bourboux et al. 2020)
        Flags the forest with bad_cont if the computation fails.

        Arguments
        ---------
        forest: Forest
        A forest instance where the continuum will be computed

        Return
        ------
        forest: Forest
        The modified forest instance
        """
        # get mean continuum
        mean_cont = self.get_mean_cont(forest.log_lambda -
                                       np.log10(1 + forest.z))

        # add transmission correction
        # (previously computed using method add_optical_depth)
        mean_cont *= forest.transmission_correction

        mean_cont_kwargs = {"mean_cont": mean_cont}
        # TODO: This can probably be replaced by forest.log_lambda[-1] and
        # forest.log_lambda[0]
        mean_cont_kwargs["log_lambda_max"] = (
            Forest.log_lambda_rest_frame_grid[-1] + np.log10(1 + forest.z))
        mean_cont_kwargs["log_lambda_min"] = (
            Forest.log_lambda_rest_frame_grid[0] + np.log10(1 + forest.z))

        leasts_squares = LeastsSquaresContModel(
            forest=forest,
            expected_flux=self,
            mean_cont_kwargs=mean_cont_kwargs,
        )

        zero_point = (forest.flux * forest.ivar).sum() / forest.ivar.sum()
        slope = 0.0

        minimizer = iminuit.Minuit(leasts_squares,
                                   zero_point=zero_point,
                                   slope=slope)
        minimizer.errors["zero_point"] = zero_point / 2.
        minimizer.errors["slope"] = zero_point / 2.
        minimizer.errordef = 1.
        minimizer.print_level = 0
        minimizer.fixed["slope"] = self.order == 0
        minimizer.migrad()

        bad_continuum_reason = None
        cont_model = self.get_continuum_model(forest,
                                              minimizer.values["zero_point"],
                                              minimizer.values["slope"],
                                              **mean_cont_kwargs)
        if not minimizer.valid:
            bad_continuum_reason = "minuit didn't converge"
        if np.any(cont_model < 0):
            bad_continuum_reason = "negative continuum"

        if bad_continuum_reason is None:
            continuum_fit_parameters = (minimizer.values["zero_point"],
                                        minimizer.values["slope"])
        ## if the continuum is negative or minuit didn't converge, then
        ## set it to None
        else:
            cont_model = None
            continuum_fit_parameters = (np.nan, np.nan)

        return cont_model, bad_continuum_reason, continuum_fit_parameters

    def compute_delta_stack(self, forests, stack_from_deltas=False):
        """Compute a stack of the delta field as a function of wavelength

        Arguments
        ---------
        forests: List of Forest
        A list of Forest from which to compute the deltas.

        stack_from_deltas: bool - default: False
        Flag to determine whether to stack from deltas or compute them
        """
        # TODO: move this to _initialize_variables (after tests are done)
        stack_delta = np.zeros_like(Forest.log_lambda_grid)
        stack_weight = np.zeros_like(Forest.log_lambda_grid)

        for forest in forests:
            if stack_from_deltas:
                delta = forest.delta
                weights = forest.weights
            else:
                # ignore forest if continuum could not be computed
                if forest.continuum is None:
                    continue
                delta = forest.flux / forest.continuum
                var_lss = self.get_var_lss(forest.log_lambda)
                eta = self.get_eta(forest.log_lambda)
                fudge = self.get_fudge(forest.log_lambda)
                var = 1. / forest.ivar / forest.continuum**2
                variance = eta * var + var_lss + fudge / var
                weights = 1. / variance

            bins = find_bins(forest.log_lambda, Forest.log_lambda_grid,
                             Forest.wave_solution)
            rebin = np.bincount(bins, weights=delta * weights)
            stack_delta[:len(rebin)] += rebin
            rebin = np.bincount(bins, weights=weights)
            stack_weight[:len(rebin)] += rebin

        w = stack_weight > 0
        stack_delta[w] /= stack_weight[w]

        self.get_stack_delta = interp1d(
            Forest.log_lambda_grid[stack_weight > 0.],
            stack_delta[stack_weight > 0.],
            kind="nearest",
            fill_value="extrapolate")
        self.get_stack_delta_weights = interp1d(
            Forest.log_lambda_grid[stack_weight > 0.],
            stack_weight[stack_weight > 0.],
            kind="nearest",
            fill_value=0.0,
            bounds_error=False)

    # TODO: We should check if we can directly compute the mean continuum
    # in particular this means:
    # 0. check the inner todo
    # 1. check that we can use forest.continuum instead of
    #    forest.flux/forest.continuum right before `mean_cont[:len(cont)] += cont`
    # 2. check that in that case we don't need to use the new_cont
    # 3. check that this is not propagated elsewhere through self.get_mean_cont
    # If this works then:
    # 1. update this function to be essentially the same as in TrueContinuum
    #    (except for the weights)
    # 2. overload `compute_continuum_weights` in TrueContinuum to compute the
    #    correct weights
    # 3. remove method compute_mean_cont from TrueContinuum
    # 4. restore min-similarity-lines in .pylintrc back to 5
    def compute_mean_cont(self, forests):
        """Compute the mean quasar continuum over the whole sample.
        Then updates the value of self.get_mean_cont to contain it

        Arguments
        ---------
        forests: List of Forest
        A list of Forest from which to compute the deltas.
        """
        mean_cont = np.zeros_like(Forest.log_lambda_rest_frame_grid)
        mean_cont_weight = np.zeros_like(Forest.log_lambda_rest_frame_grid)

        # first compute <F/C> in bins. C=Cont_old*spectrum_dependent_fitting_fct
        # (and Cont_old is constant for all spectra in a bin), thus we actually
        # compute
        #    1/Cont_old * <F/spectrum_dependent_fitting_function>
        for forest in forests:
            if forest.bad_continuum_reason is not None:
                continue
            bins = find_bins(forest.log_lambda - np.log10(1 + forest.z),
                             Forest.log_lambda_rest_frame_grid,
                             Forest.wave_solution)

            weights = self.get_continuum_weights(forest, forest.continuum)
            # this is needed as the weights from get_continuum_weights are
            # divided by the continuum model squared, in this case forest.continuum
            # TODO: check that we indeed need this or if the weights without it
            # are better
            if not self.use_constant_weight:
                weights *= forest.continuum**2
            cont = np.bincount(bins,
                               weights=forest.flux / forest.continuum * weights)
            mean_cont[:len(cont)] += cont
            cont = np.bincount(bins, weights=weights)
            mean_cont_weight[:len(cont)] += cont

        w = mean_cont_weight > 0
        mean_cont[w] /= mean_cont_weight[w]
        mean_cont /= mean_cont.mean()
        log_lambda_cont = Forest.log_lambda_rest_frame_grid[w]

        # the new mean continuum is multiplied by the previous one to recover
        # <F/spectrum_dependent_fitting_function>
        new_cont = self.get_mean_cont(log_lambda_cont) * mean_cont[w]
        self.get_mean_cont = interp1d(log_lambda_cont,
                                      new_cont,
                                      fill_value="extrapolate")
        self.get_mean_cont_weight = interp1d(log_lambda_cont,
                                             mean_cont_weight[w],
                                             fill_value=0.0,
                                             bounds_error=False)

    def compute_expected_flux(self, forests):
        """Compute the mean expected flux of the forests.
        This includes the quasar continua and the mean transimission. It is
        computed iteratively following as explained in du Mas des Bourboux et
        al. (2020)

        Arguments
        ---------
        forests: List of Forest
        A list of Forest from which to compute the deltas.
        """
        context = multiprocessing.get_context('fork')
        for iteration in range(self.num_iterations):
            self.logger.progress(
                f"Continuum fitting: starting iteration {iteration} of {self.num_iterations}"
            )
            if self.num_processors > 1:
                with context.Pool(processes=self.num_processors) as pool:
                    imap_it = pool.imap(self.compute_continuum, forests)

                    self.continuum_fit_parameters = {}
                    for forest, (cont_model, bad_continuum_reason,
                                 continuum_fit_parameters) in zip(
                                     forests, imap_it):
                        forest.bad_continuum_reason = bad_continuum_reason
                        forest.continuum = cont_model
                        self.continuum_fit_parameters[forest.los_id] = continuum_fit_parameters

            else:
                self.continuum_fit_parameters = {}
                for forest in forests:
                    (cont_model, bad_continuum_reason,
                     continuum_fit_parameters) = self.compute_continuum(forest)
                    forest.bad_continuum_reason = bad_continuum_reason
                    forest.continuum = cont_model
                    self.continuum_fit_parameters[forest.los_id] = continuum_fit_parameters
                #forests = [self.compute_continuum(f) for f in forests]

            if iteration < self.num_iterations - 1:
                # Compute mean continuum (stack in rest-frame)
                self.compute_mean_cont(forests)

                # Compute observer-frame mean quantities (var_lss, eta, fudge)
                if not (self.use_ivar_as_weight or self.use_constant_weight):
                    self.compute_var_stats(forests)

            # compute the mean deltas
            self.compute_delta_stack(forests)

            # Save the iteration step
            if iteration == self.num_iterations - 1:
                self.save_iteration_step(-1)
            else:
                self.save_iteration_step(iteration)

            self.logger.progress(
                f"Continuum fitting: ending iteration {iteration} of "
                f"{self.num_iterations}")

        # now loop over forests to populate los_ids
        self.populate_los_ids(forests)

    def compute_var_stats(self, forests):
        """Compute variance functions and statistics

        This function computes the statistics required to fit the mapping functions
        eta, var_lss, and fudge. It also computes the functions themselves. See
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
        error_fudge = np.zeros(self.num_bins_variance)
        num_pixels = np.zeros(self.num_bins_variance)
        valid_fit = np.zeros(self.num_bins_variance)

        # define an array to contain the possible values of pipeline variances
        # the measured pipeline variance of the deltas will be averaged using the
        # same binning, and the two arrays will be compared to fit the functions
        # eta, var_lss, and fudge
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

        # fit the functions eta, var_lss, and fudge
        chi2_in_bin = np.zeros(self.num_bins_variance)
        fudge_ref = 1e-7

        self.logger.progress(" Mean quantities in observer-frame")
        self.logger.progress(
            " loglam    eta      var_lss  fudge    chi2     num_pix valid_fit")
        for index in range(self.num_bins_variance):
            # pylint: disable-msg=cell-var-from-loop
            # this function is defined differntly at each step of the loop
            def chi2(eta, var_lss, fudge):
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
                variance = eta * var_pipe_values + var_lss + fudge * fudge_ref / var_pipe_values
                chi2_contribution = (
                    var_delta[index * num_var_bins:(index + 1) * num_var_bins] -
                    variance)
                weights = var2_delta[index * num_var_bins:(index + 1) *
                                     num_var_bins]
                w = num_qso[index * num_var_bins:(index + 1) *
                            num_var_bins] > self.min_num_qso_in_fit
                return np.sum(chi2_contribution[w]**2 / weights[w])

            minimizer = iminuit.Minuit(chi2,
                                       name=("eta", "var_lss", "fudge"),
                                       eta=1.,
                                       var_lss=0.1,
                                       fudge=1.)
            minimizer.errors["eta"] = 0.05
            minimizer.errors["var_lss"] = 0.05
            minimizer.errors["fudge"] = 0.05
            minimizer.errordef = 1.
            minimizer.print_level = 0
            minimizer.limits["eta"] = self.limit_eta
            minimizer.limits["var_lss"] = self.limit_var_lss
            minimizer.limits["fudge"] = (0, None)
            minimizer.migrad()

            if minimizer.valid:
                minimizer.hesse()
                eta[index] = minimizer.values["eta"]
                var_lss[index] = minimizer.values["var_lss"]
                fudge[index] = minimizer.values["fudge"] * fudge_ref
                error_eta[index] = minimizer.errors["eta"]
                error_var_lss[index] = minimizer.errors["var_lss"]
                error_fudge[index] = minimizer.errors["fudge"] * fudge_ref
                valid_fit[index] = True
            else:
                eta[index] = 1.
                var_lss[index] = 0.1
                fudge[index] = 1. * fudge_ref
                error_eta[index] = 0.
                error_var_lss[index] = 0.
                error_fudge[index] = 0.
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

    # pylint: disable=no-self-use
    # We expect this function to be changed by some child classes
    def get_continuum_model(self, forest, zero_point, slope, **kwargs):
        """Get the model for the continuum fit

        Arguments
        ---------
        forest: Forest
        The forest instance we want the model from

        zero_point: float
        Zero point of the linear function (flux mean). Referred to as $a_q$ in
        du Mas des Bourboux et al. 2020

        slope: float
        Slope of the linear function (evolution of the flux). Referred to as
        $b_q$ in du Mas des Bourboux et al. 2020

        Keyword Arguments
        -----------------
        mean_cont: array of floats
        Mean continuum. Required.

        log_lambda_max: float
        Maximum log_lambda for this forest.

        log_lambda_min: float
        Minimum log_lambda for this forest.

        Return
        ------
        cont_model: array of float
        The continuum model
        """
        # unpack kwargs
        if "mean_cont" not in kwargs:
            raise ExpectedFluxError("Function get_cont_model requires "
                                    "'mean_cont' in the **kwargs dictionary")
        mean_cont = kwargs.get("mean_cont")
        for key in ["log_lambda_max", "log_lambda_min"]:
            if key not in kwargs:
                raise ExpectedFluxError("Function get_cont_model requires "
                                        f"'{key}' in the **kwargs dictionary")
        log_lambda_max = kwargs.get("log_lambda_max")
        log_lambda_min = kwargs.get("log_lambda_min")
        # compute continuum
        line = (slope * (forest.log_lambda - log_lambda_min) /
                (log_lambda_max - log_lambda_min) + zero_point)

        return line * mean_cont

    # pylint: disable=unused-argument
    # kwargs are passed here in case this is necessary in child classes
    def get_continuum_weights(self, forest, cont_model, **kwargs):
        """Get the continuum model weights

        Arguments
        ---------
        forest: Forest
        The forest instance we want the model from

        cont_model: array of float
        The continuum model

        Return
        ------
        weights: array of float
        The continuum model weights
        """
        # force weights=1 when use-constant-weight
        if self.use_constant_weight:
            weights = np.ones_like(forest.flux)
        else:
            # pixel variance due to the Large Scale Strucure
            var_lss = self.get_var_lss(forest.log_lambda)
            # correction factor to the contribution of the pipeline
            # estimate of the instrumental noise to the variance.
            eta = self.get_eta(forest.log_lambda)
            # fudge contribution to the variance
            fudge = self.get_fudge(forest.log_lambda)

            var_pipe = 1. / forest.ivar / cont_model**2
            ## prep_del.variance is the variance of delta
            ## we want here the weights = ivar(flux)
            variance = eta * var_pipe + var_lss + fudge / var_pipe
            weights = 1.0 / cont_model**2 / variance

        return weights

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
                continue
            # get the variance functions and statistics
            eta = self.get_eta(forest.log_lambda)

            # assignment operator (=) creates a reference, such that
            # mean_expected_flux points to forest.continuum and
            # forest.continuum gets modified within if statement
            mean_expected_flux = np.copy(forest.continuum)
            if self.force_stack_delta_to_zero:
                stack_delta = self.get_stack_delta(forest.log_lambda)
                mean_expected_flux *= stack_delta
            weights = self.get_continuum_weights(forest, mean_expected_flux)
            # this is needed as the weights from get_continuum_weights are
            # divided by the continuum model squared, in this case mean_expected_flux
            # TODO: check that we indeed need this or if the weights without it
            # are better
            if not self.use_constant_weight:
                weights *= mean_expected_flux**2

            forest_info = {
                "mean expected flux": mean_expected_flux,
                "weights": weights,
                "continuum": forest.continuum,
            }
            if isinstance(forest, Pk1dForest):
                ivar = forest.ivar / (eta +
                                      (eta == 0)) * (mean_expected_flux**2)

                forest_info["ivar"] = ivar
            self.los_ids[forest.los_id] = forest_info

    def save_iteration_step(self, iteration):
        """Save the statistical properties of deltas at a given iteration
        step

        Arguments
        ---------
        iteration: int
        Iteration number. -1 for final iteration
        """
        if iteration == -1:
            iter_out_file = self.iter_out_prefix + ".fits.gz"
        else:
            iter_out_file = self.iter_out_prefix + f"_iteration{iteration+1}.fits.gz"

        with fitsio.FITS(self.out_dir + iter_out_file, 'rw',
                         clobber=True) as results:
            header = {}
            header["FITORDER"] = self.order

            # TODO: update this once the TODO in compute continua is fixed
            results.write([
                Forest.log_lambda_grid,
                self.get_stack_delta(Forest.log_lambda_grid),
                self.get_stack_delta_weights(Forest.log_lambda_grid)
            ],
                          names=['loglam', 'stack', 'weight'],
                          header=header,
                          extname='STACK_DELTAS')

            results.write([
                self.log_lambda_var_func_grid,
                self.get_eta(self.log_lambda_var_func_grid),
                self.get_var_lss(self.log_lambda_var_func_grid),
                self.get_fudge(self.log_lambda_var_func_grid),
                self.get_num_pixels(self.log_lambda_var_func_grid),
                self.get_valid_fit(self.log_lambda_var_func_grid)
            ],
                          names=[
                              'loglam', 'eta', 'var_lss', 'fudge', 'num_pixels',
                              'valid_fit'
                          ],
                          extname='VAR_FUNC')

            results.write([
                Forest.log_lambda_rest_frame_grid,
                self.get_mean_cont(Forest.log_lambda_rest_frame_grid),
                self.get_mean_cont_weight(Forest.log_lambda_rest_frame_grid),
            ],
                          names=['loglam_rest', 'mean_cont', 'weight'],
                          extname='CONT')


class LeastsSquaresContModel:
    """This class deals with the continuum fitting.

    It is passed to iminuit and when called it will return the chi2 for a given
    set of parameters

    Methods
    -------
    __init__
    __call__

    Attributes
    ----------
    forest: Forest
    Forest instance where the model is fit

    expected_flux: Dr16ExpectedFlux
    Dr16ExpectedFlux instance running the fit

    mean_cont_kwargs: dict
    kwargs passed to expected_flux.get_continuum_model

    weights_kwargs: dict
    kwargs passed to expected_flux.get_continuum_weights
    """

    def __init__(self,
                 forest,
                 expected_flux,
                 mean_cont_kwargs=None,
                 weights_kwargs=None):
        """Initialize class instances

        Arguments
        ---------
        forest: Forest
        The forest to fit

        expected_flux: Dr16ExpectedFlux
        The expected flux instance

        mean_cont_kwargs: dict or None - default = None
        kwargs needed by method get_continuum_model of expected_flux. If None
        then it will be assigned an empty dictionary

        weights_kwargs: dict or None - default = None
        kwargs needed by method get_continuum_weights of expected_flux. If None
        then it will be assigned an empty dictionary
        """
        self.forest = forest
        self.expected_flux = expected_flux
        if mean_cont_kwargs is None:
            self.mean_cont_kwargs = {}
        else:
            self.mean_cont_kwargs = mean_cont_kwargs
        if weights_kwargs is None:
            self.weights_kwargs = {}
        else:
            self.weights_kwargs = weights_kwargs

    def __call__(self, zero_point, slope):
        """
        Compute chi2 for a given set of parameters

        Arguments
        ---------
        zero_point: float
        Zero point of the linear function (flux mean). Referred to as $a_q$ in
        du Mas des Bourboux et al. 2020

        slope: float
        Slope of the linear function (evolution of the flux). Referred to as
        $b_q$ in du Mas des Bourboux et al. 2020

        Returns
        -------
        chi2: float
        The chi2 for this run
        """
        cont_model = self.expected_flux.get_continuum_model(
            self.forest, zero_point, slope, **self.mean_cont_kwargs)

        weights = self.expected_flux.get_continuum_weights(
            self.forest, cont_model, **self.weights_kwargs)

        chi2_contribution = (self.forest.flux - cont_model)**2 * weights
        return chi2_contribution.sum() - np.log(weights).sum()
