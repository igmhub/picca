"""This module defines the class Dr16ExpectedFlux"""
import logging
import multiprocessing
import time

import iminuit
import numpy as np
from scipy.interpolate import interp1d

from picca.delta_extraction.astronomical_objects.pk1d_forest import Pk1dForest
from picca.delta_extraction.errors import ExpectedFluxError
from picca.delta_extraction.expected_flux import ExpectedFlux, defaults, accepted_options
from picca.delta_extraction.expected_fluxes.utils import compute_continuum
from picca.delta_extraction.least_squares.least_squares_var_stats import (
    LeastsSquaresVarStats, FUDGE_REF)
from picca.delta_extraction.utils import (update_accepted_options,
                                          update_default_options)

accepted_options = update_accepted_options(accepted_options, [
    "force stack delta to zero", "limit eta", "limit var lss",
    "min num qso in fit", "num iterations", "order", "use constant weight",
    "use ivar as weight"
])

defaults = update_default_options(
    defaults, {
        "force stack delta to zero": True,
        "limit eta": (0.5, 1.5),
        "limit var lss": (0., 0.3),
        "num iterations": 5,
        "min num qso in fit": 100,
        "order": 1,
        "use constant weight": False,
        "use ivar as weight": False,
    })

FUDGE_FIT_START = FUDGE_REF
ETA_FIT_START = 1.
VAR_LSS_FIT_START = 0.1


class Dr16ExpectedFlux(ExpectedFlux):
    """Class to the expected flux as done in the DR16 SDSS analysys
    The mean expected flux is calculated iteratively as explained in
    du Mas des Bourboux et al. (2020)

    Methods
    -------
    (see ExpectedFlux in py/picca/delta_extraction/expected_flux.py)
    __init__
    _initialize_get_eta
    _initialize_get_fudge
    _initialize_get_var_lss
    _initialize_mean_continuum_arrays
    _initialize_variance_wavelength_array
    _initialize_variance_functions
    __parse_config
    compute_delta_stack
    compute_forest_variance
    compute_mean_cont
    compute_expected_flux
    compute_var_stats
    get_continuum_weights
    hdu_cont
    hdu_stack_deltas
    hdu_var_func
    populate_los_ids
    save_iteration_step

    Attributes
    ----------
    (see ExpectedFlux in py/picca/delta_extraction/expected_flux.py)

    continuum_fit_parameters: dict
    A dictionary containing the continuum fit parameters for each line of sight.
    Keys are the identifier for the line of sight and values are tuples with
    the best-fit zero point and slope of the linear part of the fit, the chi2
    of the fit, and the number of datapoints used in the fit..

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

    min_num_qso_in_fit: int
    Minimum number of quasars contributing to a bin of wavelength and pipeline
    variance in order to consider it in the fit. This is passed to
    LeastsSquaresVarStats.

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

        Raise
        -----
        ExpectedFluxError if Forest class variables are not set
        """
        self.logger = logging.getLogger(__name__)
        super().__init__(config)

        # load variables from config
        self.limit_eta = None
        self.limit_var_lss = None
        self.min_num_qso_in_fit = None
        self.num_iterations = None
        self.order = None
        self.use_constant_weight = None
        self.use_ivar_as_weight = None
        self.force_stack_delta_to_zero = None
        self.__parse_config(config)

        # initialize variance functions
        self.get_eta = None
        self.get_fudge = None
        self.get_num_pixels = None
        self.get_valid_fit = None
        self.get_var_lss = None
        self.fit_variance_functions = []
        self._initialize_variance_functions()

        self.continuum_fit_parameters = None

    def _initialize_get_eta(self):
        """Initialiaze function get_eta"""
        # if use_ivar_as_weight is set, we fix eta=1, var_lss=0 and fudge=0
        if self.use_ivar_as_weight:
            eta = np.ones(self.num_bins_variance)
        # if use_constant_weight is set, we fix eta=0, var_lss=1, and fudge=0
        elif self.use_constant_weight:
            eta = np.zeros(self.num_bins_variance)
        # normal initialization, starting values eta=1, var_lss=0.2 , and fudge=0
        else:
            eta = np.ones(self.num_bins_variance)
            # this bit is what is actually freeing eta for the fit
            self.fit_variance_functions.append("eta")

        self.get_eta = interp1d(self.log_lambda_var_func_grid,
                                eta,
                                fill_value='extrapolate',
                                kind='nearest')

    def _initialize_get_fudge(self):
        """Initialiaze function get_fudge"""
        # if use_ivar_as_weight is set, we fix eta=1, var_lss=0 and fudge=0
        # if use_constant_weight is set, we fix eta=0, var_lss=1, and fudge=0
        # normal initialization, starting values eta=1, var_lss=0.2 , and fudge=0
        if not self.use_ivar_as_weight and not self.use_constant_weight:
            # this bit is what is actually freeing fudge for the fit
            self.fit_variance_functions.append("fudge")
        fudge = np.zeros(self.num_bins_variance)
        self.get_fudge = interp1d(self.log_lambda_var_func_grid,
                                  fudge,
                                  fill_value='extrapolate',
                                  kind='nearest')

    def _initialize_get_var_lss(self):
        """Initialiaze function get_var_lss"""
        # if use_ivar_as_weight is set, we fix eta=1, var_lss=0 and fudge=0
        if self.use_ivar_as_weight:
            var_lss = np.zeros(self.num_bins_variance)
        # if use_constant_weight is set, we fix eta=0, var_lss=1, and fudge=0
        elif self.use_constant_weight:
            var_lss = np.ones(self.num_bins_variance)
        # normal initialization, starting values eta=1, var_lss=0.2 , and fudge=0
        else:
            var_lss = np.zeros(self.num_bins_variance) + 0.2
            # this bit is what is actually freeing var_lss for the fit
            self.fit_variance_functions.append("var_lss")
        self.get_var_lss = interp1d(self.log_lambda_var_func_grid,
                                    var_lss,
                                    fill_value='extrapolate',
                                    kind='nearest')

    def _initialize_variance_functions(self):
        """Initialize variance functions
        The initialized arrays are:
        - self.get_eta
        - self.get_fudge
        - self.get_num_pixels
        - self.get_valid_fit
        - self.get_var_lss
        """
        # if use_ivar_as_weight is set, eta, var_lss and fudge will be ignored
        # print a message to inform the user
        if self.use_ivar_as_weight:
            self.logger.info(("using ivar as weights, ignoring eta, "
                              "var_lss, fudge fits"))
            valid_fit = np.ones(self.num_bins_variance, dtype=bool)
        # if use_constant_weight is set then initialize eta, var_lss, and fudge
        # with values to have constant weights
        elif self.use_constant_weight:
            self.logger.info(("using constant weights, ignoring eta, "
                              "var_lss, fudge fits"))
            valid_fit = np.ones(self.num_bins_variance, dtype=bool)
        # normal initialization: eta, var_lss, and fudge are ignored in the
        # first iteration
        else:
            valid_fit = np.zeros(self.num_bins_variance, dtype=bool)
        num_pixels = np.zeros(self.num_bins_variance)

        self._initialize_get_eta()
        self._initialize_get_var_lss()
        self._initialize_get_fudge()
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
        ExpectedFluxError if variables are not valid
        """
        self.force_stack_delta_to_zero = config.getboolean(
            "force stack delta to zero")
        if self.force_stack_delta_to_zero is None:
            raise ExpectedFluxError(
                "Missing argument 'force stack delta to zero' required by Dr16ExpectedFlux"
            )

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
        if self.use_constant_weight:
            self.logger.warning(
                "Deprecation Warning: option 'use constant weight' is now deprecated "
                "and will be removed in future versions. Consider using class "
                "Dr16FixedEtaVarlssFudgeExpectedFlux with options 'eta = 0', "
                "'var lss = 1' and 'fudge = 0'")
            # if use_ivar_as_weight is set, we fix eta=1, var_lss=0 and fudge=0
            # if use_constant_weight is set, we fix eta=0, var_lss=1, and fudge=0

        self.use_ivar_as_weight = config.getboolean("use ivar as weight")
        if self.use_ivar_as_weight is None:
            raise ExpectedFluxError(
                "Missing argument 'use ivar as weight' required by Dr16ExpectedFlux"
            )
        if self.use_ivar_as_weight:
            self.logger.warning(
                "Deprecation Warning: option 'use ivar as weight' is now deprecated "
                "and will be removed in future versions. Consider using class "
                "Dr16FixedEtaVarlssFudgeExpectedFlux with options 'eta = 1', "
                "'var lss = 0' and 'fudge = 0'")

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
            t0 = time.time()
            self.logger.info(
                f"Computing quasar continua using {self.num_processors} processors")
            if self.num_processors > 1:
                with context.Pool(processes=self.num_processors) as pool:
                    arguments = [(forest, self.get_mean_cont, self.get_eta,
                                  self.get_var_lss, self.get_fudge,
                                  self.use_constant_weight, self.order)
                                 for forest in forests]
                    imap_it = pool.starmap(compute_continuum, arguments)

                    self.continuum_fit_parameters = {}
                    for forest, (cont_model, bad_continuum_reason,
                                 continuum_fit_parameters) in zip(
                                     forests, imap_it):
                        forest.bad_continuum_reason = bad_continuum_reason
                        forest.continuum = cont_model
                        self.continuum_fit_parameters[
                            forest.los_id] = continuum_fit_parameters

            else:
                self.continuum_fit_parameters = {}
                for forest in forests:
                    (cont_model, bad_continuum_reason,
                     continuum_fit_parameters) = compute_continuum(
                         forest, self.get_mean_cont, self.get_eta,
                         self.get_var_lss, self.get_fudge,
                         self.use_constant_weight, self.order)

                    forest.bad_continuum_reason = bad_continuum_reason
                    forest.continuum = cont_model
                    self.continuum_fit_parameters[
                        forest.los_id] = continuum_fit_parameters
            t1 = time.time()
            self.logger.info(f"Time spent computing quasar continua: {t1-t0}")

            if iteration < self.num_iterations - 1:
                # Compute mean continuum (stack in rest-frame)
                t0 = time.time()
                self.compute_mean_cont(forests)
                t1 = time.time()
                self.logger.info(f"Time spent computing the mean continuum: {t1-t0}")

                # Compute observer-frame mean quantities (var_lss, eta, fudge)
                if not (self.use_ivar_as_weight or self.use_constant_weight):
                    t0 = time.time()
                    self.compute_var_stats(forests)
                    t1 = time.time()
                    self.logger.info(
                        f"Time spent computing eta, var_lss and fudge: {t1-t0}")

            # compute the mean deltas
            t0 = time.time()
            self.compute_delta_stack(forests)
            t1 = time.time()
            self.logger.info(f"Time spent computing delta stack: {t1-t0}")

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

    def compute_forest_variance(self, forest, continuum):
        """Compute the forest variance following du Mas des Bourboux 2020

        Arguments
        ---------
        forest: Forest
        A forest instance where the variance will be computed

        continuum: array of float
        Quasar continuum associated with the forest
        """
        w = forest.ivar > 0
        variance = np.empty_like(forest.log_lambda)
        variance[~w] = np.inf

        var_pipe = 1. / forest.ivar[w] / continuum[w]**2
        var_lss = self.get_var_lss(forest.log_lambda[w])
        eta = self.get_eta(forest.log_lambda[w])
        fudge = self.get_fudge(forest.log_lambda[w])
        variance[w] = eta * var_pipe + var_lss + fudge / var_pipe

        return variance

    # TODO: We should check if we can directly compute the mean continuum
    # in particular this means:
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

        super()._compute_mean_cont(forests,
            lambda forest: forest.flux/(forest.continuum+1e-16))

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
        if "eta" in self.fit_variance_functions:
            eta = np.zeros(self.num_bins_variance) + ETA_FIT_START
        else:
            eta = self.get_eta(self.log_lambda_var_func_grid)
        if "var_lss" in self.fit_variance_functions:
            var_lss = np.zeros(self.num_bins_variance) + VAR_LSS_FIT_START
        else:
            var_lss = self.get_var_lss(self.log_lambda_var_func_grid)
        if "fudge" in self.fit_variance_functions:
            fudge = np.zeros(self.num_bins_variance) + FUDGE_FIT_START
        else:
            fudge = self.get_fudge(self.log_lambda_var_func_grid)
        num_pixels = np.zeros(self.num_bins_variance)
        valid_fit = np.zeros(self.num_bins_variance)

        chi2_in_bin = np.zeros(self.num_bins_variance)

        # initialize the fitter class
        leasts_squares = LeastsSquaresVarStats(
            self.num_bins_variance,
            forests,
            self.log_lambda_var_func_grid,
            self.min_num_qso_in_fit,
        )

        self.logger.progress(" Mean quantities in observer-frame")
        self.logger.progress(
            " loglam    eta      var_lss  fudge    chi2     num_pix valid_fit")
        for index in range(self.num_bins_variance):
            leasts_squares.set_fit_bins(index)

            minimizer = iminuit.Minuit(leasts_squares,
                                       name=("eta", "var_lss", "fudge"),
                                       eta=eta[index],
                                       var_lss=var_lss[index],
                                       fudge=fudge[index] / FUDGE_REF)
            minimizer.errors["eta"] = 0.05
            minimizer.limits["eta"] = self.limit_eta
            minimizer.errors["var_lss"] = 0.05
            minimizer.limits["var_lss"] = self.limit_var_lss
            minimizer.errors["fudge"] = 0.05
            minimizer.limits["fudge"] = (0, None)
            minimizer.errordef = 1.
            minimizer.print_level = 0
            minimizer.fixed["eta"] = "eta" not in self.fit_variance_functions
            minimizer.fixed[
                "var_lss"] = "var_lss" not in self.fit_variance_functions
            minimizer.fixed[
                "fudge"] = "fudge" not in self.fit_variance_functions
            minimizer.migrad()

            if minimizer.valid:
                minimizer.hesse()
                eta[index] = minimizer.values["eta"]
                var_lss[index] = minimizer.values["var_lss"]
                fudge[index] = minimizer.values["fudge"] * FUDGE_REF
                valid_fit[index] = True
            else:
                eta[index] = 1.
                var_lss[index] = 0.1
                fudge[index] = 1. * FUDGE_REF
                valid_fit[index] = False
            num_pixels[index] = leasts_squares.get_num_pixels()
            chi2_in_bin[index] = minimizer.fval

            self.logger.progress(
                f" {self.log_lambda_var_func_grid[index]:.3e} "
                f"{eta[index]:.2e} {var_lss[index]:.2e} {fudge[index]:.2e} "
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

    def hdu_fit_metadata(self, results):
        """Add to the results file an HDU with the fits results

        This function is a placeholder here and should be overloaded by child
        classes if they require it

        Arguments
        ---------
        results: fitsio.FITS
        The open fits file
        """
        if self.continuum_fit_parameters is not None:
            los_id_list = []
            zero_point_list = []
            slope_list = []
            chi2_list = []
            ndata_list = []
            accepted_fit = []
            for los_id in sorted(self.continuum_fit_parameters):
                cont_fit = self.continuum_fit_parameters.get(los_id)
                los_id_list.append(los_id)
                zero_point_list.append(cont_fit[0])
                slope_list.append(cont_fit[1])
                chi2_list.append(cont_fit[2])
                ndata_list.append(cont_fit[3])
                if any(np.isnan(cont_fit)):
                    accepted_fit.append("no")
                else:
                    accepted_fit.append("yes")
            values = [
                np.array(los_id_list),
                np.array(zero_point_list),
                np.array(slope_list),
                np.array(chi2_list),
                np.array(ndata_list, dtype=int),
                np.array(accepted_fit),
            ]
            names = [
                "los_id",
                "zero_point",
                "slope",
                "chi2",
                "num_datapoints",
                "accepted_fit",
            ]

            results.write(values, names=names, extname='FIT_METADATA')

    def hdu_var_func(self, results):
        """Add to the results file an HDU with the variance functions

        Arguments
        ---------
        results: fitsio.FITS
        The open fits file
        """
        values = [
            self.log_lambda_var_func_grid,
            self.get_eta(self.log_lambda_var_func_grid),
            self.get_var_lss(self.log_lambda_var_func_grid),
            self.get_fudge(self.log_lambda_var_func_grid),
            self.get_num_pixels(self.log_lambda_var_func_grid),
            self.get_valid_fit(self.log_lambda_var_func_grid)
        ]
        names = [
            "loglam",
            "eta",
            "var_lss",
            "fudge",
            "num_pixels",
            "valid_fit",
        ]

        results.write(values, names=names, extname='VAR_FUNC')

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
            # assignment operator (=) creates a reference, such that
            # mean_expected_flux points to forest.continuum and
            # forest.continuum gets modified within if statement
            mean_expected_flux = np.copy(forest.continuum)
            if self.force_stack_delta_to_zero:
                stack_delta = self.get_stack_delta(forest.log_lambda)
                mean_expected_flux *= stack_delta

            weights = 1. / self.compute_forest_variance(forest, mean_expected_flux)

            forest_info = {
                "mean expected flux": mean_expected_flux,
                "weights": weights,
                "continuum": forest.continuum,
            }
            if isinstance(forest, Pk1dForest):
                eta = self.get_eta(forest.log_lambda)
                ivar = forest.ivar / (eta +
                                      (eta == 0)) * (mean_expected_flux**2)

                forest_info["ivar"] = ivar
            self.los_ids[forest.los_id] = forest_info
