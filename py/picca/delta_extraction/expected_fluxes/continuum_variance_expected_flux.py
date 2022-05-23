"""This module defines the class Dr16ExpectedFlux"""
import logging
import multiprocessing

import fitsio
from astropy.io import fits
import iminuit
import numpy as np
from pkg_resources import resource_filename
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

from pkg_resources import resource_filename
from picca.delta_extraction.astronomical_objects.forest import Forest
from picca.delta_extraction.astronomical_objects.pk1d_forest import Pk1dForest
from picca.delta_extraction.errors import ExpectedFluxError, AstronomicalObjectError
from picca.delta_extraction.expected_fluxes.dr16_expected_flux import Dr16ExpectedFlux
from picca.delta_extraction.expected_flux import ExpectedFlux
from picca.delta_extraction.utils import find_bins

accepted_options = ["iter out prefix", "var lss file",
                    "num bins variance", "num iterations", "num processors",
                    "order", "out dir"]

defaults = {
    "iter out prefix": "delta_attributes",
    "num bins variance": 20,
    "num iterations": 5,
    "num processors": 1,
    "order": 1,
    "use constant weight": False,
    "use ivar as weight": False,
    "var lss file": None,
}

class ContinuumVarianceExpectedFlux(Dr16ExpectedFlux):
    """Class to the expected flux with a variance term dependent
    on the rest frame wavelength.

    Methods
    -------
    extract_deltas (from ExpectedFlux)
    __init__
    _initialize_arrays
    __parse_config
    compute_continuum
        get_cont_model
        chi2
    compute_delta_stack
    compute_mean_cont
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

    out_dir: str (from ExpectedFlux)
    Directory where logs will be saved.

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

    get_tq_list: scipy.interpolate.interp1d
    Interpolaton function to compute mapping function tq_list. This is a term
    in the variance of the quasar continua dependent on the rest-frame wavelength.

    get_mean_cont: scipy.interpolate.interp1d
    Interpolation function to compute the unabsorbed mean quasar continua.

    get_num_pixels: scipy.interpolate.interp1d
    Number of pixels used to fit for eta, var_lss and fudge.

    get_stack_delta: scipy.interpolate.interp1d
    Interpolation function to compute the mean delta (from stacking all lines of
    sight).

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

    lambda_: array of float or None
    Wavelengths where the variance functions and statistics are
    computed. None (and unused) for a logarithmic wavelength solution.

    limit_eta: tuple of floats
    Limits on the correction factor to the contribution of the pipeline estimate
    of the instrumental noise to the variance.

    limit_var_lss: tuple of floats
    Limits on the pixel variance due to Large Scale Structure

    log_lambda_var_func_grid: array of float
    Logarithm of the wavelengths where the variance functions and
    statistics are computed.

    num_bins_variance: int
    Number of bins to be used to compute variance functions and statistics as
    a function of wavelength.

    num_iterations: int
    Number of iterations to determine the mean continuum shape, LSS variances, etc.

    num_processors: int
    Number of processors to be used to compute the mean continua. None for no
    specified number (subprocess will take its default value).

    order: int
    Order of the polynomial for the continuum fit.

    use_constant_weight: boolean
    If "True", set all the delta weights to one (implemented as eta = 0,
    sigma_lss = 1, fudge = 0).

    use_ivar_as_weight: boolean
    If "True", use ivar as weights (implemented as eta = 1, sigma_lss = fudge = 0).
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
        ExpectedFlux.__init__(self, config) # Fer servir Dr16 instead.
        # Només declarar les variables noves
        # esborrar el config i deixar nomes el get var lss

        # load variables from config
        self.iter_out_prefix = None
        self.limit_eta = None
        self.limit_var_lss = None
        self.order = None
        self.num_bins_variance = None
        self.num_iterations = None
        self.num_processors = None
        self.use_constant_weight = False
        self.use_ivar_as_weight = False
        self.__parse_config(config)

        # initialize variables
        self.get_eta = None
        self.get_fudge = None
        self.get_mean_cont = None
        self.get_mean_cont_weight = None
        self.get_num_pixels = None
        self.get_valid_fit = None
        self.get_var_lss = None
        self.get_tq_list = None
        self.log_lambda_var_func_grid = None
        self._initialize_variables()

        self.continuum_fit_parameters = None

        self.get_stack_delta = None
        self.get_stack_delta_weights = None
        self.var_lss_filename = None
    
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
                "by ContinuumVarianceExpectedFlux")
        if "/" in self.iter_out_prefix:
            raise ExpectedFluxError(
                "Error constructing ContinuumVarianceExpectedFlux. "
                "'iter out prefix' should not incude folders. "
                f"Found: {self.iter_out_prefix}")

        self.num_bins_variance = config.getint("num bins variance")
        if self.num_bins_variance is None:
            raise ExpectedFluxError(
                "Missing argument 'num bins variance' required by ContinuumVarianceExpectedFlux")

        self.num_iterations = config.getint("num iterations")
        if self.num_iterations is None:
            raise ExpectedFluxError(
                "Missing argument 'num iterations' required by ContinuumVarianceExpectedFlux")

        self.num_processors = config.getint("num processors")
        if self.num_processors is None:
            raise ExpectedFluxError(
                "Missing argument 'num processors' required by ContinuumVarianceExpectedFlux")

        self.order = config.getint("order")
        if self.order is None:
            raise ExpectedFluxError(
                "Missing argument 'order' required by ContinuumVarianceExpectedFlux")

        self.var_lss_filename = config.get("var lss file")

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

    def _initialize_variables(self):
        """Initialize useful variables
        The initialized arrays are:
        - self.get_eta
        - self.get_fudge
        - self.get_mean_cont
        - self.get_num_pixels
        - self.get_valid_fit
        - self.get_var_lss
        - self.log_lambda_var_func_grid
        """
        # check that Forest variables are set
        try:
            Forest.class_variable_check()
        except AstronomicalObjectError:
            raise ExpectedFluxError("Forest class variables need to be set "
                                    "before initializing variables here." )

        # initialize the mean quasar continuum
        # TODO: maybe we can drop this and compute first the mean quasar
        # continuum on compute_expected_flux
        self.get_mean_cont = interp1d(Forest.log_lambda_rest_frame_grid,
                                      np.ones_like(Forest.log_lambda_rest_frame_grid),
                                      fill_value="extrapolate")
        self.get_mean_cont_weight = interp1d(Forest.log_lambda_rest_frame_grid,
                                             np.zeros_like(
                                                 Forest.log_lambda_rest_frame_grid),
                                             fill_value="extrapolate")


        # initialize the variance-related variables (see equation 4 of
        # du Mas des Bourboux et al. 2020 for details on these variables)
        self.log_lambda_var_func_grid = (
            Forest.log_lambda_grid[0] + (np.arange(self.num_bins_variance) + .5) *
            (Forest.log_lambda_grid[-1] - Forest.log_lambda_grid[0]) /
            self.num_bins_variance)

        eta = np.ones(self.num_bins_variance)
        fudge = np.zeros(self.num_bins_variance)
        tq_list = np.zeros_like(Forest.log_lambda_rest_frame_grid)
        num_pixels = np.zeros(self.num_bins_variance)
        valid_fit = np.zeros(self.num_bins_variance, dtype=bool)

        self.get_eta = interp1d(self.log_lambda_var_func_grid,
                                eta,
                                fill_value='extrapolate',
                                kind='nearest')
        self.read_var_lss()
        self.get_tq_list = interp1d(Forest.log_lambda_rest_frame_grid,
                                       tq_list,
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
                                      num_pixels,
                                      fill_value="extrapolate",
                                      kind='nearest')

    def read_var_lss(self):
        """Read var lss from mocks. We should upgrade this into computing var_lss from Pk1d."""
        log_lambda, var_lss = np.loadtxt("/global/project/projectdirs/desi/users/cramirez/Continuum_fitting/compute_var_lss/sigma_lss_val.txt")

        def var_lss_fitting_curve(lambda_, a, b, c, d):
            """Polynomical fitting curve for var_lss 
            comming from raw files

            Arguments
            ---------
            lambda_: float
            Wavelength for the fitting curve

            a,b,c,d: float
            Parameters for the polynomical curve

            """
            return a + b*lambda_ + c*lambda_**2 + d*lambda_**3
        
        msk = ~np.isnan(var_lss) & ~np.isnan(log_lambda)
        popt, pcov = curve_fit(
            var_lss_fitting_curve,
            10**log_lambda[msk],
            var_lss[msk]
        )

        self.get_var_lss = interp1d(log_lambda, 
                                    var_lss_fitting_curve(10**log_lambda, *popt),
                                    fill_value='extrapolate',
                                    kind='nearest')

        return

        if self.var_lss_filename != "":
            filename = self.var_lss_filename
        else:
            filename = resource_filename('picca', 'delta_extraction') + '/expected_fluxes/raw_stats/'
            if Forest.wave_solution == "log":
                filename += 'colore_v9_lya_log.fits.gz'
            elif Forest.wave_solution == "lin" and Forest.delta_lambda == 2.4:
                filename += 'colore_v9_lya_lin_2.4.fits.gz'
            elif Forest.wave_solution == "lin" and Forest.delta_lambda == 3.2:
                filename += 'colore_v9_lya_lin_3.2.fits.gz'
            else:
                raise ExpectedFluxError("Couldn't find compatible raw satistics file. Provide a custom one using 'raw statistics file' field.")
        self.logger.info(f'Reading raw statistics var_lss and mean_flux from file: {filename}')

        try:
            hdul = fits.open(filename)
        except:
            raise ExpectedFluxError(f"raw statistics file {filename} couldn't be loaded")

        header = hdul[1].header
        # if (
        #     (header['LINEAR'] and Forest.wave_solution != "linear" )
        #     or (header['LINEAR'] and (not np.isclose(header['DEL_LL'], 10**Forest.log_lambda_grid[1] - 10**Forest.log_lambda_grid[0])))
        #     or not header['LINEAR'] and Forest.wave_solution != "log"
        #     or not np.isclose(header['L_MIN'], 10**Forest.log_lambda_grid[0], rtol=1e-3) 
        #     or not np.isclose(header['L_MAX'], 10**Forest.log_lambda_grid[-1], rtol=1e-3)  
        #     or not np.isclose(header['LR_MIN'], 10**Forest.log_lambda_rest_frame_grid[0], rtol=1e-3)
        #     or not np.isclose(header['LR_MAX'], 10**Forest.log_lambda_rest_frame_grid[-1], rtol=1e-3)
        # ):
        #     pixelization = "lin" if header['LINEAR'] else "log"
        #     raise ExpectedFluxError(f'''raw statistics file pixelization scheme does not match input pixelization scheme. 
        #         \t\tPIX\tL_MIN\tL_MAX\tLR_MIN\tLR_MAX\tDEL_LL
        #         raw\t\t{pixelization}\t{header['L_MIN']}\t{header['L_MAX']}\t{header['LR_MIN']}\t{header['LR_MAX']}\t{header['DEL_LL']}
        #         input\t{Forest.wave_solution}\t{10**Forest.log_lambda_grid[0]}\t{10**Forest.log_lambda_grid[-1]}\t{10**Forest.log_lambda_rest_frame_grid[0]}\t{10**Forest.log_lambda_rest_frame_grid[-1]}\t{10**Forest.log_lambda_grid[1] - 10**Forest.log_lambda_grid[0]}
        #         provide a custom file in 'raw statistics file' field matching input pixelization scheme''')

        lambda_ = hdul[1].data['LAMBDA']
        flux_variance = hdul[1].data['VAR']
        mean_flux = hdul[1].data['MEANFLUX']
        hdul.close()

        var_lss = flux_variance/mean_flux**2

        def var_lss_fitting_curve(lambda_, a, b, c, d):
            """Polynomical fitting curve for var_lss 
            comming from raw files

            Arguments
            ---------
            lambda_: float
            Wavelength for the fitting curve

            a,b,c,d: float
            Parameters for the polynomical curve

            """
            return a + b*lambda_ + c*lambda_**2 + d*lambda_**3
        
        msk = ~np.isnan(var_lss) & ~np.isnan(lambda_)
        popt, pcov = curve_fit(
            var_lss_fitting_curve,
            lambda_[msk],
            var_lss[msk]
        )

        self.get_var_lss = interp1d(np.log10(lambda_), 
                                    var_lss_fitting_curve(lambda_, *popt),
                                    fill_value='extrapolate',
                                    kind='nearest')
        
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
        # initialize arrays
        num_var_bins = 1
        var_pipe_min = np.log10(1e-5)
        var_pipe_max = np.log10(2.)

        # initialize arrays to compute the statistics of deltas
        var_delta = np.zeros_like(self.log_lambda_var_func_grid)
        mean_delta = np.zeros_like(self.log_lambda_var_func_grid)
        var2_delta = np.zeros_like(self.log_lambda_var_func_grid)
        tq_list = np.zeros_like(Forest.log_lambda_rest_frame_grid)
        tq_count = np.zeros_like(Forest.log_lambda_rest_frame_grid)
        count = np.zeros_like(self.log_lambda_var_func_grid)
        num_qso = np.zeros_like(self.log_lambda_var_func_grid)

        # compute delta statistics, binning the variance according to 'ivar'
        self.logger.progress("Computing delta statistics")
        for forest in forests:
            # ignore forest if continuum could not be computed
            if forest.continuum is None:
                continue
            var_pipe = 1 / forest.ivar / forest.continuum**2
            w = ((np.log10(var_pipe) > var_pipe_min) &
                 (np.log10(var_pipe) < var_pipe_max))

            # select the wavelength bins
            log_lambda_bins = find_bins(
                forest.log_lambda,
                self.log_lambda_var_func_grid, 
                "lin",
                )[w]

            log_lambda_rest_bins = find_bins(
                forest.log_lambda - np.log10(1 + forest.z),
                Forest.log_lambda_rest_frame_grid,
                "lin",
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
            tq = delta**2 - var_lss - eta * var_pipe[w]
            
            rebin = np.bincount(log_lambda_rest_bins, weights=tq)
            tq_list[:len(rebin)] += rebin

            rebin = np.bincount(log_lambda_rest_bins)
            tq_count[:len(rebin)] += rebin

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
        
        tq_w = tq_count > 0
        tq_list[tq_w] /= tq_count[tq_w]
        
        self.get_tq_list = interp1d(Forest.log_lambda_rest_frame_grid[tq_w],
                                    tq_list[tq_w],
                                    fill_value='extrapolate',
                                    kind='nearest')

    def compute_forest_variance(self, forest, var_pipe):
        var_lss = self.get_var_lss(forest.log_lambda)
        eta = self.get_eta(forest.log_lambda)
        sigma_c = self.get_tq_list(forest.log_lambda - np.log10( 1 + forest.z ))

        return eta*var_pipe + var_lss + sigma_c

    def save_iteration_step(self, iteration):
        """Save the statistical properties of deltas at a given iteration
        step

        Arguments
        ---------
        iteration: int
        Iteration number. -1 for final iteration

        Raise
        -----
        ExpectedFluxError if Forest.wave_solution is not 'lin' or 'log'
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
                          names=['loglam', 'eta', 'var_lss', 'fudge',
                                 'num_pixels', 'valid_fit'],
                          extname='VAR_FUNC')

            results.write([
                Forest.log_lambda_rest_frame_grid,
                self.get_mean_cont(Forest.log_lambda_rest_frame_grid),
                self.get_mean_cont_weight(Forest.log_lambda_rest_frame_grid),
                self.get_tq_list(Forest.log_lambda_rest_frame_grid),
            ],
                          names=['loglam_rest', 'mean_cont', 'weight', 'cont_var'],
                          extname='CONT')