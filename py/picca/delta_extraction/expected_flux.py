"""This module defines the abstract class ExpectedFlux from which all
classes computing the mean expected flux must inherit. The mean expected flux
is the product of the unabsorbed quasar continuum and the mean transmission
"""
import multiprocessing

import fitsio
import numpy as np
from scipy.interpolate import interp1d

from picca.delta_extraction.astronomical_objects.forest import Forest
from picca.delta_extraction.astronomical_objects.pk1d_forest import Pk1dForest
from picca.delta_extraction.errors import ExpectedFluxError, AstronomicalObjectError
from picca.delta_extraction.utils import find_bins

accepted_options = [
    "iter out prefix", "num bins variance", "num processors", "out dir"
]

defaults = {
    "iter out prefix": "delta_attributes",
    "num bins variance": 20,
    "num processors": 0,
}


class ExpectedFlux:
    """Abstract class from which all classes computing the expected flux
    must inherit. Classes that inherit from this should be initialized using
    a configparser.SectionProxy instance.

    Methods
    -------
    __init__
    compute_expected_flux
    extract_deltas

    Attributes
    ----------
    los_ids: dict
    A dictionary to store the mean expected flux and the weights for each line
    of sight. Keys are the identifier for the line of sight and values are
    dictionaries with the keys "mean expected flux" (continuum times stack of
    1+delta), "weights", and "continuum" pointing to the respective arrays. If
    the given Forests are also Pk1dForests, then the key "ivar" (inverse noise
    variance from the pipeline) must be available.
    Arrays must have the same size as the flux array for the corresponding line
    of sight forest instance.

    num_processors: int
    Number of processors to use for multiprocessing-enabled tasks (will be passed
    downstream to child classes)

    out_dir: str (from ExpectedFlux)
    Directory where logs will be saved.
    """

    def __init__(self, config):
        """Initialize class instance"""
        self.los_ids = {}

        self.iter_out_prefix = None
        self.out_dir = None
        self.num_bins_variance = None
        self.num_processors = None
        self.__parse_config(config)

        # check that Forest class variables are set
        # these are required in order to initialize the arrays
        try:
            Forest.class_variable_check()
        except AstronomicalObjectError as error:
            raise ExpectedFluxError(
                "Forest class variables need to be set "
                "before initializing variables here.") from error

        # initialize wavelength array for variance functions
        self.log_lambda_var_func_grid = None
        self._initialize_variance_wavelength_array()
        # variance functions are initialized by the child classes

        # initialize mean continuum
        self.get_mean_cont = None
        self.get_mean_cont_weight = None
        self._initialize_mean_continuum_arrays()

        # to store the stack of deltas
        self.get_stack_delta = None
        self.get_stack_delta_weights = None

    def _initialize_mean_continuum_arrays(self):
        """Initialize mean continuum arrays
        The initialized arrays are:
        - self.get_mean_cont
        - self.get_mean_cont_weight
        """
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

    def _initialize_variance_wavelength_array(self):
        """Initialize the wavelength array where variance functions will be
        computed
        The initialized arrays are:
        - self.log_lambda_var_func_grid
        """
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
        self.iter_out_prefix = config.get("iter out prefix")
        if self.iter_out_prefix is None:
            raise ExpectedFluxError(
                "Missing argument 'iter out prefix' required "
                "by ExpectedFlux")
        if "/" in self.iter_out_prefix:
            raise ExpectedFluxError(
                "Error constructing ExpectedFlux. "
                "'iter out prefix' should not incude folders. "
                f"Found: {self.iter_out_prefix}")

        self.num_bins_variance = config.getint("num bins variance")
        if self.num_bins_variance is None:
            raise ExpectedFluxError(
                "Missing argument 'num bins variance' required by ExpectedFlux")

        self.num_processors = config.getint("num processors")
        if self.num_processors is None:
            raise ExpectedFluxError(
                "Missing argument 'num processors' required by ExpectedFlux")
        if self.num_processors == 0:
            self.num_processors = (multiprocessing.cpu_count() // 2)

        self.out_dir = config.get("out dir")
        if self.out_dir is None:
            raise ExpectedFluxError(
                "Missing argument 'out dir' required by ExpectedFlux")
        self.out_dir += "Log/"

    def compute_delta_stack(self, forests, stack_from_deltas=False):
        """Compute a stack of the delta field as a function of wavelength

        Arguments
        ---------
        forests: List of Forest
        A list of Forest from which to compute the deltas.

        stack_from_deltas: bool - default: False
        Flag to determine whether to stack from deltas or compute them
        """
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

                delta = np.zeros_like(forest.log_lambda)
                w = forest.ivar > 0
                delta[w] = forest.flux[w] / forest.continuum[w]
                weights = 1. / self.compute_forest_variance(forest, forest.continuum)

            bins = find_bins(forest.log_lambda, Forest.log_lambda_grid,
                             Forest.wave_solution)
            stack_delta += np.bincount(bins, weights=delta * weights, minlength=stack_delta.size)
            stack_weight += np.bincount(bins, weights=weights, minlength=stack_delta.size)

        w = stack_weight > 0
        stack_delta[w] /= stack_weight[w]

        self.get_stack_delta = interp1d(
            Forest.log_lambda_grid[w],
            stack_delta[w],
            kind="nearest",
            fill_value="extrapolate")
        self.get_stack_delta_weights = interp1d(
            Forest.log_lambda_grid[w],
            stack_weight[w],
            kind="nearest",
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

        Raise
        -----
        MeanExpectedFluxError if function was not overloaded by child class
        """
        raise ExpectedFluxError("Function 'compute_expected_flux' was not "
                                "overloaded by child class")

    def _compute_mean_cont(self, forests,
        which_cont=lambda forest: forest.continuum):
        """Compute the mean quasar continuum over the whole sample.
        Then updates the value of self.get_mean_cont to contain it

        Arguments
        ---------
        forests: List of Forest
        A list of Forest from which to compute the deltas.

        which_cont: Function or lambda
        Should return what to use as continuum given a forest
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

            weights = 1. / self.compute_forest_variance(forest, forest.continuum)
            forest_continuum = which_cont(forest)
            mean_cont += np.bincount(bins, weights=forest_continuum * weights,
                minlength=mean_cont.size)
            mean_cont_weight += np.bincount(bins, weights=weights, minlength=mean_cont.size)

        w = mean_cont_weight > 0
        mean_cont[w] /= mean_cont_weight[w]
        mean_cont /= mean_cont[w].mean()
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

    def compute_forest_variance(self, forest, continuum):
        """Compute the forest variance

        Arguments
        ---------
        forest: Forest
        A forest instance where the variance will be computed

        continuum: array of float
        Quasar continuum associated with the forest

        Raise
        -----
        MeanExpectedFluxError if function was not overloaded by child class
        """
        raise ExpectedFluxError("Function 'compute_forest_variance' was not "
                                "overloaded by child class")

    def extract_deltas(self, forest):
        """Apply the continuum to compute the delta field

        Arguments
        ---------
        forest: Forest
        A Forest instance to which the continuum is applied
        """
        if self.los_ids.get(forest.los_id) is not None:
            w = self.los_ids.get(forest.los_id).get("weights") > 0

            expected_flux = self.los_ids.get(
                forest.los_id).get("mean expected flux")
            forest.deltas = np.zeros_like(forest.flux)
            forest.deltas[w] = forest.flux[w] / expected_flux[w] - 1
            forest.weights = self.los_ids.get(forest.los_id).get("weights")
            if isinstance(forest, Pk1dForest):
                forest.ivar = self.los_ids.get(forest.los_id).get("ivar")
                forest.exposures_diff[w] /= expected_flux[w]
                forest.exposures_diff[~w] = 0

            forest.continuum = self.los_ids.get(forest.los_id).get("continuum")

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
        ],
                      names=['loglam_rest', 'mean_cont', 'weight'],
                      extname='CONT')

    def hdu_fit_metadata(self, results):
        """Add to the results file an HDU with the fits results

        This function is a placeholder here and should be overloaded by child
        classes if they require it

        Arguments
        ---------
        results: fitsio.FITS
        The open fits file
        """

    def hdu_stack_deltas(self, results):
        """Add to the results file an HDU with the delta stack

        Arguments
        ---------
        results: fitsio.FITS
        The open fits file
        """
        header = {}
        if hasattr(self, 'order'):
            header["FITORDER"] = self.order

        results.write([
            Forest.log_lambda_grid,
            self.get_stack_delta(Forest.log_lambda_grid),
            self.get_stack_delta_weights(Forest.log_lambda_grid)
        ],
                      names=['loglam', 'stack', 'weight'],
                      header=header,
                      extname='STACK_DELTAS')

    def hdu_var_func(self, results):
        """Add to the results file an HDU with the variance functions

        This function is a placeholder here and should be overloaded by child
        classes if they require it

        Arguments
        ---------
        results: fitsio.FITS
        The open fits file
        """

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
            self.hdu_stack_deltas(results)
            self.hdu_var_func(results)
            self.hdu_cont(results)
            self.hdu_fit_metadata(results)
