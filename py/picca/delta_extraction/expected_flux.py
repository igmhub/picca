"""This module defines the abstract class ExpectedFlux from which all
classes computing the mean expected flux must inherit. The mean expected flux
is the product of the unabsorbed quasar continuum and the mean transmission
"""
import fitsio
import multiprocessing
import numpy as np
from scipy.interpolate import interp1d
from picca.delta_extraction.astronomical_objects.forest import Forest
from picca.delta_extraction.astronomical_objects.pk1d_forest import Pk1dForest
from picca.delta_extraction.errors import ExpectedFluxError
from picca.delta_extraction.utils import find_bins

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

        self.out_dir = config.get("out dir")
        if self.out_dir is None:
            raise ExpectedFluxError(
                "Missing argument 'out dir' required by ExpectedFlux")
        self.out_dir += "Log/"

        self.num_processors = config.getint("num processors")
        if self.num_processors is None:
            raise ExpectedFluxError(
                "Missing argument 'num processors' required by ExpectedFlux")
        if self.num_processors == 0:
            self.num_processors = (multiprocessing.cpu_count() // 2)

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
                delta = forest.flux / forest.continuum
                variance = self.compute_forest_variance(
                    forest, forest.continuum)
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

    # pylint: disable=no-self-use
    # this method should use self in child classes
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

    def extract_deltas(self, forest):
        """Apply the continuum to compute the delta field

        Arguments
        ---------
        forest: Forest
        A Forest instance to which the continuum is applied
        """
        if self.los_ids.get(forest.los_id) is not None:
            expected_flux = self.los_ids.get(
                forest.los_id).get("mean expected flux")
            forest.deltas = forest.flux / expected_flux - 1
            forest.weights = self.los_ids.get(forest.los_id).get("weights")
            if isinstance(forest, Pk1dForest):
                forest.ivar = self.los_ids.get(forest.los_id).get("ivar")
                forest.exposures_diff /= expected_flux

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

        Arguments
        ---------
        results: fitsio.FITS
        The open fits file
        """
        values = [self.log_lambda_var_func_grid]
        names = ['loglam']

        if hasattr(self, 'get_eta'):
            values.append(self.get_eta(self.log_lambda_var_func_grid))
            names.append('eta')
        if hasattr(self, 'get_var_lss'):
            values.append(self.get_var_lss(self.log_lambda_var_func_grid))
            names.append('var_lss')
        if hasattr(self, 'get_fudge'):
            values.append(self.get_fudge(self.log_lambda_var_func_grid))
            names.append('fudge')
        if hasattr(self, 'get_num_pixels'):
            values.append(self.get_num_pixels(self.log_lambda_var_func_grid))
            names.append('num_pixels')
        if hasattr(self, 'get_valid_fit'):
            values.append(self.get_valid_fit(self.log_lambda_var_func_grid))
            names.append('valid_fit')

        results.write(values, names=names, extname='VAR_FUNC')

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
