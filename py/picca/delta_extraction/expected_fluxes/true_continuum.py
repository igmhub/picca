"""This module defines the class Dr16ExpectedFlux"""
import logging
import multiprocessing

from astropy.io import fits
from astropy.table import Table,vstack
import iminuit
import numpy as np
from scipy.interpolate import interp1d
import healpy
from pkg_resources import resource_filename

from picca.delta_extraction.astronomical_objects.forest import Forest
from picca.delta_extraction.astronomical_objects.pk1d_forest import Pk1dForest
from picca.delta_extraction.errors import ExpectedFluxError
from picca.delta_extraction.expected_flux import ExpectedFlux, defaults

defaults.update({
    "iter out prefix": "delta_attributes",
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
        self.read_var_lss()

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
        self.input_directory = config.get("input directory")
        if self.input_directory is None:
            raise ExpectedFluxError(
                "Missing argument 'input directory' required "
                "by true_continuum")

        self.iter_out_prefix = config.get("iter out prefix")
        self.num_processors = config.getint("num processors")
        self.num_iterations = config.getint("num iterations")
        if self.num_iterations is None:
            self.num_iterations = defaults.get("num iterations")




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
        input_directory=self.input_directory
        in_nside = 16
        healpix = healpy.ang2pix(in_nside, np.pi / 2 - forest.dec, forest.ra, nest=True)
        filename_truth = (
                    f"{input_directory}/{healpix//100}/{healpix}/truth-{in_nside}-{healpix}.fits")
        hdul = fits.open(filename_truth)
        wmin = hdul["TRUE_CONT"].header["WMIN"]
        wmax = hdul["TRUE_CONT"].header["WMAX"]
        dwave = hdul["TRUE_CONT"].header["DWAVE"]
        twave = np.arange(wmin,wmax+dwave,dwave)
        true_cont = hdul["TRUE_CONT"].data
        hdul.close()
        indx = np.where(true_cont["TARGETID"]==forest.targetid)
        true_continuum = interp1d(twave,true_cont["TRUE_CONT"][indx])

        if Forest.wave_solution == "log":
            forest.continuum = true_continuum(10**forest.log_lambda)[0]
        elif Forest.wave_solution == "lin":
            forest.continuum = true_continuum(forest.lambda_)[0]
        else:
            raise ExpectedFluxError("Forest.wave_solution must be "
                                            "either 'log' or 'linear'")

        #multiply by mean flux model for now, it might be changed for computing the mean flux or reading it from file
        mean_optical_depth = np.ones(forest.log_lambda.size)
        tau, gamma, lambda_rest_frame = 0.0023, 3.64, 1215.67
        w = 10.**forest.log_lambda / (1. + forest.z) <= lambda_rest_frame
        z = 10.**forest.log_lambda / lambda_rest_frame - 1.
        mean_optical_depth[w] *= np.exp(-tau * (1. + z[w])**gamma)

        forest.continuum *= mean_optical_depth
        return forest

    def read_var_lss(self):
        """

        Arguments
        ---------
        TO BE MODIFIED TO READ ANDREI's var_lss
        Raise
        -----

        """
        var_lss_file=resource_filename('picca', 'delta_extraction')+'/expected_fluxes/var_lss.dat'
        var_lss_lambda,var_lss=np.loadtxt(var_lss_file).T
        self.get_var_lss = interp1d(var_lss_lambda,
                                    var_lss,
                                    fill_value='extrapolate',
                                    kind='nearest')

    def compute_delta_stack(self, forests, stack_from_deltas=False):
        """Compute a stack of the delta field as a function of wavelength

        Arguments
        ---------
        forests: List of Forest
        A list of Forest from which to compute the deltas.

        stack_from_deltas: bool - default: False
        Flag to determine whether to stack from deltas or compute them

        Raise
        -----
        ExpectedFluxError if Forest.wave_solution is not 'lin' or 'log'
        """
        # TODO: move this to _initialize_variables_lin and
        # _initialize_variables_log (after tests are done)
        if Forest.wave_solution == "log":
            num_bins = int((Forest.log_lambda_max - Forest.log_lambda_min) /
                           Forest.delta_log_lambda) + 1
            stack_log_lambda = (Forest.log_lambda_min +
                                np.arange(num_bins) * Forest.delta_log_lambda)
        elif Forest.wave_solution == "lin":
            num_bins = int((Forest.lambda_max - Forest.lambda_min) /
                           Forest.delta_lambda) + 1
            stack_lambda = (Forest.lambda_min +
                            np.arange(num_bins) * Forest.delta_lambda)
        else:
            raise ExpectedFluxError("Forest.wave_solution must be either "
                                    "'log' or 'linear'")

        stack_delta = np.zeros(num_bins)
        stack_weight = np.zeros(num_bins)

        for forest in forests:
            if stack_from_deltas:
                delta = forest.delta
                weights = forest.weights
            else:
                # ignore forest if continuum could not be computed
                if forest.continuum is None:
                    continue
                delta = forest.flux / forest.continuum
                if Forest.wave_solution == "log":
                    var_lss = self.get_var_lss(forest.log_lambda)
                elif Forest.wave_solution == "lin":
                    var_lss = self.get_var_lss(forest.lambda_)
                else:
                    raise ExpectedFluxError(
                        "Forest.wave_solution must be either "
                        "'log' or 'linear'")
                var = 1. / forest.ivar / forest.continuum**2
                variance = var + var_lss
                weights = 1. / variance

            if Forest.wave_solution == "log":
                bins = ((forest.log_lambda - Forest.log_lambda_min) /
                        Forest.delta_log_lambda + 0.5).astype(int)
            elif Forest.wave_solution == "lin":
                bins = ((forest.lambda_ - Forest.log_lambda_min) /
                        Forest.delta_log_lambda + 0.5).astype(int)
            else:
                raise ExpectedFluxError("Forest.wave_solution must be either "
                                        "'log' or 'linear'")
            rebin = np.bincount(bins, weights=delta * weights)
            stack_delta[:len(rebin)] += rebin
            rebin = np.bincount(bins, weights=weights)
            stack_weight[:len(rebin)] += rebin

        w = stack_weight > 0
        stack_delta[w] /= stack_weight[w]

        if Forest.wave_solution == "log":
            self.get_stack_delta = interp1d(stack_log_lambda[stack_weight > 0.],
                                            stack_delta[stack_weight > 0.],
                                            kind="nearest",
                                            fill_value="extrapolate")
            self.get_stack_delta_weights = interp1d(
                stack_log_lambda[stack_weight > 0.],
                stack_weight[stack_weight > 0.],
                kind="nearest",
                fill_value=0.0,
                bounds_error=False)
        elif Forest.wave_solution == "lin":
            self.get_stack_delta = interp1d(stack_lambda[stack_weight > 0.],
                                            stack_delta[stack_weight > 0.],
                                            kind="nearest",
                                            fill_value="extrapolate")
            self.get_stack_delta_weights = interp1d(
                stack_lambda[stack_weight > 0.],
                stack_weight[stack_weight > 0.],
                kind="nearest",
                fill_value=0.0,
                bounds_error=False)
        else:
            raise ExpectedFluxError("Forest.wave_solution must be either "
                                    "'log' or 'linear'")


    def compute_expected_flux(self, forests, out_dir):
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
        context = multiprocessing.get_context('fork')
        for iteration in range(1):
            pool = context.Pool(processes=self.num_processors)
            self.logger.progress(
                f"Continuum fitting: starting iteration {iteration} of {self.num_iterations}"
            )

            forests = pool.map(self.read_true_continuum, forests)
            pool.close()


        # now loop over forests to populate los_ids
        self.compute_delta_stack(forests)
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


            mean_expected_flux = forest.continuum #* stack_delta ##not sure of this stack_delta 
            var_pipe = 1. / forest.ivar / mean_expected_flux**2
            variance = var_pipe + var_lss
            weights = 1. / variance

            if isinstance(forest, Pk1dForest):
                ivar = forest.ivar / mean_expected_flux**2

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

