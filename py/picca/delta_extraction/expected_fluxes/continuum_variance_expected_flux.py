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
from picca.delta_extraction.expected_fluxes.dr16_expected_flux import (# pylint: disable=unused-import
    accepted_options, defaults)
from picca.delta_extraction.expected_flux import ExpectedFlux
from picca.delta_extraction.utils import find_bins

accepted_options = sorted(
    list(
        set(accepted_options + accepted_options_quasar_catalogue +
            ["var lss file"])))

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

class ContinuumVarianceExpectedFlux(Dr16ExpectedFlux):
    """Class to the expected flux with a variance term dependent
    on the rest frame wavelength.

    Methods
    -------
    (see Dr16ExpectedFlux in py/picca/delta_extraction/expected_fluxes/dr16_expected_flux.py)

    Attributes
    ----------
    (see Dr16ExpectedFlux in py/picca/delta_extraction/expected_fluxes/dr16_expected_flux.py)

    self.get_tq_list: scipy.interpolate.interp1d
    ??????

    var_lss_filename: string or None
    File containing the LSS variance contribution. If not passed to the constructor
    it is set to None. In this case,

    Unused attributes from parent
    -----------------------------
    get_fudge
    limit_eta
    limit_var_lss
    use_constant_weight
    use_ivar_as_weight
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

        self.var_lss_filename = None
        self.__parse_config(config)

        self.get_tq_list = None
        super().__init__(self, config)

    def __initialize_variance_functions(self):
        """Initialize variance functions
        The initialized arrays are:
        - self.get_eta
        - self.get_tq_list
        - self.get_num_pixels
        - self.get_valid_fit
        - self.get_var_lss
        """
        eta = np.ones(self.num_bins_variance)
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
        self.get_num_pixels = interp1d(self.log_lambda_var_func_grid,
                                       num_pixels,
                                       fill_value="extrapolate",
                                       kind='nearest')
        self.get_valid_fit = interp1d(self.log_lambda_var_func_grid,
                                      num_pixels,
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

    def read_var_lss(self):
        """Read var lss from mocks. We should upgrade this into computing var_lss from Pk1d."""
        log_lambda, var_lss = np.loadtxt("/global/project/projectdirs/desi/users/cramirez/Continuum_fitting/compute_var_lss/sigma_lss_val.txt")

        mask = ~np.isnan(var_lss) & ~np.isnan(log_lambda)
        popt, pcov = curve_fit(
            var_lss_fitting_curve,
            10**log_lambda[mask],
            var_lss[mask]
        )

        self.get_var_lss = interp1d(log_lambda,
                                    var_lss_fitting_curve(10**log_lambda, *popt),
                                    fill_value='extrapolate',
                                    kind='nearest')

        return

        if self.var_lss_filename is not None:
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

        mask = ~np.isnan(var_lss) & ~np.isnan(lambda_)
        popt, pcov = curve_fit(
            var_lss_fitting_curve,
            lambda_[mask],
            var_lss[mask]
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

    def compute_forest_variance(self, forest, continuum):
        """Compute the forest variance following Du Mas 2020

        Arguments
        ---------
        forest: Forest
        A forest instance where the variance will be computed

        var_pipe: float
        Pipeline variances that will be used to compute the full variance
        """
        var_pipe = 1. / forest.ivar / continuum**2
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
                self.get_num_pixels(self.log_lambda_var_func_grid),
                self.get_valid_fit(self.log_lambda_var_func_grid)
            ],
                          names=['loglam', 'eta', 'var_lss',
                                 'num_pixels', 'valid_fit'],
                          extname='VAR_FUNC_OBS_FRAME')

            results.write([
                Forest.log_lambda_rest_frame_grid,
                self.get_mean_cont(Forest.log_lambda_rest_frame_grid),
                self.get_mean_cont_weight(Forest.log_lambda_rest_frame_grid),
                self.get_tq_list(Forest.log_lambda_rest_frame_grid),
            ],
                          names=['loglam_rest', 'mean_cont', 'weight', 'cont_var'],
                          extname='VAR_FUNC_REST_FRAME')
