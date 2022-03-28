"""This module defines the class TrueContinuum"""
import logging
import multiprocessing

import fitsio
from astropy.io import fits
import numpy as np
from scipy.interpolate import interp1d
import healpy
from pkg_resources import resource_filename

from picca.delta_extraction.astronomical_objects.forest import Forest
from picca.delta_extraction.astronomical_objects.pk1d_forest import Pk1dForest
from picca.delta_extraction.errors import ExpectedFluxError
from picca.delta_extraction.expected_flux import ExpectedFlux
from picca.delta_extraction.utils import find_bins

accepted_options = ["input directory", "iter out prefix",
                    "num processors", "out dir",
                    "raw statistics file"]

defaults = {
    "iter out prefix": "delta_attributes",
    "raw statistics file": "",
}


class TrueContinuum(ExpectedFlux):
    """Class to compute the expected flux using the true unabsorbed contiuum
    for mocks.
    It uses var_lss pre-computed from mocks and the mean flux modeled from a 2nd order polinomial in effective optical depth.

    Methods
    -------
    extract_deltas (from ExpectedFlux)
    __init__
    __parse_config
    compute_expected_flux
    compute_mean_cont_lin
    compute_mean_cont_log
    read_true_continuum
    read_raw_statistics
    populate_los_ids
    save_delta_attributes

    Attributes
    ----------
    los_ids: dict (from ExpectedFlux)
    A dictionary to store the mean expected flux, the weights, and
    the inverse variance for each line of sight. Keys are the identifier for the
    line of sight and values are dictionaries with the keys "mean expected flux",
    and "weights" pointing to the respective arrays. If the given Forests are
    also Pk1dForests, then the key "ivar" must be available. Arrays have the same
    size as the flux array for the corresponding line of sight forest instance.

    out_dir: str (from ExpectedFlux)
    Directory where logs will be saved.

    get_var_lss: scipy.interpolate.interp1d
    Interpolation function to compute mapping functions var_lss. See equation 4 of
    du Mas des Bourboux et al. 2020 for details. Data for interpolation is read from a file.

    input_directory: str
    Directory where true continum data is store

    iter_out_prefix: str
    Prefix of the iteration files. These file contain the statistical properties
    of deltas at a given iteration step. Intermediate files will add
    '_iteration{num}.fits.gz' to the prefix for intermediate steps and '.fits.gz'
    for the final results.

    num_processors: int or None
    Number of processors to be used to compute the mean continua. None for no
    specified number (subprocess will take its default value).
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
        super().__init__(config)

        # load variables from config
        self.input_directory = None
        self.iter_out_prefix = None
        self.num_processors = None
        self.__parse_config(config)

        # read large scale structure variance and mean flux
        self.get_var_lss = None
        self.get_mean_flux = None
        self.read_raw_statistics()


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
        self.input_directory = config.get("input directory")
        if self.input_directory is None:
            raise ExpectedFluxError(
                "Missing argument 'input directory' required "
                "by TrueContinuum")

        self.iter_out_prefix = config.get("iter out prefix")
        if self.iter_out_prefix is None:
            raise ExpectedFluxError(
                "Missing argument 'iter out prefix' required "
                "by TrueContinuum")
        if "/" in self.iter_out_prefix:
            raise ExpectedFluxError(
                "Error constructing TrueContinuum. "
                "'iter out prefix' should not incude folders. "
                f"Found: {self.iter_out_prefix}")

        self.num_processors = config.getint("num processors")

        self.raw_statistics_filename = config.get("raw statistics file")


    def compute_expected_flux(self, forests):
        """

        Arguments
        ---------
        forests: List of Forest
        A list of Forest from which to compute the deltas.

        Raise
        -----
        ExpectedFluxError if Forest.wave_solution is not 'lin' or 'log'
        """
        context = multiprocessing.get_context('fork')
        for iteration in range(1):
            pool = context.Pool(processes=self.num_processors)
            self.logger.progress(
                f"Reading continum with {self.num_processors} processors"
            )

            forests = pool.map(self.read_true_continuum, forests)
            pool.close()

        self.compute_mean_cont(forests)

        # now loop over forests to populate los_ids
        self.populate_los_ids(forests)
        # Save delta atributes
        self.save_delta_attributes()

    def compute_mean_cont(self, forests):
        """Compute the mean quasar continuum over the whole sample.
        Then updates the value of self.get_mean_cont to contain it

        Arguments
        ---------
        forests: List of Forest
        A list of Forest from which to compute the deltas.
        """
        mean_cont = np.zeros_like(Forest.log_lambda_rest_frame_grid.size)
        mean_cont_weight = np.zeros_like(Forest.log_lambda_rest_frame_grid.size)

        for forest in forests:
            if forest.bad_continuum_reason is not None:
                continue
            bins = find_bins(
                forest.log_lambda -  np.log10(1 + forest.z),
                Forest.log_lambda_rest_frame_grid,
                Forest.wave_solution
            )

            var_lss = self.get_var_lss(forest.log_lambda)
            var_pipe = 1. / forest.ivar / forest.continuum**2
            variance = var_lss + var_pipe
            weights = 1 / variance
            cont = np.bincount(bins,
                               weights= forest.continuum * weights)
            mean_cont[:len(cont)] += cont
            cont = np.bincount(bins, weights=weights)
            mean_cont_weight[:len(cont)] += cont

        w = mean_cont_weight > 0
        mean_cont[w] /= mean_cont_weight[w]
        mean_cont /= mean_cont.mean()
        log_lambda_cont = Forest.log_lambda_rest_frame_grid[w]

        self.get_mean_cont = interp1d(log_lambda_cont,
                                      mean_cont,
                                      fill_value="extrapolate")
        self.get_mean_cont_weight = interp1d(log_lambda_cont,
                                             mean_cont_weight,
                                             fill_value=0.0,
                                             bounds_error=False)

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
        in_nside = 16
        healpix = healpy.ang2pix(in_nside, np.pi / 2 - forest.dec, forest.ra,
                                 nest=True)
        filename_truth = (
            f"{self.input_directory}/{healpix//100}/{healpix}/truth-{in_nside}-"
            f"{healpix}.fits")
        hdul = fits.open(filename_truth)
        lambda_min = hdul["TRUE_CONT"].header["WMIN"]
        lambda_max = hdul["TRUE_CONT"].header["WMAX"]
        delta_lambda = hdul["TRUE_CONT"].header["DWAVE"]
        lambda_ = np.arange(lambda_min, lambda_max + delta_lambda, delta_lambda)
        true_cont = hdul["TRUE_CONT"].data
        hdul.close()
        indx = np.where(true_cont["TARGETID"]==forest.targetid)
        true_continuum = interp1d(lambda_, true_cont["TRUE_CONT"][indx])

        forest.continuum = true_continuum(10**forest.log_lambda)[0]
        mean_optical_depth = np.ones(forest.log_lambda.size)
        tau, gamma, lambda_rest_frame = 0.0023, 3.64, 1215.67
        w = 10.**forest.log_lambda / (1. + forest.z) <= lambda_rest_frame
        z = 10.**forest.log_lambda / lambda_rest_frame - 1.

        return forest

    def read_raw_statistics(self):
        """Read the LSS delta variance and mean transmitted flux from files written by the raw analysis
        """
        #files are only for lya so far, this will need to be updated so that regions other than Lya are available

        if self.raw_statistics_filename != "":
            filename = self.raw_statistics_filename
        else:
            filename = resource_filename('picca', 'delta_extraction') + '/expected_fluxes/raw_stats/'
            if Forest.wave_solution == "log":
                filename += 'colore_v9_lya_log.fits.gz'
            elif Forest.wave_solution == "lin" and (10**Forest.log_lambda_grid[1]-10**Forest.log_lambda_grid[0]) == 2.4:
                filename += 'colore_v9_lya_lin_2.4.fits.gz'
            elif Forest.wave_solution == "lin" and (10**Forest.log_lambda_grid[1]-10**Forest.log_lambda_grid[0]) == 3.2:
                filename += 'colore_v9_lya_lin_3.2.fits.gz'
            else:
                raise ExpectedFluxError("Couldn't find compatible raw satistics file. Provide a custom one using 'raw statistics file' field.")
        self.logger.info(f'Reading raw statistics var_lss and mean_flux from file: {filename}')

        try:
            hdul = fits.open(filename)
        except:
            raise ExpectedFluxError(f"raw statistics file {filename} couldn't be loaded")

        header = hdul[1].header
        if Forest.wave_solution == "log":
            if (
                header['LINEAR']
                or not np.isclose(header['L_MIN'], 10**Forest.log_lambda_grid[0], rtol=1e-3)
                or not np.isclose(header['L_MAX'], 10**Forest.log_lambda_grid[-1], rtol=1e-3)
                or not np.isclose(header['LR_MIN'], 10**Forest.log_lambda_rest_frame_grid[0], rtol=1e-3)
                or not np.isclose(header['LR_MAX'], 10**Forest.log_lambda_rest_frame_grid[-1], rtol=1e-3)
                or not np.isclose(header['DEL_LL'], Forest.log_lambda_grid[1]-Forest.log_lambda_grid[0], rtol=1e-3)
            ):
                raise ExpectedFluxError(f'''raw statistics file pixelization scheme does not match input pixelization scheme.
                \t\tL_MIN\tL_MAX\tLR_MIN\tLR_MAX\tDEL_LL
                raw\t{header['L_MIN']}\t{header['L_MAX']}\t{header['LR_MIN']}\t{header['LR_MAX']}\t{header['DEL_LL']}
                input\t{10**Forest.log_lambda_grid[0]}\t{10**Forest.log_lambda_grid[-1]}\t{10**Forest.log_lambda_rest_frame_grid[0]}\t{10**Forest.log_lambda_rest_frame_grid[-1]}\t{Forest.delta_log_lambda}
                provide a custom file in 'raw statistics file' field matching input pixelization scheme''')
        elif Forest.wave_solution == "lin":
            if (
                not header['LINEAR']
                or not np.isclose(header['L_MIN'], 10**Forest.log_lambda_grid[0] , rtol=1e-3)
                or not np.isclose(header['L_MAX'], 10**Forest.log_lambda_grid[-1] , rtol=1e-3)
                or not np.isclose(header['LR_MIN'], 10**Forest.log_lambda_rest_frame_grid[0], rtol=1e-3)
                or not np.isclose(header['LR_MAX'], 10**Forest.log_lambda_rest_frame_grid[-1], rtol=1e-3)
                or not np.isclose(header['DEL_LL'], 10**Forest.log_lambda_grid[1]-10**Forest.log_lambda_grid[0], rtol=1e-3)
            ):
                raise ExpectedFluxError(f'''raw statistics file pixelization scheme does not match input pixelization scheme.
                \t\tL_MIN\tL_MAX\tLR_MIN\tLR_MAX\tDEL_L
                raw\t{header['L_MIN']}\t{header['L_MAX']}\t{header['LR_MIN']}\t{header['LR_MAX']}\t{header['DEL_LL']}
                input\t{10**Forest.log_lambda_grid[0]}\t{10**Forest.log_lambda_grid[-1]}\t{ 10**Forest.log_lambda_rest_frame_grid[0]}\t{ 10**Forest.log_lambda_rest_frame_grid[-1]}\t{10**Forest.log_lambda_grid[1]-10**Forest.log_lambda_grid[0]}
                provide a custom file in 'raw statistics file' field matching input pixelization scheme''')

        lambda_ = hdul[1].data['LAMBDA']
        flux_variance = hdul[1].data['VAR']
        mean_flux = hdul[1].data['MEANFLUX']
        hdul.close()

        var_lss = flux_variance/mean_flux**2

        self.get_var_lss = interp1d(lambda_,
                                    var_lss,
                                    fill_value='extrapolate',
                                    kind='nearest')

        self.get_mean_flux = interp1d(lambda_,
                                      mean_flux,
                                      fill_value='extrapolate',
                                      kind='nearest')

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
            # get the variance functions
            var_lss = self.get_var_lss(forest.log_lambda)

            mean_expected_flux = forest.continuum
            var_pipe = 1. / forest.ivar/ forest.continuum**2
            variance =  var_lss + var_pipe
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

    def save_delta_attributes(self):
        """Save mean continuum in the delta attributes file.

        Raise
        -----
        ExpectedFluxError if Forest.wave_solution is not 'lin' or 'log'
        """
        iter_out_file = self.iter_out_prefix + ".fits.gz"

        with fitsio.FITS(self.out_dir + iter_out_file, 'rw',
                         clobber=True) as results:
            header = {}
            header["FITORDER"] = -1
            results.write([
                Forest.log_lambda_rest_frame_grid,
                self.get_mean_cont(Forest.log_lambda_rest_frame_grid),
                self.get_mean_cont_weight(Forest.log_lambda_rest_frame_grid),
            ],
                          names=['loglam_rest', 'mean_cont', 'weight'],
                          extname='CONT',
                          header=header)
