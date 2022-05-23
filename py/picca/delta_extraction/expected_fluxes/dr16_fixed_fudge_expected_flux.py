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
        - self.get_fudge
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

        # initialize fudge factor
        if self.fudge.endswith(".fits") or self.fudge.endswith(".fits.gz"):
            hdu = fitsio.read(self.fudge, ext="VAR_FUNC")
            self.get_fudge = interp1d(hdu["loglam"],
                                      hdu["fudge"],
                                      fill_value='extrapolate',
                                      kind='nearest')
        else:
            fudge = np.ones(self.num_bins_variance) * float(self.fudge_value)
            self.get_fudge = interp1d(self.log_lambda_var_func_grid,
                                      fudge,
                                      fill_value='extrapolate',
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
            raise ExpectedFluxError("Missing argument 'fudge value' required "
                                    "by Dr16FixFudgeExpectedFlux")
        if not (self.fudge.endswith(".fits") or
                self.fudge.endswith(".fits.gz")):
            try:
                _ = float(self.fudge)
            except ValueError as error:
                raise ExpectedFluxError(
                    "Wrong argument 'fudge value'. Expected a fits file or "
                    "a float. Found {self.fudge}") from error

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
        num_pixels = np.zeros(self.num_bins_variance)
        valid_fit = np.zeros(self.num_bins_variance)
        chi2_in_bin = np.zeros(self.num_bins_variance)

        # initialize the fitter class
        leasts_squares = LeastsSquaresVarStatsFixFudge(
            self.num_bins_variance,
            forests,
            self.log_lambda_var_func_grid,
        )

        self.logger.progress(" Mean quantities in observer-frame")
        self.logger.progress(
            " loglam    eta      var_lss    chi2     num_pix valid_fit")
        for index in range(self.num_bins_variance):
            leasts_squares.set_fit_bins(index)

            minimizer = iminuit.Minuit(leasts_squares,
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
                valid_fit[index] = True
            else:
                eta[index] = 1.
                var_lss[index] = 0.1
                valid_fit[index] = False
            num_pixels[index] = leasts_squares.get_num_pixels()
            chi2_in_bin[index] = minimizer.fval

            self.logger.progress(
                f" {self.log_lambda_var_func_grid[index]:.3e} "
                f"{eta[index]:.2e} {var_lss[index]:.2e} "
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
