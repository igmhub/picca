"""This module defines the class Dr16FixedFudgeExpectedFlux"""
import logging

import fitsio
import numpy as np
from scipy.interpolate import interp1d

from picca.delta_extraction.errors import ExpectedFluxError
from picca.delta_extraction.expected_fluxes.dr16_expected_flux import Dr16ExpectedFlux

accepted_options = [
    "fudge_value",
]

defaults = {
    "fudge_value": 0.0,
}

FIT_VARIANCE_FUNCTIONS = ["eta", "var_lss"]

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
        """
        self.logger = logging.getLogger(__name__)

        # load variables from config
        self.fudge_value = None
        self.__parse_config(config)

        super().__init__(config)

    def _initialize_variance_functions(self):
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
        self.fit_variance_functions = FIT_VARIANCE_FUNCTIONS

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
        if self.fudge_value.endswith(".fits") or self.fudge_value.endswith(".fits.gz"):
            hdu = fitsio.read(self.fudge_value, ext="VAR_FUNC")
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
        if not (self.fudge_value.endswith(".fits") or
                self.fudge_value.endswith(".fits.gz")):
            try:
                _ = float(self.fudge_value)
            except ValueError as error:
                raise ExpectedFluxError(
                    "Wrong argument 'fudge value'. Expected a fits file or "
                    "a float. Found {self.fudge}") from error
