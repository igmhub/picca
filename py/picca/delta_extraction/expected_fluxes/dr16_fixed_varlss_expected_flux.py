"""This module defines the class Dr16FixedVarlssExpectedFlux"""
import logging

import fitsio
import numpy as np
from scipy.interpolate import interp1d

from picca.delta_extraction.errors import ExpectedFluxError
from picca.delta_extraction.expected_fluxes.dr16_expected_flux import Dr16ExpectedFlux
from picca.delta_extraction.expected_fluxes.dr16_expected_flux import (
    defaults, accepted_options)
from picca.delta_extraction.utils import update_accepted_options, update_default_options

accepted_options = update_accepted_options(accepted_options, ["var lss value"])
accepted_options = update_accepted_options(
    accepted_options,
    ["limit var lss", "use constant weight", "use ivar as weight"],
    remove=True)

defaults = update_default_options(defaults, {
    "var lss value": 0.15,
})


class Dr16FixedVarlssExpectedFlux(Dr16ExpectedFlux):
    """Class to the expected flux similar to Dr16ExpectedFlux but fixing

    Methods
    -------
    (see Dr16ExpectedFlux in py/picca/delta_extraction/expected_fluxes/dr16_expected_flux.py)
    __init__
    __initialize_get_eta
    __parse_config

    Attributes
    ----------
    (see Dr16ExpectedFlux in py/picca/delta_extraction/expected_fluxes/dr16_expected_flux.py)

    var_lss_value: float or string
    If a string, name of the file containing the var_lss values as a function of
    wavelength. If a float, the var_lss value will be applied to all wavelengths
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
        self.var_lss_value = None
        self.__parse_config(config)

        super().__init__(config)

    def _initialize_get_var_lss(self):
        """Initialiaze function get_var_lss"""
        # initialize fudge factor
        if self.var_lss_value.endswith(".fits") or self.var_lss_value.endswith(
                ".fits.gz"):
            hdu = fitsio.read(self.var_lss_value, ext="VAR_FUNC")
            self.get_var_lss = interp1d(hdu["loglam"],
                                        hdu["var_lss"],
                                        fill_value='extrapolate',
                                        kind='nearest')
        else:
            var_lss = np.ones(self.num_bins_variance) * float(
                self.var_lss_value)
            self.get_var_lss = interp1d(self.log_lambda_var_func_grid,
                                        var_lss,
                                        fill_value='extrapolate',
                                        kind='nearest')
        # note that for var_lss to be fitted, we need to include it to
        # self.fit_variance_functions:
        # self.fit_variance_functions.append("var_lss")
        # since we do not do it here, var_lss is fixed

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
        self.var_lss_value = config.get("var lss value")
        if self.var_lss_value is None:
            raise ExpectedFluxError("Missing argument 'var lss value' required "
                                    "by Dr16FixedVarlssExpectedFlux")
        if not (self.var_lss_value.endswith(".fits") or
                self.var_lss_value.endswith(".fits.gz")):
            try:
                _ = float(self.var_lss_value)
            except ValueError as error:
                raise ExpectedFluxError(
                    "Wrong argument 'var_lss value' passed to "
                    "Dr16FixedVarlssExpectedFlux. Expected a fits file or "
                    f"a float. Found {self.var_lss_value}") from error
