"""This module defines the class Dr16FixedFudgeExpectedFlux"""
import logging

import fitsio
import numpy as np
from scipy.interpolate import interp1d

from picca.delta_extraction.errors import ExpectedFluxError
from picca.delta_extraction.expected_fluxes.dr16_expected_flux import Dr16ExpectedFlux
from picca.delta_extraction.expected_fluxes.dr16_expected_flux import (
    defaults, accepted_options)
from picca.delta_extraction.utils import update_accepted_options, update_default_options

accepted_options = update_accepted_options(accepted_options, ["fudge value"])
accepted_options = update_accepted_options(
    accepted_options, ["use constant weight", "use ivar as weight"],
    remove=True)

defaults = update_default_options(defaults, {
    "fudge value": 0.0,
})


class Dr16FixedFudgeExpectedFlux(Dr16ExpectedFlux):
    """Class to the expected flux similar to Dr16ExpectedFlux but fixing the
    fudge factor

    Methods
    -------
    (see Dr16ExpectedFlux in py/picca/delta_extraction/expected_fluxes/dr16_expected_flux.py)
    __init__
    _initialize_get_fudge
    __parse_config

    Attributes
    ----------
    (see Dr16ExpectedFlux in py/picca/delta_extraction/expected_fluxes/dr16_expected_flux.py)

    fudge_value: float or string
    If a string, name of the file containing the fudge values as a function of
    wavelength. If a float, the fudge value will be applied to all wavelengths
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

    def _initialize_get_fudge(self):
        """Initialiaze function get_fudge"""
        # initialize fudge factor
        if self.fudge_value.endswith(".fits") or self.fudge_value.endswith(
                ".fits.gz"):
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
        # note that for fudge to be fitted, we need to include it to
        # self.fit_variance_functions:
        # self.fit_variance_functions.append("fudge")
        # since we do not do it here, fudge is fixed

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
                    "Wrong argument 'fudge value' passed to "
                    "Dr16FixFudgeExpectedFlux. Expected a fits file or "
                    f"a float. Found {self.fudge_value}") from error
