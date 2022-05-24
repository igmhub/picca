"""This module defines the class Dr16FixedFudgeExpectedFlux"""
import logging

import fitsio
import numpy as np
from scipy.interpolate import interp1d

from picca.delta_extraction.errors import ExpectedFluxError
from picca.delta_extraction.expected_fluxes.dr16_expected_flux import Dr16ExpectedFlux


class Dr16FixedEtaFudgeExpectedFlux(Dr16FixedEtaExpectedFlux,
                                    Dr16FixedFudgeExpectedFlux):
    """Class to the expected flux similar to Dr16ExpectedFlux but fixing both
    eta and fudge

    Methods
    -------
    (see Dr16FixedEtaExpectedFlux in py/picca/delta_extraction/expected_fluxes/dr16_fix_eta_expected_flux.py)
    (see Dr16FixedFudgeExpectedFlux in py/picca/delta_extraction/expected_fluxes/dr16_fix_fudge_expected_flux.py)

    Attributes
    ----------
    (see Dr16FixedEtaExpectedFlux in py/picca/delta_extraction/expected_fluxes/dr16_fix_eta_expected_flux.py)
    (see Dr16FixedFudgeExpectedFlux in py/picca/delta_extraction/expected_fluxes/dr16_fix_fudge_expected_flux.py)
    """
    pass
