"""This module defines the class Dr16FixedVarlssFudgeExpectedFlux"""
from picca.delta_extraction.expected_fluxes.dr16_fixed_varlss_expected_flux import (
    Dr16FixedVarlssExpectedFlux)
from picca.delta_extraction.expected_fluxes.dr16_fixed_varlss_expected_flux import (
    defaults, accepted_options)
from picca.delta_extraction.expected_fluxes.dr16_fixed_fudge_expected_flux import (
    Dr16FixedFudgeExpectedFlux)
from picca.delta_extraction.expected_fluxes.dr16_fixed_fudge_expected_flux import (
    defaults as defaults2, accepted_options as accepted_options2)

accepted_options = sorted(
    list(
        set(accepted_options +
            [item for item in accepted_options2 if item not in accepted_options])))

defaults = defaults.copy()
defaults.update(defaults2)

class Dr16FixedVarlssFudgeExpectedFlux(Dr16FixedVarlssExpectedFlux,
                                       Dr16FixedFudgeExpectedFlux):
    """Class to the expected flux similar to Dr16ExpectedFlux but fixing both
    var_lss and fudge

    Methods
    -------
    (see Dr16FixedVarlssExpectedFlux in
     py/picca/delta_extraction/expected_fluxes/dr16_fix_varlss_expected_flux.py)
    (see Dr16FixedFudgeExpectedFlux in
     py/picca/delta_extraction/expected_fluxes/dr16_fix_fudge_expected_flux.py)

    Attributes
    ----------
    (see Dr16FixedVarlssExpectedFlux in
     py/picca/delta_extraction/expected_fluxes/dr16_fix_varlss_expected_flux.py)
    (see Dr16FixedFudgeExpectedFlux in
     py/picca/delta_extraction/expected_fluxes/dr16_fix_fudge_expected_flux.py)
    """
