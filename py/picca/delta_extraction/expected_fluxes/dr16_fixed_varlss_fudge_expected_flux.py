"""This module defines the class Dr16FixedVarlssFudgeExpectedFlux"""
from picca.delta_extraction.expected_fluxes.dr16_fixed_varlss_expected_flux import (
    Dr16FixedVarlssExpectedFlux)
from picca.delta_extraction.expected_fluxes.dr16_fixed_varlss_expected_flux import (
    defaults, accepted_options)
from picca.delta_extraction.expected_fluxes.dr16_fixed_fudge_expected_flux import (
    Dr16FixedFudgeExpectedFlux)
from picca.delta_extraction.expected_fluxes.dr16_fixed_fudge_expected_flux import (
    defaults as defaults2, accepted_options as accepted_options2)
from picca.delta_extraction.utils import update_accepted_options, update_default_options

accepted_options = update_accepted_options(accepted_options, accepted_options2)
accepted_options = update_accepted_options(
    accepted_options,
    ["limit var lss", "use constant weight", "use ivar as weight"],
    remove=True)

defaults = update_default_options(defaults, defaults2)


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
