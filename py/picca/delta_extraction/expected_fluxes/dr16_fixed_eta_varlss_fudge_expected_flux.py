"""This module defines the class Dr16FixedEtaVarlssFudgeExpectedFlux"""
from picca.delta_extraction.expected_fluxes.dr16_fixed_eta_expected_flux import (
    Dr16FixedEtaExpectedFlux)
from picca.delta_extraction.expected_fluxes.dr16_fixed_eta_expected_flux import (
    defaults, accepted_options)
from picca.delta_extraction.expected_fluxes.dr16_fixed_fudge_expected_flux import (
    Dr16FixedFudgeExpectedFlux)
from picca.delta_extraction.expected_fluxes.dr16_fixed_fudge_expected_flux import (
    defaults as defaults2, accepted_options as accepted_options2)
from picca.delta_extraction.expected_fluxes.dr16_fixed_varlss_expected_flux import (
    Dr16FixedVarlssExpectedFlux)
from picca.delta_extraction.expected_fluxes.dr16_fixed_varlss_expected_flux import (
    defaults as defaults3, accepted_options as accepted_options3)

accepted_options = sorted(
    list(
        set(accepted_options +
            [item for item in accepted_options2 if item not in accepted_options])))
accepted_options = sorted(
    list(
        set(accepted_options +
            [item for item in accepted_options3 if item not in accepted_options])))

defaults = defaults.copy()
defaults.update(defaults2)
defaults.update(defaults3)


class Dr16FixedEtaVarlssFudgeExpectedFlux(Dr16FixedEtaExpectedFlux,
                                          Dr16FixedVarlssExpectedFlux,
                                          Dr16FixedFudgeExpectedFlux):
    """Class to the expected flux similar to Dr16ExpectedFlux but fixing
    eta, var_lss and fudge

    Methods
    -------
    (see Dr16FixedEtaExpectedFlux in
     py/picca/delta_extraction/expected_fluxes/dr16_fix_eta_expected_flux.py)
    (see Dr16FixedVarlssExpectedFlux in
     py/picca/delta_extraction/expected_fluxes/dr16_fix_varlss_expected_flux.py)
    (see Dr16FixedFudgeExpectedFlux in
     py/picca/delta_extraction/expected_fluxes/dr16_fix_fudge_expected_flux.py)

    Attributes
    ----------
    (see Dr16FixedEtaExpectedFlux in
     py/picca/delta_extraction/expected_fluxes/dr16_fix_eta_expected_flux.py)
    (see Dr16FixedVarlssExpectedFlux in
     py/picca/delta_extraction/expected_fluxes/dr16_fix_varlss_expected_flux.py)
    (see Dr16FixedFudgeExpectedFlux in
     py/picca/delta_extraction/expected_fluxes/dr16_fix_fudge_expected_flux.py)
    """
