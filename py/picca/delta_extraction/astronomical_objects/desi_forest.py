"""This module defines the abstract class SdssForest to represent
SDSS forests
"""
import numpy as np

from picca.delta_extraction.errors import AstronomicalObjectError

from picca.delta_extraction.astronomical_objects.forest import Forest


class DesiForest(Forest):
    """Forest Object

    Methods
    -------
    __gt__ (from AstronomicalObject)
    __eq__ (from AstronomicalObject)
    rebin (from Forest)
    __init__
    coadd

    Class Attributes
    ----------------
    delta_lambda: float or None (from Forest)
    Variation of the wavelength (in Angs) between two pixels. This should not
    be None if wave_solution is "lin". Ignored if wave_solution is "log".

    delta_log_lambda: float or None (from Forest)
    Variation of the logarithm of the wavelength (in Angs) between two pixels.
    This should not be None if wave_solution is "log". Ignored if wave_solution
    is "lin".

    lambda_max: float or None (from Forest)
    Maximum wavelength (in Angs) to be considered in a forest. This should not
    be None if wave_solution is "lin". Ignored if wave_solution is "log".

    lambda_min: float or None (from Forest)
    Minimum wavelength (in Angs) to be considered in a forest. This should not
    be None if wave_solution is "lin". Ignored if wave_solution is "log".

    lambda_max_rest_frame: float or None (from Forest)
    As wavelength_max but for rest-frame wavelength. This should not
    be None if wave_solution is "lin". Ignored if wave_solution is "log".

    lambda_min_rest_frame: float or None (from Forest)
    As wavelength_min but for rest-frame wavelength. This should not
    be None if wave_solution is "lin". Ignored if wave_solution is "log".

    log_lambda_max: float or None (from Forest)
    Logarithm of the maximum wavelength (in Angs) to be considered in a forest.
    This should not be None if wave_solution is "log". Ignored if wave_solution
    is "lin".

    log_lambda_min: float or None (from Forest)
    Logarithm of the minimum wavelength (in Angs) to be considered in a forest.
    This should not be None if wave_solution is "log". Ignored if wave_solution
    is "lin".

    log_lambda_max_rest_frame: float or None (from Forest)
    As log_lambda_max but for rest-frame wavelength. This should not be None if
    wave_solution is "log". Ignored if wave_solution is "lin".

    log_lambda_min_rest_frame: float or None (from Forest)
    As log_lambda_min but for rest-frame wavelength. This should not be None if
    wave_solution is "log". Ignored if wave_solution is "lin".

    wave_solution: "lin" or "log" (from Forest)
    Determines whether the wavelength solution has linear spacing ("lin") or
    logarithmic spacing ("log").

    Attributes
    ----------
    dec: float (from AstronomicalObject)
    Declination (in rad)

    healpix: int (from AstronomicalObject)
    Healpix number associated with (ra, dec)

    los_id: longint (from AstronomicalObject)
    Line-of-sight id. Same as targetid

    ra: float (from AstronomicalObject)
    Right ascention (in rad)

    z: float (from AstronomicalObject)
    Redshift

    continuum: array of float or None (from Forest)
    Quasar continuum. None for no information

    deltas: array of float or None (from Forest)
    Flux-transmission field (delta field). None for no information

    flux: array of float (from Forest)
    Flux

    ivar: array of float (from Forest)
    Inverse variance

    lambda_: array of float or None (from Forest)
    Wavelength (in Angstroms)

    log_lambda: array of float or None (from Forest)
    Logarithm of the wavelength (in Angstroms)

    mask_fields: list of str (from Forest)
    Names of the fields that are affected by masking. In general it will
    be "flux" and "ivar" but some child classes might add more.

    mean_snf: float (from Forest)
    Mean signal-to-noise of the forest

    night: int or None
    Identifier of the night where the observation was made. None for no info

    petal: int or None
    Identifier of the spectrograph used in the observation. None for no info

    targetid: int
    Targetid of the object

    tile: int or None
    Identifier of the tile used in the observation. None for no info
    """
    def __init__(self, **kwargs):
        """Initialize instance

        Arguments
        ---------
        **kwargs: dict
        Dictionary contiaing the information
        """
        self.night = kwargs.get("night")
        if self.night is not None:
            del kwargs["night"]

        self.petal = kwargs.get("petal")
        if self.petal is not None:
            del kwargs["petal"]

        self.targetid = kwargs.get("targetid")
        if self.targetid is None:
            raise AstronomicalObjectError("Error constructing DesiForest. "
                                          "Missing variable 'targetid'")
        del kwargs["targetid"]

        self.tile = kwargs.get("tile")
        if self.tile is not None:
            del kwargs["tile"]

        # call parent constructor
        kwargs["los_id"] = self.targetid
        super().__init__(**kwargs)

        # rebin arrays
        super().rebin()
