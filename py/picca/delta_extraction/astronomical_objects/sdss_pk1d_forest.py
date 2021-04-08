"""This module defines the abstract class SdssForest to represent
SDSS forests
"""
from picca.delta_extraction.astronomical_objects.pk1d_forest import Pk1dForest
from picca.delta_extraction.astronomical_objects.sdss_forest import SdssForest

class SdssPk1dForest(Pk1dForest, SdssForest):
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
    Line-of-sight id. Same as thingid

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

    fiberid: int (from SdssForest)
    Fiberid of the observation

    mjd: int (from SdssForest)
    Modified Julian Date of the observation

    plate: int (from SdssForest)
    Plate of the observation

    thingid: int (from SdssForest)
    Thingid of the object

    exposures_diff: array of floats (from Pk1dForest)
    Difference between exposures

    reso: array of floats or None (from Pk1dForest)
    Resolution of the forest
    """
    def __init__(self, **kwargs):
        """Initialize instance

        Arguments
        ---------
        **kwargs: dict
        Dictionary contiaing the information
        """
        super().__init__(**kwargs)

        # rebin arrays
        # this needs to happen after flux, ivar arrays are initialized by
        # Forest constructor
        super().rebin()

    def get_header(self):
        """Returns line-of-sight data to be saved as a fits file header

        Returns
        -------
        header : list of dict
        A list of dictionaries containing 'name', 'value' and 'comment' fields
        """
        return super().get_header()
