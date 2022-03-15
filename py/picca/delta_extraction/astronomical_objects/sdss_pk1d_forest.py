"""This module defines the class DesiPk1dForest to represent SDSS forests
in the Pk1D analysis
"""
from picca.delta_extraction.astronomical_objects.pk1d_forest import Pk1dForest
from picca.delta_extraction.astronomical_objects.sdss_forest import SdssForest

class SdssPk1dForest(SdssForest, Pk1dForest):
    """Forest Object

    Methods
    -------
    __gt__ (from AstronomicalObject)
    __eq__ (from AstronomicalObject)
    class_variable_check (from Forest, Pk1dForest)
    consistency_check (from Forest, Pk1dForest)
    get_data (from Forest, Pk1dForest)
    rebin (from Forest)
    coadd (from SdssForest, Pk1dForest)
    get_header (from SdssForest, Pk1dForest)
    __init__

    Class Attributes
    ----------------
    blinding: str (from Forest)
    Name of the blinding strategy used

    log_lambda_grid: array of float or None (from Forest)
    Common grid in log_lambda based on the specified minimum and maximum
    wavelengths, and delta_log_lambda.

    log_lambda_rest_frame_grid: array of float or None (from Forest)
    Same as log_lambda_grid but for rest-frame wavelengths.

    log_lambda_rest_frame_grid: array of float (from Forest)
    Same as log_lambda_grid but for rest-frame wavelengths

    mask_fields: list of str (from Forest)
    Names of the fields that are affected by masking. In general it will
    be "flux" and "ivar" but some child classes might add more.

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

    bad_continuum_reason: str or None
    Reason as to why the continuum fit is not acceptable. None for acceptable
    contiuum.

    continuum: array of float or None (from Forest)
    Quasar continuum. None for no information

    deltas: array of float or None (from Forest)
    Flux-transmission field (delta field). None for no information

    flux: array of float (from Forest)
    Flux

    ivar: array of float (from Forest)
    Inverse variance

    log_lambda: array of float or None (from Forest)
    Logarithm of the wavelength (in Angstroms)

    mean_snr: float (from Forest)
    Mean signal-to-noise of the forest

    transmission_correction: array of float (from Forest)
    Transmission correction.

    weights: array of float or None (from Forest)
    Weights associated to the delta field. None for no information

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

    mean_z: float
    Mean redshift of the forest (from Pk1dForest)

    reso: array of floats or None (from Pk1dForest)
    Resolution of the forest
    """
    def __init__(self, **kwargs):
        """Initialize instance

        Arguments
        ---------
        **kwargs: dict
        Dictionary contiaing the information

        Raise
        -----
        AstronomicalObjectError if there are missing variables
        """
        super().__init__(**kwargs)
