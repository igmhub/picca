"""This module defines the class DesiPk1dForest to represent DESI forests
in the Pk1D analysis
"""
from picca.delta_extraction.astronomical_objects.desi_forest import DesiForest
from picca.delta_extraction.astronomical_objects.pk1d_forest import Pk1dForest


class DesiPk1dForest(DesiForest, Pk1dForest):
    """Forest Object

    Methods
    -------
    __gt__ (from AstronomicalObject)
    __eq__ (from AstronomicalObject)
    class_variable_check (from Forest, Pk1dForest)
    consistency_check (from Forest, Pk1dForest)
    get_data (from Forest, Pk1dForest)
    rebin (from Forest)
    coadd (from DesiForest, Pk1dForest)
    get_header (from DesiForest, Pk1dForest)
    __init__


    Class Attributes
    ----------------
    log_lambda_grid: array of float (from Forest)
    Common grid in log_lambda based on the specified minimum and maximum
    wavelengths, the step size and the wavelength solution (lin or log).

    log_lambda_rest_frame_grid: array of float (from Forest)
    Same as log_lambda_grid but for rest-frame wavelengths

    mask_fields: list of str (from Forest)
    Names of the fields that are affected by masking. In general it will
    be "flux" and "ivar" but some child classes might add more.

    wave_solution: "lin" or "log" (from Forest)
    Determines whether the wavelength solution has linear spacing ("lin") or
    logarithmic spacing ("log").

    lambda_abs_igm: float (from Pk1dForest)
    Wavelength of the IGM absorber

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

    night: list of int (from DesiForest)
    Identifier of the night where the observation was made. None for no info

    petal: list of int (from DesiForest)
    Identifier of the spectrograph used in the observation. None for no info

    targetid: int (from DesiForest)
    Targetid of the object

    tile: list of int (from DesiForest)
    Identifier of the tile used in the observation. None for no info

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
