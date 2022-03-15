"""This module defines the class SdssForest to represent SDSS forests"""
from picca.delta_extraction.astronomical_objects.forest import Forest
from picca.delta_extraction.errors import AstronomicalObjectError

class SdssForest(Forest):
    """Forest Object

    Methods
    -------
    __gt__ (from AstronomicalObject)
    __eq__ (from AstronomicalObject)
    class_variable_check (from Forest)
    consistency_check (from Forest)
    get_data (from Forest)
    rebin (from Forest)
    __init__
    coadd
    get_header

    Class Attributes
    ----------------
    blinding: str (from Forest)
    Name of the blinding strategy used

    log_lambda_grid: array of float or None (from Forest)
    Common grid in log_lambda based on the specified minimum and maximum
    wavelengths, and delta_log_lambda.

    log_lambda_rest_frame_grid: array of float or None (from Forest)
    Same as log_lambda_grid but for rest-frame wavelengths.

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

    fiberid: list of int
    Fiberid of the observation

    mjd: list of int
    Modified Julian Date of the observation

    plate: list of int
    Plate of the observation

    thingid: int
    Thingid of the object
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
        if kwargs.get("fiberid") is None:
            raise AstronomicalObjectError("Error constructing SdssForest. "
                                          "Missing variable 'fiberid'")
        self.fiberid = [kwargs.get("fiberid")]
        del kwargs["fiberid"]

        if kwargs.get("mjd") is None:
            raise AstronomicalObjectError("Error constructing SdssForest. "
                                          "Missing variable 'mjd'")
        self.mjd = [kwargs.get("mjd")]
        del kwargs["mjd"]

        if kwargs.get("plate") is None:
            raise AstronomicalObjectError("Error constructing SdssForest. "
                                          "Missing variable 'plate'")
        self.plate = [kwargs.get("plate")]
        del kwargs["plate"]

        self.thingid = kwargs.get("thingid")
        if self.thingid is None:
            raise AstronomicalObjectError("Error constructing SdssForest. "
                                          "Missing variable 'thingid'")
        del kwargs["thingid"]

        # call parent constructor
        kwargs["los_id"] = self.thingid
        super().__init__(**kwargs)

    def coadd(self, other):
        """Coadd the information of another forest.

        Forests are coadded by calling the coadd function from Forest.
        SDSS fiberid, mjd and plate from other are added to the current list

        Arguments
        ---------
        other: Forest
        The forest instance to be coadded.

        Raise
        -----
        AstronomicalObjectError if other is not a DesiForest instance
        """
        if not isinstance(other, SdssForest):
            raise AstronomicalObjectError("Error coadding SdssForest. Expected "
                                          "SdssForest instance in other. Found: "
                                          f"{type(other)}")

        self.fiberid += other.fiberid
        self.mjd += other.mjd
        self.plate += other.plate
        super().coadd(other)

    def get_header(self):
        """Return line-of-sight data to be saved as a fits file header

        Adds to specific SDSS keys to general header (defined in class Forsest)

        Return
        ------
        header : list of dict
        A list of dictionaries containing 'name', 'value' and 'comment' fields
        """
        header = super().get_header()
        header += [
            {
                'name': 'THING_ID',
                'value': self.thingid,
                'comment': 'Object identification'
            },
            {
                'name': 'PLATE',
                'value': "-".join(f"{plate:04d}" for plate in self.plate),
                'comment': 'SDSS plate(s)',
            },
            {
                'name': 'MJD',
                'value': "-".join(f"{mjd:05d}" for mjd in self.mjd),
                'comment': 'Modified Julian date'
            },
            {
                'name': 'FIBERID',
                'value': "-".join(f"{fiberid:04d}" for fiberid in self.fiberid),
                'comment': 'SDSS fiber id(s)',
            },
        ]

        return header
