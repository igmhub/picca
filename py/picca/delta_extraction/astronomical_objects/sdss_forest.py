"""This module defines the class SdssForest to represent SDSS forests"""
from picca.delta_extraction.astronomical_objects.forest import Forest
from picca.delta_extraction.errors import AstronomicalObjectError

class SdssForest(Forest):
    """Forest Object

    Class Methods
    -------------
    (see Forest in py/picca/delta_extraction/astronomical_objects/forest.py)
    get_metadata_dtype
    get_metadata_units

    Methods
    -------
    (see Forest in py/picca/delta_extraction/astronomical_objects/forest.py)
    __init__
    coadd
    get_header
    get_metadata

    Class Attributes
    ----------------
    (see Forest in py/picca/delta_extraction/astronomical_objects/forest.py)

    Attributes
    ----------
    (see Forest in py/picca/delta_extraction/astronomical_objects/forest.py)

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
                                          f"{type(other).__name__}")

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

    def get_metadata(self):
        """Return line-of-sight data as a list. Names and types of the variables
        are given by SdssForest.get_metadata_dtype. Units are given by
        SdssForest.get_metadata_units

        Return
        ------
        metadata: list
        A list containing the line-of-sight data
        """
        metadata = super().get_metadata()
        metadata += [
            self.thingid,
            "-".join(f"{plate:04d}" for plate in self.plate),
            "-".join(f"{mjd:05d}" for mjd in self.mjd),
            "-".join(f"{fiberid:04d}" for fiberid in self.fiberid),
        ]
        return metadata

    @classmethod
    def get_metadata_dtype(cls):
        """Return the types and names of the line-of-sight data returned by
        method self.get_metadata

        Return
        ------
        metadata_dtype: list
        A list with tuples containing the name and data type of the line-of-sight
        data
        """
        dtype = super().get_metadata_dtype()
        dtype += [('THING_ID', int), ('PLATE', 'S12'), ('MJD', 'S12'), ('FIBERID', 'S12')]
        return dtype

    @classmethod
    def get_metadata_units(cls):
        """Return the units of the line-of-sight data returned by
        method self.get_metadata

        Return
        ------
        metadata_units: list
        A list with the units of the line-of-sight data
        """
        units = super().get_metadata_units()
        units += ["", "", "DAYS", ""]
        return units
