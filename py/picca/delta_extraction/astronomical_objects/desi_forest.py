"""This module defines the class DesiForest to represent DESI forests"""
from picca.delta_extraction.astronomical_objects.forest import Forest
from picca.delta_extraction.errors import AstronomicalObjectError

class DesiForest(Forest):
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

    night: list of int
    Identifier of the night where the observation was made. None for no info

    petal: list of int
    Identifier of the spectrograph used in the observation. None for no info

    targetid: int
    Targetid of the object

    tile: list of int
    Identifier of the tile used in the observation. None for no info
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
        self.night = []
        if kwargs.get("night") is not None:
            self.night.append(kwargs.get("night"))
            del kwargs["night"]

        self.petal = []
        if kwargs.get("petal") is not None:
            self.petal.append(kwargs.get("petal"))
            del kwargs["petal"]

        self.targetid = kwargs.get("targetid")
        if self.targetid is None:
            raise AstronomicalObjectError("Error constructing DesiForest. "
                                          "Missing variable 'targetid'")
        del kwargs["targetid"]

        self.tile = []
        if kwargs.get("tile") is not None:
            self.tile.append(kwargs.get("tile"))
            del kwargs["tile"]

        # call parent constructor
        kwargs["los_id"] = self.targetid
        super().__init__(**kwargs)

    def coadd(self, other):
        """Coadd the information of another forest.

        Forests are coadded by calling the coadd function from Forest.
        DESI night, petal and night from other are added to the current list

        Arguments
        ---------
        other: DesiForest
        The forest instance to be coadded.

        Raise
        -----
        AstronomicalObjectError if other is not a DesiForest instance
        """
        if not isinstance(other, DesiForest):
            raise AstronomicalObjectError("Error coadding DesiForest. Expected "
                                          "DesiForest instance in other. Found: "
                                          f"{type(other).__name__}")
        self.night += other.night
        self.petal += other.petal
        self.tile += other.tile
        super().coadd(other)

    def get_header(self):
        """Return line-of-sight data to be saved as a fits file header

        Adds specific DESI keys to general header (defined in class Forest)

        Return
        ------
        header : list of dict
        A list of dictionaries containing 'name', 'value' and 'comment' fields
        """
        header = super().get_header()
        header += [
            {
                'name': 'TARGETID',
                'value': self.targetid,
                'comment': 'Object identification'
            },
            {
                'name': 'NIGHT',
                'value': "-".join(str(night) for night in self.night),
                'comment': "Observation night(s)"
            },
            {
                'name': 'PETAL',
                'value': "-".join(str(petal) for petal in self.petal),
                'comment': 'Observation petal(s)'
            },
            {
                'name': 'TILE',
                'value': "-".join(str(tile) for tile in self.tile),
                'comment': 'Observation tile(s)'
            },
        ]

        return header

    def get_metadata(self):
        """Return line-of-sight data as a list. Names and types of the variables
        are given by DesiForest.get_metadata_dtype. Units are given by
        DesiForest.get_metadata_units

        Return
        ------
        metadata: list
        A list containing the line-of-sight data
        """
        metadata = super().get_metadata()
        metadata += [
            self.targetid,
            "-".join(str(night) for night in self.night),
            "-".join(str(petal) for petal in self.petal),
            "-".join(str(tile) for tile in self.tile),
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
        dtype += [('TARGETID', int), ('NIGHT', 'S12'), ('PETAL', 'S12'), ('TILE', 'S12')]
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
        units += ["", "", "", ""]
        return units
