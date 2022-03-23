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
    (see DesiForest in py/picca/delta_extraction/astronomical_objects/desi_forest.py)
    (see Pk1dForest in py/picca/delta_extraction/astronomical_objects/pk1d_forest.py)

    Attributes
    ----------
    (see DesiForest in py/picca/delta_extraction/astronomical_objects/desi_forest.py)
    (see Pk1dForest in py/picca/delta_extraction/astronomical_objects/pk1d_forest.py)
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
