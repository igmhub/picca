"""This module defines the class DesiPk1dForest to represent SDSS forests
in the Pk1D analysis
"""
from picca.delta_extraction.astronomical_objects.pk1d_forest import Pk1dForest
from picca.delta_extraction.astronomical_objects.sdss_forest import SdssForest

class SdssPk1dForest(SdssForest, Pk1dForest):
    """Forest Object

    Class Methods
    -------------
    (see SdssForest in py/picca/delta_extraction/astronomical_objects/sdss_forest.py)
    (see Pk1dForest in py/picca/delta_extraction/astronomical_objects/pk1d_forest.py)

    Methods
    -------
    (see SdssForest in py/picca/delta_extraction/astronomical_objects/sdss_forest.py)
    (see Pk1dForest in py/picca/delta_extraction/astronomical_objects/pk1d_forest.py)
    __init__

    Class Attributes
    ----------------
    (see SdssForest in py/picca/delta_extraction/astronomical_objects/sdss_forest.py)
    (see Pk1dForest in py/picca/delta_extraction/astronomical_objects/pk1d_forest.py)

    Attributes
    ----------
    (see SdssForest in py/picca/delta_extraction/astronomical_objects/sdss_forest.py)
    (see Pk1dForest in py/picca/delta_extraction/astronomical_objects/pk1d_forest.py)
    """
