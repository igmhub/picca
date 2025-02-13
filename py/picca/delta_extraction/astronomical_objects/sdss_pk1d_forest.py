"""This module defines the class DesiPk1dForest to represent SDSS forests
in the Pk1D analysis
"""
from picca.delta_extraction.astronomical_objects.pk1d_forest import Pk1dForest
from picca.delta_extraction.astronomical_objects.sdss_forest import SdssForest
from picca.delta_extraction.astronomical_objects.forest import Forest

class SdssPk1dForest(SdssForest, Pk1dForest):
    """Forest Object

    Class Methods
    -------------
    (see SdssForest in py/picca/delta_extraction/astronomical_objects/sdss_forest.py)
    (see Pk1dForest in py/picca/delta_extraction/astronomical_objects/pk1d_forest.py)
    update_class_variables

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

    @classmethod
    def update_class_variables(cls):
        """Update class variable mask_fields (from Forest) to also contain the
        necessary fields for this class to work properly.
        """
        cls.class_variable_check()
        for field in ["exposures_diff", "reso", "reso_pix"]:
            if field not in Forest.mask_fields:
                cls.mask_fields.append(field)
