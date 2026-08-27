"""This module defines the class FourmostPk1dForest to represent 4MOST
forests in the Pk1D analysis
"""
from picca.delta_extraction.astronomical_objects.forest import Forest
from picca.delta_extraction.astronomical_objects.pk1d_forest import Pk1dForest


class FourmostPk1dForest(Pk1dForest):
    """Forest Object

    4MOST forests carry no per-object resolution information: the delivery has
    no resolution matrix and no wdisp-like column. The resolution arrays are
    instead synthesised from the published LRS R(lambda) curves, which depend
    only on wavelength, so every forest at a given wavelength shares the same
    resolution. See FOURMOST_ARM_RESOLUTION in
    py/picca/delta_extraction/data_catalogues/fourmost_data.py.

    The delivery also has no independent sub-exposures -- the repeat rows of
    an object are nested cumulative stacks of the same photons -- so
    exposures_diff is filled with zeros. Downstream, only
    'picca_Pk1D.py --noise-estimate pipeline' (or 'mean_pipeline') is
    meaningful; the diff-based estimators would read those zeros as a
    noise-free spectrum, and picca_Pk1D defaults to 'mean_diff'.

    Class Methods
    -------------
    (see Pk1dForest in py/picca/delta_extraction/astronomical_objects/pk1d_forest.py)
    update_class_variables

    Methods
    -------
    (see Pk1dForest in py/picca/delta_extraction/astronomical_objects/pk1d_forest.py)

    Class Attributes
    ----------------
    (see Pk1dForest in py/picca/delta_extraction/astronomical_objects/pk1d_forest.py)

    Attributes
    ----------
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
                Forest.mask_fields.append(field)
