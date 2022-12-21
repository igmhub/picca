"""This module defines the abstract class Correction from which all
Corrections must inherit
"""
from picca.delta_extraction.errors import CorrectionError

class Correction:
    """Abstract class from which all Corrections must inherit.
    Classes that inherit from this should be initialized using
    a configparser.SectionProxy instance.

    Methods
    -------
    apply_correction
    """
    def apply_correction(self, forest):
        """Applies the correction. This function should be
        overloaded with the correct functionallity by any child
        of this class

        Arguments
        ---------
        forest: Forest
        A Forest instance to which the correction is applied

        Raise
        -----
        CorrectionError if function was not overloaded by child class
        """
        raise CorrectionError("Function 'apply_correction' was not overloaded "
                              "by child class")
