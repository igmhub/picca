"""This module defines the abstract class Mask from which all
Masks must inherit
"""
from picca.delta_extraction.errors import MaskError

class Mask:
    """Abstract class from which all Masks must inherit.
    Classes that inherit from this should be initialized using
    a configparser.SectionProxy instance.

    Methods
    -------
    apply_mask
    """
    # pylint: disable=no-self-use
    # this method should use self in child classes
    def apply_mask(self, forest):
        """Applies the mask. This function should be
        overloaded with the correct functionallity by any child
        of this class

        Arguments
        ---------
        forest: Forest
        A Forest instance to which the correction is applied

        Raises
        ------
        MaskError if function was not overloaded by child class
        """
        raise MaskError("Function 'apply_mask' was not overloaded by child class")
