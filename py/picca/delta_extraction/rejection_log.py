"""This module defines the class RejectionLog."""
from picca.delta_extraction.errors import RejectionLogError


class RejectionLog:
    """Class to handle rejection logs.

    Methods
    -------
    add_to_rejection_log
    initialize_rejection_log
    save_rejection_log

    Attributes
    ----------
    file: str
    Filename of the rejection log

    initialized: bool
    Boolean determining whether the log is fully initialized or not
    """

    def __init__(self, file):
        """Initialize class instance

        Arguments
        ---------
        file:str
        Filename of the rejection log
        """
        self.file = file
        self.initialized = False

    def initialize_rejection_log(self, forest):
        """Initialize rejection log

        Arguments
        ---------
        forest: Forest
        Forest to obtain metadata dtypes
        """
        raise RejectionLogError(
            "Function 'initialize_rejection_log' was not overloaded by child class"
        )

    def add_to_rejection_log(self, forest, rejection_status):
        """Adds to the rejection log arrays.
        In the log forest metadata will be saved along with the forest size and
        rejection status.

        Arguments
        ---------
        forest: Forest
        Forest to include in the rejection log

        rejection_status: str
        Rejection status
        """
        raise RejectionLogError(
            "Function 'add_to_rejection_log' was not overloaded by child class")

    def save_rejection_log(self):
        """Saves the rejection log arrays.
        In the log forest metadata will be saved along with the forest size and
        rejection status.
        """
        raise RejectionLogError(
            "Function 'save_rejection_log' was not overloaded by child class")
