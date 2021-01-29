"""This module defines the class UserPrint to manage printing options
"""
import sys

class UserPrint:
    """Class to manage printing options

    Class Methods
    -------------
    initialize_log
    quietprint
    verboseprint

    Methods
    -------
    __call__

    Class Attributes
    ----------
    print_type: str
    String specifying the method to call for printing.
    Allowed values are "verboseprint", "quietprint".

    log_file: file or None
    Opened instance of a log file where information is printed. None for
    no logging
    """
    print_type = "verboseprint"
    log_file = None

    def __call__(self, *args, **kwargs):
        """Applies the mask. This function should be
        overloaded with the correct functionallity by any child
        of this class

        Arguments
        ---------
        data: Forest
        A Forest instance to which the correction is applied

        Raises
        ------
        MaskError if function was not overloaded by child class
        """
        if UserPrint.print_type == "verboseprint":
            UserPrint.verboseprint(*args, **kwargs)
        elif UserPrint.print_type == "quietprint":
            UserPrint.quietprint(*args, **kwargs)
        if UserPrint.log_file is not None:
            UserPrint.verboseprint(*args, **kwargs, file=UserPrint.log_file)

    @classmethod
    def initialize_log(cls, log_filename):
        """Open log file

        Arguments
        ---------
        log_filename: str
        Name of the file where the log will be printed

        Raises
        ------
        IOError if file could not be opened
        """
        cls.log_file = open(log_filename, "w")

    @classmethod
    def quietprint(cls, *args, **kwargs):
        """ Don't print anything """

    @classmethod
    def reset_log(cls):
        """Closes log file and resets cls.log_file to None"""
        if cls.log_file is not None:
            cls.log_file.close()
            cls.log_file = None

    @classmethod
    def verboseprint(cls, *args, **kwargs):
        """Function to use user-specified prints.
        Default is to use an extension of the normal print function,
        but this function can be overloaded based on the options passed in
        the configuration file

        Arguments
        ---------
        args: arguments passed to print

        kwargs: keyword arguments passed to print
        """
        print(*args, **kwargs)
        sys.stdout.flush()


userprint = UserPrint()
