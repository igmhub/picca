"""This module defined the class RejectionLogTable to manage
rejection log when data is saved in BinTableHDU format"""
import numpy as np
import fitsio

from picca.delta_extraction.rejection_log import RejectionLog


class RejectionLogFromTable(RejectionLog):
    """Class to handle rejection logs for data in Table format

    Methods
    -------
    add_to_rejection_log
    initialize_rejection_log
    save_rejection_log

    Attributes
    ----------
    (see RejectionLog in py/picca/delta_extraction/rejection_log.py)

    cols: list of list
    Each list contains the data of each of the fields saved in the rejection log

    comments: list of list
    Description of each of the fields saved in the rejection log

    names: list of list
    Name of each of the fields saved in the rejection log
    """

    def __init__(self, file):
        """Initialize class instance

        Arguments
        ---------
        file: str
        Filename of the rejection log

        """
        super().__init__(file)

        self.cols = None
        self.names = None
        self.comments = None

    def initialize_rejection_log(self, forest):
        """Intiialize rejection log

        Arguments
        ---------
        forest: Forest
        forest to obtain header information
        """
        self.cols = [[], []]
        self.names = ["FOREST_SIZE", "REJECTION_STATUS"]
        self.comments = ["num pixels in forest", "rejection status"]

        for item in forest.get_header():
            self.cols.append([])
            self.names.append(item.get("name"))
            self.comments.append(item.get("comment"))

        self.initialized = True

    def add_to_rejection_log(self, forest, rejection_status):
        """Adds to the rejection log arrays.
        In the log forest headers will be saved along with the forest size and
        the rejection status.

        Arguments
        ---------
        forest: Forest
        Forest to include in the rejection log

        rejection_status: str
        Rejection status
        """
        if not self.initialized:
            self.initialize_rejection_log(forest)

        header = forest.get_header()
        size = forest.flux.size

        for col, name in zip(self.cols, self.names):
            if name == "FOREST_SIZE":
                col.append(size)
            elif name == "REJECTION_STATUS":
                col.append(rejection_status)
            else:
                # this loop will always end with the break
                # the break is introduced to avoid useless checks
                for item in header:  # pragma: no branch
                    if item.get("name") == name:
                        col.append(item.get("value"))
                        break

    def save_rejection_log(self):
        """Saves the rejection log arrays.
        In the log forest headers will be saved along with the forest size and
        the rejection status.
        """
        rejection_log = fitsio.FITS(self.file, 'rw', clobber=True)

        rejection_log.write([np.array(item) for item in self.cols],
                            names=self.names,
                            comment=self.comments,
                            extname="rejection_log")

        rejection_log.close()
