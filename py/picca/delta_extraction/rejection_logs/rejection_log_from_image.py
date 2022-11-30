"""This module defined the class RejectionLogImage to manage
rejection log when data is saved in ImageHDU format"""

import numpy as np
import fitsio

from picca.delta_extraction.rejection_log import RejectionLog


class RejectionLogFromImage(RejectionLog):
    """Class to handle rejection logs for data in Image format.

    Methods
    -------
    add_to_rejection_log
    initialize_rejection_log
    save_rejection_log

    Attributes
    ----------
    (see RejectionLog in py/picca/delta_extraction/rejection_log.py)

    data: list of list
    List containing the rejection information for each of the forests

    dtypes: list
    List containing dtype information for each of the columns

    """

    def __init__(self, file):
        """Initialize class instance

        Arguments
        ---------

        file: str
        Filename of the rejection log
        """
        super().__init__(file)

        self.dtypes = None
        self.data = []

    def initialize_rejection_log(self, forest):
        """Initialize rejection log

        Arguments
        ---------
        forest: Forest
        Forest to obtain metadata dtypes
        """
        self.dtypes = forest.get_metadata_dtype() + [('FOREST_SIZE', int),
                                                     ('REJECTION_STATUS', 'S12')
                                                    ]
        self.initialized = True

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
        if not self.initialized:
            self.initialize_rejection_log(forest)

        size = forest.flux.size

        self.data.append(tuple(forest.get_metadata() +
                               [size, rejection_status]))

    def save_rejection_log(self):
        """Saves the rejection log arrays.
        In the log forest metadata will be saved along with the forest size and
        rejection status.
        """
        rejection_log = fitsio.FITS(self.file, 'rw', clobber=True)

        rejection_log.write(
            np.array(
                self.data,
                dtype=self.dtypes,
            ),
            extname="rejection_log",
        )
