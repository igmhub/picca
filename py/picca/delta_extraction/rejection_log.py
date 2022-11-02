"""This module defines the class RejectionLog."""
import numpy as np
import fitsio

class RejectionLogTable:
    """Class to handle rejection logs for data in Table format
    
    Methods
    -------
    add_to_rejection_log
    save_rejection_log
    
    Attributes
    ----------
    file: str
    Filename of the rejection log

    cols: list of list
    Each list contains the data of each of the fields saved in the rejection log

    comments: list of list
    Description of each of the fields saved in the rejection log

    names: list of list
    Name of each of the fields saved in the rejection log
    """
    def __init__(self, header, file):
        """Initialize class instance
        
        Arguments
        ---------
        header: Fits header
        header to initialize rejection attributes

        file: str
        Filename of the rejection log

        """
        self.file = file
        
        self.cols = [[], []]
        self.names = ["FOREST_SIZE", "REJECTION_STATUS"]
        self.comments = [
            "num pixels in forest", "rejection status"
        ]

        for item in header:
            self.cols.append([])
            self.names.append(item.get("name"))
            self.comments.append(item.get("comment"))
    
    
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
                for item in header:   # pragma: no branch
                    if item.get("name") == name:
                        col.append(item.get("value"))
                        break

    def save_rejection_log(self):
        """Saves the rejection log arrays.
        In the log forest headers will be saved along with the forest size and
        the rejection status.
        """
        rejection_log = fitsio.FITS(self.file, 'rw', clobber=True)

        rejection_log.write(
            [np.array(item) for item in self.cols],
            names=self.names,
            comment=self.comments,
            extname="rejection_log")

        rejection_log.close()

class RejectionLogImage:
    """Class to handle rejection logs for data in Image format.
    
    Methods
    -------
    add_to_rejection_log
    save_rejection_log
    
    Attributes
    ----------
    file: str
    Filename of the rejection log

    data: list of list
    List containing the rejection information for each of the forests

    dtypes: list 
    List containing dtype information for each of the columns
    """
    def __init__(self, dtypes, file):
        """Initialize class instance
        
        Arguments
        ---------
        dtypes: list
        dtypes of each forest metadata
        
        file: str
        Filename of the rejection log
        """
        self.file = file
        self.dtypes = dtypes + [('FOREST_SIZE', int), ('REJECTION_STATUS', 'S12')]
        self.data = []

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
        size = forest.flux.size

        self.data.append(
            tuple(forest.get_metadata() + [size, rejection_status])
        )

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