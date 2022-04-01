"""This module defines the class DesiData to load DESI data
"""
import multiprocessing
import logging

import fitsio
import healpy
import numpy as np

from picca.delta_extraction.data_catalogues.desi_healpix import DesiHealpix, defaults, accepted_options
from picca.delta_extraction.errors import DataError

class DesisimMocks(DesiHealpix):
    """Reads the spectra from DESI using healpix mode and formats its data as a
    list of Forest instances.

    Should work for both data and mocks. This is specified using the 'mode'
    keyword. It is required to set the in_nside member.

    Methods
    -------
    filter_forests (from Data)
    set_blinding (from Data)
    read_file (from DesiHealpix)
    __init__
    read_data

    Attributes
    ----------
    analysis_type: str (from Data)
    Selected analysis type. Current options are "BAO 3D" or "PK 1D"

    forests: list of Forest (from Data)
    A list of Forest from which to compute the deltas.

    min_num_pix: int (from Data)
    Minimum number of pixels in a forest. Forests with less pixels will be dropped.

    blinding: str (from DesiData)
    A string specifying the chosen blinding strategies. Must be one of the
    accepted values in ACCEPTED_BLINDING_STRATEGIES

    catalogue: astropy.table.Table (from DesiData)
    The quasar catalogue

    input_directory: str (from DesiData)
    Directory to spectra files.

    logger: logging.Logger
    Logger object
    """
    def __init__(self, config):
        """Initialize class instance

        Arguments
        ---------
        config: configparser.SectionProxy
        Parsed options to initialize class
        """

        self.logger = logging.getLogger(__name__)
        super().__init__(config)

    def read_data(self):
        """Read the spectra and formats its data as Forest instances.

        Method used to read healpix-based survey data.

        Return
        ------
        is_mock: bool
        True as we are loading mocks

        is_sv: bool
        False as mocks data is not part of DESI SV data

        Raise
        -----
        DataError if no quasars were found
        """
        in_nside = 16

        healpix = [
            healpy.ang2pix(in_nside, np.pi / 2 - row["DEC"], row["RA"], nest=True)
            for row in self.catalogue
        ]
        self.catalogue["HEALPIX"] = healpix
        self.catalogue.sort("HEALPIX")

        #Current mocks don't have this "SURVEY" column in the catalog
        #but its not clear future ones will not have it, so I think is good to leave it for now.
        if not "SURVEY" in self.catalogue.colnames:
             self.catalogue["SURVEY"]=np.ma.masked

        grouped_catalogue = self.catalogue.group_by(["HEALPIX", "SURVEY"])
        arguments=[]
       
        context = multiprocessing.get_context('fork')
        pool = context.Pool(processes=num_processors)
        manager =  multiprocessing.Manager()
        forests_by_targetid = manager.dict()

        for (index,
             (healpix, survey)), group in zip(enumerate(grouped_catalogue.groups.keys),
                                    grouped_catalogue.groups):

            filename = (
                f"{self.input_directory}/{healpix//100}/{healpix}/spectra-"
                f"{in_nside}-{healpix}.fits")
            arguments.append((filename,group,forests_by_targetid))

        self.logger.info(f"reading data from {len(arguments)} files")
        if num_processors>1:
            pool.starmap(self.read_file,arguments)
        else:

        pool.close()
        if len(forests_by_targetid) == 0:
            raise DataError("No Quasars found, stopping here")
        self.forests = list(forests_by_targetid.values())

        return True, False
