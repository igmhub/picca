"""This module defines the class DesiData to load DESI data
"""
import multiprocessing
import logging

import fitsio
import healpy
import numpy as np

from picca.delta_extraction.data_catalogues.desi_healpix import DesiHealpix, ParallelReader, defaults, accepted_options
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
        if self.use_non_coadded_spectra:
            self.logger.warning('the "use_non_coadded_spectra" option was set, '
                                'but has no effect on Mocks, will proceed as normal')

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
        
        forests_by_targetid = {}

        arguments = []
        for group in grouped_catalogue.groups:
            # healpix, survey = group["HEALPIX", "SURVEY"][0]#
            healpix = group["HEALPIX"][0]

            filename = (f"{self.input_directory}/{healpix//100}/{healpix}/spectra-"
                    f"{in_nside}-{healpix}.fits")

            arguments.append((filename,group))

        self.logger.info(f"reading data from {len(arguments)} files")

        if self.num_processors>1:
            context = multiprocessing.get_context('fork')
            with context.Pool(processes=self.num_processors) as pool:
                imap_it = pool.imap_unordered(ParallelReader(self.analysis_type, self.use_non_coadded_spectra, self.logger), arguments)
                for forests_by_pe in imap_it:
                    # Merge each dict to master forests_by_targetid
                    ParallelReader.merge_new_forest(forests_by_pe, forests_by_targetid)
        else:
            reader = ParallelReader(self.analysis_type, self.use_non_coadded_spectra, self.logger)
            for index, this_arg in enumerate(arguments):
                self.logger.progress(
                    f"Read {index} of {len(arguments)}. "
                    f"num_data: {len(forests_by_targetid)}"
                    )
                ParallelReader.merge_new_forest(reader(this_arg), forests_by_targetid)

        if len(forests_by_targetid) == 0:
            raise DataError("No quasars found, stopping here")
        self.forests = list(forests_by_targetid.values())

        return True, False
