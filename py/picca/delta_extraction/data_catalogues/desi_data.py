"""This module defines the class DesiData to load DESI data
"""
import logging
import time
import numpy as np

from picca.delta_extraction.astronomical_objects.desi_forest import DesiForest
from picca.delta_extraction.astronomical_objects.desi_pk1d_forest import DesiPk1dForest
from picca.delta_extraction.astronomical_objects.forest import Forest
from picca.delta_extraction.data import Data, defaults, accepted_options
from picca.delta_extraction.errors import DataError
from picca.delta_extraction.quasar_catalogues.desi_quasar_catalogue import DesiQuasarCatalogue
from picca.delta_extraction.quasar_catalogues.desi_quasar_catalogue import (
    accepted_options as accepted_options_quasar_catalogue)
from picca.delta_extraction.quasar_catalogues.desi_quasar_catalogue import (
    defaults as defaults_quasar_catalogue)
from picca.delta_extraction.utils import (
    ACCEPTED_BLINDING_STRATEGIES)
from picca.delta_extraction.utils_pk1d import spectral_resolution_desi, exp_diff_desi
from picca.delta_extraction.utils import (
    ABSORBER_IGM, update_accepted_options, update_default_options)

accepted_options = update_accepted_options(accepted_options, accepted_options_quasar_catalogue)
accepted_options = update_accepted_options(accepted_options,
    ["use non-coadded spectra",
     "uniquify night targetid",
     "keep single exposures", 
     "wave solution"])

defaults = update_default_options(defaults, {
    "delta lambda": 0.8,
    "delta log lambda": 3e-4,
    "use non-coadded spectra": False,
    "uniquify night targetid": False,
    "keep single exposures": False,
    "wave solution": "lin",
})
defaults = update_default_options(defaults, defaults_quasar_catalogue)

def verify_exposures_shape(forests_by_targetid):
    """Verify that the exposures have the same shape. 
    If not, it removes them from the dictionnary of forests by targetid.
    Only works for use_non_coadded_spectra and keep_single_exposures options.

    Arguments
    ---------
    forests_by_targetid: dict
    Dictionary were forests are stored. Its content is modified by this
    function.

    """
    full_los_ids = np.array(list(forests_by_targetid.keys()))
    if isinstance(full_los_ids[0],int):
        raise ValueError(
            "The key of this dictionnary should be str"
            "in order to verify if exposures have "
            "different shapes"
        )
    los_ids = [int(key.split("_")[1]) for key in full_los_ids]
    unique_los_ids = np.unique(los_ids)
    full_los_ids_to_remove = []
    for los_id in unique_los_ids:
        select_forest = full_los_ids[los_ids == los_id]
        size_forests = np.unique(
            [forests_by_targetid[key].flux.size for key in select_forest]
        )
        if len(size_forests) > 1:
            full_los_ids_to_remove.append(select_forest)

    if len(full_los_ids_to_remove) != 0:
        full_los_ids_to_remove = np.concatenate(full_los_ids_to_remove, axis=0)
        for los_id in full_los_ids_to_remove:
            forests_by_targetid.pop(los_id)
    return forests_by_targetid


def merge_new_forest(forests_by_targetid, new_forests_by_targetid):
    """Merge new_forests_by_targetid and forests_by_targetid as Forest instances.

    Arguments
    ---------
    forests_by_targetid: dict
    Dictionary were forests are stored. Its content is modified by this
    function with the new forests.

    new_forests_by_targetid: dict
    Dictionary were new forests are stored. Its content will be merged with
    forests_by_targetid
    """
    parent_targetids = set(forests_by_targetid.keys())
    existing_targetids = parent_targetids.intersection(
        new_forests_by_targetid.keys())
    new_targetids = new_forests_by_targetid.keys() - existing_targetids

    # Does not fail if existing_targetids is empty
    for tid in existing_targetids:
        forests_by_targetid[tid].coadd(new_forests_by_targetid[tid])
    for tid in new_targetids:
        forests_by_targetid[tid] = new_forests_by_targetid[tid]


class DesiData(Data):
    """Abstract class to read DESI data and format it as a list of
    Forest instances.

    Methods
    -------
    (see Data in py/picca/delta_extraction/data.py)
    __init__
    __parse_config
    format_data
    read_data
    set_blinding

    Attributes
    ----------
    (see Data in py/picca/delta_extraction/data.py)

    blinding: str
    A string specifying the chosen blinding strategies. Must be one of the
    accepted values in ACCEPTED_BLINDING_STRATEGIES

    catalogue: astropy.table.Table
    The quasar catalogue

    logger: logging.Logger
    Logger object

    use_non_coadded_spectra: bool
    If True, load data from non-coadded spectra. Otherwise,
    load coadded data

    uniquify_night_targetid: bool
    If True, remove the quasars taken on the same night.

    keep_single_exposures: bool
    If True, the date loadded from non-coadded spectra are not coadded. 
    Otherwise, coadd the spectra here.
    """

    def __init__(self, config):
        """Initialize class instance

        Arguments
        ---------
        config: configparser.SectionProxy
        Parsed options to initialize class
        """
        self.logger = logging.getLogger(__name__)

        config["flux units"] = "10**-17 erg/(s cm2 Angstrom)"
        super().__init__(config)

        # load variables from config
        self.__parse_config(config)

        # load z_truth catalogue
        t0 = time.time()
        self.logger.progress("Reading quasar catalogue")
        self.catalogue = DesiQuasarCatalogue(config).catalogue
        t1 = time.time()
        self.logger.progress(f"Time spent reading quasar catalogue: {t1-t0}")

        # read data
        t0 = time.time()
        self.logger.progress("Reading data")
        is_mock = self.read_data()
        t1 = time.time()
        self.logger.progress(f"Time spent reading data: {t1-t0}")

        # set blinding
        self.blinding = None
        self.set_blinding(is_mock)

    def __parse_config(self, config):
        """Parse the configuration options

        Arguments
        ---------
        config: configparser.SectionProxy
        Parsed options to initialize class

        Raise
        -----
        DataError upon missing required variables
        """
        # instance variables
        self.keep_single_exposures = config.getboolean(
            "keep single exposures")
        if self.keep_single_exposures is None:
            raise DataError(
                "Missing argument 'keep single exposures' required by DesiData"
            )

        self.use_non_coadded_spectra = config.getboolean(
            "use non-coadded spectra")
        if self.use_non_coadded_spectra is None:
            raise DataError(
                "Missing argument 'use non-coadded spectra' required by DesiData"
            )

        self.uniquify_night_targetid = config.getboolean(
            "uniquify night targetid")
        if self.uniquify_night_targetid is None:
            raise DataError(
                "Missing argument 'uniquify night targetid' required by DesiData"
            )



    def read_data(self):
        """Read the spectra and formats its data as Forest instances.

        Method to be implemented by child classes.

        Return
        ------
        is_mock: bool
        True if mocks are read, False otherwise

        Raise
        -----
        DataError if no quasars were found
        """
        raise DataError(
            "Function 'read_data' was not overloaded by child class")

    def set_blinding(self, is_mock):
        """Set the blinding in Forest.

        Update the stored value if necessary.

        Attributes
        ----------
        is_mock: boolean
        True if reading mocks, False otherwise
        """
        # do not blind mocks
        if is_mock:
            self.blinding = "none"
        # do not blind metal forests (not lya)
        elif Forest.log_lambda_rest_frame_grid[0] > ABSORBER_IGM["LYA"]:
            self.blinding = "none"
        # figure out blinding
        else:
            if all(self.catalogue["LASTNIGHT"] < 20210514):
                # sv data, no blinding
                self.blinding = "none"
            elif all(self.catalogue["LASTNIGHT"] < 20210801):
                self.blinding = "desi_m2"
            elif all(self.catalogue["LASTNIGHT"] < 20220801):
                self.blinding = "desi_y1"
            else:
                self.blinding = "desi_y3"

        if self.blinding not in ACCEPTED_BLINDING_STRATEGIES:
            raise DataError(
                "Unrecognized blinding strategy. Accepted strategies "
                f"are {ACCEPTED_BLINDING_STRATEGIES}. "
                f"Found '{self.blinding}'")

        # set blinding strategy
        Forest.blinding = self.blinding


class DesiDataFileHandler():
    """File handler for class DesiHealpix
    This implementation is based on the understanding that imap in multiprocessing
    cannot be applied to class methods due to `pickle`ing limitations. Each child
    process creates an instance of this class, then imap calls each instance with
    an argument in parallel. imap is limited to single-argument functions, but it
    can be overcome by making that argument a tuple.

    Methods
    -------
    __init__
    __call__
    format_data
    read_file

    Attributes
    ----------
    analysis_type: str
    Selected analysis type. See class Data from py/picca/delta_extraction/data.py
    for details

    logger: logging.Logger
    Logger object

    use_non_coadded_spectra: bool
    If True, load data from non-coadded spectra. Otherwise,
    load coadded data

    uniquify_night_targetid: bool
    If True, remove the quasars taken on the same night.

    keep_single_exposures: bool
    If True, the date loadded from non-coadded spectra are not coadded. 
    Otherwise, coadd the spectra here.
    """

    def __init__(self,
                 analysis_type,
                 use_non_coadded_spectra,
                 uniquify_night_targetid,
                 keep_single_exposures,
                 logger):
        """Initialize file handler

        Arguments
        ---------
        analysis_type: str
        Selected analysis type. See class Data from py/picca/delta_extraction/data.py
        for details

        keep_single_exposures: bool
        If True, the date loadded from non-coadded spectra are not coadded. 
        Otherwise, coadd the spectra here.

        uniquify_night_targetid: bool
        If True, remove the quasars taken on the same night.

        use_non_coadded_spectra: bool
        If True, load data from non-coadded spectra. Otherwise,
        load coadded data

        logger: logging.Logger
        Logger object from parent class. Trying to initialize it here
        without copying failed data_tests.py
        """
        # The next line gives failed tests
        # self.logger = logging.getLogger(__name__)
        self.logger = logger
        self.analysis_type = analysis_type
        self.keep_single_exposures = keep_single_exposures
        self.uniquify_night_targetid = uniquify_night_targetid
        self.use_non_coadded_spectra = use_non_coadded_spectra

    def __call__(self, args):
        """Call method read_file. Note imap can be called with
        only one argument, hence tuple as argument.

        Arguments
        ---------
        args: tuple
        Arguments to be passed to read_file. Should contain a string with the
        filename and a astropy.table.Table with the quasar catalogue
        """
        return self.read_file(*args)

    def format_data(self,
                    catalogue,
                    spectrographs_data,
                    targetid_spec,
                    reso_from_truth=False):
        """After data has been read, format it into DesiForest instances

        Instances will be DesiForest or DesiPk1dForest depending on analysis_type

        Arguments
        ---------
        catalogue: astropy.table.Table
        The quasar catalogue fragment associated with this data

        spectrographs_data: dict
        The read data

        targetid_spec: int
        Targetid of the objects to format

        reso_from_truth: bool - Default: False
        Specifies whether resolution matrixes are read from truth files (True)
        or directly from data (False)

        Return
        ------
        forests_by_targetid: dict
        Dictionary were forests are stored.

        num_data: int
        The number of instances loaded
        """
        num_data = 0
        forests_by_targetid = {}
        # Loop over quasars in catalogue fragment
        for row in catalogue:
            # Find which row in tile contains this quasar
            # It should be there by construction
            targetid = row["TARGETID"]
            w_t = np.where(targetid_spec == targetid)[0]
            if len(w_t) == 0:
                self.logger.warning(
                    f"Error reading {targetid}. Ignoring object")
                continue
            if len(w_t) > 1:
                if not self.use_non_coadded_spectra:
                    self.logger.warning(
                        "Warning: more than one spectrum in this file "
                        f"for {targetid}")
            else:
                w_t = w_t[0]
            # Construct DesiForest instance
            # Fluxes from the different spectrographs will be coadded
            for spec in spectrographs_data.values():
                if self.use_non_coadded_spectra and not self.keep_single_exposures:
                    ivar = np.atleast_2d(spec['IVAR'][w_t])
                    ivar_coadded_flux = np.atleast_2d(
                            ivar * spec['FLUX'][w_t]).sum(axis=0)
                    ivar = ivar.sum(axis=0)
                    flux = ivar_coadded_flux / ivar
                else:
                    flux = spec['FLUX'][w_t].copy()
                    ivar = spec['IVAR'][w_t].copy()

                # If keep_single_exposures is activated, all exposures are kept.
                # To avoid coadding exposures, we store the forest with
                # targetid_i (i = exposure index) instead of targetid
                # The variable forests_by_targetid is modified by update_forest_dictionary.
                if self.use_non_coadded_spectra & self.keep_single_exposures:
                    # One exposure case
                    if len(flux.shape) != 2:
                        flux , ivar = np.array([flux]), np.array([ivar])
                    for i, (flux_i,ivar_i) in enumerate(zip(flux,ivar)):
                        forests_by_targetid, num_data = self.update_forest_dictionary(
                                forests_by_targetid,
                                f"{targetid}_{i}",
                                targetid,
                                spec,
                                row,
                                flux_i,
                                ivar_i,
                                w_t,
                                reso_from_truth,
                                num_data)
                else:
                    forests_by_targetid, num_data  = self.update_forest_dictionary(
                            forests_by_targetid,
                            targetid,
                            targetid,
                            spec,
                            row,
                            flux,
                            ivar,
                            w_t,
                            reso_from_truth,
                            num_data)
        return forests_by_targetid, num_data

    def update_forest_dictionary(self,
                                 forests_by_targetid,
                                 key_update,
                                 targetid,
                                 spec,
                                 row,
                                 flux,
                                 ivar,
                                 w_t,
                                 reso_from_truth,
                                 num_data):
        """Add new forests to the current forest dictonary

        Arguments
        ---------
        forests_by_targetid: dict
        Dictionary were forests are stored.

        key_update: int or str
        The key to update in the forest dictionary

        targetid: int
        The los_id of the forest

        spec: dict
        Dictionary containing the spectrum information.

        row: dict
        Metadata of the spectrum to treat.

        flux: numpy.array
        Flux of the spectrum to add.

        ivar: numpy.array
        Inverse variance of the spectrum to add.

        w_t: numpy.array
        The index of spectrum with the same targetid.

        reso_from_truth: bool - Default: False
        Specifies whether resolution matrixes are read from truth files (True)
        or directly from data (False)

        num_data: int
        The number of instances loaded
        
        Return
        ------
        forests_by_targetid: dict
        Dictionary were forests are stored.

        num_data: int
        The number of instances loaded
        """
        args = {
            "flux": flux,
            "ivar": ivar,
            "targetid": targetid,
            "ra": row['RA'],
            "dec": row['DEC'],
            "z": row['Z'],
        }
        args["log_lambda"] = np.log10(spec['WAVELENGTH'])


        if self.analysis_type == "BAO 3D":
            forest = DesiForest(**args)
        elif self.analysis_type == "PK 1D":
            if (self.use_non_coadded_spectra) & (not self.keep_single_exposures):
                exposures_diff = exp_diff_desi(spec, w_t)
                if exposures_diff is None:
                    return forests_by_targetid, num_data
            else:
                exposures_diff = np.zeros(spec['WAVELENGTH'].shape)
            if reso_from_truth:
                reso_sum = spec['RESO'][:, :]
            else:
                if len(spec['RESO'][w_t].shape) < 3:
                    reso_sum = spec['RESO'][w_t].copy()
                else:
                    reso_sum = spec['RESO'][w_t].sum(axis=0)
            reso_in_pix, reso_in_km_per_s = spectral_resolution_desi(
                reso_sum, spec['WAVELENGTH'])
            args["exposures_diff"] = exposures_diff
            args["reso"] = reso_in_km_per_s
            args["resolution_matrix"] = reso_sum
            args["reso_pix"] = reso_in_pix

            forest = DesiPk1dForest(**args)
        # this should never be entered added here in case at some point
        # we add another analysis type
        else:  # pragma: no cover
            raise DataError("Unkown analysis type. Expected 'BAO 3D'"
                            f"or 'PK 1D'. Found '{self.analysis_type}'")

        # rebin arrays
        # this needs to happen after all arrays are initialized by
        # Forest constructor
        forest.rebin()

        if key_update in forests_by_targetid:
            existing_forest = forests_by_targetid[key_update]
            existing_forest.coadd(forest)
            forests_by_targetid[key_update] = existing_forest
        else:
            forests_by_targetid[key_update] = forest

        num_data += 1

        return forests_by_targetid, num_data

    def read_file(self, filename, catalogue):
        """Read the spectra and formats its data as Forest instances.

        Arguments
        ---------
        filename: str
        Name of the file to read

        catalogue: astropy.table.Table
        The quasar catalogue fragment associated with this file

        Returns:
        ---------
        forests_by_targetid: dict
        Dictionary were forests are stored.

        num_data: int
        The number of instances loaded

        Raise
        -----
        DataError if the analysis type is PK 1D and resolution data is not present
        """
        raise DataError(
            "Function 'read_data' was not overloaded by child class")
