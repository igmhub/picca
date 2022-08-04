"""This module defines the abstract class Data from which all
classes loading data must inherit
"""
import logging
import multiprocessing

import numpy as np
import fitsio
import healpy

from picca.delta_extraction.astronomical_objects.forest import Forest
from picca.delta_extraction.astronomical_objects.pk1d_forest import Pk1dForest
from picca.delta_extraction.errors import DataError
from picca.delta_extraction.utils import ABSORBER_IGM

accepted_options = [
    "analysis type",
    "delta lambda",
    "delta log lambda",
    "delta lambda rest frame",
    "input directory",
    "lambda abs IGM",
    "lambda max",
    "lambda max rest frame",
    "lambda min",
    "lambda min rest frame",
    "minimum number pixels in forest",
    "out dir",
    "rejection log file",
    "minimal snr",
    # these options are allowed but will be overwritten by
    # minimal snr (only needed to allow running on a .config
    # with default options)
    "minimal snr pk1d",
    "minimal snr bao3d",
    "num processors",
]

defaults = {
    "analysis type": "BAO 3D",
    "lambda abs IGM": "LYA",
    "lambda max": 5500.0,
    "lambda max rest frame": 1200.0,
    "lambda min": 3600.0,
    "lambda min rest frame": 1040.0,
    "minimum number pixels in forest": 50,
    "rejection log file": "rejection_log.fits.gz",
    "minimal snr pk1d": 1,
    "minimal snr bao3d": 0,
}

accepted_analysis_type = ["BAO 3D", "PK 1D"]

def _save_deltas_one_healpix(out_dir, healpix, forests):
    """Saves the deltas that belong to one healpix.

    Arguments
    ---------
    out_dir: str
    Parent directory to save deltas.

    healpix: int

    forests: List of Forests
    List of forests to save into one file.

    Returns:
    ---------
    header_n_size: List of (header, size)
    List of forest.header and forest.size to later
    add to rejection log as accepted.
    """
    results = fitsio.FITS(
        f"{out_dir}/Delta/delta-{healpix}.fits.gz",
        'rw',
        clobber=True)

    header_n_size = []
    for forest in forests:
        header = forest.get_header()
        cols, names, units, comments = forest.get_data()
        results.write(cols,
                      names=names,
                      header=header,
                      comment=comments,
                      units=units,
                      extname=str(forest.los_id))

        # store information for logs
        header_n_size.append((header, forest.flux.size))
        # self.add_to_rejection_log(header, forest.flux.size, "accepted")
    results.close()

    return header_n_size

class Data:
    """Abstract class from which all classes loading data must inherit.
    Classes that inherit from this should be initialized using
    a configparser.SectionProxy instance.

    Methods
    -------
    __parse_config
    add_to_rejection_log
    initialize_rejection_log
    filter_bad_cont_forests
    filter_forests
    find_nside
    save_deltas
    save_rejection_log

    Attributes
    ----------
    analysis_type: str
    Selected analysis type. Current options are "BAO 3D" or "PK 1D"

    forests: list of Forest
    A list of Forest from which to compute the deltas.

    input_directory: str
    Directory where input data is stored.

    logger: logging.Logger
    Logger object

    min_num_pix: int
    Minimum number of pixels in a forest. Forests with less pixels will be dropped.

    min_snr: float
    Minimum signal-to-noise ratio to accept a Forest instance.

    out_dir: str
    Directory where data will be saved. Log info will be saved in out_dir+"Log/"
    and deltas will be saved in out_dir+"Delta/"

    rejection_log_file: str
    Filelame of the rejection log

    rejection_log_initialized: bool
    Flag specifying if the rejection log has been initialized

    rejection_log_cols: list of list
    Each list contains the data of each of the fields saved in the rejection log

    rejection_log_comments: list of list
    Description of each of the fields saved in the rejection log

    rejection_log_names: list of list
    Name of each of the fields saved in the rejection log
    """

    def __init__(self, config):
        """Initialize class instance"""
        self.logger = logging.getLogger('picca.delta_extraction.data.Data')
        self.forests = []

        self.analysis_type = None
        self.input_directory = None
        self.min_num_pix = None
        self.out_dir = None
        self.rejection_log_file = None
        self.min_snr = None
        self.num_processors = None
        self.__parse_config(config)

        # rejection log arays
        self.rejection_log_initialized = False
        self.rejection_log_cols = None
        self.rejection_log_names = None
        self.rejection_log_comments = None

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
        # setup Forest class variables
        wave_solution = config.get("wave solution")

        if wave_solution is None:
            raise DataError("Missing argument 'wave solution' required by Data")
        if wave_solution not in ["lin", "log"]:
            raise DataError(
                "Unrecognised value for 'wave solution'. Expected either "
                f"'lin' or 'log'. Found '{wave_solution}'.")

        if wave_solution == "log":
            pixel_step = config.getfloat("delta log lambda")
            if pixel_step is None:
                raise DataError(
                    "Missing argument 'delta log lambda' required by "
                    "Data when 'wave solution' is set to 'log'")
            pixel_step_rest_frame = config.getfloat(
                "delta log lambda rest frame")
            if pixel_step_rest_frame is None:
                pixel_step_rest_frame = pixel_step
                self.logger.info("'delta log lambda rest frame' not set, using "
                                 "the same value as for 'delta log lambda' "
                                 f"({pixel_step_rest_frame})")
        elif wave_solution == "lin":
            pixel_step = config.getfloat("delta lambda")
            if pixel_step is None:
                raise DataError("Missing argument 'delta lambda' required by "
                                "Data when 'wave solution' is set to 'lin'")
            pixel_step_rest_frame = config.getfloat("delta lambda rest frame")
            if pixel_step_rest_frame is None:
                pixel_step_rest_frame = pixel_step
                self.logger.info(
                    "'delta lambda rest frame' not set, using "
                    f"the same value as for 'delta lambda' ({pixel_step_rest_frame})"
                )
        # this should not be reached as wave_solution is either "lin" or "log"
        # added here only in case we add another wave_solution in the future
        else:  # pragma: no cover
            raise DataError(
                "Unrecognised value for 'wave solution'. Expected either "
                f"'lin' or 'log'. Found '{wave_solution}'.")

        lambda_max = config.getfloat("lambda max")
        if lambda_max is None:
            raise DataError("Missing argument 'lambda max' required by Data")
        lambda_max_rest_frame = config.getfloat("lambda max rest frame")
        if lambda_max_rest_frame is None:
            raise DataError(
                "Missing argument 'lambda max rest frame' required by Data")
        lambda_min = config.getfloat("lambda min")
        if lambda_min is None:
            raise DataError("Missing argument 'lambda min' required by Data")
        lambda_min_rest_frame = config.getfloat("lambda min rest frame")
        if lambda_min_rest_frame is None:
            raise DataError(
                "Missing argument 'lambda min rest frame' required by Data")

        Forest.set_class_variables(lambda_min, lambda_max,
                                   lambda_min_rest_frame, lambda_max_rest_frame,
                                   pixel_step, pixel_step_rest_frame,
                                   wave_solution)

        # instance variables
        self.analysis_type = config.get("analysis type")
        if self.analysis_type is None:
            raise DataError("Missing argument 'analysis type' required by Data")
        if self.analysis_type not in accepted_analysis_type:
            raise DataError("Invalid argument 'analysis type' required by "
                            f"Data. Found: '{self.analysis_type}'. Accepted "
                            "values: " + ",".join(accepted_analysis_type))

        if self.analysis_type == "PK 1D":
            lambda_abs_igm_name = config.get("lambda abs IGM")
            if lambda_abs_igm_name is None:
                raise DataError(
                    "Missing argument 'lambda abs IGM' required by Data "
                    "when 'analysys type' is 'PK 1D'")
            Pk1dForest.lambda_abs_igm = ABSORBER_IGM.get(lambda_abs_igm_name)
            if Pk1dForest.lambda_abs_igm is None:
                raise DataError(
                    "Invalid argument 'lambda abs IGM' required by "
                    f"Data. Found: '{lambda_abs_igm_name}'. Accepted "
                    "values: " + ", ".join(ABSORBER_IGM))

        self.input_directory = config.get("input directory")
        if self.input_directory is None:
            raise DataError(
                "Missing argument 'input directory' required by Data")

        self.min_num_pix = config.getint("minimum number pixels in forest")
        if self.min_num_pix is None:
            raise DataError(
                "Missing argument 'minimum number pixels in forest' "
                "required by Data")

        self.num_processors = config.getint("num processors")
        if self.num_processors is None:
            raise DataError(
                "Missing argument 'num processors' required by Data")
        if self.num_processors == 0:
            self.num_processors = (multiprocessing.cpu_count() // 2)

        self.out_dir = config.get("out dir")
        if self.out_dir is None:
            raise DataError("Missing argument 'out dir' required by Data")

        self.rejection_log_file = config.get("rejection log file")
        if self.rejection_log_file is None:
            raise DataError(
                "Missing argument 'rejection log file' required by Data")
        if "/" in self.rejection_log_file:
            raise DataError("Error constructing Data. "
                            "'rejection log file' should not incude folders. "
                            f"Found: {self.rejection_log_file}")
        if not (self.rejection_log_file.endswith(".fits") or
                self.rejection_log_file.endswith(".fits.gz")):
            raise DataError("Error constructing Data. Invalid extension for "
                            "'rejection log file'. Filename "
                            "should en with '.fits' or '.fits.gz'. Found "
                            f"'{self.rejection_log_file}'")

        if self.analysis_type == "BAO 3D":
            self.min_snr = config.getfloat("minimal snr bao3d")
        elif self.analysis_type == "PK 1D":
            self.min_snr = config.getfloat("minimal snr pk1d")
        # this should not be reached as analysis_type is either "BAO 3D" or
        # "PK 1D" added here only in case we add another analysis_type in the
        # future
        else:  # pragma: no cover
            raise DataError("Invalid argument 'analysis type' required by "
                            f"Data. Found: '{self.analysis_type}'. Accepted "
                            "values: " + ",".join(accepted_analysis_type))
        if self.min_snr is None:
            raise DataError(
                "Missing argument 'minimal snr bao3d' (if 'analysis type' = "
                "'BAO 3D') or ' minimal snr pk1d' (if 'analysis type' = 'Pk1d') "
                "required by Data")

    def add_to_rejection_log(self, header, size, rejection_status):
        """Adds to the rejection log arrays.
        In the log forest headers will be saved along with the forest size and
        the rejection status.

        Arguments
        ---------
        header: list of dict
        Output of forest.get_header()

        size: int
        Size of the forest

        rejection_status: str
        Rejection status
        """
        # if necessary initialize arrays to save rejected quasars in the log
        if not self.rejection_log_initialized:
            self.initialize_rejection_log()

        for col, name in zip(self.rejection_log_cols, self.rejection_log_names):
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

    def initialize_rejection_log(self):
        """Initializes the rejection log arrays.
        In the log forest headers will be saved along with the forest size and
        the rejection status.
        """
        self.rejection_log_cols = [[], []]
        self.rejection_log_names = ["FOREST_SIZE", "REJECTION_STATUS"]
        self.rejection_log_comments = [
            "num pixels in forest", "rejection status"
        ]

        for item in self.forests[0].get_header():
            self.rejection_log_cols.append([])
            self.rejection_log_names.append(item.get("name"))
            self.rejection_log_comments.append(item.get("comment"))
        self.rejection_log_initialized = True

    def filter_bad_cont_forests(self):
        """Remove forests where continuum could not be computed"""
        remove_indexs = []
        for index, forest in enumerate(self.forests):
            if forest.bad_continuum_reason is not None:
                # store information for logs
                self.add_to_rejection_log(forest.get_header(), forest.flux.size,
                                          forest.bad_continuum_reason)

                self.logger.progress(
                    f"Rejected forest with los_id {forest.los_id} "
                    "due to continuum fitting problems. Reason: "
                    f"{forest.bad_continuum_reason}")

                remove_indexs.append(index)

        for index in sorted(remove_indexs, reverse=True):
            del self.forests[index]

        self.logger.progress(f"Accepted sample has {len(self.forests)} forests")

    def filter_forests(self):
        """Remove forests that do not meet quality standards"""
        self.logger.progress(f"Input sample has {len(self.forests)} forests")

        remove_indexs = []
        for index, forest in enumerate(self.forests):
            if forest.flux.size < self.min_num_pix:
                # store information for logs
                self.add_to_rejection_log(forest.get_header(), forest.flux.size,
                                          "short_forest")
                self.logger.progress(
                    f"Rejected forest with los_id {forest.los_id} "
                    f"due to forest being too short ({forest.flux.size})")
            elif np.isnan((forest.flux * forest.ivar).sum()):
                self.add_to_rejection_log(forest.get_header(), forest.flux.size,
                                          "nan_forest")
                self.logger.progress(
                    f"Rejected forest with los_id {forest.los_id} "
                    "due to finding nan")
            elif forest.mean_snr < self.min_snr:
                self.add_to_rejection_log(forest.get_header(), forest.flux.size,
                                          f"low SNR ({forest.mean_snr})")
                self.logger.progress(
                    f"Rejected forest with los_id {forest.los_id} "
                    f"due to low SNR ({forest.mean_snr} < {self.min_snr})")
            else:
                continue

            remove_indexs.append(index)

        # remove forests
        for index in sorted(remove_indexs, reverse=True):
            del self.forests[index]

        self.logger.progress("Removed forests that are too short")
        self.logger.progress(
            f"Remaining sample has {len(self.forests)} forests")

    def find_nside(self):
        """Determines nside such that there are 500 objs per pixel on average."""

        self.logger.progress("determining nside")
        nside = 256
        target_mean_num_obj = 500
        ra = np.array([forest.ra for forest in self.forests])
        dec = np.array([forest.dec for forest in self.forests])
        healpixs = healpy.ang2pix(nside, np.pi / 2 - dec, ra)

        mean_num_obj = len(healpixs) / len(np.unique(healpixs))
        nside_min = 8
        while mean_num_obj < target_mean_num_obj and nside >= nside_min:
            nside //= 2
            healpixs = healpy.ang2pix(nside, np.pi / 2 - dec, ra)
            mean_num_obj = len(healpixs) / len(np.unique(healpixs))

        self.logger.progress(f"nside = {nside} -- mean #obj per pixel = "
                             f"{mean_num_obj}")

        for forest, healpix in zip(self.forests, healpixs):
            forest.healpix = healpix

    def save_deltas(self):
        """Save the deltas."""
        healpixs = np.array([forest.healpix for forest in self.forests])
        unique_healpixs = np.unique(healpixs)

        arguments = []
        for healpix in unique_healpixs:
            this_idx = np.nonzero(healpix == healpixs)[0]
            grouped_forests = [self.forests[i] for i in this_idx]
            arguments.append((self.out_dir, healpix, grouped_forests))

        if self.num_processors > 1:
            context = multiprocessing.get_context('fork')
            with context.Pool(processes=self.num_processors) as pool:
                header_n_sizes = pool.starmap(_save_deltas_one_healpix,
                                              arguments)
        else:
            header_n_sizes = []
            for args in arguments:
                header_n_sizes.append(_save_deltas_one_healpix(*args))

        # store information for logs
        for header_n_size in header_n_sizes:
            for header, size in header_n_size:
                self.add_to_rejection_log(header, size, "accepted")

        self.save_rejection_log()

    def save_rejection_log(self):
        """Saves the rejection log arrays.
        In the log forest headers will be saved along with the forest size and
        the rejection status.
        """
        rejection_log = fitsio.FITS(self.out_dir + "Log/" +
                                    self.rejection_log_file,
                                    'rw',
                                    clobber=True)

        rejection_log.write(
            [np.array(item) for item in self.rejection_log_cols],
            names=self.rejection_log_names,
            comment=self.rejection_log_comments,
            extname="rejection_log")

        rejection_log.close()
