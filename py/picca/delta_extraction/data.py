"""This module defines the abstract class Data from which all
classes loading data must inherit
"""
import logging

import numpy as np
import fitsio

from picca.delta_extraction.astronomical_objects.pk1d_forest import Pk1dForest
from picca.delta_extraction.errors import DataError
from picca.delta_extraction.utils import ABSORBER_IGM

defaults = {
    "analysis type": "BAO 3D",
    "lambda abs IGM": ABSORBER_IGM.get("LYA"),
    "minimum number pixels in forest": 50,
}

accepted_analysis_type = ["BAO 3D", "PK 1D"]

class Data:
    """Abstract class from which all classes loading data must inherit.
    Classes that inherit from this should be initialized using
    a configparser.SectionProxy instance.

    Methods
    -------
    _parse_config
    filter_forests
    save_deltas

    Attributes
    ----------
    analysis_type: str
    Selected analysis type. Current options are "BAO 3D" or "PK 1D"

    forests: list of Forest
    A list of Forest from which to compute the deltas.

    logger: logging.Logger
    Logger object

    min_num_pix: int
    Minimum number of pixels in a forest. Forests with less pixels will be dropped.
    """

    def __init__(self, config):
        """Initialize class instance"""
        self.logger = logging.getLogger('picca.delta_extraction.data.Data')
        self.forests = []

        self.analysis_type = config.get("analysis type")
        if self.analysis_type is None:
            self.analysis_type = defaults.get("analysis type")
        if self.analysis_type not in accepted_analysis_type:
            raise DataError("Invalid argument 'analysis type' required by "
                            "DesiData. Accepted values: " +
                            ",".join(accepted_analysis_type))

        if self.analysis_type == "BAO 3D":
            if config.get("absorber IGM") is None:
                Pk1dForest.lambda_abs_igm = defaults.get("lambda abs IGM")
            else:
                Pk1dForest.lambda_abs_igm = ABSORBER_IGM.get(config.get("absorber IGM"))

        self.min_num_pix = config.getint("minimum number pixels in forest")
        if self.min_num_pix is None:
            self.min_num_pix = defaults.get("minimum number pixels in forest")

    def filter_forests(self):
        """Remove forests that do not meet quality standards"""
        self.logger.progress(f"Input sample has {len(self.forests)} forests")
        remove_indexs = []
        for index, forest in enumerate(self.forests):
            print(f"Forest {forest.los_id} has length {len(forest.log_lambda)}")
            if forest.flux.size < self.min_num_pix:
                self.logger.progress(
                    f"Rejected forest with thingid {forest.thingid} "
                    "due to forest being too short")
            elif np.isnan((forest.flux * forest.ivar).sum()):
                self.logger.progress(
                    f"Rejected forest with thingid {forest.thingid} "
                    "due to finding nan")
            else:
                continue
            remove_indexs.append(index)

        for index in sorted(remove_indexs, reverse=True):
            del self.forests[index]

        self.logger.progress(f"Remaining sample has {len(self.forests)} forests")

    def save_deltas(self, out_dir):
        """Save the deltas.

        Attributes
        ----------
        out_dir: str
        Directory where data will be saved
        """
        healpixs = np.array([forest.healpix for forest in self.forests])
        unique_healpixs = np.unique(healpixs)
        healpixs_indexs = {healpix: np.where(healpixs == healpix)
                           for healpix in unique_healpixs}

        for healpix, indexs in sorted(healpixs_indexs.items()):
            results = fitsio.FITS(out_dir + "/delta-{}".format(healpix) +
                                  ".fits.gz",
                                  'rw',
                                  clobber=True)
            for index in indexs:
                forest = self.forests[index]
                cols, names, units, comments = forest.get_data()
                results.write(cols,
                              names=names,
                              header=forest.get_header(),
                              comment=comments,
                              units=units,
                              extname=str(forest.los_id))

            results.close()
