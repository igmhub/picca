"""This module defines the abstract class Data from which all
classes loading data must inherit
"""
import numpy as np

from picca.delta_extraction.userprint import userprint

from picca.delta_extraction.astronomical_objects.forest import Forest

defaults = {
    "minimum number pixels in forest": 50,
}


class Data:
    """Abstract class from which all classes loading data must inherit.
    Classes that inherit from this should be initialized using
    a configparser.SectionProxy instance.

    Methods
    -------
    _parse_config
    filter_forests

    Attributes
    ----------
    forests: list of Forest
    A list of Forest from which to compute the deltas.
    """

    def __init__(self, config):
        """Initialize class instance"""
        self.forests = []
        self.min_num_pix = None
        self._parse_config(config)

    # pylint: disable=no-self-use
    # this method should use self in child classes
    def _parse_config(self, config):
        """Parse the configuration options for the parent type.

        Arguments
        ---------
        config: configparser.SectionProxy
        Parsed options to initialize class
        """
        self.min_num_pix = config.getint("minimum number pixels in forest")
        if self.min_num_pix is None:
            self.min_num_pix = defaults.get("minimum number pixels in forest")

    def filter_forests(self):
        """Removes forests that do not meet quality standards"""
        ## Apply cuts
        userprint(f"INFO: Input sample has {len(self.forests)} forests")
        remove_indexs = []
        for index, forest in enumerate(self.forests):
            if ((Forest.wave_solution == "log" and
                 len(forest.log_lambda) < self.min_num_pix) or
                    (Forest.wave_solution == "lin" and
                     len(forest.lambda_) < self.min_num_pix)):
                userprint(
                    f"INFO: Rejected forest with thingid {forest.thingid} "
                    "due to forest being too short")
            elif np.isnan((forest.flux * forest.ivar).sum()):
                userprint(
                    f"INFO: Rejected forest with thingid {forest.thingid} "
                    "due to finding nan")
            else:
                continue
            remove_indexs.append(index)

        for index in sorted(remove_indexs, reverse=True):
            del self.forests[index]

        userprint(f"INFO: Remaining sample has {len(self.forests)} forests")
