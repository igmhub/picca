"""This module defines the abstract class OpticalDepthCorrection"""
import logging

import numpy as np

from picca.delta_extraction.correction import Correction
from picca.delta_extraction.errors import CorrectionError
from picca.delta_extraction.utils import ABSORBER_IGM

accepted_options = ["optical depth absorber", "optical depth gamma",
                    "optical depth tau"]

class OpticalDepthCorrection(Correction):
    """Class to correct for optical depths contribution

    Methods
    -------
    __init__
    apply_correction

    Attributes
    ----------
    correct_flux: scipy.interpolate.interp1d
    Interpolation function to adapt the correction to slightly different
    grids of wavelength

    gamma_list: list of float
    List of gamma factors for each of the optical depth absorbers

    lambda_rest_frame_list: list of float
    List of rest frame wavelengths for each of the optical depth absorbers

    logger: logging.Logger
    Logger object

    tau_list: list of float
    List of tau factors for each of the optical depth absorbers
    """

    def __init__(self, config):
        """Initialize class instance.

        Arguments
        ---------
        config: configparser.SectionProxy
        Parsed options to initialize class

        Raises
        ------
        CorrectionError if any of the variables 'optical depths tau', 'optical
        depth gamma' and 'optical depth absorber' are not present or have
        different lengths
        """
        self.logger = logging.getLogger(__name__)

        tau_list = config.get("optical depth tau")
        if tau_list is None:
            raise CorrectionError(
                "Missing argument 'optical depth tau' required by SdssOpticalDepthCorrection")
        self.tau_list = [float(item) for item in tau_list.split()]
        gamma_list = config.get("optical depth gamma")
        if gamma_list is None:
            raise CorrectionError(
                "Missing argument 'optical depth gamma' required by SdssOpticalDepthCorrection")
        self.gamma_list = [float(item) for item in gamma_list.split()]
        absorber_list = config.get("optical depth absorber")
        if absorber_list is None:
            raise CorrectionError(
                "Missing argument 'optical depth absorber' required by SdssOpticalDepthCorrection")
        absorber_list = [item.upper() for item in absorber_list.split()]
        self.lambda_rest_frame_list = [
            ABSORBER_IGM[absorber] for absorber in absorber_list
        ]
        if not (len(self.tau_list) == len(self.gamma_list) and
                len(self.tau_list) == len(self.lambda_rest_frame_list)):
            raise CorrectionError("Variables 'optical depth tau', 'optical "
                                  "depth gamma' and 'optical depth absorber' "
                                  "should have the same number of entries")

        self.logger.info(f"Adding {len(self.tau_list)} optical depths")
        for tau, gamma, lambda_rest_frame in zip(self.tau_list, self.gamma_list,
                                                 self.lambda_rest_frame_list):
            self.logger.info(f"   tau = {tau}, gamma = {gamma}, "
                             f"lambda_rest_frame = {lambda_rest_frame}")

    def apply_correction(self, forest):
        """Apply the correction. Correction is applied by dividing the
        data flux by the loaded correction, and the subsequent correction
        of the inverse variance

        Arguments
        ---------
        forest: Forest
        A Forest instance to which the correction is applied
        """
        mean_optical_depth = np.ones(forest.log_lambda.size)
        for tau, gamma, lambda_rest_frame in zip(self.tau_list, self.gamma_list,
                                                 self.lambda_rest_frame_list):

            w = 10.**forest.log_lambda / (1. + forest.z) <= lambda_rest_frame
            z = 10.**forest.log_lambda / lambda_rest_frame - 1.
            mean_optical_depth[w] *= np.exp(-tau * (1. + z[w])**gamma)

        forest.transmission_correction *= mean_optical_depth
