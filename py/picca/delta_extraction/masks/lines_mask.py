"""This module defines the class LinesMask in the masking of (sky) lines"""
from astropy.table import Table
import numpy as np

from picca.delta_extraction.astronomical_objects.forest import Forest
from picca.delta_extraction.errors import MaskError
from picca.delta_extraction.mask import (
    Mask, _add_mask_intervals, _create_deltas, _finalize_deltas_to_mask)
from picca.delta_extraction.mask import ( # pylint: disable=unused-import
    accepted_options, defaults)
from picca.delta_extraction.utils import update_accepted_options

accepted_options = update_accepted_options(accepted_options, [
    "filename"
])

class LinesMask(Mask):
    """Class to mask (sky) lines

    Methods
    -------
    __init__
    apply_mask

    Attributes
    ----------
    (see Mask in py/picca/delta_extraction/mask.py)

    mask_rest_frame: astropy.Table
    Table with the rest-frame wavelength of the lines to mask

    mask_obs_frame: astropy.Table
    Table with the observed-frame wavelength of the lines to mask. This usually
    contains the sky lines.
    """
    def __init__(self, config):
        """Initialize class instance.

        Arguments
        ---------
        config: configparser.SectionProxy
        Parsed options to initialize class
        """
        super().__init__(config)

        mask_file = config.get("filename")
        if mask_file is None:
            raise MaskError(
                "Missing argument 'filename' required by LinesMask")
        try:
            mask = Table.read(mask_file,
                              names=('type', 'wave_min', 'wave_max', 'frame'),
                              format='ascii')
            mask['log_wave_min'] = np.log10(mask['wave_min'])
            mask['log_wave_max'] = np.log10(mask['wave_max'])

            select_rest_frame_mask = mask['frame'] == 'RF'
            select_obs_mask = mask['frame'] == 'OBS'

            self.mask_rest_frame = mask[select_rest_frame_mask]
            self.mask_obs_frame = mask[select_obs_mask]

            self._obs_log_wave_min = np.asarray(self.mask_obs_frame['log_wave_min'])
            self._obs_log_wave_max = np.asarray(self.mask_obs_frame['log_wave_max'])
            self._rest_log_wave_min = np.asarray(self.mask_rest_frame['log_wave_min'])
            self._rest_log_wave_max = np.asarray(self.mask_rest_frame['log_wave_max'])
        except (OSError, ValueError) as error:
            raise MaskError(
                "Error loading SkyMask. Unable to read mask file. "
                f"File {mask_file}"
            ) from error

    def apply_mask(self, forest):
        """Apply the mask. The mask is done by removing the affected
        pixels from the arrays in Forest.mask_fields

        Arguments
        ---------
        forest: Forest
        A Forest instance to which the correction is applied

        Raise
        -----
        CorrectionError if Forest.wave_solution is not 'lin' or 'log'
        """
        # find masking array
        deltas = _create_deltas(forest.log_lambda.size)

        if self._obs_log_wave_min.size > 0:
            _add_mask_intervals(
                deltas,
                forest.log_lambda,
                self._obs_log_wave_min,
                self._obs_log_wave_max,
            )

        if self._rest_log_wave_min.size > 0:
            log_lambda_rest_frame = forest.log_lambda - np.log10(1.0 + forest.z)
            _add_mask_intervals(
                deltas,
                log_lambda_rest_frame,
                self._rest_log_wave_min,
                self._rest_log_wave_max,
            )

        w = _finalize_deltas_to_mask(deltas)

        # do the actual masking
        for param in Forest.mask_fields:
            self._masker(forest, param, w)
