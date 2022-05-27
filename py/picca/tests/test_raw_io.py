"""
Test module for picca.raw_io
"""

import unittest
from pathlib import Path
from picca.tests.delta_extraction.abstract_test import AbstractTest
from picca.raw_io import read_transmission_file, convert_transmission_to_deltas
import numpy as np

THIS_DIR = Path(__file__).parent

class TestRaw(AbstractTest):
    """Test module picca.raw_io

    Methods
    -------

    """

    def test_read_transmission_file(self):
        data_dir = THIS_DIR / "data" / "test_raw_io"
        filename = (
            data_dir / "0" / "0" / "transmission-16-0.fits.gz"
        )

        deltas, stack_flux, stack_weight = read_transmission_file(
            filename=str(filename),
            objs_thingid=[
                60298, 59167, 60909, 59973, 60602,  60908, 
                60295, 59410, 60611, 60629, 0, 1,
                ],
            num_bins=475,
            lambda_min=3600.,
            lambda_max=5500.,
            delta_lambda=4,
            lin_spaced=True,)

        deltas_values = np.concatenate([delta.delta for delta in deltas['0']])
        
        deltas_target = np.loadtxt(data_dir / 'deltas.txt')
        stack_flux_target = np.loadtxt(data_dir / 'stack_flux.txt')
        stack_weight_target = np.loadtxt(data_dir / 'stack_weight.txt')

        np.testing.assert_almost_equal(deltas_values, deltas_target)
        np.testing.assert_almost_equal(stack_flux, stack_flux_target)
        np.testing.assert_almost_equal(stack_weight, stack_weight_target)

    def test_convert_transmission_to_deltas(self):
        data_dir = THIS_DIR / "data" / "test_raw_io"
        filename = (
            data_dir / "0" / "0" / "transmission-16-0.fits.gz"
        )
        results = THIS_DIR / 'results' / 'raw'
        results.mkdir(parents=True, exist_ok=True)

        convert_transmission_to_deltas(
            obj_path = str(data_dir / 'test_master.fits'),
            out_dir = str(results),
            in_dir = str(data_dir))

        self.compare_fits(
            data_dir / 'raw-stats.fits.gz',
            results.parent / 'raw-stats.fits.gz'
        )

        self.compare_fits(
            data_dir / 'delta-1448.fits.gz',
            results / 'delta-1448.fits.gz',
        )
