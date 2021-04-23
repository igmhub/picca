"""This file contains tests related to Data and its childs"""
import os
import unittest
import copy
import numpy as np
from scipy.interpolate import interp1d
from configparser import ConfigParser

from picca.delta_extraction.errors import ExpectedFluxError
from picca.delta_extraction.expected_flux import ExpectedFlux
from picca.delta_extraction.data_catalogues.sdss_data import SdssData
from picca.delta_extraction.expected_fluxes.dr16_expected_flux import Dr16ExpectedFlux
from picca.delta_extraction.tests.abstract_test import AbstractTest
from picca.delta_extraction.tests.test_utils import forest1
from picca.delta_extraction.tests.test_utils import setup_forest, reset_forest

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class ExpectedFluxTest(AbstractTest):
    """Test class ExpectedFlux and its childs."""

    def setUp(self):
        reset_forest()

    def tearDown(self):
        reset_forest()

    def test_expected_flux(self):
        """Test Abstract class ExpectedFlux

        Load an ExpectedFlux instance.
        """
        expected_flux = ExpectedFlux()

        # compute_expected_flux should not be defined
        with self.assertRaises(ExpectedFluxError):
            expected_flux.compute_expected_flux([])

        forest = copy.deepcopy(forest1)
        self.assertTrue(forest.deltas is None)

        # los_id not in dictionary: extract deltas does nothing
        expected_flux.extract_deltas(forest)
        self.assertTrue(forest.deltas is None)

        # los_id in dictionary: extract deltas
        expected_flux.los_ids = {forest1.los_id: {"mean expected flux": np.ones_like(forest1.flux),
                                                  "weights": np.ones_like(forest1.flux)}}
        expected_flux.extract_deltas(forest)
        self.assertTrue(all(forest.deltas == np.zeros_like(forest1.flux)))

    def test_dr16_expected_flux(self):
        """Test constructor for class Dr16ExpectedFlux

        Load an Dr16ExpectedFlux instance.
        """
        config = ConfigParser()
        config.read_dict({"expected flux":
            {"iter out prefix": "results/iter_out_prefix"}})
        # this should raise an error as iter out prefix should not have a folder
        with self.assertRaises(ExpectedFluxError):
            expected_flux = Dr16ExpectedFlux(config["expected flux"])

        config = ConfigParser()
        config.read_dict({"expected flux": {"iter out prefix": "iter_out_prefix"}})
        # this should also raise an error as Forest variables are not defined
        with self.assertRaises(ExpectedFluxError):
            expected_flux = Dr16ExpectedFlux(config["expected flux"])

        # setup Forest variables; case: logarithmic wavelength solution
        setup_forest("log")
        expected_flux = Dr16ExpectedFlux(config["expected flux"])

        self.assertTrue(isinstance(expected_flux.get_eta, interp1d))
        self.assertTrue(isinstance(expected_flux.get_fudge, interp1d))
        self.assertTrue(isinstance(expected_flux.get_mean_cont, interp1d))
        self.assertTrue(isinstance(expected_flux.get_var_lss, interp1d))
        self.assertTrue(expected_flux.lambda_ is None)
        self.assertTrue(expected_flux.lambda_rest_frame is None)
        self.assertTrue(isinstance(expected_flux.log_lambda, np.ndarray))
        self.assertTrue(isinstance(expected_flux.log_lambda_rest_frame, np.ndarray))

        # setup Forest variables; case: linear wavelength solution
        reset_forest()
        setup_forest("lin")
        expected_flux = Dr16ExpectedFlux(config["expected flux"])

        self.assertTrue(isinstance(expected_flux.get_eta, interp1d))
        self.assertTrue(isinstance(expected_flux.get_fudge, interp1d))
        self.assertTrue(isinstance(expected_flux.get_mean_cont, interp1d))
        self.assertTrue(isinstance(expected_flux.get_var_lss, interp1d))
        self.assertTrue(isinstance(expected_flux.lambda_, np.ndarray))
        self.assertTrue(isinstance(expected_flux.lambda_rest_frame, np.ndarray))
        self.assertTrue(expected_flux.log_lambda is None)
        self.assertTrue(expected_flux.log_lambda_rest_frame is None)

    def test_dr16_expected_flux_compute_continuum(self):
        """Test method compute_continuum for class Dr16ExpectedFlux"""
        # setup Forest variables; case: logarithmic wavelength solution
        setup_forest("log")

        test_file = f"{THIS_DIR}/data/continua_log.txt"

        # initialize Data and Dr16ExpectedFlux instances
        config = ConfigParser()
        config.read_dict({
        "data": {
            "input directory":
                f"{THIS_DIR}/data",
            "output directory":
                f"{THIS_DIR}/results",
            "drq catalogue":
                f"{THIS_DIR}/data/cat_for_clustering_plate3655.fits.gz",
        },
        "expected flux": {"iter out prefix": "iter_out_prefix"},
        })
        data = SdssData(config["data"])
        expected_flux = Dr16ExpectedFlux(config["expected flux"])

        # compute the forest continua
        for forest in data.forests:
            expected_flux.compute_continuum(forest)

        # load expected forest continua
        continua = {}
        f = open(test_file)
        for line in f.readlines():
            if line.startswith("#"):
                continue
            cols = line.split()
            los_id = int(cols[0])
            continuum = np.array([float(item) for item in cols[1:]])
            continua[los_id] = continuum
        f.close()

        # compare the results
        correct_forests = 0
        for forest in data.forests:
            self.assertTrue(np.allclose(forest.continuum,
                                        continua.get(forest.los_id)))
            correct_forests += 1

        # check that we loaded all quasars
        self.assertTrue(correct_forests == len(continua))

        # setup Forest variables; case: linear wavelength solution
        reset_forest()
        setup_forest("lin")
        #TODO: add linear wavelength solution test

    def test_dr16_expected_flux_compute_delta_stack(self):
        """Test method compute_delta_stack for class Dr16ExpectedFlux"""
        # setup Forest variables; case: logarithmic wavelength solution
        setup_forest("log")

        test_file = f"{THIS_DIR}/data/delta_stack_log.txt"

        # initialize Data and Dr16ExpectedFlux instances
        config = ConfigParser()
        config.read_dict({
        "data": {
            "input directory":
                f"{THIS_DIR}/data",
            "output directory":
                f"{THIS_DIR}/results",
            "drq catalogue":
                f"{THIS_DIR}/data/cat_for_clustering_plate3655.fits.gz",
        },
        "expected flux": {"iter out prefix": "iter_out_prefix"},
        })
        data = SdssData(config["data"])
        expected_flux = Dr16ExpectedFlux(config["expected flux"])

        # compute the forest continua
        for forest in data.forests:
            expected_flux.compute_continuum(forest)

        # compute variance functions and statistics
        expected_flux.compute_delta_stack(data.forests)

        # load expected delta stack
        expectations = np.genfromtxt(test_file, names=True)

        # compare with obtained results
        stack_delta = expected_flux.get_stack_delta(expectations["log_lambda"])
        self.assertTrue(np.allclose(stack_delta, expectations["delta"]))

    def test_dr16_expected_flux_compute_mean_cont_lin(self):
        """Test method compute_mean_cont_lin for class Dr16ExpectedFlux"""
        # setup Forest variables; case: logarithmic wavelength solution
        setup_forest("lin")
        #TODO: add test

    def test_dr16_expected_flux_compute_mean_cont_log(self):
        """Test method compute_mean_cont_log for class Dr16ExpectedFlux"""
        # setup Forest variables; case: logarithmic wavelength solution
        setup_forest("log")

        test_file = f"{THIS_DIR}/data/mean_cont_log.txt"

        # initialize Data and Dr16ExpectedFlux instances
        config = ConfigParser()
        config.read_dict({
        "data": {
            "input directory":
                f"{THIS_DIR}/data",
            "output directory":
                f"{THIS_DIR}/results",
            "drq catalogue":
                f"{THIS_DIR}/data/cat_for_clustering_plate3655.fits.gz",
        },
        "expected flux": {"iter out prefix": "iter_out_prefix"},
        })
        data = SdssData(config["data"])
        expected_flux = Dr16ExpectedFlux(config["expected flux"])

        # compute the forest continua
        for forest in data.forests:
            expected_flux.compute_continuum(forest)

        # compute variance functions and statistics
        expected_flux.compute_mean_cont_log(data.forests)

        # load the expected results
        expectations = np.genfromtxt(test_file, names=True)

        # compare with obtained results
        mean_cont = expected_flux.get_mean_cont(expectations["log_lambda"])
        self.assertTrue(np.allclose(mean_cont, expectations["mean_cont"]))

    def test_dr16_expected_flux_compute_expected_flux(self):
        """Test method compute_var_stats for class Dr16ExpectedFlux"""
        # setup Forest variables; case: logarithmic wavelength solution
        setup_forest("log")

        out_file = f"{THIS_DIR}/results/iter_out_prefix_compute_expected_flux_log.fits.gz"
        test_file = f"{THIS_DIR}/data/iter_out_prefix_compute_expected_flux_log.fits.gz"
        out_dir = "results/"

        # initialize Data and Dr16ExpectedFlux instances
        config = ConfigParser()
        config.read_dict({
        "data": {
            "input directory":
                f"{THIS_DIR}/data",
            "output directory":
                f"{THIS_DIR}/results",
            "drq catalogue":
                f"{THIS_DIR}/data/cat_for_clustering_plate3655.fits.gz",
        },
        "expected flux": {"iter out prefix": "iter_out_prefix_compute_expected_flux"},
        })
        data = SdssData(config["data"])
        expected_flux = Dr16ExpectedFlux(config["expected flux"])

        # compute the expected flux
        expected_flux.compute_expected_flux(data.forests, out_dir)

        #TODO: add some tests to check the results

        # setup Forest variables; case: linear wavelength solution
        reset_forest()
        setup_forest("lin")
        #TODO: add linear wavelength solution test

    def test_dr16_expected_flux_compute_var_stats(self):
        """Test method compute_var_stats for class Dr16ExpectedFlux"""
        # setup Forest variables; case: logarithmic wavelength solution
        setup_forest("log")

        # initialize Data and Dr16ExpectedFlux instances
        config = ConfigParser()
        config.read_dict({
        "data": {
            "input directory":
                f"{THIS_DIR}/data",
            "output directory":
                f"{THIS_DIR}/results",
            "drq catalogue":
                f"{THIS_DIR}/data/cat_for_clustering_plate3655.fits.gz",
        },
        "expected flux": {"iter out prefix": "iter_out_prefix"},
        })
        data = SdssData(config["data"])
        expected_flux = Dr16ExpectedFlux(config["expected flux"])

        # compute the forest continua
        for forest in data.forests:
            expected_flux.compute_continuum(forest)

        # compute variance functions and statistics
        expected_flux.compute_var_stats(data.forests)

        #TODO: add some tests to check the results

        # setup Forest variables; case: linear wavelength solution
        reset_forest()
        setup_forest("lin")
        #TODO: add linear wavelength solution test

    def test_dr16_expected_flux_populate_los_ids(self):
        """Test method populate_los_ids for class Dr16ExpectedFlux"""
        # setup Forest variables; case: logarithmic wavelength solution
        setup_forest("log")

        out_file = f"{THIS_DIR}/results/iter_out_prefix_log_iteration0.fits.gz"
        out_file2 = f"{THIS_DIR}/results/iter_out_prefix_log.fits.gz"
        test_file = f"{THIS_DIR}/data/iter_out_prefix_log_iteration0.fits.gz"
        test_file2 = f"{THIS_DIR}/data/iter_out_prefix_log.fits.gz"
        out_dir = "results/"

        # initialize Data and Dr16ExpectedFlux instances
        config = ConfigParser()
        config.read_dict({
        "data": {
            "input directory":
                f"{THIS_DIR}/data",
            "output directory":
                f"{THIS_DIR}/results",
            "drq catalogue":
                f"{THIS_DIR}/data/cat_for_clustering_plate3655.fits.gz",
        },
        "expected flux": {"iter out prefix": "iter_out_prefix_log"},
        })
        data = SdssData(config["data"])
        expected_flux = Dr16ExpectedFlux(config["expected flux"])

        # compute the forest continua
        for forest in data.forests:
            expected_flux.compute_continuum(forest)

        # compute variance functions and statistics
        expected_flux.compute_delta_stack(data.forests)

        # save iter_out_prefix for iteration 0
        expected_flux.populate_los_ids(data.forests)

        # TODO: check the results

    def test_dr16_expected_flux_save_iteration_step(self):
        """Test method save_iteration_step for class Dr16ExpectedFlux"""
        # setup Forest variables; case: logarithmic wavelength solution
        setup_forest("log")

        out_file = f"{THIS_DIR}/results/iter_out_prefix_log_iteration0.fits.gz"
        out_file2 = f"{THIS_DIR}/results/iter_out_prefix_log.fits.gz"
        test_file = f"{THIS_DIR}/data/iter_out_prefix_log_iteration0.fits.gz"
        test_file2 = f"{THIS_DIR}/data/iter_out_prefix_log.fits.gz"
        out_dir = "results/"

        # initialize Data and Dr16ExpectedFlux instances
        config = ConfigParser()
        config.read_dict({
        "data": {
            "input directory":
                f"{THIS_DIR}/data",
            "output directory":
                f"{THIS_DIR}/results",
            "drq catalogue":
                f"{THIS_DIR}/data/cat_for_clustering_plate3655.fits.gz",
        },
        "expected flux": {"iter out prefix": "iter_out_prefix_log"},
        })
        data = SdssData(config["data"])
        expected_flux = Dr16ExpectedFlux(config["expected flux"])

        # compute the forest continua
        for forest in data.forests:
            expected_flux.compute_continuum(forest)

        # compute variance functions and statistics
        expected_flux.compute_delta_stack(data.forests)

        # save iter_out_prefix for iteration 0
        expected_flux.save_iteration_step(0, out_dir)
        self.compare_fits(out_file, test_file)

        # save iter_out_prefix for final iteration
        expected_flux.save_iteration_step(-1, out_dir)
        self.compare_fits(out_file2, test_file2)

        # setup Forest variables; case: linear wavelength solution
        setup_forest("lin")

        out_file = f"{THIS_DIR}/results/iter_out_prefix_lin_iteration0.fits.gz"
        out_file2 = f"{THIS_DIR}/results/iter_out_prefix_lin.fits.gz"
        test_file = f"{THIS_DIR}/data/iter_out_prefix_lin_iteration0.fits.gz"
        test_file2 = f"{THIS_DIR}/data/iter_out_prefix_lin.fits.gz"
        out_dir = "results/"

        # TODO: add test

if __name__ == '__main__':
    unittest.main()
