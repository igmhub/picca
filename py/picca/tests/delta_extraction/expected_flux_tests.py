"""This file contains tests related to ExpectedFlux and its childs"""
from configparser import ConfigParser
import copy
import os
import unittest

import numpy as np
from scipy.interpolate import interp1d

from picca.delta_extraction.errors import ExpectedFluxError
from picca.delta_extraction.data_catalogues.sdss_data import SdssData
from picca.delta_extraction.data_catalogues.sdss_data import defaults as defaults_sdss_data
from picca.delta_extraction.expected_flux import ExpectedFlux
from picca.delta_extraction.expected_fluxes.dr16_expected_flux import Dr16ExpectedFlux
from picca.delta_extraction.expected_fluxes.dr16_expected_flux import (
    defaults as defaults_dr16_expected_flux)
from picca.tests.delta_extraction.abstract_test import AbstractTest
from picca.tests.delta_extraction.test_utils import forest1
from picca.tests.delta_extraction.test_utils import setup_forest, reset_forest
from picca.tests.delta_extraction.test_utils import sdss_data_kwargs

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class ExpectedFluxTest(AbstractTest):
    """Test class ExpectedFlux and its childs.

    Methods
    -------
    compare_ascii (from AbstractTest)
    compare_fits (from AbstractTest)
    setUp
    tearDown
    test_expected_flux
    test_dr16_expected_flux
    test_dr16_expected_flux_compute_continuum
    test_dr16_expected_flux_compute_delta_stack
    test_dr16_expected_flux_compute_expected_flux
    test_dr16_expected_flux_compute_mean_cont_lin
    test_dr16_expected_flux_compute_mean_cont_log
    test_dr16_expected_flux_compute_var_stats
    test_dr16_expected_flux_populate_los_ids
    test_dr16_expected_flux_save_iteration_step
    """
    def setUp(self):
        reset_forest()
        super().setUp()

    def tearDown(self):
        reset_forest()

    def test_dr16_expected_flux(self):
        """Test constructor for class Dr16ExpectedFlux

        Load an Dr16ExpectedFlux instance.
        """
        config = ConfigParser()
        config.read_dict(
            {"expected flux": {
                "iter out prefix": f"{THIS_DIR}/results/iter_out_prefix",
                "out dir": f"{THIS_DIR}/results",
                "num processors": 1
            }})
        for key, value in defaults_dr16_expected_flux.items():
            if key not in config["expected flux"]:
                config["expected flux"][key] = str(value)
        # this should raise an error as iter out prefix should not have a folder
        with self.assertRaises(ExpectedFluxError):
            expected_flux = Dr16ExpectedFlux(config["expected flux"])

        config = ConfigParser()
        config.read_dict(
            {"expected flux": {
                "iter out prefix": "iter_out_prefix",
                "out dir": f"{THIS_DIR}/results/",
                "num processors": 1
            }})
        for key, value in defaults_dr16_expected_flux.items():
            if key not in config["expected flux"]:
                config["expected flux"][key] = str(value)
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
        self.assertTrue(isinstance(expected_flux.log_lambda, np.ndarray))

        # setup Forest variables; case: linear wavelength solution
        reset_forest()
        setup_forest("lin")
        expected_flux = Dr16ExpectedFlux(config["expected flux"])

        self.assertTrue(isinstance(expected_flux.get_eta, interp1d))
        self.assertTrue(isinstance(expected_flux.get_fudge, interp1d))
        self.assertTrue(isinstance(expected_flux.get_mean_cont, interp1d))
        self.assertTrue(isinstance(expected_flux.get_var_lss, interp1d))
        self.assertTrue(isinstance(expected_flux.lambda_, np.ndarray))
        self.assertTrue(expected_flux.log_lambda is None)

    def test_dr16_expected_flux_compute_continuum(self):
        """Test method compute_continuum for class Dr16ExpectedFlux"""
        # setup Forest variables; case: logarithmic wavelength solution
        setup_forest("log")

        out_file = f"{THIS_DIR}/results/continua_log.txt"
        test_file = f"{THIS_DIR}/data/continua_log.txt"

        # initialize Data and Dr16ExpectedFlux instances
        config = ConfigParser()
        config.read_dict({
            "data": sdss_data_kwargs,
            "expected flux": {
                "iter out prefix": "iter_out_prefix",
                "out dir": f"{THIS_DIR}/results/",
                "num processors": 1
            },
        })
        for key, value in defaults_dr16_expected_flux.items():
            if key not in config["expected flux"]:
                config["expected flux"][key] = str(value)
        for key, value in defaults_sdss_data.items():
            if key not in config["data"]:
                config["data"][key] = str(value)
        data = SdssData(config["data"])
        expected_flux = Dr16ExpectedFlux(config["expected flux"])

        # compute the forest continua
        for forest in data.forests:
            expected_flux.compute_continuum(forest)

        # save the results
        f = open(out_file, "w")
        f.write("# thingid cont[0] ... cont[N]\n")
        for forest in data.forests:
            f.write(f"{forest.los_id} ")
            for item in forest.continuum:
                f.write(f"{item} ")
            f.write("\n")
        f.close()

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
            if not np.allclose(forest.continuum, continua.get(forest.los_id)):
                print("Difference found in forest.continuum")
                print(f"forest.los_id: {forest.los_id}")
                print(f"result test are_close result-test")
                for i1, i2 in zip(forest.continuum, continua.get(forest.los_id)):
                    print(i1, i2, np.isclose(i1, i2), i1-i2)
            self.assertTrue(
                np.allclose(forest.continuum, continua.get(forest.los_id)))
            correct_forests += 1

        # check that we loaded all quasars
        self.assertTrue(correct_forests == len(continua))

        # setup Forest variables; case: linear wavelength solution
        reset_forest()
        setup_forest("lin")
        #TODO: compute_continuum: add linear wavelength solution test

    def test_dr16_expected_flux_compute_delta_stack(self):
        """Test method compute_delta_stack for class Dr16ExpectedFlux"""
        # setup Forest variables; case: logarithmic wavelength solution
        setup_forest("log")

        out_file = f"{THIS_DIR}/results/delta_stack_log.txt"
        test_file = f"{THIS_DIR}/data/delta_stack_log.txt"

        # initialize Data and Dr16ExpectedFlux instances
        config = ConfigParser()
        config.read_dict({
            "data": sdss_data_kwargs,
            "expected flux": {
                "iter out prefix": "iter_out_prefix",
                "out dir": f"{THIS_DIR}/results/",
                "num processors": 1
            },
        })
        for key, value in defaults_dr16_expected_flux.items():
            if key not in config["expected flux"]:
                config["expected flux"][key] = str(value)
        for key, value in defaults_sdss_data.items():
            if key not in config["data"]:
                config["data"][key] = str(value)
        data = SdssData(config["data"])
        expected_flux = Dr16ExpectedFlux(config["expected flux"])

        # compute the forest continua
        for forest in data.forests:
            expected_flux.compute_continuum(forest)

        # compute variance functions and statistics
        expected_flux.compute_delta_stack(data.forests)

        # save results
        f = open(out_file, "w")
        f.write("# log_lambda delta\n")
        for log_lambda in np.arange(3.5563025, 3.7123025 + 3e-4, 3e-4):
            f.write(f"{log_lambda} {expected_flux.get_stack_delta(log_lambda)}\n")
        f.close()

        # load expected delta stack
        expectations = np.genfromtxt(test_file, names=True)

        # compare with obtained results
        stack_delta = expected_flux.get_stack_delta(expectations["log_lambda"])
        if not np.allclose(stack_delta, expectations["delta"]):
            print(f"\nOriginal file: {test_file}")
            print(f"New file: {out_file}")
            print("Difference found in delta stack")
            print(f"result test are_close result-test")
            for i1, i2 in zip(stack_delta, expectations["delta"]):
                print(i1, i2, np.isclose(i1, i2), i1-i2)
        self.assertTrue(np.allclose(stack_delta, expectations["delta"]))

        # setup Forest variables; case: linear wavelength solution
        reset_forest()
        setup_forest("lin")

        # TODO: compute_delta_stack: add linear wavelength solution test

    def test_dr16_expected_flux_compute_expected_flux(self):
        """Test method compute_var_stats for class Dr16ExpectedFlux"""
        # setup Forest variables; case: logarithmic wavelength solution
        setup_forest("log")

        out_file = f"{THIS_DIR}/results/Log/iter_out_prefix_compute_expected_flux_log.fits.gz"
        test_file = f"{THIS_DIR}/data/iter_out_prefix_compute_expected_flux_log.fits.gz"

        # initialize Data and Dr16ExpectedFlux instances
        config = ConfigParser()
        config.read_dict({
            "data": sdss_data_kwargs,
            "expected flux": {
                "iter out prefix": "iter_out_prefix_compute_expected_flux_log",
                "out dir": f"{THIS_DIR}/results/",
                "num processors": 1
            },
        })
        for key, value in defaults_dr16_expected_flux.items():
            if key not in config["expected flux"]:
                config["expected flux"][key] = str(value)
        for key, value in defaults_sdss_data.items():
            if key not in config["data"]:
                config["data"][key] = str(value)
        data = SdssData(config["data"])
        expected_flux = Dr16ExpectedFlux(config["expected flux"])

        # compute the expected flux
        expected_flux.compute_expected_flux(data.forests)

        # check the results
        for iteration in range(1, 5):
            self.compare_fits(
                test_file.replace(".fits", f"_iteration{iteration}.fits"),
                out_file.replace(".fits", f"_iteration{iteration}.fits"))
        self.compare_fits(test_file, out_file)

        # setup Forest variables; case: linear wavelength solution
        reset_forest()
        setup_forest("lin")
        #TODO: compute_expected_flux: add linear wavelength solution test

    def test_dr16_expected_flux_compute_mean_cont_lin(self):
        """Test method compute_mean_cont_lin for class Dr16ExpectedFlux"""
        # setup Forest variables; case: logarithmic wavelength solution
        setup_forest("lin")

        #TODO: add test: compute_mean_cont_lin
        with self.assertRaises(Exception):
            raise NotImplementedError()

    def test_dr16_expected_flux_compute_mean_cont_log(self):
        """Test method compute_mean_cont_log for class Dr16ExpectedFlux"""
        # setup Forest variables; case: logarithmic wavelength solution
        setup_forest("log")

        out_file = f"{THIS_DIR}/results/mean_cont_log.txt"
        test_file = f"{THIS_DIR}/data/mean_cont_log.txt"

        # initialize Data and Dr16ExpectedFlux instances
        config = ConfigParser()
        config.read_dict({
            "data": sdss_data_kwargs,
            "expected flux": {
                "iter out prefix": "iter_out_prefix",
                "out dir": f"{THIS_DIR}/results/",
                "num processors": 1
            },
        })
        for key, value in defaults_dr16_expected_flux.items():
            if key not in config["expected flux"]:
                config["expected flux"][key] = str(value)
        for key, value in defaults_sdss_data.items():
            if key not in config["data"]:
                config["data"][key] = str(value)
        data = SdssData(config["data"])
        expected_flux = Dr16ExpectedFlux(config["expected flux"])

        # compute the forest continua
        for forest in data.forests:
            expected_flux.compute_continuum(forest)

        # compute mean quasar continuum
        expected_flux.compute_mean_cont_log(data.forests)

        # save results
        f = open(out_file, "w")
        f.write("# log_lambda delta\n")
        for log_lambda in np.arange(3.0171, 3.079 + 3e-4, 3e-4):
            f.write(f"{log_lambda} {expected_flux.get_mean_cont(log_lambda)}\n")
        f.close()

        # load the expected results
        expectations = np.genfromtxt(test_file, names=True)
        f = open(f"{THIS_DIR}/results/mean_cont_log.txt", "w")
        f.write("# log_lambda mean_cont\n")
        for item in expectations["log_lambda"]:
            f.write(f"{item} {expected_flux.get_mean_cont(item)}\n")
        f.close()

        # compare with obtained results
        mean_cont = expected_flux.get_mean_cont(expectations["log_lambda"])
        if not np.allclose(mean_cont, expectations["mean_cont"]):
            print(f"\nOriginal file: {test_file}")
            print(f"New file: {out_file}")
            print("Difference found in mean_cont")
            print(f"result test are_close result-test")
            for i1, i2 in zip(mean_cont, expectations["mean_cont"]):
                print(i1, i2, np.isclose(i1, i2), i1-i2)
        self.assertTrue(np.allclose(mean_cont, expectations["mean_cont"]))

    def test_dr16_expected_flux_compute_var_stats(self):
        """Test method compute_var_stats for class Dr16ExpectedFlux"""
        # setup Forest variables; case: logarithmic wavelength solution
        setup_forest("log")

        test_file = f"{THIS_DIR}/data/eta_var_lss_fudge.txt"

        # initialize Data and Dr16ExpectedFlux instances
        config = ConfigParser()
        config.read_dict({
            "data": sdss_data_kwargs,
            "expected flux": {
                "iter out prefix": "iter_out_prefix",
                "out dir": f"{THIS_DIR}/results/",
                "num processors": 1
            },
        })
        for key, value in defaults_dr16_expected_flux.items():
            if key not in config["expected flux"]:
                config["expected flux"][key] = str(value)
        for key, value in defaults_sdss_data.items():
            if key not in config["data"]:
                config["data"][key] = str(value)
        data = SdssData(config["data"])
        expected_flux = Dr16ExpectedFlux(config["expected flux"])

        # compute the forest continua
        for forest in data.forests:
            expected_flux.compute_continuum(forest)

        # compute variance functions and statistics
        expected_flux.compute_var_stats(data.forests)

        # load the expected results
        expectations = np.genfromtxt(test_file, names=True)

        # compare with obtained results
        eta = expected_flux.get_eta(expectations["log_lambda"])
        var_lss = expected_flux.get_var_lss(expectations["log_lambda"])
        fudge = expected_flux.get_fudge(expectations["log_lambda"])
        self.assertTrue(np.allclose(eta, expectations["eta"]))
        self.assertTrue(np.allclose(var_lss, expectations["var_lss"]))
        self.assertTrue(np.allclose(fudge, expectations["fudge"]))

        # setup Forest variables; case: linear wavelength solution
        reset_forest()
        setup_forest("lin")
        #TODO: compute_var_stats: add linear wavelength solution test

    def test_dr16_expected_flux_populate_los_ids(self):
        """Test method populate_los_ids for class Dr16ExpectedFlux"""
        # setup Forest variables; case: logarithmic wavelength solution
        setup_forest("log")

        # initialize Data and Dr16ExpectedFlux instances
        config = ConfigParser()
        config.read_dict({
            "data": sdss_data_kwargs,
            "expected flux": {
                "iter out prefix": "iter_out_prefix_log",
                "out dir": f"{THIS_DIR}/results/",
                "num processors": 1
            },
        })
        for key, value in defaults_dr16_expected_flux.items():
            if key not in config["expected flux"]:
                config["expected flux"][key] = str(value)
        for key, value in defaults_sdss_data.items():
            if key not in config["data"]:
                config["data"][key] = str(value)
        data = SdssData(config["data"])
        expected_flux = Dr16ExpectedFlux(config["expected flux"])

        # compute the forest continua
        for forest in data.forests:
            expected_flux.compute_continuum(forest)

        # compute variance functions and statistics
        expected_flux.compute_delta_stack(data.forests)

        # save iter_out_prefix for iteration 0
        expected_flux.populate_los_ids(data.forests)

        # TODO: populate_los_ids: check the results
        with self.assertRaises(Exception):
            raise NotImplementedError()

    def test_dr16_expected_flux_save_iteration_step(self):
        """Test method save_iteration_step for class Dr16ExpectedFlux"""
        # setup Forest variables; case: logarithmic wavelength solution
        setup_forest("log")

        out_file = f"{THIS_DIR}/results/Log/iter_out_prefix_log_iteration1.fits.gz"
        out_file2 = f"{THIS_DIR}/results/Log/iter_out_prefix_log.fits.gz"
        test_file = f"{THIS_DIR}/data/iter_out_prefix_log_iteration1.fits.gz"
        test_file2 = f"{THIS_DIR}/data/iter_out_prefix_log.fits.gz"

        # initialize Data and Dr16ExpectedFlux instances
        config = ConfigParser()
        config.read_dict({
            "data": sdss_data_kwargs,
            "expected flux": {
                "iter out prefix": "iter_out_prefix_log",
                "out dir": f"{THIS_DIR}/results/",
                "num processors": 1
            },
        })
        for key, value in defaults_dr16_expected_flux.items():
            if key not in config["expected flux"]:
                config["expected flux"][key] = str(value)
        for key, value in defaults_sdss_data.items():
            if key not in config["data"]:
                config["data"][key] = str(value)
        data = SdssData(config["data"])
        expected_flux = Dr16ExpectedFlux(config["expected flux"])

        # compute the forest continua
        for forest in data.forests:
            expected_flux.compute_continuum(forest)

        # compute variance functions and statistics
        expected_flux.compute_delta_stack(data.forests)

        # save iter_out_prefix for iteration 0
        expected_flux.save_iteration_step(0)
        self.compare_fits(test_file, out_file)

        # save iter_out_prefix for final iteration
        expected_flux.save_iteration_step(-1)
        self.compare_fits(test_file2, out_file2)

        # setup Forest variables; case: linear wavelength solution
        setup_forest("lin")

        out_file = f"{THIS_DIR}/results/iter_out_prefix_lin_iteration1.fits.gz"
        out_file2 = f"{THIS_DIR}/results/iter_out_prefix_lin.fits.gz"
        test_file = f"{THIS_DIR}/data/iter_out_prefix_lin_iteration1.fits.gz"
        test_file2 = f"{THIS_DIR}/data/iter_out_prefix_lin.fits.gz"
        out_dir = f"{THIS_DIR}/results/"

        # TODO: save_iteration_step: add linear wavelength solution test

    def test_expected_flux(self):
        """Test Abstract class ExpectedFlux

        Load an ExpectedFlux instance.
        """
        # initialize ExpectedFlux instance
        config = ConfigParser()
        config.read_dict({
            "expected flux": {
                "out dir": f"{THIS_DIR}/results/",
                "num processors": 1
            },
        })
        expected_flux = ExpectedFlux(config["expected flux"])

        # compute_expected_flux should not be defined
        with self.assertRaises(ExpectedFluxError):
            expected_flux.compute_expected_flux([], "")

        forest = copy.deepcopy(forest1)
        self.assertTrue(forest.deltas is None)

        # los_id not in dictionary: extract deltas does nothing
        expected_flux.extract_deltas(forest)
        self.assertTrue(forest.deltas is None)

        # los_id in dictionary: extract deltas
        expected_flux.los_ids = {
            forest1.los_id: {
                "mean expected flux": np.ones_like(forest1.flux),
                "weights": np.ones_like(forest1.flux)
            }
        }
        expected_flux.extract_deltas(forest)
        self.assertTrue(all(forest.deltas == np.zeros_like(forest1.flux)))


if __name__ == '__main__':
    unittest.main()
