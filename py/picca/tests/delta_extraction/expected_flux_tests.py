"""This file contains tests related to ExpectedFlux and its childs"""
from configparser import ConfigParser
import copy
import os
import unittest

import numpy as np
from scipy.interpolate import interp1d

from pathlib import Path
from picca.delta_extraction.errors import ExpectedFluxError
from picca.delta_extraction.data_catalogues.desi_healpix import DesiHealpix
from picca.delta_extraction.data_catalogues.desi_healpix import defaults as defaults_desi_healpix
from picca.delta_extraction.data_catalogues.sdss_data import SdssData
from picca.delta_extraction.data_catalogues.sdss_data import defaults as defaults_sdss_data
from picca.delta_extraction.data_catalogues.desisim_mocks import DesisimMocks
from picca.delta_extraction.data_catalogues.desisim_mocks import defaults as defaults_desisim_data
from picca.delta_extraction.expected_flux import ExpectedFlux
from picca.delta_extraction.expected_fluxes.dr16_expected_flux import Dr16ExpectedFlux
from picca.delta_extraction.expected_fluxes.dr16_expected_flux import (
    defaults as defaults_dr16_expected_flux)
from picca.delta_extraction.expected_fluxes.true_continuum import (
    TrueContinuum, defaults as defaults_true_continuum)
from picca.tests.delta_extraction.abstract_test import AbstractTest
from picca.tests.delta_extraction.test_utils import forest1
from picca.tests.delta_extraction.test_utils import setup_forest, reset_forest
from picca.tests.delta_extraction.test_utils import desi_healpix_kwargs
from picca.tests.delta_extraction.test_utils import sdss_data_kwargs
from picca.tests.delta_extraction.test_utils import desi_mock_data_kwargs

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
    def test_dr16_expected_flux(self):
        """Test constructor for class Dr16ExpectedFlux

        Load an Dr16ExpectedFlux instance.
        """
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
        # this should raise an error as Forest variables are not defined
        expected_message = (
            "Forest class variables need to be set before initializing "
            "variables here.")
        with self.assertRaises(ExpectedFluxError) as context_manager:
            expected_flux = Dr16ExpectedFlux(config["expected flux"])
        self.compare_error_message(context_manager, expected_message)

        # setup Forest variables; case: logarithmic wavelength solution
        setup_forest("log", rebin=3)
        expected_flux = Dr16ExpectedFlux(config["expected flux"])

        self.assertTrue(isinstance(expected_flux.get_eta, interp1d))
        self.assertTrue(isinstance(expected_flux.get_fudge, interp1d))
        self.assertTrue(isinstance(expected_flux.get_mean_cont, interp1d))
        self.assertTrue(isinstance(expected_flux.get_var_lss, interp1d))
        self.assertTrue(isinstance(expected_flux.log_lambda_var_func_grid, np.ndarray))

        # setup Forest variables; case: linear wavelength solution
        reset_forest()
        setup_forest("lin")
        expected_flux = Dr16ExpectedFlux(config["expected flux"])

        self.assertTrue(isinstance(expected_flux.get_eta, interp1d))
        self.assertTrue(isinstance(expected_flux.get_fudge, interp1d))
        self.assertTrue(isinstance(expected_flux.get_mean_cont, interp1d))
        self.assertTrue(isinstance(expected_flux.get_var_lss, interp1d))
        self.assertTrue(isinstance(expected_flux.log_lambda_var_func_grid, np.ndarray))

    def test_dr16_expected_flux_compute_continuum(self):
        """Test method compute_continuum for class Dr16ExpectedFlux"""
        # setup Forest variables; case: logarithmic wavelength solution
        setup_forest("log", rebin=3)

        out_file = f"{THIS_DIR}/results/continua_log.txt"
        test_file = f"{THIS_DIR}/data/continua_log.txt"

        # initialize Data and Dr16ExpectedFlux instances
        config = ConfigParser()
        config.read_dict({
            "data": sdss_data_kwargs,
            "expected flux": {
                "iter out prefix": "iter_out_prefix",
                "out dir": f"{THIS_DIR}/results/",
                "num processors": 1,
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

        #self.compare_ascii(test_file, out_file)

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

        out_file = f"{THIS_DIR}/results/continua_lin.txt"
        test_file = f"{THIS_DIR}/data/continua_lin.txt"

        # initialize Data and Dr16ExpectedFlux instances
        config = ConfigParser()
        config.read_dict({
            "data": desi_healpix_kwargs,
            "expected flux": {
                "iter out prefix": "iter_out_prefix",
                "out dir": f"{THIS_DIR}/results/",
                "num processors": 1,
            },
        })
        for key, value in defaults_dr16_expected_flux.items():
            if key not in config["expected flux"]:
                config["expected flux"][key] = str(value)
        for key, value in defaults_desi_healpix.items():
            if key not in config["data"]:
                config["data"][key] = str(value)
        data = DesiHealpix(config["data"])
        expected_flux = Dr16ExpectedFlux(config["expected flux"])

        # compute the forest continua
        for forest in data.forests:
            expected_flux.compute_continuum(forest)

        # save the results
        f = open(out_file, "w")
        f.write("# thingid cont[0] ... cont[N]\n")
        for forest in data.forests:
            f.write(f"{forest.los_id} ")
            if forest.continuum is not None:
                for item in forest.continuum:
                    f.write(f"{item} ")
            f.write("\n")
        f.close()

        #self.compare_ascii(test_file, out_file)

        # load expected forest continua
        continua = {}
        f = open(test_file)
        for line in f.readlines():
            if line.startswith("#"):
                continue
            cols = line.split()
            los_id = int(cols[0])
            if len(cols) == 1:
                continuum = None
            else:
                continuum = np.array([float(item) for item in cols[1:]])
            continua[los_id] = continuum
        f.close()

        # compare the results
        correct_forests = 0
        for forest in data.forests:
            if forest.continuum is None:
                if continua.get(forest.los_id) is not None:
                    print(f"For forest with los_id {forest.los_id}, new continuum "
                          "is None. Expected continua:")
                    print(continua.get(forest.los_id))
                self.assertTrue(continua.get(forest.los_id) is None)
            elif continua.get(forest.los_id) is None:
                self.assertTrue(forest.continuum is None)
            else:
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

    def test_dr16_expected_flux_compute_delta_stack(self):
        """Test method compute_delta_stack for class Dr16ExpectedFlux"""
        # setup Forest variables; case: logarithmic wavelength solution
        setup_forest("log", rebin=3)

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

        out_file = f"{THIS_DIR}/results/delta_stack_lin.txt"
        test_file = f"{THIS_DIR}/data/delta_stack_lin.txt"

        # initialize Data and Dr16ExpectedFlux instances
        config = ConfigParser()
        config.read_dict({
            "data": desi_healpix_kwargs,
            "expected flux": {
                "iter out prefix": "iter_out_prefix",
                "out dir": f"{THIS_DIR}/results/",
                "num processors": 1,
            },
        })
        for key, value in defaults_dr16_expected_flux.items():
            if key not in config["expected flux"]:
                config["expected flux"][key] = str(value)
        for key, value in defaults_desi_healpix.items():
            if key not in config["data"]:
                config["data"][key] = str(value)
        data = DesiHealpix(config["data"])
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

    def test_dr16_expected_flux_compute_expected_flux(self):
        """Test method compute_var_stats for class Dr16ExpectedFlux"""
        # setup Forest variables; case: logarithmic wavelength solution
        setup_forest("log", rebin=3)

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
        self.compare_fits(test_file, out_file)        # setup Forest variables; case: linear wavelength solution
        reset_forest()
        setup_forest("lin")

        out_file = f"{THIS_DIR}/results/Log/iter_out_prefix_compute_expected_flux_lin.fits.gz"
        test_file = f"{THIS_DIR}/data/iter_out_prefix_compute_expected_flux_lin.fits.gz"

        # initialize Data and Dr16ExpectedFlux instances
        config = ConfigParser()
        config.read_dict({
            "data": desi_healpix_kwargs,
            "expected flux": {
                "iter out prefix": "iter_out_prefix_compute_expected_flux_lin",
                "out dir": f"{THIS_DIR}/results/",
                "num processors": 1,
            },
        })
        for key, value in defaults_dr16_expected_flux.items():
            if key not in config["expected flux"]:
                config["expected flux"][key] = str(value)
        for key, value in defaults_desi_healpix.items():
            if key not in config["data"]:
                config["data"][key] = str(value)
        data = DesiHealpix(config["data"])
        expected_flux = Dr16ExpectedFlux(config["expected flux"])

        # compute the expected flux
        expected_flux.compute_expected_flux(data.forests)

        # check the results
        for iteration in range(1, 5):
            self.compare_fits(
                test_file.replace(".fits", f"_iteration{iteration}.fits"),
                out_file.replace(".fits", f"_iteration{iteration}.fits"))
        self.compare_fits(test_file, out_file)

    def test_dr16_expected_flux_compute_mean_cont_lin(self):
        """Test method compute_mean_cont_lin for class Dr16ExpectedFlux"""
        # setup Forest variables; case: logarithmic wavelength solution
        setup_forest("lin")

        out_file = f"{THIS_DIR}/results/mean_cont_lin.txt"
        test_file = f"{THIS_DIR}/data/mean_cont_lin.txt"

        # initialize Data and Dr16ExpectedFlux instances
        config = ConfigParser()
        config.read_dict({
            "data": desi_healpix_kwargs,
            "expected flux": {
                "iter out prefix": "iter_out_prefix",
                "out dir": f"{THIS_DIR}/results/",
                "num processors": 1,
            },
        })
        for key, value in defaults_dr16_expected_flux.items():
            if key not in config["expected flux"]:
                config["expected flux"][key] = str(value)
        for key, value in defaults_desi_healpix.items():
            if key not in config["data"]:
                config["data"][key] = str(value)
        data = DesiHealpix(config["data"])
        expected_flux = Dr16ExpectedFlux(config["expected flux"])

        # compute the forest continua
        for forest in data.forests:
            expected_flux.compute_continuum(forest)

        # compute mean quasar continuum
        expected_flux.compute_mean_cont(data.forests)

        # save results
        f = open(out_file, "w")
        f.write("# log_lambda mean_cont\n")
        for log_lambda in np.arange(3.0171, 3.079 + 3e-4, 3e-4):
            f.write(f"{log_lambda} {expected_flux.get_mean_cont(log_lambda)}\n")
        f.close()

        # load the expected results
        expectations = np.genfromtxt(test_file, names=True)

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

    def test_dr16_expected_flux_compute_mean_cont_log(self):
        """Test method compute_mean_cont_log for class Dr16ExpectedFlux"""
        # setup Forest variables; case: logarithmic wavelength solution
        setup_forest("log", rebin=3)

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
        expected_flux.compute_mean_cont(data.forests)

        # save results
        f = open(out_file, "w")
        f.write("# log_lambda mean_cont\n")
        for log_lambda in np.arange(3.0171, 3.079 + 3e-4, 3e-4):
            f.write(f"{log_lambda} {expected_flux.get_mean_cont(log_lambda)}\n")
        f.close()

        # load the expected results
        expectations = np.genfromtxt(test_file, names=True)

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
        setup_forest("log", rebin=3)

        out_file = f"{THIS_DIR}/results/var_stats_log.txt"
        test_file = f"{THIS_DIR}/data/var_stats_log.txt"

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

        # save results
        f = open(out_file, "w")
        f.write("#log_lambda eta var_lss fudge num_pixels valid_fit\n")
        for log_lambda in expected_flux.log_lambda_var_func_grid:
            f.write(f"{log_lambda} ")
            f.write(f"{expected_flux.get_eta(log_lambda)} ")
            f.write(f"{expected_flux.get_var_lss(log_lambda)} ")
            f.write(f"{expected_flux.get_fudge(log_lambda)} ")
            f.write(f"{expected_flux.get_num_pixels(log_lambda)} ")
            f.write(f"{expected_flux.get_valid_fit(log_lambda)} ")
            f.write("\n")
        f.close()

        # load the expected results
        expectations = np.genfromtxt(test_file, names=True)

        # compare with obtained results
        eta = expected_flux.get_eta(expectations["log_lambda"])
        var_lss = expected_flux.get_var_lss(expectations["log_lambda"])
        fudge = expected_flux.get_fudge(expectations["log_lambda"])
        num_pixels = expected_flux.get_num_pixels(expectations["log_lambda"])
        valid_fit = expected_flux.get_valid_fit(expectations["log_lambda"])
        self.assertTrue(np.allclose(eta, expectations["eta"]))
        self.assertTrue(np.allclose(var_lss, expectations["var_lss"]))
        self.assertTrue(np.allclose(fudge, expectations["fudge"]))
        self.assertTrue(np.allclose(num_pixels, expectations["num_pixels"]))
        self.assertTrue(np.allclose(valid_fit, expectations["valid_fit"]))
        # setup Forest variables; case: linear wavelength solution
        reset_forest()
        setup_forest("lin")

        out_file = f"{THIS_DIR}/results/var_stats_lin.txt"
        test_file = f"{THIS_DIR}/data/var_stats_lin.txt"

        # initialize Data and Dr16ExpectedFlux instances
        config = ConfigParser()
        config.read_dict({
            "data": desi_healpix_kwargs,
            "expected flux": {
                "iter out prefix": "iter_out_prefix",
                "out dir": f"{THIS_DIR}/results/",
                "num processors": 1,
            },
        })
        for key, value in defaults_dr16_expected_flux.items():
            if key not in config["expected flux"]:
                config["expected flux"][key] = str(value)
        for key, value in defaults_desi_healpix.items():
            if key not in config["data"]:
                config["data"][key] = str(value)
        data = DesiHealpix(config["data"])
        expected_flux = Dr16ExpectedFlux(config["expected flux"])

        # compute the forest continua
        for forest in data.forests:
            expected_flux.compute_continuum(forest)

        # compute variance functions and statistics
        expected_flux.compute_var_stats(data.forests)

        # save results
        f = open(out_file, "w")
        f.write("#log_lambda eta var_lss fudge num_pixels valid_fit\n")
        for log_lambda in expected_flux.log_lambda_var_func_grid:
            f.write(f"{log_lambda} ")
            f.write(f"{expected_flux.get_eta(log_lambda)} ")
            f.write(f"{expected_flux.get_var_lss(log_lambda)} ")
            f.write(f"{expected_flux.get_fudge(log_lambda)} ")
            f.write(f"{expected_flux.get_num_pixels(log_lambda)} ")
            f.write(f"{expected_flux.get_valid_fit(log_lambda)} ")
            f.write("\n")
        f.close()

        # load the expected results
        expectations = np.genfromtxt(test_file, names=True)

        # compare with obtained results
        eta = expected_flux.get_eta(expectations["log_lambda"])
        var_lss = expected_flux.get_var_lss(expectations["log_lambda"])
        fudge = expected_flux.get_fudge(expectations["log_lambda"])
        num_pixels = expected_flux.get_num_pixels(expectations["log_lambda"])
        valid_fit = expected_flux.get_valid_fit(expectations["log_lambda"])
        self.assertTrue(np.allclose(eta, expectations["eta"]))
        self.assertTrue(np.allclose(var_lss, expectations["var_lss"]))
        self.assertTrue(np.allclose(fudge, expectations["fudge"]))
        self.assertTrue(np.allclose(num_pixels, expectations["num_pixels"]))
        self.assertTrue(np.allclose(valid_fit, expectations["valid_fit"]))

    def test_dr16_expected_flux_parse_config(self):
        """Test method __parse_config for class Dr16ExpectedFlux"""
        # create a Dr16ExpectedFlux with missing ExpectedFlux Options
        config = ConfigParser()
        config.read_dict({"expected_flux": {
        }})
        expected_message = (
            "Missing argument 'out dir' required by ExpectedFlux"
        )
        with self.assertRaises(ExpectedFluxError) as context_manager:
            Dr16ExpectedFlux(config["expected_flux"])
        self.compare_error_message(context_manager, expected_message)

        # create a Dr16ExpectedFlux with missing iter_out_prefix
        config = ConfigParser()
        config.read_dict({"expected_flux": {
            "out dir": f"{THIS_DIR}/results/",
            "num processors": 1
        }})
        expected_message = (
            "Missing argument 'iter out prefix' required by Dr16ExpectedFlux"
        )
        with self.assertRaises(ExpectedFluxError) as context_manager:
            Dr16ExpectedFlux(config["expected_flux"])
        self.compare_error_message(context_manager, expected_message)

        # create a Dr16ExpectedFlux with invalid iter_out_prefix
        config = ConfigParser()
        config.read_dict(
            {"expected flux": {
                "iter out prefix": f"{THIS_DIR}/results/iter_out_prefix",
                "out dir": f"{THIS_DIR}/results/",
                "num processors": 1
            }})
        for key, value in defaults_dr16_expected_flux.items():
            if key not in config["expected flux"]:
                config["expected flux"][key] = str(value)
        expected_message = (
            "Error constructing Dr16ExpectedFlux. 'iter out prefix' should not "
            f"incude folders. Found: {THIS_DIR}/results/iter_out_prefix")
        with self.assertRaises(ExpectedFluxError) as context_manager:
            Dr16ExpectedFlux(config["expected flux"])
        self.compare_error_message(context_manager, expected_message)

        # create a Dr16ExpectedFlux with missing 'limit eta'
        config = ConfigParser()
        config.read_dict({"expected_flux": {
            "iter out prefix": f"iter_out_prefix",
            "out dir": f"{THIS_DIR}/results/",
            "num processors": 1
        }})
        expected_message = (
            "Missing argument 'limit eta' required by Dr16ExpectedFlux"
        )
        with self.assertRaises(ExpectedFluxError) as context_manager:
            Dr16ExpectedFlux(config["expected_flux"])
        self.compare_error_message(context_manager, expected_message)

        # create a Dr16ExpectedFlux with missing 'limit var lss'
        config = ConfigParser()
        config.read_dict({"expected_flux": {
            "iter out prefix": f"iter_out_prefix",
            "out dir": f"{THIS_DIR}/results/",
            "num processors": 1,
            "limit eta": "0.0, 1.90",
        }})
        expected_message = (
            "Missing argument 'limit var lss' required by Dr16ExpectedFlux"
        )
        with self.assertRaises(ExpectedFluxError) as context_manager:
            Dr16ExpectedFlux(config["expected_flux"])
        self.compare_error_message(context_manager, expected_message)

        # create a Dr16ExpectedFlux with missing num_bins_vairance
        config = ConfigParser()
        config.read_dict({"expected_flux": {
            "iter out prefix": f"iter_out_prefix",
            "out dir": f"{THIS_DIR}/results/",
            "num processors": 1,
            "limit eta": "0.0, 1.90",
            "limit var lss": "0.0, 1.90",
        }})
        expected_message = (
            "Missing argument 'num bins variance' required by Dr16ExpectedFlux"
        )
        with self.assertRaises(ExpectedFluxError) as context_manager:
            Dr16ExpectedFlux(config["expected_flux"])
        self.compare_error_message(context_manager, expected_message)

        # create a Dr16ExpectedFlux with missing num_iterations
        config = ConfigParser()
        config.read_dict({"expected_flux": {
            "iter out prefix": f"iter_out_prefix",
            "out dir": f"{THIS_DIR}/results/",
            "num processors": 1,
            "limit eta": "0.0, 1.90",
            "limit var lss": "0.0, 1.90",
            "num bins variance": 20,
        }})
        expected_message = (
            "Missing argument 'num iterations' required by Dr16ExpectedFlux"
        )
        with self.assertRaises(ExpectedFluxError) as context_manager:
            Dr16ExpectedFlux(config["expected_flux"])
        self.compare_error_message(context_manager, expected_message)

        # create a Dr16ExpectedFlux with missing order
        config = ConfigParser()
        config.read_dict({"expected_flux": {
            "iter out prefix": f"iter_out_prefix",
            "out dir": f"{THIS_DIR}/results/",
            "num processors": 1,
            "limit eta": "0.0, 1.90",
            "limit var lss": "0.0, 1.90",
            "num bins variance": 20,
            "num iterations": 5,
        }})
        expected_message = (
            "Missing argument 'order' required by Dr16ExpectedFlux"
        )
        with self.assertRaises(ExpectedFluxError) as context_manager:
            Dr16ExpectedFlux(config["expected_flux"])
        self.compare_error_message(context_manager, expected_message)

        # create a Dr16ExpectedFlux with missing use_constant_weight
        config = ConfigParser()
        config.read_dict({"expected_flux": {
            "iter out prefix": f"iter_out_prefix",
            "out dir": f"{THIS_DIR}/results/",
            "num processors": 1,
            "limit eta": "0.0, 1.90",
            "limit var lss": "0.0, 1.90",
            "num bins variance": 20,
            "num iterations": 5,
            "order": 1,
        }})
        expected_message = (
            "Missing argument 'use constant weight' required by Dr16ExpectedFlux"
        )
        with self.assertRaises(ExpectedFluxError) as context_manager:
            Dr16ExpectedFlux(config["expected_flux"])
        self.compare_error_message(context_manager, expected_message)

        # create a Dr16ExpectedFlux with missing use_ivar_as_weight
        config = ConfigParser()
        config.read_dict({"expected_flux": {
            "iter out prefix": f"iter_out_prefix",
            "out dir": f"{THIS_DIR}/results/",
            "num processors": 1,
            "limit eta": "0.0, 1.90",
            "limit var lss": "0.0, 1.90",
            "num bins variance": 20,
            "num iterations": 5,
            "order": 1,
            "use constant weight": False,
        }})
        expected_message = (
            "Missing argument 'use ivar as weight' required by Dr16ExpectedFlux"
        )
        with self.assertRaises(ExpectedFluxError) as context_manager:
            Dr16ExpectedFlux(config["expected_flux"])
        self.compare_error_message(context_manager, expected_message)


        setup_forest("log", rebin=3)
        # create a Dr16ExpectedFlux instance; case: limit eta and limit var_lss with ()
        config = ConfigParser()
        config.read_dict({"expected_flux": {
            "iter out prefix": f"iter_out_prefix",
            "out dir": f"{THIS_DIR}/results/",
            "num processors": 1,
            "limit eta": "(0.0, 1.90)",
            "limit var lss": "(0.5, 1.40)",
            "num bins variance": 20,
            "num iterations": 5,
            "order": 1,
            "use constant weight": False,
            "use ivar as weight": False,
        }})
        expected_flux = Dr16ExpectedFlux(config["expected_flux"])
        self.assertTrue(np.allclose(expected_flux.limit_eta, (0.0, 1.9)))
        self.assertTrue(np.allclose(expected_flux.limit_var_lss, (0.5, 1.4)))

        # create a Dr16ExpectedFlux instance; case: limit eta and limit var_lss with []
        config = ConfigParser()
        config.read_dict({"expected_flux": {
            "iter out prefix": f"iter_out_prefix",
            "out dir": f"{THIS_DIR}/results/",
            "num processors": 1,
            "limit eta": "[0.0, 1.90]",
            "limit var lss": "[0.5, 1.40]",
            "num bins variance": 20,
            "num iterations": 5,
            "order": 1,
            "use constant weight": False,
            "use ivar as weight": False,
        }})
        expected_flux = Dr16ExpectedFlux(config["expected_flux"])
        self.assertTrue(np.allclose(expected_flux.limit_eta, (0.0, 1.9)))
        self.assertTrue(np.allclose(expected_flux.limit_var_lss, (0.5, 1.4)))

        # create a Dr16ExpectedFlux instance; case: limit eta and limit var_lss
        # without parenthesis
        config = ConfigParser()
        config.read_dict({"expected_flux": {
            "iter out prefix": f"iter_out_prefix",
            "out dir": f"{THIS_DIR}/results/",
            "num processors": 1,
            "limit eta": "0.0, 1.90",
            "limit var lss": "0.5, 1.40",
            "num bins variance": 20,
            "num iterations": 5,
            "order": 1,
            "use constant weight": False,
            "use ivar as weight": False,
        }})
        expected_flux = Dr16ExpectedFlux(config["expected_flux"])
        self.assertTrue(np.allclose(expected_flux.limit_eta, (0.0, 1.9)))
        self.assertTrue(np.allclose(expected_flux.limit_var_lss, (0.5, 1.4)))

    def test_dr16_expected_flux_populate_los_ids(self):
        """Test method populate_los_ids for class Dr16ExpectedFlux"""
        # setup Forest variables; case: logarithmic wavelength solution
        setup_forest("log", rebin=3)

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

    def test_dr16_expected_flux_save_iteration_step(self):
        """Test method save_iteration_step for class Dr16ExpectedFlux"""
        # setup Forest variables; case: logarithmic wavelength solution
        setup_forest("log", rebin=3)

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
        reset_forest()
        setup_forest("lin")

        out_file = f"{THIS_DIR}/results/Log/iter_out_prefix_lin_iteration1.fits.gz"
        out_file2 = f"{THIS_DIR}/results/Log/iter_out_prefix_lin.fits.gz"
        test_file = f"{THIS_DIR}/data/iter_out_prefix_lin_iteration1.fits.gz"
        test_file2 = f"{THIS_DIR}/data/iter_out_prefix_lin.fits.gz"

        # initialize Data and Dr16ExpectedFlux instances
        config = ConfigParser()
        config.read_dict({
            "data": desi_healpix_kwargs,
            "expected flux": {
                "iter out prefix": "iter_out_prefix_lin",
                "out dir": f"{THIS_DIR}/results/",
                "num processors": 1,
            },
        })
        for key, value in defaults_dr16_expected_flux.items():
            if key not in config["expected flux"]:
                config["expected flux"][key] = str(value)
        for key, value in defaults_desi_healpix.items():
            if key not in config["data"]:
                config["data"][key] = str(value)
        data = DesiHealpix(config["data"])
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
        expected_message = ("Function 'compute_expected_flux' was not "
                            "overloaded by child class")
        with self.assertRaises(ExpectedFluxError) as context_manager:
            expected_flux.compute_expected_flux([])
        self.compare_error_message(context_manager, expected_message)

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

    def test_dr16_expected_flux_parse_config(self):
        """Test method __parse_config for class Dr16ExpectedFlux"""
        # create a ExpectedFlux with missing out_dir
        config = ConfigParser()
        config.read_dict({"expected_flux": {
        }})
        expected_message = (
            "Missing argument 'out dir' required by ExpectedFlux"
        )
        with self.assertRaises(ExpectedFluxError) as context_manager:
            ExpectedFlux(config["expected_flux"])
        self.compare_error_message(context_manager, expected_message)

        # create a ExpectedFlux with missing num_processors
        config = ConfigParser()
        config.read_dict({"expected_flux": {
            "out dir": f"{THIS_DIR}/results/",
        }})
        expected_message = (
            "Missing argument 'num processors' required by ExpectedFlux"
        )
        with self.assertRaises(ExpectedFluxError) as context_manager:
            ExpectedFlux(config["expected_flux"])
        self.compare_error_message(context_manager, expected_message)

    def test_true_continuum(self):
        """Test constructor for class TrueContinuum

        Load a TrueContinuum instance.
        """
        config = ConfigParser()
        config.read_dict({
                "expected flux": {
                     "input directory": f"{THIS_DIR}/data",
                     "iter out prefix": f"{THIS_DIR}/results/iter_out_prefix",
                     "out dir": f"{THIS_DIR}/results",
                     "num processors": 1,
                }})
        for key, value in defaults_true_continuum.items():
            if key not in config["expected flux"]:
                config["expected flux"][key] = str(value)
        # this should raise an error as iter out prefix should not have a folder
        with self.assertRaises(ExpectedFluxError) as context_manager:
            expected_flux = TrueContinuum(config["expected flux"])
        expected_message = (
            "Error constructing TrueContinuum. 'iter out prefix' should not incude folders. Found: "
            )
        self.compare_error_message(context_manager, expected_message, startswith=True)


        config = ConfigParser()
        config.read_dict(
            {"expected flux": {
                "input directory": f"{THIS_DIR}/data",
                "iter out prefix": "iter_out_prefix",
                "out dir": f"{THIS_DIR}/results/",
                "num processors": 1,
            }})
        for key, value in defaults_true_continuum.items():
            if key not in config["expected flux"]:
                config["expected flux"][key] = str(value)
        # this should also raise an error as Forest variables are not defined
        with self.assertRaises(ExpectedFluxError) as context_manager:
            expected_flux = TrueContinuum(config["expected flux"])
        expected_message = (
            "Forest class variables need to be set before initializing variables here."
        )
        self.compare_error_message(context_manager, expected_message)

        # setup Forest variables; case: logarithmic wavelength solution
        setup_forest("log", rebin=3)
        expected_flux = TrueContinuum(config["expected flux"])

        self.assertTrue(isinstance(expected_flux.get_var_lss, interp1d))

        # setup Forest variables; case: linear wavelength solution
        reset_forest()
        setup_forest("lin", pixel_step=2.4)
        expected_flux = TrueContinuum(config["expected flux"])

        self.assertTrue(isinstance(expected_flux.get_var_lss, interp1d))

        # Assert invalid binning raises ExpectedFluxError
        reset_forest()
        setup_forest("lin", pixel_step=0.4)
        with self.assertRaises(ExpectedFluxError) as context_manager:
            expected_flux = TrueContinuum(config["expected flux"])
        expected_message = (
            "Couldn't find compatible raw satistics file. Provide a custom one using"
            " 'raw statistics file' field."
            )
        self.compare_error_message(context_manager, expected_message)

    def test_true_continuum_read_raw_statistics(self):
        """Test reading raw statistics files"""
        # setup Forest variables; case: linear wavelength solution
        setup_forest("lin", pixel_step=2.4)

        results_dir = Path(THIS_DIR) / "results"
        data_dir = Path(THIS_DIR) / "data"

        # initialize Data and Dr16ExpectedFlux instances
        config = ConfigParser()
        config.read_dict({
            "data": desi_mock_data_kwargs,
            "expected flux": {
                "type": "TrueContinuum",
                "input directory": f"{THIS_DIR}/data/data",
                "iter out prefix": "iter_out_prefix",
                "out dir": f"{THIS_DIR}/results/",
                "num processors": 1,
            },
        })

        for key, value in defaults_true_continuum.items():
            if key not in config["expected flux"]:
                config["expected flux"][key] = str(value)
        for key, value in defaults_desisim_data.items():
            if key not in config["data"]:
                config["data"][key] = str(value)
        expected_flux = TrueContinuum(config["expected flux"])

        log_lambda_grid = np.loadtxt(data_dir / "true_log_lambda_grid.txt")
        var_lss = expected_flux.get_var_lss(log_lambda_grid)
        mean_flux = expected_flux.get_mean_flux(log_lambda_grid)

        np.savetxt(results_dir / "true_var_lss.txt", var_lss)
        np.savetxt(results_dir / "true_mean_flux.txt", mean_flux)
        var_lss_target =   np.loadtxt(data_dir / "true_var_lss.txt")
        mean_flux_target = np.loadtxt(data_dir / "true_mean_flux.txt")
        if not np.allclose(var_lss, var_lss_target):
            filename = "true_var_lss.txt"
            print(f"\nOriginal file: {data_dir / filename}")
            print(f"New file: {results_dir / filename}")
            print("Difference found in var_lss")
            print(f"result test are_close result-test")
            for i1, i2 in zip(var_lss, var_lss_target):
                print(i1, i2, np.isclose(i1, i2), i1-i2)
            self.assertTrue(np.allclose(var_lss, var_lss_target))

        if not np.allclose(mean_flux, mean_flux_target):
            filename = "true_mean_flux.txt"
            print(f"\nOriginal file: {data_dir / filename}")
            print(f"New file: {results_dir / filename}")
            print("Difference found in mean_flux")
            print(f"result test are_close result-test")
            for i1, i2 in zip(mean_flux, mean_flux_target):
                print(i1, i2, np.isclose(i1, i2), i1-i2)
            self.assertTrue(np.allclose(mean_flux, mean_flux_target))

        # setup Forest variables; case: log wavelength solution
        reset_forest()
        setup_forest("log", rebin=3)

        # initialize Data and Dr16ExpectedFlux instances
        config = ConfigParser()
        config.read_dict({
            "data": {**desi_mock_data_kwargs, **{"wave solution": "log"}},
            "expected flux": {
                "type": "TrueContinuum",
                "input directory": f"{THIS_DIR}/data/data",
                "iter out prefix": "iter_out_prefix",
                "num processors": 1,
                "out dir": f"{THIS_DIR}/results/",
            },
        })

        for key, value in defaults_true_continuum.items():
            if key not in config["expected flux"]:
                config["expected flux"][key] = str(value)
        for key, value in defaults_desisim_data.items():
            if key not in config["data"]:
                config["data"][key] = str(value)
        expected_flux = TrueContinuum(config["expected flux"])

        log_lambda_ = np.arange(3.5563025007672873, 3.7403626894942437, 0.0003)
        var_lss = expected_flux.get_var_lss(log_lambda_)
        mean_flux = expected_flux.get_mean_flux(log_lambda_)

        np.savetxt(results_dir / "true_var_lss_log.txt", var_lss)
        np.savetxt(results_dir / "true_mean_flux_log.txt", mean_flux)
        var_lss_target =   np.loadtxt(data_dir / "true_var_lss_log.txt")
        mean_flux_target = np.loadtxt(data_dir / "true_mean_flux_log.txt")
        if not np.allclose(var_lss, var_lss_target):
            filename = "true_var_lss_log.txt"
            print(f"\nOriginal file: {data_dir / filename}")
            print(f"New file: {results_dir / filename}")
            print("Difference found in var_lss")
            print(f"result test are_close result-test")
            for i1, i2 in zip(var_lss, var_lss_target):
                print(i1, i2, np.isclose(i1, i2), i1-i2)
            self.assertTrue(np.allclose(var_lss, var_lss_target))

        if not np.allclose(mean_flux, mean_flux_target):
            filename = "true_mean_flux.txt"
            print(f"\nOriginal file: {data_dir / filename}")
            print(f"New file: {results_dir / filename}")
            print("Difference found in mean_flux")
            print(f"result test are_close result-test")
            for i1, i2 in zip(mean_flux, mean_flux_target):
                print(i1, i2, np.isclose(i1, i2), i1-i2)
            self.assertTrue(np.allclose(mean_flux, mean_flux_target))


    def test_true_continuum_read_true_continuum(self):
        """Test reading true continuum from mocks"""
        # setup Forest variables; case: linear wavelength solution
        setup_forest("lin", pixel_step=2.4)

        results_dir = Path(THIS_DIR) / "results"
        data_dir = Path(THIS_DIR) / "data"

        # initialize Data and Dr16ExpectedFlux instances
        config = ConfigParser()
        config.read_dict({
            "data": desi_mock_data_kwargs,
            "expected flux": {
                "type": "TrueContinuum",
                "input directory": f"{THIS_DIR}/data",
                "iter out prefix": "iter_out_prefix",
                "out dir": f"{THIS_DIR}/results/",
                "num processors": 1,
            },
        })

        for key, value in defaults_true_continuum.items():
            if key not in config["expected flux"]:
                config["expected flux"][key] = str(value)
        for key, value in defaults_desisim_data.items():
            if key not in config["data"]:
                config["data"][key] = str(value)
        data = DesisimMocks(config["data"])
        expected_flux = TrueContinuum(config["expected flux"])

        out_file = results_dir / "continua_true_lin.txt"
        # save the results
        f = open(out_file, "w")
        f.write("# thingid cont[0] ... cont[N]\n")
        for forest in data.forests:
            expected_flux.read_true_continuum(forest)
            f.write(f"{forest.los_id} ")
            if forest.continuum is not None:
                for item in forest.continuum:
                    f.write(f"{item} ")
            f.write("\n")
        f.close()

        test_file = data_dir / "continua_true_lin.txt"
        continua = {}
        f = open(test_file)
        for line in f.readlines():
            if line.startswith("#"):
                continue
            cols = line.split()
            los_id = int(cols[0])
            if len(cols) == 1:
                continuum = None
            else:
                continuum = np.array([float(item) for item in cols[1:]])
            continua[los_id] = continuum
        f.close()

        # compare the results
        correct_forests = 0
        for forest in data.forests:
            if forest.continuum is None:
                if continua.get(forest.los_id) is not None:
                    print(f"For forest with los_id {forest.los_id}, new continuum "
                          "is None. Expected continua:")
                    print(continua.get(forest.los_id))
                self.assertTrue(continua.get(forest.los_id) is None)
            elif continua.get(forest.los_id) is None:
                self.assertTrue(forest.continuum is None)
            else:
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

        # setup Forest variables; case: log wavelength solution
        setup_forest("log", rebin=3)

        data_dir = Path(THIS_DIR) / "data"

        # initialize Data and Dr16ExpectedFlux instances
        config = ConfigParser()
        config.read_dict({
            "data": {**desi_mock_data_kwargs, **{"wave solution": "log"}},
            "expected flux": {
                "type": "TrueContinuum",
                "input directory": f"{THIS_DIR}/data",
                "iter out prefix": "iter_out_prefix",
                "out dir": f"{THIS_DIR}/results/",
                "num processors": 1,
            },
        })

        for key, value in defaults_true_continuum.items():
            if key not in config["expected flux"]:
                config["expected flux"][key] = str(value)
        for key, value in defaults_desisim_data.items():
            if key not in config["data"]:
                config["data"][key] = str(value)
        data = DesisimMocks(config["data"])
        expected_flux = TrueContinuum(config["expected flux"])

        out_file = results_dir / "continua_true_log.txt"
        # save the results
        f = open(out_file, "w")
        f.write("# thingid cont[0] ... cont[N]\n")
        for forest in data.forests:
            expected_flux.read_true_continuum(forest)
            f.write(f"{forest.los_id} ")
            if forest.continuum is not None:
                for item in forest.continuum:
                    f.write(f"{item} ")
            f.write("\n")
        f.close()

        test_file = data_dir / "continua_true_log.txt"
        continua = {}
        f = open(test_file)
        for line in f.readlines():
            if line.startswith("#"):
                continue
            cols = line.split()
            los_id = int(cols[0])
            if len(cols) == 1:
                continuum = None
            else:
                continuum = np.array([float(item) for item in cols[1:]])
            continua[los_id] = continuum
        f.close()

        # compare the results
        correct_forests = 0
        for forest in data.forests:
            if forest.continuum is None:
                if continua.get(forest.los_id) is not None:
                    print(f"For forest with los_id {forest.los_id}, new continuum "
                          "is None. Expected continua:")
                    print(continua.get(forest.los_id))
                self.assertTrue(continua.get(forest.los_id) is None)
            elif continua.get(forest.los_id) is None:
                self.assertTrue(forest.continuum is None)
            else:
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

    def test_true_continuum_expected_flux(self):
        """Test method compute expected flux for class TrueContinuum"""
        # setup Forest variables; case: logarithmic wavelength solution
        setup_forest("log", rebin=3)

        results_dir = Path(THIS_DIR) / "results"
        data_dir = Path(THIS_DIR) / "data"

        out_file = Path(THIS_DIR) / "results" / "Log" / "iter_out_prefix_compute_expected_flux_log.fits.gz"
        test_file = data_dir / "true_iter_out_prefix_compute_expected_flux_log.fits.gz"

        # initialize Data and Dr16ExpectedFlux instances
        config = ConfigParser()
        config.read_dict({
            "data": {**desi_mock_data_kwargs, **{"wave solution": "log"}},
            "expected flux": {
                "type": "TrueContinuum",
                "input directory": f"{THIS_DIR}/data",
                "iter out prefix": "iter_out_prefix",
                "out dir": f"{THIS_DIR}/results/",
                "num processors": 1,
            },
        })

        for key, value in defaults_true_continuum.items():
            if key not in config["expected flux"]:
                config["expected flux"][key] = str(value)
        for key, value in defaults_desisim_data.items():
            if key not in config["data"]:
                config["data"][key] = str(value)
        data = DesisimMocks(config["data"])
        expected_flux = TrueContinuum(config["expected flux"])

        # compute the expected flux
        expected_flux.compute_expected_flux(data.forests)

        # check the results
        for iteration in range(1,5):
            self.compare_fits(
                str(test_file).replace(".fits", f"_iteration{iteration}.fits"),
                str(out_file).replace(".fits", f"_iteration{iteration}.fits"))
        self.compare_fits(test_file, out_file)

        # setup Forest variables; case: linear wavelength solution
        setup_forest("lin", pixel_step=2.4)

        data_dir = Path(THIS_DIR) / "data"

        out_file = Path(THIS_DIR) / "results" / "Log" / "iter_out_prefix_compute_expected_flux_log.fits.gz"
        test_file = data_dir / "true_iter_out_prefix_compute_expected_flux_log_lin.fits.gz"

        # initialize Data and Dr16ExpectedFlux instances
        config = ConfigParser()
        config.read_dict({
            "data": desi_mock_data_kwargs,
            "expected flux": {
                "type": "TrueContinuum",
                "input directory": f"{THIS_DIR}/data",
                "iter out prefix": "iter_out_prefix",
                "num processors": 1,
                "out dir": f"{THIS_DIR}/results/",
            },
        })

        for key, value in defaults_true_continuum.items():
            if key not in config["expected flux"]:
                config["expected flux"][key] = str(value)
        for key, value in defaults_desisim_data.items():
            if key not in config["data"]:
                config["data"][key] = str(value)
        data = DesisimMocks(config["data"])
        expected_flux = TrueContinuum(config["expected flux"])

        expected_flux.compute_expected_flux(data.forests)

        # check the results
        for iteration in range(1,5):
            self.compare_fits(
                str(test_file).replace(".fits", f"_iteration{iteration}.fits"),
                str(out_file).replace(".fits", f"_iteration{iteration}.fits"))
        self.compare_fits(test_file, out_file)

    def test_true_cont_compute_mean_cont_linear_wave_solution(self):
        """Test method compute_mean_cont for class TrueContinuum using linear wave solution"""
        # setup Forest variables; case: linear wavelength solution
        setup_forest("lin", pixel_step=2.4)

        results_dir = Path(THIS_DIR) / "results"
        data_dir = Path(THIS_DIR) / "data"

        # initialize Data and Dr16ExpectedFlux instances
        config = ConfigParser()
        config.read_dict({
            "data": desi_mock_data_kwargs,
            "expected flux": {
                "type": "TrueContinuum",
                "input directory": f"{THIS_DIR}/data",
                "iter out prefix": "iter_out_prefix",
                "num processors": 1,
                "out dir": f"{THIS_DIR}/results/",
            },
        })

        for key, value in defaults_true_continuum.items():
            if key not in config["expected flux"]:
                config["expected flux"][key] = str(value)
        for key, value in defaults_desisim_data.items():
            if key not in config["data"]:
                config["data"][key] = str(value)
        data = DesisimMocks(config["data"])
        expected_flux = TrueContinuum(config["expected flux"])

        expected_flux.compute_expected_flux(data.forests)

        log_lambda_grid = np.loadtxt(data_dir / "true_log_lambda_rest_grid.txt")
        mean_cont = expected_flux.get_mean_cont(log_lambda_grid)
        mean_cont_weight = expected_flux.get_mean_cont(log_lambda_grid)

        np.savetxt(results_dir / "true_mean_cont_lin.txt", mean_cont)
        np.savetxt(results_dir / "true_mean_cont_weight_lin.txt", mean_cont_weight)
        mean_cont_target =   np.loadtxt(data_dir / "true_mean_cont_lin.txt")
        mean_cont_weight_target = np.loadtxt(data_dir / "true_mean_cont_weight_lin.txt")
        if not np.allclose(mean_cont, mean_cont_target):
            filename = "true_mean_cont_lin.txt"
            print(f"\nOriginal file: {data_dir / filename}")
            print(f"New file: {results_dir / filename}")
            print("Difference found in mean cont")
            print(f"result test are_close result-test")
            for i1, i2 in zip(mean_cont, mean_cont_target):
                print(i1, i2, np.isclose(i1, i2), i1-i2)
            self.assertTrue(np.allclose(mean_cont_target, mean_cont_target))

        if not np.allclose(mean_cont_weight, mean_cont_weight_target):
            filename = "true_mean_cont_weight_lin.txt"
            print(f"\nOriginal file: {data_dir / filename}")
            print(f"New file: {results_dir / filename}")
            print("Difference found in mean cont weight")
            print(f"result test are_close result-test")
            for i1, i2 in zip(mean_cont_weight, mean_cont_weight_target):
                print(i1, i2, np.isclose(i1, i2), i1-i2)
            self.assertTrue(np.allclose(mean_cont_weight_target, mean_cont_weight_target))


    def test_true_cont_compute_mean_cont_log_wave_solution(self):
        """Test method compute_mean_cont for class TrueContinuum using logarithmic wave solution"""
        # setup Forest variables; case: logarithmic wavelength solution
        setup_forest("log", rebin=3)

        results_dir = Path(THIS_DIR) / "results"
        data_dir = Path(THIS_DIR) / "data"

        # initialize Data and Dr16ExpectedFlux instances
        config = ConfigParser()
        config.read_dict({
            "data": {**desi_mock_data_kwargs, **{"wave solution": "log"}},
            "expected flux": {
                "type": "TrueContinuum",
                "input directory": f"{THIS_DIR}/data",
                "iter out prefix": "iter_out_prefix",
                "out dir": f"{THIS_DIR}/results/",
                "num processors": 1,
            },
        })

        for key, value in defaults_true_continuum.items():
            if key not in config["expected flux"]:
                config["expected flux"][key] = str(value)
        for key, value in defaults_desisim_data.items():
            if key not in config["data"]:
                config["data"][key] = str(value)
        data = DesisimMocks(config["data"])
        expected_flux = TrueContinuum(config["expected flux"])

        # compute the expected flux
        expected_flux.compute_expected_flux(data.forests)

        log_lambda_grid = np.loadtxt(data_dir / "true_log_lambda_rest_grid_log.txt")
        mean_cont = expected_flux.get_mean_cont(log_lambda_grid)
        mean_cont_weight = expected_flux.get_mean_cont(log_lambda_grid)

        np.savetxt(results_dir / "true_mean_cont_log.txt", mean_cont)
        np.savetxt(results_dir / "true_mean_cont_weight_log.txt", mean_cont_weight)
        mean_cont_target =   np.loadtxt(data_dir / "true_mean_cont_log.txt")
        mean_cont_weight_target = np.loadtxt(data_dir / "true_mean_cont_weight_log.txt")
        if not np.allclose(mean_cont, mean_cont_target):
            filename = "true_mean_cont_log.txt"
            print(f"\nOriginal file: {data_dir / filename}")
            print(f"New file: {results_dir / filename}")
            print("Difference found in mean cont")
            print(f"result test are_close result-test")
            for i1, i2 in zip(mean_cont, mean_cont_target):
                print(i1, i2, np.isclose(i1, i2), i1-i2)
            self.assertTrue(np.allclose(mean_cont_target, mean_cont_target))

        if not np.allclose(mean_cont_weight, mean_cont_weight_target):
            filename = "true_mean_cont_weight_log.txt"
            print(f"\nOriginal file: {data_dir / filename}")
            print(f"New file: {results_dir / filename}")
            print("Difference found in mean cont weight")
            print(f"result test are_close result-test")
            for i1, i2 in zip(mean_cont_weight, mean_cont_weight_target):
                print(i1, i2, np.isclose(i1, i2), i1-i2)
            self.assertTrue(np.allclose(mean_cont_weight_target, mean_cont_weight_target))


    def test_true_continuum_populate_los_ids(self):
        """Test method populate_los_ids for class TrueContinuum"""
        # setup Forest variables; case: logarithmic wavelength solution
        setup_forest("log", rebin=3)

        data_dir = Path(THIS_DIR) / "data"

        # initialize Data and Dr16ExpectedFlux instances
        config = ConfigParser()
        config.read_dict({
            "data": {**desi_mock_data_kwargs, **{"wave solution": "log"}},
            "expected flux": {
                "type": "TrueContinuum",
                "input directory": f"{THIS_DIR}/data",
                "iter out prefix": "iter_out_prefix",
                "out dir": f"{THIS_DIR}/results/",
                "num processors": 1,
            },
        })

        for key, value in defaults_true_continuum.items():
            if key not in config["expected flux"]:
                config["expected flux"][key] = str(value)
        for key, value in defaults_desisim_data.items():
            if key not in config["data"]:
                config["data"][key] = str(value)
        data = DesisimMocks(config["data"])
        expected_flux = TrueContinuum(config["expected flux"])

        # compute the forest continua
        for forest in data.forests:
            expected_flux.read_true_continuum(forest)

        # save iter_out_prefix for iteration 0
        expected_flux.populate_los_ids(data.forests)

        for i, key in enumerate(("mean expected flux", "weights", "continuum")):
            self.assertTrue(np.allclose(
                expected_flux.los_ids[59152][key],
                np.loadtxt( data_dir / f"los_ids_{i}.txt")
            ))

if __name__ == '__main__':
    unittest.main()
