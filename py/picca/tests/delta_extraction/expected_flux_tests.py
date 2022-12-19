"""This file contains tests related to ExpectedFlux and its childs"""
from configparser import ConfigParser
import copy
import os
import unittest

import numpy as np
from scipy.interpolate import interp1d

from picca.delta_extraction.errors import ExpectedFluxError
from picca.delta_extraction.data_catalogues.desi_healpix import DesiHealpix
from picca.delta_extraction.data_catalogues.desi_healpix import defaults as defaults_desi_healpix
from picca.delta_extraction.data_catalogues.sdss_data import SdssData
from picca.delta_extraction.data_catalogues.sdss_data import defaults as defaults_sdss_data
from picca.delta_extraction.data_catalogues.desisim_mocks import DesisimMocks
from picca.delta_extraction.data_catalogues.desisim_mocks import defaults as defaults_desisim_data
from picca.delta_extraction.expected_flux import ExpectedFlux
from picca.delta_extraction.expected_fluxes.dr16_expected_flux import (
    Dr16ExpectedFlux, compute_continuum)
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

    def test_dr16_expected_flux_compute_continuum_lin(self):
        """Test method compute_continuum for class Dr16ExpectedFlux for
        linear wavelength solution"""
        setup_forest("lin")

        out_file = f"{THIS_DIR}/results/continua_lin.txt"
        test_file = f"{THIS_DIR}/data/continua_lin.txt"

        # initialize DesiHealpix and Dr16ExpectedFlux instances
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
            (cont_model, bad_continuum_reason,
             continuum_fit_parameters) = compute_continuum(forest,
                                                           expected_flux.get_mean_cont,
                                                           expected_flux.get_eta,
                                                           expected_flux.get_var_lss,
                                                           expected_flux.get_fudge,
                                                           expected_flux.use_constant_weight,
                                                           expected_flux.order)
            forest.bad_continuum_reason = bad_continuum_reason
            forest.continuum = cont_model

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

    def test_dr16_expected_flux_compute_continuum_log(self):
        """Test method compute_continuum for class Dr16ExpectedFlux for
        logarithmic wavelength solution"""
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
            (cont_model, bad_continuum_reason,
             continuum_fit_parameters) = compute_continuum(forest,
                                                           expected_flux.get_mean_cont,
                                                           expected_flux.get_eta,
                                                           expected_flux.get_var_lss,
                                                           expected_flux.get_fudge,
                                                           expected_flux.use_constant_weight,
                                                           expected_flux.order)
            forest.bad_continuum_reason = bad_continuum_reason
            forest.continuum = cont_model

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

    def test_dr16_expected_flux_compute_delta_stack_lin(self):
        """Test method compute_delta_stack for class Dr16ExpectedFlux for
        linear wavelength solution"""
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
            (cont_model, bad_continuum_reason,
             continuum_fit_parameters) = compute_continuum(forest,
                                                           expected_flux.get_mean_cont,
                                                           expected_flux.get_eta,
                                                           expected_flux.get_var_lss,
                                                           expected_flux.get_fudge,
                                                           expected_flux.use_constant_weight,
                                                           expected_flux.order)
            forest.bad_continuum_reason = bad_continuum_reason
            forest.continuum = cont_model

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

    def test_dr16_expected_flux_compute_delta_stack_log(self):
        """Test method compute_delta_stack for class Dr16ExpectedFlux for
        logarithmic wavelength solution"""
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
            (cont_model, bad_continuum_reason,
             continuum_fit_parameters) = compute_continuum(forest,
                                                           expected_flux.get_mean_cont,
                                                           expected_flux.get_eta,
                                                           expected_flux.get_var_lss,
                                                           expected_flux.get_fudge,
                                                           expected_flux.use_constant_weight,
                                                           expected_flux.order)
            forest.bad_continuum_reason = bad_continuum_reason
            forest.continuum = cont_model

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

    def test_dr16_expected_flux_compute_expected_flux_lin(self):
        """Test method compute_var_stats for class Dr16ExpectedFlux for
        linear wavelength solution"""
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

    def test_dr16_expected_flux_compute_expected_flux_log(self):
        """Test method compute_var_stats for class Dr16ExpectedFlux for
        logarithmic wavelength solution"""
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
        self.compare_fits(test_file, out_file)

    def test_dr16_expected_flux_compute_mean_cont_lin(self):
        """Test method compute_mean_cont_lin for class Dr16ExpectedFlux
        for linear wavelength solution"""
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
            (cont_model, bad_continuum_reason,
             continuum_fit_parameters) = compute_continuum(forest,
                                                           expected_flux.get_mean_cont,
                                                           expected_flux.get_eta,
                                                           expected_flux.get_var_lss,
                                                           expected_flux.get_fudge,
                                                           expected_flux.use_constant_weight,
                                                           expected_flux.order)
            forest.bad_continuum_reason = bad_continuum_reason
            forest.continuum = cont_model

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
        """Test method compute_mean_cont_log for class Dr16ExpectedFlux for
        logarithmic wavelength solution"""
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
            (cont_model, bad_continuum_reason,
             continuum_fit_parameters) = compute_continuum(forest,
                                                           expected_flux.get_mean_cont,
                                                           expected_flux.get_eta,
                                                           expected_flux.get_var_lss,
                                                           expected_flux.get_fudge,
                                                           expected_flux.use_constant_weight,
                                                           expected_flux.order)
            forest.bad_continuum_reason = bad_continuum_reason
            forest.continuum = cont_model

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

    def test_dr16_expected_flux_compute_var_stats_lin(self):
        """Test method compute_var_stats for class Dr16ExpectedFlux with
        linear wavelength solution
        """
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
            (cont_model, bad_continuum_reason,
             continuum_fit_parameters) = compute_continuum(forest,
                                                           expected_flux.get_mean_cont,
                                                           expected_flux.get_eta,
                                                           expected_flux.get_var_lss,
                                                           expected_flux.get_fudge,
                                                           expected_flux.use_constant_weight,
                                                           expected_flux.order)
            forest.bad_continuum_reason = bad_continuum_reason
            forest.continuum = cont_model

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

    def test_dr16_expected_flux_compute_var_stats_log(self):
        """Test method compute_var_stats for class Dr16ExpectedFlux with
        logarithmic wavelength solution
        """
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
            (cont_model, bad_continuum_reason,
             continuum_fit_parameters) = compute_continuum(forest,
                                                           expected_flux.get_mean_cont,
                                                           expected_flux.get_eta,
                                                           expected_flux.get_var_lss,
                                                           expected_flux.get_fudge,
                                                           expected_flux.use_constant_weight,
                                                           expected_flux.order)
            forest.bad_continuum_reason = bad_continuum_reason
            forest.continuum = cont_model

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
        # Forest variables need to be initialize to finish ExpectedFlux.__init__
        setup_forest("log", rebin=3)

        # create a Dr16ExpectedFlux with missing ExpectedFlux Options
        config = ConfigParser()
        config.read_dict({"expected_flux": {
        }})
        expected_message = (
            "Missing argument 'iter out prefix' required by ExpectedFlux"
        )
        with self.assertRaises(ExpectedFluxError) as context_manager:
            Dr16ExpectedFlux(config["expected_flux"])
        self.compare_error_message(context_manager, expected_message)

        # create a Dr16ExpectedFlux with missing 'force stack delta to zero'
        config = ConfigParser()
        config.read_dict({"expected_flux": {
            "iter out prefix": f"iter_out_prefix",
            "out dir": f"{THIS_DIR}/results/",
            "num bins variance": 20,
            "num processors": 1
        }})
        expected_message = (
            "Missing argument 'force stack delta to zero' required by Dr16ExpectedFlux"
        )
        with self.assertRaises(ExpectedFluxError) as context_manager:
            Dr16ExpectedFlux(config["expected_flux"])
        self.compare_error_message(context_manager, expected_message)

        # create a Dr16ExpectedFlux with missing 'limit eta'
        config = ConfigParser()
        config.read_dict({"expected_flux": {
            "iter out prefix": f"iter_out_prefix",
            "out dir": f"{THIS_DIR}/results/",
            "num bins variance": 20,
            "num processors": 1,
            "force stack delta to zero": True,
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
            "num bins variance": 20,
            "num processors": 1,
            "force stack delta to zero": True,
            "limit eta": "0.0, 1.90",
        }})
        expected_message = (
            "Missing argument 'limit var lss' required by Dr16ExpectedFlux"
        )
        with self.assertRaises(ExpectedFluxError) as context_manager:
            Dr16ExpectedFlux(config["expected_flux"])
        self.compare_error_message(context_manager, expected_message)

        # create a Dr16ExpectedFlux with missing num_iterations
        config = ConfigParser()
        config.read_dict({"expected_flux": {
            "iter out prefix": f"iter_out_prefix",
            "out dir": f"{THIS_DIR}/results/",
            "num bins variance": 20,
            "num processors": 1,
            "force stack delta to zero": True,
            "limit eta": "0.0, 1.90",
            "limit var lss": "0.0, 1.90",
        }})
        expected_message = (
            "Missing argument 'min qso in fit' required by Dr16ExpectedFlux"
        )
        with self.assertRaises(ExpectedFluxError) as context_manager:
            Dr16ExpectedFlux(config["expected_flux"])
        self.compare_error_message(context_manager, expected_message)

        # create a Dr16ExpectedFlux with missing num_iterations
        config = ConfigParser()
        config.read_dict({"expected_flux": {
            "iter out prefix": f"iter_out_prefix",
            "out dir": f"{THIS_DIR}/results/",
            "num bins variance": 20,
            "num processors": 1,
            "force stack delta to zero": True,
            "limit eta": "0.0, 1.90",
            "limit var lss": "0.0, 1.90",
            "min num qso in fit": 100,
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
            "num bins variance": 20,
            "num processors": 1,
            "force stack delta to zero": True,
            "limit eta": "0.0, 1.90",
            "limit var lss": "0.0, 1.90",
            "min num qso in fit": 100,
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
            "num bins variance": 20,
            "num processors": 1,
            "force stack delta to zero": True,
            "limit eta": "0.0, 1.90",
            "limit var lss": "0.0, 1.90",
            "min num qso in fit": 100,
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
            "num bins variance": 20,
            "num processors": 1,
            "force stack delta to zero": True,
            "limit eta": "0.0, 1.90",
            "limit var lss": "0.0, 1.90",
            "min num qso in fit": 100,
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

        # create a Dr16ExpectedFlux instance; case: limit eta and limit var_lss with ()
        config = ConfigParser()
        config.read_dict({"expected_flux": {
            "iter out prefix": f"iter_out_prefix",
            "out dir": f"{THIS_DIR}/results/",
            "num bins variance": 20,
            "num processors": 1,
            "force stack delta to zero": True,
            "limit eta": "(0.0, 1.90)",
            "limit var lss": "(0.5, 1.40)",
            "min num qso in fit": 100,
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
            "num bins variance": 20,
            "num processors": 1,
            "force stack delta to zero": True,
            "limit eta": "[0.0, 1.90]",
            "limit var lss": "[0.5, 1.40]",
            "min num qso in fit": 100,
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
            "num bins variance": 20,
            "num processors": 1,
            "force stack delta to zero": True,
            "limit eta": "0.0, 1.90",
            "limit var lss": "0.5, 1.40",
            "min num qso in fit": 100,
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
            (cont_model, bad_continuum_reason,
             continuum_fit_parameters) = compute_continuum(forest,
                                                           expected_flux.get_mean_cont,
                                                           expected_flux.get_eta,
                                                           expected_flux.get_var_lss,
                                                           expected_flux.get_fudge,
                                                           expected_flux.use_constant_weight,
                                                           expected_flux.order)
            forest.bad_continuum_reason = bad_continuum_reason
            forest.continuum = cont_model

        # compute variance functions and statistics
        expected_flux.compute_delta_stack(data.forests)

        # save iter_out_prefix for iteration 0
        expected_flux.populate_los_ids(data.forests)

    def test_dr16_expected_flux_save_iteration_step_lin(self):
        """Test method save_iteration_step for class Dr16ExpectedFlux for
        linear wavelength solution"""
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
        continuum_fit_parameters_dict = {}
        for forest in data.forests:
            (cont_model, bad_continuum_reason,
             continuum_fit_parameters) = compute_continuum(forest,
                                                           expected_flux.get_mean_cont,
                                                           expected_flux.get_eta,
                                                           expected_flux.get_var_lss,
                                                           expected_flux.get_fudge,
                                                           expected_flux.use_constant_weight,
                                                           expected_flux.order)
            forest.bad_continuum_reason = bad_continuum_reason
            forest.continuum = cont_model
            continuum_fit_parameters_dict[forest.los_id] = continuum_fit_parameters
        expected_flux.continuum_fit_parameters = continuum_fit_parameters_dict

        # compute variance functions and statistics
        expected_flux.compute_delta_stack(data.forests)

        # save iter_out_prefix for iteration 0
        expected_flux.save_iteration_step(0)
        self.compare_fits(test_file, out_file)


        # save iter_out_prefix for final iteration
        expected_flux.save_iteration_step(-1)
        self.compare_fits(test_file2, out_file2)

    def test_dr16_expected_flux_save_iteration_step_log(self):
        """Test method save_iteration_step for class Dr16ExpectedFlux for
        logarithmic wavelength solution"""
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
            (cont_model, bad_continuum_reason,
             continuum_fit_parameters) = compute_continuum(forest,
                                                           expected_flux.get_mean_cont,
                                                           expected_flux.get_eta,
                                                           expected_flux.get_var_lss,
                                                           expected_flux.get_fudge,
                                                           expected_flux.use_constant_weight,
                                                           expected_flux.order)
            forest.bad_continuum_reason = bad_continuum_reason
            forest.continuum = cont_model

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
                "iter out prefix": "delta_attributes",
                "out dir": f"{THIS_DIR}/results/",
                "num bins variance": 20,
                "num processors": 1,
            },
        })
        # this should raise an error as Forest variables are not defined
        expected_message = (
            "Forest class variables need to be set before initializing "
            "variables here.")
        with self.assertRaises(ExpectedFluxError) as context_manager:
            expected_flux = ExpectedFlux(config["expected flux"])
        self.compare_error_message(context_manager, expected_message)

        # setup Forest variables; case: logarithmic wavelength solution
        setup_forest("log", rebin=3)
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

    def test_expected_flux_parse_config(self):
        """Test method __parse_config for class Dr16ExpectedFlux"""
        # create a ExpectedFlux with missing iter_out_prefix
        config = ConfigParser()
        config.read_dict({"expected_flux": {
        }})
        expected_message = (
            "Missing argument 'iter out prefix' required by ExpectedFlux"
        )
        with self.assertRaises(ExpectedFluxError) as context_manager:
            ExpectedFlux(config["expected_flux"])
        self.compare_error_message(context_manager, expected_message)

        # create a ExpectedFlux with invalid iter_out_prefix
        config = ConfigParser()
        config.read_dict({"expected flux": {
            "iter out prefix": f"{THIS_DIR}/results/iter_out_prefix",
        }})
        for key, value in defaults_dr16_expected_flux.items():
            if key not in config["expected flux"]:
                config["expected flux"][key] = str(value)
        expected_message = (
            "Error constructing ExpectedFlux. 'iter out prefix' should not "
            f"incude folders. Found: {THIS_DIR}/results/iter_out_prefix")
        with self.assertRaises(ExpectedFluxError) as context_manager:
            ExpectedFlux(config["expected flux"])
        self.compare_error_message(context_manager, expected_message)

        # create a ExpectedFlux with missing num_bins_variance
        config = ConfigParser()
        config.read_dict({"expected_flux": {
            "iter out Prefix": f"delta_attributes",
        }})
        expected_message = (
            "Missing argument 'num bins variance' required by ExpectedFlux"
        )
        with self.assertRaises(ExpectedFluxError) as context_manager:
            ExpectedFlux(config["expected_flux"])
        self.compare_error_message(context_manager, expected_message)

        # create a ExpectedFlux with missing num_processors
        config = ConfigParser()
        config.read_dict({"expected_flux": {
            "iter out Prefix": f"delta_attributes",
            "num bins variance": 20,
        }})
        expected_message = (
            "Missing argument 'num processors' required by ExpectedFlux"
        )
        with self.assertRaises(ExpectedFluxError) as context_manager:
            ExpectedFlux(config["expected_flux"])
        self.compare_error_message(context_manager, expected_message)

        # create a ExpectedFlux with missing out_dir
        config = ConfigParser()
        config.read_dict({"expected_flux": {
            "iter out Prefix": f"delta_attributes",
            "num bins variance": 20,
            "num processors": 1,
        }})
        expected_message = (
            "Missing argument 'out dir' required by ExpectedFlux"
        )
        with self.assertRaises(ExpectedFluxError) as context_manager:
            ExpectedFlux(config["expected_flux"])
        self.compare_error_message(context_manager, expected_message)

    def test_true_continuum(self):
        """Test constructor for class TrueContinuum
        Load a TrueContinuum instance.
        """
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

    def test_true_continuum_compute_mean_cont_lin(self):
        """Test method compute_mean_cont for class TrueContinuum using
        linear wave solution"""
        setup_forest("lin", pixel_step=2.4)

        out_file = f"{THIS_DIR}/results/true_continuum_mean_cont_lin.txt"
        test_file = f"{THIS_DIR}/data/true_continuum_mean_cont_lin.txt"

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

        # save results
        f = open(out_file, "w")
        f.write("# log_lambda mean_cont mean_cont_weight\n")
        for log_lambda in np.arange(3.0171, 3.079 + 3e-4, 3e-4):
            f.write(f"{log_lambda} {expected_flux.get_mean_cont(log_lambda)} "
                    f"{expected_flux.get_mean_cont_weight(log_lambda)}\n")
        f.close()

        # load the expected results
        expectations = np.genfromtxt(test_file, names=True)

        # compare mean_cont data with obtained results
        mean_cont = expected_flux.get_mean_cont(expectations["log_lambda"])
        if not np.allclose(mean_cont, expectations["mean_cont"]):
            print(f"\nOriginal file: {test_file}")
            print(f"New file: {out_file}")
            print("Difference found in mean_cont")
            print(f"result test are_close result-test")
            for i1, i2 in zip(mean_cont, expectations["mean_cont"]):
                print(i1, i2, np.isclose(i1, i2), i1-i2)
        self.assertTrue(np.allclose(mean_cont, expectations["mean_cont"]))

        # compare mean_flux data with obtained results
        mean_cont_weight = expected_flux.get_mean_cont_weight(expectations["log_lambda"])
        if not np.allclose(mean_cont_weight, expectations["mean_cont_weight"]):
            print(f"\nOriginal file: {test_file}")
            print(f"New file: {out_file}")
            print("Difference found in mean_cont_weight")
            print(f"result test are_close result-test")
            for i1, i2 in zip(mean_cont_weight, expectations["mean_cont_weight"]):
                print(i1, i2, np.isclose(i1, i2), i1-i2)
        self.assertTrue(np.allclose(mean_cont_weight, expectations["mean_cont_weight"]))

    def test_true_continuum_compute_mean_cont_log(self):
        """Test method compute_mean_cont for class TrueContinuum using
        logarithmic wave solution"""
        setup_forest("log", rebin=3)

        out_file = f"{THIS_DIR}/results/true_continuum_mean_cont_log.txt"
        test_file = f"{THIS_DIR}/data/true_continuum_mean_cont_log.txt"

        # initialize Data and Dr16ExpectedFlux instances
        config = ConfigParser()
        kwargs = desi_mock_data_kwargs.copy()
        kwargs["wave solution"] = "log"
        config.read_dict({
            "data": kwargs,
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

        # save results
        f = open(out_file, "w")
        f.write("# log_lambda mean_cont mean_cont_weight\n")
        for log_lambda in np.arange(3.0171, 3.079 + 3e-4, 3e-4):
            f.write(f"{log_lambda} {expected_flux.get_mean_cont(log_lambda)} "
                    f"{expected_flux.get_mean_cont_weight(log_lambda)}\n")
        f.close()

        # load the expected results
        expectations = np.genfromtxt(test_file, names=True)

        # compare mean_cont data with obtained results
        mean_cont = expected_flux.get_mean_cont(expectations["log_lambda"])
        if not np.allclose(mean_cont, expectations["mean_cont"]):
            print(f"\nOriginal file: {test_file}")
            print(f"New file: {out_file}")
            print("Difference found in mean_cont")
            print(f"result test are_close result-test")
            for i1, i2 in zip(mean_cont, expectations["mean_cont"]):
                print(i1, i2, np.isclose(i1, i2), i1-i2)
        self.assertTrue(np.allclose(mean_cont, expectations["mean_cont"]))

        # compare mean_flux data with obtained results
        mean_cont_weight = expected_flux.get_mean_cont_weight(expectations["log_lambda"])
        if not np.allclose(mean_cont_weight, expectations["mean_cont_weight"]):
            print(f"\nOriginal file: {test_file}")
            print(f"New file: {out_file}")
            print("Difference found in mean_cont_weight")
            print(f"result test are_close result-test")
            for i1, i2 in zip(mean_cont_weight, expectations["mean_cont_weight"]):
                print(i1, i2, np.isclose(i1, i2), i1-i2)
        self.assertTrue(np.allclose(mean_cont_weight, expectations["mean_cont_weight"]))

    def test_true_continuum_expected_flux_lin(self):
        """Test method compute expected flux for class TrueContinuum for
        linear wavelength solution"""
        setup_forest("lin", pixel_step=2.4)

        out_file = f"{THIS_DIR}/results/Log/true_iter_out_prefix_compute_expected_flux_lin.fits.gz"
        test_file = f"{THIS_DIR}/data/true_iter_out_prefix_compute_expected_flux_lin.fits.gz"

        # initialize Data and Dr16ExpectedFlux instances
        config = ConfigParser()
        config.read_dict({
            "data": desi_mock_data_kwargs,
            "expected flux": {
                "type": "TrueContinuum",
                "input directory": f"{THIS_DIR}/data",
                "iter out prefix": "true_iter_out_prefix_compute_expected_flux_lin",
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
        self.compare_fits(test_file, out_file)

    def test_true_continuum_expected_flux_log(self):
        """Test method compute expected flux for class TrueContinuum for
        logarithmic wavelength solution"""
        setup_forest("log", rebin=3)

        out_file = f"{THIS_DIR}/results/Log/true_iter_out_prefix_compute_expected_flux_log.fits.gz"
        test_file = f"{THIS_DIR}/data/true_iter_out_prefix_compute_expected_flux_log.fits.gz"

        # initialize Data and Dr16ExpectedFlux instances
        config = ConfigParser()
        kwargs = desi_mock_data_kwargs.copy()
        kwargs["wave solution"] = "log"
        config.read_dict({
            "data": kwargs,
            "expected flux": {
                "type": "TrueContinuum",
                "input directory": f"{THIS_DIR}/data",
                "iter out prefix": "true_iter_out_prefix_compute_expected_flux_log",
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
        self.compare_fits(test_file, out_file)

    def test_true_continuum_read_raw_statistics_lin(self):
        """Test reading raw statistics files for linear wavelength solution"""
        setup_forest("lin", pixel_step=2.4)

        out_file = f"{THIS_DIR}/results/true_continuum_raw_stats_lin.txt"
        test_file = f"{THIS_DIR}/data/true_continuum_raw_stats_lin.txt"

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

        # save results
        f = open(out_file, "w")
        f.write("# log_lambda var_lss mean_flux\n")
        for log_lambda in np.arange(3.0171, 3.079 + 3e-4, 3e-4):
            f.write(f"{log_lambda} {expected_flux.get_var_lss(log_lambda)} "
                    f"{expected_flux.get_mean_flux(log_lambda)}\n")
        f.close()

        # load the expected results
        expectations = np.genfromtxt(test_file, names=True)

        # compare var_lss data with obtained results
        var_lss = expected_flux.get_var_lss(expectations["log_lambda"])
        if not np.allclose(var_lss, expectations["var_lss"]):
            print(f"\nOriginal file: {test_file}")
            print(f"New file: {out_file}")
            print("Difference found in var_lss")
            print(f"result test are_close result-test")
            for i1, i2 in zip(mean_cont, expectations["var_lss"]):
                print(i1, i2, np.isclose(i1, i2), i1-i2)
        self.assertTrue(np.allclose(var_lss, expectations["var_lss"]))

        # compare mean_flux data with obtained results
        mean_flux = expected_flux.get_mean_flux(expectations["log_lambda"])
        if not np.allclose(mean_flux, expectations["mean_flux"]):
            print(f"\nOriginal file: {test_file}")
            print(f"New file: {out_file}")
            print("Difference found in mean_flux")
            print(f"result test are_close result-test")
            for i1, i2 in zip(mean_cont, expectations["mean_flux"]):
                print(i1, i2, np.isclose(i1, i2), i1-i2)
        self.assertTrue(np.allclose(mean_flux, expectations["mean_flux"]))

    def test_true_continuum_read_raw_statistics_log(self):
        """Test reading raw statistics files for logarithmic wavelength solution"""
        setup_forest("log", rebin=3)

        out_file = f"{THIS_DIR}/results/true_continuum_raw_stats_log.txt"
        test_file = f"{THIS_DIR}/data/true_continuum_raw_stats_log.txt"

        # initialize Data and Dr16ExpectedFlux instances
        config = ConfigParser()
        kwargs = desi_mock_data_kwargs.copy()
        kwargs["wave solution"] = "log"
        config.read_dict({
            "data": kwargs,
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

        # save results
        f = open(out_file, "w")
        f.write("# log_lambda var_lss mean_flux\n")
        for log_lambda in np.arange(3.0171, 3.079 + 3e-4, 3e-4):
            f.write(f"{log_lambda} {expected_flux.get_var_lss(log_lambda)} "
                    f"{expected_flux.get_mean_flux(log_lambda)}\n")
        f.close()

        # load the expected results
        expectations = np.genfromtxt(test_file, names=True)

        # compare var_lss data with obtained results
        var_lss = expected_flux.get_var_lss(expectations["log_lambda"])
        if not np.allclose(var_lss, expectations["var_lss"]):
            print(f"\nOriginal file: {test_file}")
            print(f"New file: {out_file}")
            print("Difference found in var_lss")
            print(f"result test are_close result-test")
            for i1, i2 in zip(mean_cont, expectations["var_lss"]):
                print(i1, i2, np.isclose(i1, i2), i1-i2)
        self.assertTrue(np.allclose(var_lss, expectations["var_lss"]))

        # compare mean_flux data with obtained results
        mean_flux = expected_flux.get_mean_flux(expectations["log_lambda"])
        if not np.allclose(mean_flux, expectations["mean_flux"]):
            print(f"\nOriginal file: {test_file}")
            print(f"New file: {out_file}")
            print("Difference found in mean_flux")
            print(f"result test are_close result-test")
            for i1, i2 in zip(mean_cont, expectations["mean_flux"]):
                print(i1, i2, np.isclose(i1, i2), i1-i2)
        self.assertTrue(np.allclose(mean_flux, expectations["mean_flux"]))

    def test_true_continuum_read_true_continuum_lin(self):
        """Test reading true continuum from mocks for linear wavelength solution"""
        setup_forest("lin", pixel_step=2.4)

        out_file = f"{THIS_DIR}/results/continua_true_lin.txt"
        test_file = f"{THIS_DIR}/data/continua_true_lin.txt"

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

        # save the results
        f = open(out_file, "w")
        f.write("# thingid cont[0] ... cont[N]\n")
        for forest in data.forests:
            expected_flux.read_true_continuum_one_healpix([forest])
            f.write(f"{forest.los_id} ")
            if forest.continuum is not None:
                for item in forest.continuum:
                    f.write(f"{item} ")
            f.write("\n")
        f.close()

        # load test results
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

    def test_true_continuum_read_true_continuum_log(self):
        """Test reading true continuum from mocks for logarithmic wavelength solution"""
        setup_forest("log", rebin=3)

        out_file = f"{THIS_DIR}/results/continua_true_log.txt"
        test_file = f"{THIS_DIR}/data/continua_true_log.txt"

        # initialize Data and Dr16ExpectedFlux instances
        config = ConfigParser()
        kwargs = desi_mock_data_kwargs.copy()
        kwargs["wave solution"] = "log"
        config.read_dict({
            "data": kwargs,
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

        # save the results
        f = open(out_file, "w")
        f.write("# thingid cont[0] ... cont[N]\n")
        for forest in data.forests:
            expected_flux.read_true_continuum_one_healpix([forest])
            f.write(f"{forest.los_id} ")
            if forest.continuum is not None:
                for item in forest.continuum:
                    f.write(f"{item} ")
            f.write("\n")
        f.close()

        # load the test results
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

    def test_true_continuum_populate_los_ids(self):
        """Test method populate_los_ids for class TrueContinuum"""
        # setup Forest variables; case: logarithmic wavelength solution
        setup_forest("log", rebin=3)

        test_folder = f"{THIS_DIR}/data"

        # initialize Data and Dr16ExpectedFlux instances
        config = ConfigParser()
        kwargs = desi_mock_data_kwargs.copy()
        kwargs["wave solution"] = "log"
        config.read_dict({
            "data": kwargs,
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
        data.forests = expected_flux.read_all_true_continua(data.forests)

        # run populate_los_ids
        expected_flux.populate_los_ids(data.forests)

        for i, key in enumerate(("mean expected flux", "weights", "continuum")):
            self.assertTrue(np.allclose(
                expected_flux.los_ids[59152][key],
                np.loadtxt(f"{test_folder}/los_ids_{i}.txt")
            ))

if __name__ == '__main__':
    unittest.main()
