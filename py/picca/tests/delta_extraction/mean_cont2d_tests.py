"""This file contains tests related to ExpectedFlux and its childs"""
from configparser import ConfigParser
import copy
import os
import unittest

import numpy as np
from scipy.interpolate import interp1d, RegularGridInterpolator

from picca.delta_extraction.errors import ExpectedFluxError
from picca.delta_extraction.data_catalogues.desi_healpix import DesiHealpix
from picca.delta_extraction.data_catalogues.desi_healpix import defaults as defaults_desi_healpix
from picca.delta_extraction.data_catalogues.sdss_data import SdssData
from picca.delta_extraction.data_catalogues.sdss_data import defaults as defaults_sdss_data
from picca.delta_extraction.expected_fluxes.dr16_expected_flux import (
    compute_continuum)
from picca.delta_extraction.expected_fluxes.dr16_expected_flux import (
    defaults as defaults_dr16_expected_flux)
from picca.delta_extraction.expected_fluxes.mean_continuum2d_expected_flux import (
    MeanContinuum2dExpectedFlux, defaults as defaults_mean_continuum2d_expected_flux)
from picca.tests.delta_extraction.abstract_test import AbstractTest
from picca.tests.delta_extraction.test_utils import setup_forest, reset_forest
from picca.tests.delta_extraction.test_utils import desi_healpix_kwargs
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
    def test_mean_continuum2d_expected_flux(self):
        """Test constructor for class MeanContinuum2dExpectedFlux
        Load an MeanContinuum2dExpectedFlux instance.
        """
        config = ConfigParser()
        config.read_dict(
            {"expected flux": {
                "iter out prefix": "iter_out_prefix",
                "limit z": "(1.94, 4.5)",
                "num processors": 1,
                "num z bins": 2,
                "out dir": f"{THIS_DIR}/results/",
            }})
        for key, value in defaults_dr16_expected_flux.items():
            if key not in config["expected flux"]:
                config["expected flux"][key] = str(value)
        # this should raise an error as Forest variables are not defined
        expected_message = (
            "Forest class variables need to be set before initializing "
            "variables here.")
        with self.assertRaises(ExpectedFluxError) as context_manager:
            expected_flux = MeanContinuum2dExpectedFlux(config["expected flux"])
        self.compare_error_message(context_manager, expected_message)

        # setup Forest variables; case: logarithmic wavelength solution
        setup_forest("log", rebin=3)
        expected_flux = MeanContinuum2dExpectedFlux(config["expected flux"])

        self.assertTrue(isinstance(expected_flux.get_eta, interp1d))
        self.assertTrue(isinstance(expected_flux.get_fudge, interp1d))
        self.assertTrue(isinstance(expected_flux.get_mean_cont, RegularGridInterpolator))
        self.assertTrue(isinstance(expected_flux.get_var_lss, interp1d))
        self.assertTrue(isinstance(expected_flux.log_lambda_var_func_grid, np.ndarray))

        # setup Forest variables; case: linear wavelength solution
        reset_forest()
        setup_forest("lin")
        expected_flux = MeanContinuum2dExpectedFlux(config["expected flux"])

        self.assertTrue(isinstance(expected_flux.get_eta, interp1d))
        self.assertTrue(isinstance(expected_flux.get_fudge, interp1d))
        self.assertTrue(isinstance(expected_flux.get_mean_cont, RegularGridInterpolator))
        self.assertTrue(isinstance(expected_flux.get_var_lss, interp1d))
        self.assertTrue(isinstance(expected_flux.log_lambda_var_func_grid, np.ndarray))

    def test_mean_continuum2d_expected_flux_compute_continuum_lin(self):
        """Test method compute_continuum for class MeanContinuum2dExpectedFlux for
        linear wavelength solution"""
        setup_forest("lin")

        out_file = f"{THIS_DIR}/results/continua_2d_lin.txt"
        test_file = f"{THIS_DIR}/data/continua_2d_lin.txt"

        # initialize DesiHealpix and MeanContinuum2dExpectedFlux instances
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
        expected_flux = MeanContinuum2dExpectedFlux(config["expected flux"])

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

    def test_mean_continuum2d_expected_flux_compute_continuum_log(self):
        """Test method compute_continuum for class MeanContinuum2dExpectedFlux for
        logarithmic wavelength solution"""
        setup_forest("log", rebin=3)

        out_file = f"{THIS_DIR}/results/continua_2d_log.txt"
        test_file = f"{THIS_DIR}/data/continua_2d_log.txt"

        # initialize Data and MeanContinuum2dExpectedFlux instances
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
        expected_flux = MeanContinuum2dExpectedFlux(config["expected flux"])

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

    def test_mean_continuum2d_expected_flux_compute_delta_stack_lin(self):
        """Test method compute_delta_stack for class MeanContinuum2dExpectedFlux for
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
                "var lss mod": 1.0,
            },
        })
        for key, value in defaults_dr16_expected_flux.items():
            if key not in config["expected flux"]:
                config["expected flux"][key] = str(value)
        for key, value in defaults_desi_healpix.items():
            if key not in config["data"]:
                config["data"][key] = str(value)
        data = DesiHealpix(config["data"])
        expected_flux = MeanContinuum2dExpectedFlux(config["expected flux"])

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

    def test_mean_continuum2d_expected_flux_compute_delta_stack_log(self):
        """Test method compute_delta_stack for class MeanContinuum2dExpectedFlux for
        logarithmic wavelength solution"""
        setup_forest("log", rebin=3)

        out_file = f"{THIS_DIR}/results/delta_stack_2d_log.txt"
        test_file = f"{THIS_DIR}/data/delta_stack_2d_log.txt"

        # initialize Data and MeanContinuum2dExpectedFlux instances
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
        expected_flux = MeanContinuum2dExpectedFlux(config["expected flux"])

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

    def test_mean_continuum2d_expected_flux_compute_expected_flux_lin(self):
        """Test method compute_var_stats for class MeanContinuum2dExpectedFlux for
        linear wavelength solution"""
        setup_forest("lin")

        out_file = f"{THIS_DIR}/results/Log/iter_out_prefix_compute_expected_flux_2d_lin.fits.gz"
        test_file = f"{THIS_DIR}/data/iter_out_prefix_compute_expected_flux_2d_lin.fits.gz"

        # initialize Data and MeanContinuum2dExpectedFlux instances
        config = ConfigParser()
        config.read_dict({
            "data": desi_healpix_kwargs,
            "expected flux": {
                "iter out prefix": "iter_out_prefix_compute_expected_flux_2d_lin",
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
        expected_flux = MeanContinuum2dExpectedFlux(config["expected flux"])

        # compute the expected flux
        expected_flux.compute_expected_flux(data.forests)

        # check the results
        for iteration in range(1, 5):
            self.compare_fits(
                test_file.replace(".fits", f"_iteration{iteration}.fits"),
                out_file.replace(".fits", f"_iteration{iteration}.fits"))
        self.compare_fits(test_file, out_file)

    def test_mean_continuum2d_expected_flux_compute_expected_flux_log(self):
        """Test method compute_var_stats for class MeanContinuum2dExpectedFlux for
        logarithmic wavelength solution"""
        setup_forest("log", rebin=3)

        out_file = f"{THIS_DIR}/results/Log/iter_out_prefix_compute_expected_flux_2d_log.fits.gz"
        test_file = f"{THIS_DIR}/data/iter_out_prefix_compute_expected_flux_2d_log.fits.gz"

        # initialize Data and Dr16ExpectedFlux instances
        config = ConfigParser()
        config.read_dict({
            "data": sdss_data_kwargs,
            "expected flux": {
                "iter out prefix": "iter_out_prefix_compute_expected_flux_2d_log",
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
        expected_flux = MeanContinuum2dExpectedFlux(config["expected flux"])

        # compute the expected flux
        expected_flux.compute_expected_flux(data.forests)

        # check the results
        for iteration in range(1, 5):
            self.compare_fits(
                test_file.replace(".fits", f"_iteration{iteration}.fits"),
                out_file.replace(".fits", f"_iteration{iteration}.fits"))
        self.compare_fits(test_file, out_file)

    def test_mean_continuum2d_expected_flux_compute_mean_cont_lin(self):
        """Test method compute_mean_cont_lin for class MeanContinuum2dExpectedFlux
        for linear wavelength solution"""
        setup_forest("lin")

        out_file = f"{THIS_DIR}/results/mean_cont_2d_lin.txt"
        test_file = f"{THIS_DIR}/data/mean_cont_2d_lin.txt"

        # initialize Data and MeanContinuum2dExpectedFlux instances
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
        expected_flux = MeanContinuum2dExpectedFlux(config["expected flux"])

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

    def test_mean_continuum2d_expected_flux_compute_mean_cont_log(self):
        """Test method compute_mean_cont_log for class MeanContinuum2dExpectedFlux for
        logarithmic wavelength solution"""
        setup_forest("log", rebin=3)

        out_file = f"{THIS_DIR}/results/mean_cont_2d_log.txt"
        test_file = f"{THIS_DIR}/data/mean_cont_2d_log.txt"

        # initialize Data and MeanContinuum2dExpectedFlux instances
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
        expected_flux = MeanContinuum2dExpectedFlux(config["expected flux"])

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

    def test_mean_continuum2d_expected_flux_compute_var_stats_lin(self):
        """Test method compute_var_stats for class MeanContinuum2dExpectedFlux with
        linear wavelength solution
        """
        setup_forest("lin")

        out_file = f"{THIS_DIR}/results/var_stats_2d_lin.txt"
        test_file = f"{THIS_DIR}/data/var_stats_2d_lin.txt"

        # initialize Data and MeanContinuum2dExpectedFlux instances
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
        expected_flux = MeanContinuum2dExpectedFlux(config["expected flux"])

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

    def test_mean_continuum2d_expected_flux_compute_var_stats_log(self):
        """Test method compute_var_stats for class MeanContinuum2dExpectedFlux with
        logarithmic wavelength solution
        """
        setup_forest("log", rebin=3)

        out_file = f"{THIS_DIR}/results/var_stats_2d_log.txt"
        test_file = f"{THIS_DIR}/data/var_stats_2d_log.txt"

        # initialize Data and MeanContinuum2dExpectedFlux instances
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
        expected_flux = MeanContinuum2dExpectedFlux(config["expected flux"])

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

    def test_mean_continuum2d_expected_flux_parse_config(self):
        """Test method __parse_config for class MeanContinuum2dExpectedFlux"""
        # Forest variables need to be initialize to finish ExpectedFlux.__init__
        setup_forest("log", rebin=3)

        # create a MeanContinuum2dExpectedFlux with missing 'limit z'
        config = ConfigParser()
        config.read_dict({"expected_flux": {
        }})
        expected_message = (
            "Missing argument 'limit z' required by MeanContinuum2dExpectedFlux"
        )
        with self.assertRaises(ExpectedFluxError) as context_manager:
            MeanContinuum2dExpectedFlux(config["expected_flux"])
        self.compare_error_message(context_manager, expected_message)

        # create a MeanContinuum2dExpectedFlux with missing 'num z bins'
        config = ConfigParser()
        config.read_dict({"expected_flux": {
            "limit z": "(1.94, 4.5)",
        }})
        expected_message = (
            "Missing argument 'num z bins' required by MeanContinuum2dExpectedFlux"
        )
        with self.assertRaises(ExpectedFluxError) as context_manager:
            MeanContinuum2dExpectedFlux(config["expected_flux"])
        self.compare_error_message(context_manager, expected_message)

        # create a MeanContinuum2dExpectedFlux with missing ExpectedFlux Options
        config = ConfigParser()
        config.read_dict({"expected_flux": {
            "limit z": "(1.94, 4.5)",
            "num z bins": 2,
        }})
        expected_message = (
            "Missing argument 'iter out prefix' required by ExpectedFlux"
        )
        with self.assertRaises(ExpectedFluxError) as context_manager:
            MeanContinuum2dExpectedFlux(config["expected_flux"])
        self.compare_error_message(context_manager, expected_message)

        # create a MeanContinuum2dExpectedFlux with missing Dr16ExpectedFlux Options
        config = ConfigParser()
        config.read_dict({"expected_flux": {
            "iter out prefix": f"iter_out_prefix",
            "out dir": f"{THIS_DIR}/results/",
            "limit z": "(1.94, 4.5)",
            "num bins variance": 20,
            "num processors": 1,
            "num z bins": 2,
            "var lss mod": 1.0,
        }})
        expected_message = (
            "Missing argument 'force stack delta to zero' required by Dr16ExpectedFlux"
        )
        with self.assertRaises(ExpectedFluxError) as context_manager:
            MeanContinuum2dExpectedFlux(config["expected_flux"])
        self.compare_error_message(context_manager, expected_message)

    def test_mean_continuum2d_expected_flux_populate_los_ids(self):
        """Test method populate_los_ids for class MeanContinuum2dExpectedFlux"""
        # setup Forest variables; case: logarithmic wavelength solution
        setup_forest("log", rebin=3)

        # initialize Data and MeanContinuum2dExpectedFlux instances
        config = ConfigParser()
        config.read_dict({
            "data": sdss_data_kwargs,
            "expected flux": {
                "iter out prefix": "iter_out_prefix_2d_log",
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
        expected_flux = MeanContinuum2dExpectedFlux(config["expected flux"])

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
    
if __name__ == '__main__':
    unittest.main()
