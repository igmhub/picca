"""This module defines the class MeanContinuumInterpExpectedFlux"""
import logging

import numba
import numpy as np
from scipy.interpolate import interp1d, RegularGridInterpolator

from picca.delta_extraction.errors import ExpectedFluxError
from picca.delta_extraction.astronomical_objects.forest import Forest
from picca.delta_extraction.expected_fluxes.dr16_fixed_fudge_expected_flux import (
    Dr16FixedFudgeExpectedFlux, defaults, accepted_options)
from picca.delta_extraction.utils import (update_accepted_options,
                                          update_default_options)

accepted_options = update_accepted_options(
    accepted_options, ["interpolation type", "limit z", "num z bins"])

defaults = update_default_options(defaults, {
    "interpolation type": "1D",
    "limit z": (1.8, 5.),
    "num z bins": 3,
})

ACCEPTED_INTERPOLATION_TYPES = ["1D", "2D"]


class MeanContinuumInterpExpectedFlux(Dr16FixedFudgeExpectedFlux):
    """Class to compute the expected flux as done in the DR16 SDSS analysis
    The mean expected flux is calculated iteratively as explained in
    du Mas des Bourboux et al. (2020) except that the we don't use
    the stacking technique to compute the mean quasar continuum.
    Instead, we build an interpolator.

    Additionally, the mean continuum can be computed in 2D, i.e. the
    mean continuum is computed as a function of both the wavelength and
    the redshift.

    Methods
    -------
    (see Dr16ExpectedFlux in py/picca/delta_extraction/expected_fluxes/dr16_expected_flux.py)

    Attributes
    ----------
    (see Dr16ExpectedFlux in py/picca/delta_extraction/expected_fluxes/dr16_expected_flux.py)

    get_mean_cont: scipy.interpolate.interp1d or scipy.interpolate.RegularGridInterpolator
    Interpolation function to compute the unabsorbed mean quasar continua.

    lambda_abs_igm: float
    Wavelength in Angstroms at which the IGM absorption is computed.

    limit_z: tuple of float
    Minimum and maximum redshift limits for the analysis.

    num_z_bins: int
    Number of redshift bins to use for the analysis.

    z_bin_edges: np.ndarray
    Bins in redshift used to compute the mean continuum.
    """

    def __init__(self, config):
        """Initialize class instance.

        Arguments
        ---------
        config: configparser.SectionProxy
        Parsed options to initialize class

        Raise
        -----
        ExpectedFluxError if Forest class variables are not set
        """
        self.logger = logging.getLogger(__name__)

        # load variables from config
        self.interpolation_type = None
        self.limit_z = None
        self.num_z_bins = None
        self.z_bin_edges = None
        self.__parse_config(config)

        self.mean_cont = None
        super().__init__(config)

    def __parse_config(self, config):
        """Parse the configuration options

        Arguments
        ---------
        config: configparser.SectionProxy
        Parsed options to initialize class

        Raises
        ------
        ExpectedFluxError if variables are not valid
        """

        # this one needs to go first, as it constrains some of the other
        self.interpolation_type = config.get("interpolation type")
        if self.interpolation_type is None:
            raise ExpectedFluxError(
                "Missing argument 'interpolation type' required by MeanContinuumInterpExpectedFlux"
            )
        if self.interpolation_type not in ACCEPTED_INTERPOLATION_TYPES:
            raise ExpectedFluxError(
                f"Invalid interpolation type '{self.interpolation_type}' "
                f"required by MeanContinuumInterpExpectedFlux. "
                f"Accepted values are {ACCEPTED_INTERPOLATION_TYPES}")

        if self.interpolation_type == "2D":
            limit_z_string = config.get("limit z")
            if limit_z_string is None:
                raise ExpectedFluxError(
                    "Missing argument 'limit z' required by MeanContinuumInterpExpectedFlux"
                )
            limit_z = limit_z_string.split(",")
            if limit_z[0].startswith("(") or limit_z[0].startswith("["):
                z_min = float(limit_z[0][1:])
            else:
                z_min = float(limit_z[0])
            if limit_z[1].endswith(")") or limit_z[1].endswith("]"):
                z_max = float(limit_z[1][:-1])
            else:
                z_max = float(limit_z[1])
            self.limit_z = (z_min, z_max)

            num_z_bins = config.getint("num z bins")
            if num_z_bins is None or num_z_bins < 1:
                raise ExpectedFluxError(
                    "Missing argument 'num z bins' required by MeanContinuumInterpExpectedFlux"
                )
            self.num_z_bins = num_z_bins

            self.z_bin_edges = np.linspace(self.limit_z[0], self.limit_z[1],
                                           self.num_z_bins + 1)

    def _initialize_mean_continuum_arrays(self):
        """Initialize mean continuum arrays
        The initialized arrays are:
        - self.get_mean_cont
        """
        # initialize the mean quasar continuum
        if self.interpolation_type == "2D":
            mean_cont = np.ones(
                (self.z_bin_edges.size, Forest.log_lambda_rest_frame_grid.size))
            # fill_value cannot be "extrapolate" for RegularGridInterpolator
            # so we use 0.0 instead
            self.get_mean_cont = RegularGridInterpolator(
                (self.z_bin_edges, Forest.log_lambda_rest_frame_grid),
                mean_cont,
                bounds_error=False,
                fill_value=0.0)
        elif self.interpolation_type == "1D":
            self.mean_cont = np.ones(Forest.log_lambda_rest_frame_grid.size)

            self.get_mean_cont = interp1d(Forest.log_lambda_rest_frame_grid,
                                          self.mean_cont,
                                          fill_value='extrapolate')
        # this should never happen, but just in case
        else:  # pragma: no cover
            raise ExpectedFluxError(
                f"Invalid interpolation type '{self.interpolation_type}' "
                f"required by MeanContinuumInterpExpectedFlux. "
                f"Accepted values are {ACCEPTED_INTERPOLATION_TYPES}")

    def compute_mean_cont(self,
                          forests):
        """Compute the mean quasar continuum over the whole sample.
        Then updates the value of self.get_mean_cont to contain it

        Arguments
        ---------
        forests: List of Forest
        A list of Forest from which to compute the deltas.
        """
        if self.interpolation_type == "1D":
            self.compute_mean_cont_1d(forests)
        elif self.interpolation_type == "2D":
            self.compute_mean_cont_2d(forests)
        # this should never happen, but just in case
        else:  # pragma: no cover
            raise ExpectedFluxError(
                f"Invalid interpolation type '{self.interpolation_type}' "
                f"required by MeanContinuumInterpExpectedFlux. "
                f"Accepted values are {ACCEPTED_INTERPOLATION_TYPES}")

    def compute_mean_cont_1d(self,
                             forests):
        """Compute the mean quasar continuum over the whole sample.
        Then updates the value of self.get_mean_cont to contain it
        The mean continuum is computed as a function of the rest-frame
        wavelength.

        Arguments
        ---------
        forests: List of Forest
        A list of Forest from which to compute the deltas.
        """
        # implementation without numba and without parallelization
        A_matrix = np.zeros((Forest.log_lambda_rest_frame_grid.size,
                             Forest.log_lambda_rest_frame_grid.size))
        B_matrix = np.zeros(Forest.log_lambda_rest_frame_grid.size)

        for forest in forests:
            if forest.bad_continuum_reason is not None:
                continue

            log_lambda_rf = forest.log_lambda - np.log10(1 + forest.z)
            weights = self.compute_forest_weights(forest, forest.continuum)
            coeffs, rf_wavelength_bin = interp_coeff_lambda(
                log_lambda_rf, Forest.log_lambda_rest_frame_grid)

            w = np.where(forest.continuum > 0)
            B_matrix[rf_wavelength_bin[w]] += weights[w] * coeffs[
                w] * forest.flux[w] / forest.continuum[w]

            w = np.where((forest.continuum > 0) & (
                rf_wavelength_bin < Forest.log_lambda_rest_frame_grid.size - 1))
            B_matrix[rf_wavelength_bin[w] + 1] += weights[w] * (
                1 - coeffs[w]) * forest.flux[w] / forest.continuum[w]

            A_matrix[rf_wavelength_bin,
                     rf_wavelength_bin] += weights * coeffs * coeffs
            w = np.where(
                rf_wavelength_bin < Forest.log_lambda_rest_frame_grid.size - 1)
            A_matrix[rf_wavelength_bin[w] + 1,
                     rf_wavelength_bin[w]] += weights[w] * coeffs[w] * (
                         1 - coeffs[w])
            A_matrix[rf_wavelength_bin[w], rf_wavelength_bin[w] +
                     1] += weights[w] * coeffs[w] * (1 - coeffs[w])
            A_matrix[rf_wavelength_bin[w] + 1, rf_wavelength_bin[w] +
                     1] += weights[w] * (1 - coeffs[w]) * (1 - coeffs[w])

        # Take care of unstable solutions
        # If the diagonal of A_matrix is zero, we set it to 1.0
        # This is a workaround for the case where there is no coverage
        # for some wavelengths.
        w = np.diagonal(A_matrix) == 0
        A_matrix[w, w] = 1.0

        # Solve the linear system A_matrix * mean_cont = B_matrix
        try:
            self.mean_cont = np.linalg.solve(A_matrix, B_matrix)
        except np.linalg.LinAlgError as error:
            raise ExpectedFluxError(
                "The linear system could not be solved. "
                "This may be due to a lack of coverage for some "
                "wavelengths.") from error

        # update the interpolator with the mean continuum
        self.get_mean_cont = interp1d(Forest.log_lambda_rest_frame_grid,
                                      self.mean_cont,
                                      fill_value='extrapolate')

    def compute_mean_cont_2d(self,
                             forests):
        """Compute the mean quasar continuum over the whole sample.
        Then updates the value of self.get_mean_cont to contain it
        The mean continuum is computed as a function of the rest-frame
        wavelength and redshift.

        Arguments
        ---------
        forests: List of Forest
        A list of Forest from which to compute the deltas.
        """
        self.logger.debug("Entering compute_mean_cont_2d")
        # for simplicity we introduce a new index
        # combined_bin = z_bin + N_z_bin_edges * rf_wavelength_bin
        # where z_bin is the index of the redshift bin and rf_wavelength_bin
        # is the index of the rest-frame wavelength bin.
        # This allows us to use a similar logic as in the 1D case.
        matrix_size = self.z_bin_edges.size * Forest.log_lambda_rest_frame_grid.size

        A_matrix = np.zeros((matrix_size, matrix_size))
        B_matrix = np.zeros(matrix_size)

        for forest in forests:
            if forest.bad_continuum_reason is not None:
                continue

            log_lambda_rf = forest.log_lambda - np.log10(1 + forest.z)

            # get the mean continuum
            points = np.column_stack(
                [np.full_like(log_lambda_rf, forest.z), log_lambda_rf])
            forest_mean_cont = self.get_mean_cont(points)

            weights = self.compute_forest_weights(forest, forest.continuum)
            rf_wavelength_coeffs, rf_wavelength_bin = interp_coeff_lambda(
                log_lambda_rf, Forest.log_lambda_rest_frame_grid)
            one_minus_rf_wavelength_coeffs = 1 - rf_wavelength_coeffs
            z_coeffs, z_bin = interp_coeff_z(forest.z, self.z_bin_edges)
            one_minus_z_coeffs = 1 - z_coeffs

            if any(rf_wavelength_coeffs < 0) or any(
                    one_minus_rf_wavelength_coeffs < 0):
                raise ExpectedFluxError(
                    "Negative coefficients found in the rest-frame wavelength interpolation. "
                    "This should not happen, please report this issue.")
            if z_coeffs < 0 or one_minus_z_coeffs < 0:
                raise ExpectedFluxError(
                    "Negative coefficients found in the redshift interpolation. "
                    "This should not happen, please report this issue.")
            if any(weights < 0):
                raise ExpectedFluxError(
                    "Negative weights found in the forest weights. "
                    "This should not happen, please report this issue.")

            # combined_bin is the index of the bin in the 2D matrix
            combined_bin = z_bin + self.z_bin_edges.size * rf_wavelength_bin
            combined_bin_plus_wavelength = z_bin + self.z_bin_edges.size * (
                rf_wavelength_bin + 1)
            combined_bin_plus_z = z_bin + 1 + self.z_bin_edges.size * (
                rf_wavelength_bin)
            combined_bin_plus_both = z_bin + 1 + self.z_bin_edges.size * (
                rf_wavelength_bin + 1)

            # Fill the B_matrix
            w = np.where((forest.continuum != 0) &
                         (combined_bin < matrix_size) &
                         (combined_bin_plus_wavelength < matrix_size) &
                         (combined_bin_plus_z < matrix_size) &
                         (combined_bin_plus_both < matrix_size))

            # we should divide only by the qso multiplicative term, not the
            # whole continuum, so we multiply back by the mean continuum
            flux_over_cont = (forest.flux[w] /
                              forest.continuum[w]) * forest_mean_cont[w]

            # diagonal elements
            B_matrix[combined_bin[w]] += weights[
                w] * z_coeffs * rf_wavelength_coeffs[w] * flux_over_cont
            # off-diagonal elements
            B_matrix[combined_bin_plus_wavelength[
                w]] += weights[w] * z_coeffs * one_minus_rf_wavelength_coeffs[
                    w] * flux_over_cont
            B_matrix[combined_bin_plus_z[
                w]] += weights[w] * one_minus_z_coeffs * rf_wavelength_coeffs[
                    w] * flux_over_cont
            B_matrix[combined_bin_plus_both[w]] += weights[
                w] * one_minus_z_coeffs * one_minus_rf_wavelength_coeffs[
                    w] * flux_over_cont

            # Fill the A_matrix
            # diagonal elements

            A_matrix[combined_bin[w], combined_bin[w]] += weights[
                w] * z_coeffs * z_coeffs * rf_wavelength_coeffs[
                    w] * rf_wavelength_coeffs[w]
            # off-diagonal elements - wl
            aux = weights[w] * z_coeffs * z_coeffs * rf_wavelength_coeffs[
                w] * one_minus_rf_wavelength_coeffs[w]
            A_matrix[combined_bin[w], combined_bin_plus_wavelength[w]] += aux
            A_matrix[combined_bin_plus_wavelength[w], combined_bin[w]] += aux
            A_matrix[
                combined_bin_plus_wavelength[w],
                combined_bin_plus_wavelength[w]] += weights[
                    w] * z_coeffs * z_coeffs * one_minus_rf_wavelength_coeffs[
                        w] * one_minus_rf_wavelength_coeffs[w]
            # off-diagonal elements - z
            aux = weights[
                w] * z_coeffs * one_minus_z_coeffs * rf_wavelength_coeffs[
                    w] * rf_wavelength_coeffs[w]
            A_matrix[combined_bin[w], combined_bin_plus_z[w]] += aux
            A_matrix[combined_bin_plus_z[w], combined_bin[w]] += aux
            A_matrix[combined_bin_plus_z[w], combined_bin_plus_z[w]] += weights[
                w] * one_minus_z_coeffs * one_minus_z_coeffs * rf_wavelength_coeffs[
                    w] * rf_wavelength_coeffs[w]
            # off-diagonal elements - wl + z
            aux = weights[
                w] * z_coeffs * one_minus_z_coeffs * rf_wavelength_coeffs[
                    w] * one_minus_rf_wavelength_coeffs[w]
            A_matrix[combined_bin[w], combined_bin_plus_both[w]] += aux
            A_matrix[combined_bin_plus_both[w], combined_bin[w]] += aux
            A_matrix[
                combined_bin_plus_both[w],
                combined_bin_plus_both[w]] += weights[
                    w] * one_minus_z_coeffs * one_minus_z_coeffs * one_minus_rf_wavelength_coeffs[
                        w] * one_minus_rf_wavelength_coeffs[w]
            # cross terms - wl, z
            aux = weights[
                w] * z_coeffs * one_minus_z_coeffs * rf_wavelength_coeffs[
                    w] * one_minus_rf_wavelength_coeffs[w]
            A_matrix[combined_bin_plus_wavelength[w],
                     combined_bin_plus_z[w]] += aux
            A_matrix[combined_bin_plus_z[w],
                     combined_bin_plus_wavelength[w]] += aux
            # cross terms - wl, wl + z
            aux = weights[
                w] * z_coeffs * one_minus_z_coeffs * one_minus_rf_wavelength_coeffs[
                    w] * one_minus_rf_wavelength_coeffs[w]
            A_matrix[combined_bin_plus_wavelength[w],
                     combined_bin_plus_both[w]] += aux
            A_matrix[combined_bin_plus_both[w],
                     combined_bin_plus_wavelength[w]] += aux
            # cross terms - z, wl + z
            aux = weights[
                w] * one_minus_z_coeffs * one_minus_z_coeffs * rf_wavelength_coeffs[
                    w] * one_minus_rf_wavelength_coeffs[w]
            A_matrix[combined_bin_plus_z[w], combined_bin_plus_both[w]] += aux
            A_matrix[combined_bin_plus_both[w], combined_bin_plus_z[w]] += aux

        # check that A is symmetric
        if not np.allclose(A_matrix, A_matrix.T):
            raise ExpectedFluxError(
                "A_matrix is not symmetric. "
                "This should not happen, please report this issue.")
        # check that the diagonal in A is positive or 0
        if not np.all(np.diagonal(A_matrix) >= 0):
            raise ExpectedFluxError(
                "A_matrix diagonal is not positive or 0. "
                "This should not happen, please report this issue.")

        # Take care of unstable solutions
        # If the diagonal of A_matrix is zero, we set it to 1.0
        # This is a workaround for the case where there is no coverage
        # for some wavelengths.
        w = np.diagonal(A_matrix) == 0
        A_matrix[w, w] = 1.0

        # Solve the linear system A_matrix * mean_cont = B_matrix
        mean_cont = np.linalg.solve(A_matrix, B_matrix)
        # Undo the new indexing (needed to add transposition)
        self.mean_cont = mean_cont.reshape(
            (Forest.log_lambda_rest_frame_grid.size, self.z_bin_edges.size)).T

        # update the interpolator with the mean continuum
        self.get_mean_cont = RegularGridInterpolator(
            (self.z_bin_edges, Forest.log_lambda_rest_frame_grid),
            self.mean_cont,
            bounds_error=False,
            fill_value=0.0,
        )

    def hdu_cont(self, results):
        """Add to the results file an HDU with the continuum information

        Arguments
        ---------
        results: fitsio.FITS
        The open fits file
        """
        if self.interpolation_type == "2D":
            # Create meshgrid for evaluation
            z_meshgrid, log_lam_mesh_grid = np.meshgrid(
                self.z_bin_edges,
                Forest.log_lambda_rest_frame_grid,
                indexing='ij')
            points = np.stack([z_meshgrid.ravel(),
                               log_lam_mesh_grid.ravel()],
                              axis=-1)
            mean_cont_2d = self.get_mean_cont(points).reshape(z_meshgrid.shape)

            results.write([
                z_meshgrid,
                log_lam_mesh_grid,
                mean_cont_2d,
            ],
                          names=['Z_BIN_EDGE', 'LOGLAM_REST', 'MEAN_CONT'],
                          units=['', 'log(Angstrom)', Forest.flux_units],
                          extname='CONT')
            results["CONT"].write_comment(
                "2D mean quasar continuum (z, loglam)")
            results["CONT"].write_checksum()
        elif self.interpolation_type == "1D":
            results.write([
                Forest.log_lambda_rest_frame_grid,
                self.get_mean_cont(Forest.log_lambda_rest_frame_grid),
            ],
                          names=['LOGLAM_REST', 'MEAN_CONT'],
                          units=['log(Angstrom)', Forest.flux_units],
                          extname='CONT')
            results["CONT"].write_comment("Mean quasar continuum")
            results["CONT"].write_checksum()
            # this should never happen, but just in case
        else:  # pragma: no cover
            raise ExpectedFluxError(
                f"Invalid interpolation type '{self.interpolation_type}' "
                f"required by MeanContinuumInterpExpectedFlux. "
                f"Accepted values are {ACCEPTED_INTERPOLATION_TYPES}")


@numba.njit()
def interp_coeff_lambda(rf_wavelength, rf_wavelength_grid):
    """Compute the interpolation coefficients for a given rest-frame wavelength.

    Arguments
    ---------
    rf_wavelength: float or np.ndarray
    Rest-frame wavelength in Angstroms

    rf_wavelength_grid: np.ndarray
    Rest-frame wavelength nodes where the interpolation is defined

    Returns
    -------
    coeff: np.ndarray
    Interpolation coefficients for the given rest-frame wavelength

    rf_wavelength_bin: int or np.ndarray
    Indices of the rf_wavelength bins for the given rf_wavelength value
    """
    rf_wavelength_bin = np.digitize(rf_wavelength, rf_wavelength_grid) - 1
    rf_wavelength_low = rf_wavelength_grid[rf_wavelength_bin]
    rf_wavelength_high = rf_wavelength_grid[rf_wavelength_bin + 1]

    coeff = (rf_wavelength_high - rf_wavelength) / (rf_wavelength_high -
                                                    rf_wavelength_low)

    return coeff, rf_wavelength_bin


@numba.njit()
def interp_coeff_z(z, z_grid):
    """Compute the interpolation coefficients for a given redshift.

    Arguments
    ---------
    z: float
    Redshift

    z_grid: np.ndarray
    Redshift grid where the interpolation is defined

    Returns
    -------
    coeff: np.ndarray
    Interpolation coefficients for the given redshift

    z_bin: int or np.ndarray
    Indices of the z bins for the given z value
    """
    z_bin = np.digitize(z, z_grid) - 1
    z_low = z_grid[z_bin]
    z_high = z_grid[z_bin + 1]

    coeff = (z_high - z) / (z_high - z_low)

    return coeff, z_bin
