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
    accepted_options, ["infer z bins", "interpolation type", "limit z", "num z bins",
                       "remove mean cont slope",
                       "remove mean cont slope last iteration",
                       "slope buffer bins"])

defaults = update_default_options(defaults, {
    "infer z bins": False,
    "interpolation type": "1D",
    "limit z": (1.8, 5.),
    "num z bins": 3,
    "remove mean cont slope": True,
    "remove mean cont slope last iteration": True,
    "slope buffer bins": 5,
})

ACCEPTED_INTERPOLATION_TYPES = ["1D", "2D"]

# When removing the slope of the mean continuum, the fitted line must stay
# above this fraction of its mean over the covered region; otherwise the slope
# is not removed (see _remove_mean_cont_slope_1d)
MIN_NORM_LINE_FRACTION = 0.1


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

    remove_mean_cont_slope: bool
    If True, remove the slope (and normalisation) of the mean continuum
    after each computation of the mean continuum.

    remove_mean_cont_slope_last_iteration: bool
    If False (and remove_mean_cont_slope is True), do not remove the slope
    of the mean continuum the last time it is computed (in iteration
    num_iterations - 1; the last iteration only refits the quasar continua),
    so that the final continuum fits use the mean continuum as solved.

    slope_buffer_bins: int
    Number of bins at the edges of the non-zero continuum region to ignore
    when evaluating the slope.

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
        self.remove_mean_cont_slope = None
        self.remove_mean_cont_slope_last_iteration = None
        self.slope_buffer_bins = None
        self.z_bin_edges = None
        self.__parse_config(config)

        # number of times the mean continuum has been computed, and whether
        # the slope is removed in the current computation
        self.num_mean_cont_computations = 0
        self.remove_slope_this_computation = self.remove_mean_cont_slope

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
            # this one constraints the redshift parameters, so we need to check it first
            self.infer_z_bins = config.getboolean("infer z bins")
            if self.infer_z_bins is None:
                raise ExpectedFluxError(
                    "Missing argument 'infer z bins' required by MeanContinuumInterpExpectedFlux"
                )

            if self.infer_z_bins:
                self.logger.info(
                    "In MeanContinuumInterpExpectedFlux, 'infer z bins' is set to True. "
                    "Redshift bins will be inferred from the data. 'limit z' will be ignored."
                )
                self.limit_z = None # this is redundant, but we set it to None to avoid confusion
            else:
                limit_z_string = config.get("limit z")
                if limit_z_string is None and not self.infer_z_bins:
                    raise ExpectedFluxError(
                        "Missing argument 'limit z' required by MeanContinuumInterpExpectedFlux " 
                        "when 'infer z bins' is set to False. If you want to infer the redshift bins, set "
                        "'infer z bins' to True."
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

            self.num_z_bins = config.getint("num z bins")
            if self.num_z_bins is None:
                raise ExpectedFluxError(
                    "Missing argument 'num z bins' required by MeanContinuumInterpExpectedFlux"
                )
            if self.num_z_bins < 1:
                raise ExpectedFluxError(
                    "Invalid argument 'num z bins' required by MeanContinuumInterpExpectedFlux. "
                    "It must be a positive integer.")

            if self.infer_z_bins:
                self.z_bin_edges = None # this is redundant, but we set it to None to avoid confusion
            else:
                self.z_bin_edges = np.linspace(self.limit_z[0], self.limit_z[1],
                                            self.num_z_bins + 1)

        self.remove_mean_cont_slope = config.getboolean("remove mean cont slope")
        if self.remove_mean_cont_slope is None:
            raise ExpectedFluxError(
                "Missing argument 'remove mean cont slope' required by MeanContinuumInterpExpectedFlux"
            )

        self.remove_mean_cont_slope_last_iteration = config.getboolean(
            "remove mean cont slope last iteration")
        if self.remove_mean_cont_slope_last_iteration is None:
            raise ExpectedFluxError(
                "Missing argument 'remove mean cont slope last iteration' required by MeanContinuumInterpExpectedFlux"
            )

        self.slope_buffer_bins = config.getint("slope buffer bins")
        if self.slope_buffer_bins is None:
            raise ExpectedFluxError(
                "Missing argument 'slope buffer bins' required by MeanContinuumInterpExpectedFlux"
            )
        if self.slope_buffer_bins < 0:
            raise ExpectedFluxError(
                "Invalid argument 'slope buffer bins' required by MeanContinuumInterpExpectedFlux. "
                "It must be a non-negative integer.")

    def _initialize_mean_continuum_arrays(self):
        """Initialize mean continuum arrays
        The initialized arrays are:
        - self.get_mean_cont
        """
        # initialize the mean quasar continuum
        if self.interpolation_type == "2D":
            if self.z_bin_edges is None:
                self.logger.info(
                    "In MeanContinuumInterpExpectedFlux, 'z_bin_edges' is None. "
                    "Arrays will need to be initialized at a later time")
            else:
                self.logger.info(
                    "Initializing mean continuum arrays for 2D interpolation with z bins: "
                    "%s", self.z_bin_edges)
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
            self.logger.info(
                "Initializing mean continuum arrays for 1D interpolation")
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

    def compute_expected_flux(self, forests):
        """Compute the expected flux for a list of Forests

        Arguments
        ---------
        forests: list of Forest
        A list of Forest from which to compute the expected flux.
        """
        if self.interpolation_type == "2D" and self.infer_z_bins:
            # this sets the z_bin_edges attribute based on the data in forests
            # otherwise the z_bin_edges attribute is set in the __parse_config method
            # and this call is not needed
            self.infer_z_bins_from_data(forests)

            # initialize the mean continuum arrays with the inferred z_bin_edges
            # (failed before since the z_bin_edges were not set in the __parse_config method)
            self._initialize_mean_continuum_arrays()

        super().compute_expected_flux(forests)

    @staticmethod
    def _remove_mean_cont_slope_1d(mean_cont, buffer_bins=0):
        """Remove the slope of a 1D mean continuum by dividing by a normalized linear fit.

        Zeros occur at at most one end of the spectrum (never at both).
        Points with zero values (no coverage) and edge buffer points are excluded
        when evaluating the slope. Points with value 0 remain 0 in the returned array.

        Arguments
        ---------
        mean_cont: np.ndarray
        1D mean continuum array along Forest.log_lambda_rest_frame_grid

        buffer_bins: int - default=0
        Number of bins to trim on each side of the contiguous non-zero region
        when fitting the slope.

        Returns
        -------
        np.ndarray
        1D mean continuum array with slope removed and zero values preserved
        """
        if mean_cont.size < 2:
            return mean_cont.copy()

        if mean_cont[0] == 0:
            if mean_cont[-1] == 0:
                return mean_cont.copy()
            start = int(np.argmax(mean_cont != 0))
            end = mean_cont.size
        elif mean_cont[-1] == 0:
            start = 0
            end = int(np.argmax(mean_cont == 0))
        else:
            start = 0
            end = mean_cont.size

        if end - start < 2:
            return mean_cont.copy()

        fit_start = start + buffer_bins
        fit_end = end - buffer_bins

        if fit_end - fit_start < 2:
            fit_start = start
            fit_end = end

        grid_fit = Forest.log_lambda_rest_frame_grid[fit_start:fit_end]
        cont_fit = mean_cont[fit_start:fit_end]
        covered = cont_fit != 0
        if np.count_nonzero(covered) < 2:
            return mean_cont.copy()
        grid_fit = grid_fit[covered]
        cont_fit = cont_fit[covered]

        mean_x = np.mean(grid_fit)
        mean_y = np.mean(cont_fit)
        if mean_y == 0:
            return mean_cont.copy()

        dx = grid_fit - mean_x
        denom = np.dot(dx, dx)
        if denom == 0:
            return mean_cont.copy()

        slope = np.dot(dx, cont_fit) / denom

        grid_active = Forest.log_lambda_rest_frame_grid[start:end]
        norm_line = mean_y + slope * (grid_active - mean_x)

        # Guard against badly constrained mean continua (e.g. a redshift bin
        # edge with almost no quasars): if the fitted line is not positive, or
        # gets close to zero, over the covered region, dividing by it would
        # amplify the noise and can make the iterations diverge.
        # Leave the mean continuum unchanged in that case.
        if mean_y <= 0 or np.min(norm_line) <= MIN_NORM_LINE_FRACTION * mean_y:
            logging.getLogger(__name__).warning(
                "Mean continuum slope not removed: the fitted line is not "
                "positive or too close to zero (mean %g, min %g, max %g). "
                "This mean continuum is probably badly constrained.",
                mean_y, np.min(norm_line), np.max(norm_line))
            return mean_cont.copy()

        result = mean_cont.copy()
        result[start:end] /= norm_line
        return result

    @classmethod
    def _remove_mean_cont_slope(cls, mean_cont, buffer_bins=0):
        """Remove the slope of the mean continuum by dividing by a normalized linear fit.

        Arguments
        ---------
        mean_cont: np.ndarray
        Mean continuum array along Forest.log_lambda_rest_frame_grid (1D or 2D)

        buffer_bins: int - default=0
        Number of bins to trim on each side of the non-zero region when fitting the slope.

        Returns
        -------
        np.ndarray
        Mean continuum array with the slope removed

        Raises
        ------
        ExpectedFluxError if mean_cont does not have 1 or 2 dimensions
        """
        if mean_cont.ndim == 1:
            return cls._remove_mean_cont_slope_1d(mean_cont, buffer_bins)

        if mean_cont.ndim == 2:
            result = np.empty_like(mean_cont)
            for i in range(mean_cont.shape[0]):
                result[i] = cls._remove_mean_cont_slope_1d(mean_cont[i], buffer_bins)
            return result

        raise ExpectedFluxError(
            f"Invalid dimensions for mean_cont: {mean_cont.ndim}. "
            "Expected 1D or 2D array.")

    def compute_mean_cont(self,
                          forests):
        """Compute the mean quasar continuum over the whole sample.
        Then updates the value of self.get_mean_cont to contain it

        Arguments
        ---------
        forests: List of Forest
        A list of Forest from which to compute the deltas.
        """
        # the mean continuum is computed in all iterations but the last one
        # (see Dr16ExpectedFlux.compute_expected_flux), so the last
        # computation is number num_iterations - 1
        self.num_mean_cont_computations += 1
        is_last_computation = (self.num_mean_cont_computations ==
                               self.num_iterations - 1)
        self.remove_slope_this_computation = self.remove_mean_cont_slope and (
            self.remove_mean_cont_slope_last_iteration or
            not is_last_computation)
        if self.remove_mean_cont_slope and not self.remove_slope_this_computation:
            self.logger.progress(
                "Last computation of the mean continuum: its slope is not "
                "removed ('remove mean cont slope last iteration' = False)")

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

        # remove the slope of the mean continuum
        if self.remove_slope_this_computation:
            self.mean_cont = self._remove_mean_cont_slope(
                self.mean_cont, buffer_bins=self.slope_buffer_bins)

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

        # remove the slope of the mean continuum for each redshift bin
        if self.remove_slope_this_computation:
            self.mean_cont = self._remove_mean_cont_slope(
                self.mean_cont, buffer_bins=self.slope_buffer_bins)

        # update the interpolator with the mean continuum
        self.get_mean_cont = RegularGridInterpolator(
            (self.z_bin_edges, Forest.log_lambda_rest_frame_grid),
            self.mean_cont,
            bounds_error=False,
            fill_value=0.0,
        )

    def infer_z_bins_from_data(self, forests):
        """Infer the redshift bins from the data in forests.
        The redshift bins are set to self.z_bin_edges.

        Arguments
        ---------
        forests: List of Forest
        A list of Forest from which to compute the deltas.
        """
        self.logger.info(
            "Inferring redshift bins from the data in forests. "
        )
        z_dist = [forest.z for forest in forests]
        percentiles = np.linspace(0, 100, self.num_z_bins + 1)
        self.z_bin_edges = np.percentile(z_dist, percentiles)
        if np.any(np.diff(self.z_bin_edges) <= 0):
            raise ExpectedFluxError(
                "Cannot infer strictly increasing redshift bins; reduce 'num z bins' "
                "or use a sample with more redshift variation.")
        self.logger.info("Inferred redshift bins: %s", self.z_bin_edges)


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
    z_bin = min(max(z_bin, 0), len(z_grid) - 2)
    z_low = z_grid[z_bin]
    z_high = z_grid[z_bin + 1]

    coeff = (z_high - z) / (z_high - z_low)

    return coeff, z_bin
