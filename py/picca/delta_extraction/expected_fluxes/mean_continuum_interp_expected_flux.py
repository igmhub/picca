"""This module defines the class MeanContinuumInterpExpectedFlux"""
import logging
import multiprocessing

import numba
import numpy as np
from scipy.interpolate import interp1d, RegularGridInterpolator

from picca.delta_extraction.errors import ExpectedFluxError
from picca.delta_extraction.astronomical_objects.forest import Forest
#from picca.delta_extraction.expected_fluxes.dr16_expected_flux import Dr16ExpectedFlux, defaults, accepted_options
from picca.delta_extraction.expected_fluxes.dr16_fixed_fudge_expected_flux import Dr16FixedFudgeExpectedFlux, defaults, accepted_options
from picca.delta_extraction.utils import (update_accepted_options,
                                          update_default_options,
                                          ABSORBER_IGM)

accepted_options = update_accepted_options(accepted_options, [
    "interpolation type", "limit z", "num z bins"
])

defaults = update_default_options(
    defaults, {
        "interpolation type": "1D",
        "limit z": (1.94, 4.5),
        "num z bins": 10,
    })

ACCEPTED_INTERPOLATION_TYPES = ["1D", "2D"]

class MeanContinuumInterpExpectedFlux(Dr16FixedFudgeExpectedFlux):
    """Class to the expected flux as done in the DR16 SDSS analysys
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

    z_bins: np.ndarray
    Bins in redshift used to compute the mean continuum.

    z_centers: np.ndarray
    Centers of the redshift bins used to compute the mean continuum.
    This is used to interpolate the mean continuum and its weights.
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
        self.z_bins = None
        self.z_centers = None
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
                "Missing argument 'interpolation type' required by MeanContinuum2dExpectedFlux")
        if self.interpolation_type not in ACCEPTED_INTERPOLATION_TYPES:
            raise ExpectedFluxError(
                f"Invalid interpolation type '{self.interpolation_type}' "
                f"required by MeanContinuum2dExpectedFlux. "
                f"Accepted values are {ACCEPTED_INTERPOLATION_TYPES}")
        
        if self.interpolation_type == "2D":
            limit_z_string = config.get("limit z")
            if limit_z_string is None:
                raise ExpectedFluxError(
                    "Missing argument 'limit z' required by MeanContinuum2dExpectedFlux")
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
                    "Missing argument 'num z bins' required by MeanContinuum2dExpectedFlux")
            self.num_z_bins = num_z_bins

            self.z_bins = np.linspace(self.limit_z[0], self.limit_z[1],
                                    self.num_z_bins + 1)
            self.z_centers = (self.z_bins[:-1] + self.z_bins[1:]) / 2

    def _initialize_mean_continuum_arrays(self):
        """Initialize mean continuum arrays
        The initialized arrays are:
        - self.get_mean_cont
        """
        # initialize the mean quasar continuum
        if self.interpolation_type == "2D":
            mean_cont = np.ones(
                (self.z_bins.size - 1, Forest.log_lambda_rest_frame_grid.size))
            # fill_value cannot be "extrapolate" for RegularGridInterpolator
            # so we use 0.0 instead
            self.get_mean_cont = RegularGridInterpolator(
                (self.z_centers, Forest.log_lambda_rest_frame_grid), 
                mean_cont, bounds_error=False, fill_value=0.0
            )
        elif self.interpolation_type == "1D":
            self.mean_cont = np.ones(Forest.log_lambda_rest_frame_grid.size)
            
            self.get_mean_cont = interp1d(
                Forest.log_lambda_rest_frame_grid,
                self.mean_cont,
                fill_value='extrapolate'
            )
        # this should never happen, but just in case
        else: # pragma: no cover
            raise ExpectedFluxError(
                f"Invalid interpolation type '{self.interpolation_type}' "
                f"required by MeanContinuum2dExpectedFlux. "
                f"Accepted values are {ACCEPTED_INTERPOLATION_TYPES}")

    def compute_mean_cont(self, forests, which_cont=lambda forest: forest.continuum):
        """Compute the mean quasar continuum over the whole sample.
        Then updates the value of self.get_mean_cont to contain it

        Arguments
        ---------
        forests: List of Forest
        A list of Forest from which to compute the deltas.

        which_cont: Function or lambda
        Should return what to use as continuum given a forest
        """
        if self.interpolation_type == "1D":
            self.compute_mean_cont_1d(forests, which_cont)
        elif self.interpolation_type == "2D":
            self.compute_mean_cont_2d(forests, which_cont)
        # this should never happen, but just in case
        else: # pragma: no cover
            raise ExpectedFluxError(
                f"Invalid interpolation type '{self.interpolation_type}' "
                f"required by MeanContinuum2dExpectedFlux. "
                f"Accepted values are {ACCEPTED_INTERPOLATION_TYPES}")
        
    def compute_mean_cont_1d(self, forests, which_cont=lambda forest: forest.continuum):
        """Compute the mean quasar continuum over the whole sample.
        Then updates the value of self.get_mean_cont to contain it
        The mean continuum is computed as a function of the rest-frame 
        wavelength.

        Arguments
        ---------
        forests: List of Forest
        A list of Forest from which to compute the deltas.

        which_cont: Function or lambda
        Should return what to use as continuum given a forest
        """
        
        """
        # numba implementation shows a memory leak when using
        A_matrix = np.zeros(
            (Forest.log_lambda_rest_frame_grid.size, Forest.log_lambda_rest_frame_grid.size)
        )
        B_matrix = np.zeros(Forest.log_lambda_rest_frame_grid.size)   

        context = multiprocessing.get_context('fork')
        with context.Pool(processes=self.num_processors) as pool:
            arguments = [(
                    Forest.log_lambda_rest_frame_grid, 
                    forest.log_lambda, 
                    forest.flux, 
                    forest.continuum, 
                    forest.z, 
                    self.compute_forest_weights(forest, forest.continuum)
                    )
                    for forest in forests if forest.bad_continuum_reason is None]
            imap_it = pool.starmap(compute_mean_cont_1d, arguments)

            for partial_A_matrix, partial_B_matrix in imap_it:
                A_matrix += partial_A_matrix
                B_matrix += partial_B_matrix

        # Take care of unstable solutions
        # If the diagonal of A_matrix is zero, we set it to 1.0
        # This is a workaround for the case where there is no coverage
        # for some wavelengths.
        w = np.diagonal(A_matrix) == 0
        A_matrix[w, w] = 1.0

        # Solve the linear system A_matrix * mean_cont = B_matrix
        self.mean_cont = np.linalg.solve(A_matrix, B_matrix)
        
        # update the interpolator with the mean continuum
        self.get_mean_cont = interp1d(
            Forest.log_lambda_rest_frame_grid,
            self.mean_cont,
            fill_value='extrapolate'
        )

        return
        """
        # Old implementation without numba and without parallelization
        A_matrix = np.zeros(
            (Forest.log_lambda_rest_frame_grid.size, Forest.log_lambda_rest_frame_grid.size)
        )
        B_matrix = np.zeros(Forest.log_lambda_rest_frame_grid.size)   

        for forest in forests:
            if forest.bad_continuum_reason is not None:
                continue

            log_lambda_rf = forest.log_lambda - np.log10(1 + forest.z)
            weights = self.compute_forest_weights(forest, forest.continuum)
            coeffs, rf_wavelength_bin = interp_coeff_lambda(
                log_lambda_rf,
                Forest.log_lambda_rest_frame_grid)
            
            w = np.where(forest.continuum > 0)
            B_matrix[rf_wavelength_bin[w]] += weights[w] * coeffs[w] * forest.flux[w] / forest.continuum[w]
            
            w = np.where((forest.continuum > 0) & (rf_wavelength_bin < Forest.log_lambda_rest_frame_grid.size - 1))
            B_matrix[rf_wavelength_bin[w] + 1] += weights[w] * (1 - coeffs[w]) * forest.flux[w] / forest.continuum[w]
            
            A_matrix[rf_wavelength_bin, rf_wavelength_bin] += weights * coeffs * coeffs
            w = np.where(rf_wavelength_bin < Forest.log_lambda_rest_frame_grid.size - 1)
            A_matrix[rf_wavelength_bin[w] + 1, rf_wavelength_bin[w]] += weights[w] * coeffs[w] * (1 - coeffs[w])
            A_matrix[rf_wavelength_bin[w], rf_wavelength_bin[w] + 1] += weights[w] * coeffs[w] * (1 - coeffs[w])
            A_matrix[rf_wavelength_bin[w] + 1, rf_wavelength_bin[w] + 1] += weights[w] * (1 - coeffs[w]) * (1 - coeffs[w])


        # Take care of unstable solutions
        # If the diagonal of A_matrix is zero, we set it to 1.0
        # This is a workaround for the case where there is no coverage
        # for some wavelengths.
        w = np.diagonal(A_matrix) == 0
        A_matrix[w, w] = 1.0

        # Solve the linear system A_matrix * mean_cont = B_matrix
        self.mean_cont = np.linalg.solve(A_matrix, B_matrix)
        try:
            mean_cont = np.linalg.solve(A_matrix, B_matrix)
        except np.linalg.LinAlgError:
            raise ExpectedFluxError(
                "The linear system could not be solved. "
                "This may be due to a lack of coverage for some wavelengths."
            )
            mean_cont, *_ = np.linalg.lstsq(A_matrix, B_matrix, rcond=None)
        
        # update the interpolator with the mean continuum
        self.get_mean_cont = interp1d(
            Forest.log_lambda_rest_frame_grid,
            self.mean_cont,
            fill_value='extrapolate'
        )

    def compute_mean_cont_2d(self, forests, which_cont=lambda forest: forest.continuum):
        """Compute the mean quasar continuum over the whole sample.
        Then updates the value of self.get_mean_cont to contain it
        The mean continuum is computed as a function of the rest-frame 
        wavelength and redshift.

        Arguments
        ---------
        forests: List of Forest
        A list of Forest from which to compute the deltas.

        which_cont: Function or lambda
        Should return what to use as continuum given a forest
        """
        self.logger.debug("Entering compute_mean_cont_2d")
        # for simplicity we introduce a new index 
        # combined_bin = z_bin + N_z_bins * rf_wavelength_bin
        # where z_bin is the index of the redshift bin and rf_wavelength_bin
        # is the index of the rest-frame wavelength bin.
        # This allows us to use a similar logic as in the 1D case.
        matrix_size = self.num_z_bins * Forest.log_lambda_rest_frame_grid.size

        A_matrix = np.zeros(
            (matrix_size, matrix_size)
        )
        B_matrix = np.zeros(matrix_size)   

        for forest in forests:
            if forest.bad_continuum_reason is not None:
                continue

            log_lambda_rf = forest.log_lambda - np.log10(1 + forest.z)
            weights = self.compute_forest_weights(forest, forest.continuum)
            rf_wavelength_coeffs, rf_wavelength_bin = interp_coeff_lambda(
                log_lambda_rf,
                Forest.log_lambda_rest_frame_grid)
            z_coeffs, z_bin = interp_coeff_z(
                forest.z,
                self.z_centers)
            
            # combined_bin is the index of the bin in the 2D matrix
            combined_bin = z_bin + self.num_z_bins * rf_wavelength_bin
            combined_bin_plus_wavelength = z_bin + self.num_z_bins * (rf_wavelength_bin + 1)
            combined_bin_plus_z = z_bin + 1 + self.num_z_bins * (rf_wavelength_bin)
            combined_bin_plus_both = z_bin + 1 + self.num_z_bins * (rf_wavelength_bin + 1)

            # Fill the B_matrix
            # diagonal elements
            w = np.where(forest.continuum > 0)
            B_matrix[combined_bin[w]] += weights[w] * z_coeffs * rf_wavelength_coeffs[w] * forest.flux[w] / forest.continuum[w]
            # off-diagonal elements
            w = np.where((forest.continuum > 0) & (combined_bin_plus_wavelength < matrix_size))
            B_matrix[combined_bin_plus_wavelength[w] + 1] += weights[w] * z_coeffs * (1 - rf_wavelength_coeffs[w]) * forest.flux[w] / forest.continuum[w]
            w = np.where((forest.continuum > 0) & (combined_bin_plus_z < matrix_size))
            B_matrix[combined_bin_plus_z[w]] += weights[w] * (1 - z_coeffs) * rf_wavelength_coeffs[w] * forest.flux[w] / forest.continuum[w]
            w = np.where((forest.continuum > 0) & (combined_bin_plus_both < matrix_size))
            B_matrix[combined_bin_plus_both[w]] += weights[w] * (1 - z_coeffs) * (1 - rf_wavelength_coeffs[w]) * forest.flux[w] / forest.continuum[w]

            # Fill the A_matrix
            # diagonal elements
            A_matrix[combined_bin, combined_bin] += weights * z_coeffs * z_coeffs * rf_wavelength_coeffs * rf_wavelength_coeffs
            # off-diagonal elements
            w = np.where(combined_bin_plus_wavelength < matrix_size)
            A_matrix[combined_bin[w], combined_bin_plus_wavelength[w]] += weights[w] * z_coeffs * z_coeffs * rf_wavelength_coeffs[w] * (1 - rf_wavelength_coeffs[w])
            A_matrix[combined_bin_plus_wavelength[w], combined_bin[w]] += weights[w] * z_coeffs * z_coeffs * rf_wavelength_coeffs[w] * (1 - rf_wavelength_coeffs[w])
            A_matrix[combined_bin_plus_wavelength[w], combined_bin_plus_wavelength[w]] += weights[w] * z_coeffs * z_coeffs * (1 - rf_wavelength_coeffs[w]) * (1 - rf_wavelength_coeffs[w])
            w = np.where(combined_bin_plus_z < matrix_size)
            A_matrix[combined_bin[w], combined_bin_plus_z[w]] += weights[w] * z_coeffs * (1 - z_coeffs) * rf_wavelength_coeffs[w] * rf_wavelength_coeffs[w]
            A_matrix[combined_bin_plus_z[w], combined_bin[w]] += weights[w] * z_coeffs * (1 - z_coeffs) * rf_wavelength_coeffs[w] * rf_wavelength_coeffs[w]
            A_matrix[combined_bin_plus_z[w], combined_bin_plus_z[w]] += weights[w] * (1 - z_coeffs) * (1 - z_coeffs) * rf_wavelength_coeffs[w] * rf_wavelength_coeffs[w]
            w = np.where(combined_bin_plus_both < matrix_size)
            A_matrix[combined_bin[w], combined_bin_plus_both[w]] += weights[w] * z_coeffs * (1 - z_coeffs) * rf_wavelength_coeffs[w] * (1 - rf_wavelength_coeffs[w])
            A_matrix[combined_bin_plus_both[w], combined_bin[w]] += weights[w] * z_coeffs * (1 - z_coeffs) * rf_wavelength_coeffs[w] * (1 - rf_wavelength_coeffs[w])
            A_matrix[combined_bin_plus_both[w], combined_bin_plus_both[w]] += weights[w] * (1 - z_coeffs) * (1 - z_coeffs) * (1 - rf_wavelength_coeffs[w]) * (1 - rf_wavelength_coeffs[w])
            # cross terms
            w = np.where((combined_bin_plus_wavelength < matrix_size) & (combined_bin_plus_z < matrix_size))
            A_matrix[combined_bin_plus_wavelength[w], combined_bin_plus_z[w]] += weights[w] * z_coeffs * (1 - z_coeffs) * rf_wavelength_coeffs[w] * (1 - rf_wavelength_coeffs[w])
            A_matrix[combined_bin_plus_z[w], combined_bin_plus_wavelength[w]] += weights[w] * z_coeffs * (1 - z_coeffs) * rf_wavelength_coeffs[w] * (1 - rf_wavelength_coeffs[w])
            w = np.where((combined_bin_plus_wavelength < matrix_size) & (combined_bin_plus_both < matrix_size))
            A_matrix[combined_bin_plus_wavelength[w], combined_bin_plus_both[w]] += weights[w] * z_coeffs * (1 - z_coeffs) * (1 - rf_wavelength_coeffs[w]) * (1 - rf_wavelength_coeffs[w])
            A_matrix[combined_bin_plus_both[w], combined_bin_plus_wavelength[w]] += weights[w] * z_coeffs * (1 - z_coeffs) * (1 - rf_wavelength_coeffs[w]) * (1 - rf_wavelength_coeffs[w])
            w = np.where((combined_bin_plus_z < matrix_size) & (combined_bin_plus_both < matrix_size))
            A_matrix[combined_bin_plus_z[w], combined_bin_plus_both[w]] += weights[w] * (1 - z_coeffs) * (1 - z_coeffs) * rf_wavelength_coeffs[w] * (1 - rf_wavelength_coeffs[w])
            A_matrix[combined_bin_plus_both[w], combined_bin_plus_z[w]] += weights[w] * (1 - z_coeffs) * (1 - z_coeffs) * rf_wavelength_coeffs[w] * (1 - rf_wavelength_coeffs[w])

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
        # Undo the new indexing
        self.mean_cont = mean_cont.reshape(
            (self.num_z_bins, Forest.log_lambda_rest_frame_grid.size))
        
        # update the interpolator with the mean continuum
        self.get_mean_cont = RegularGridInterpolator(
            (self.z_centers, Forest.log_lambda_rest_frame_grid), 
            self.mean_cont, bounds_error=False, fill_value=0.0,
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
            z_meshgrid, log_lam_mesh_grid = np.meshgrid(self.z_centers, Forest.log_lambda_rest_frame_grid, indexing='ij')
            points = np.stack([z_meshgrid.ravel(), log_lam_mesh_grid.ravel()], axis=-1)
            mean_cont_2d = self.get_mean_cont(points).reshape(z_meshgrid.shape)
            
            results.write([
                z_meshgrid,
                log_lam_mesh_grid,
                mean_cont_2d,
            ],
                names=['Z_CENTER', 'LOGLAM_REST', 'MEAN_CONT'],
                units=['', 'log(Angstrom)', Forest.flux_units],
                extname='CONT')
            results["CONT"].write_comment("2D mean quasar continuum (z, loglam)")
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
        else: # pragma: no cover
            raise ExpectedFluxError(
                f"Invalid interpolation type '{self.interpolation_type}' "
                f"required by MeanContinuum2dExpectedFlux. "
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

    coeff = (rf_wavelength_high - rf_wavelength) / (rf_wavelength_high - rf_wavelength_low)

    return coeff, rf_wavelength_bin

@numba.njit()
def interp_coeff_z(z, z_grid):
    """Compute the interpolation coefficients for a given redshift.

    Arguments
    ---------
    z: float
    Redshift
    
    z: np.ndarray
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

@numba.njit()
def compute_mean_cont_1d(log_lambda_rest_frame_grid, log_lambda, flux, continuum, redshift, weight):
    """Compute the mean quasar continuum over the whole sample.
    Then updates the value of self.get_mean_cont to contain it
    The mean continuum is computed as a function of the rest-frame 
    wavelength.

    Arguments
    ---------
    log_lambda_rest_frame_grid: np.ndarray
    A 1D array of rest-frame wavelengths (in Angstroms) where the continuum is defined.

    which_cont: Function or lambda
    Should return what to use as continuum given a forest
    """
    A_matrix = np.zeros(
        (log_lambda_rest_frame_grid.size, log_lambda_rest_frame_grid.size)
    )
    B_matrix = np.zeros(log_lambda_rest_frame_grid.size)

    log_lambda_rf = log_lambda - np.log10(1 + redshift)
    coeffs, rf_wavelength_bin = interp_coeff_lambda(
            log_lambda_rf,
            log_lambda_rest_frame_grid)
    
    w = np.where(continuum > 0)
    B_matrix[rf_wavelength_bin[w]] += weight[w] * coeffs[w] * flux[w] / continuum[w]

    w = np.where((continuum > 0) & (rf_wavelength_bin < log_lambda_rest_frame_grid.size - 1))
    B_matrix[rf_wavelength_bin[w] + 1] += weight[w] * (1 - coeffs[w]) * flux[w] / continuum[w]

    # diagonal elements
    #A_matrix[rf_wavelength_bin, rf_wavelength_bin] += weight * coeffs * coeffs
    for index in range(rf_wavelength_bin.size):
        A_matrix[rf_wavelength_bin[index], rf_wavelength_bin[index]] += weight[index] * coeffs[index] * coeffs[index]

    # Off-diagonal elements
    w = np.where(rf_wavelength_bin < log_lambda_rest_frame_grid.size - 1)
    for index in w[0]:
        A_matrix[rf_wavelength_bin[index] + 1, rf_wavelength_bin[index]] += weight[index] * coeffs[index] * (1 - coeffs[index])
        A_matrix[rf_wavelength_bin[index], rf_wavelength_bin[index] + 1] += weight[index] * coeffs[index] * (1 - coeffs[index])
        A_matrix[rf_wavelength_bin[index] + 1, rf_wavelength_bin[index] + 1] += weight[index] * (1 - coeffs[index]) * (1 - coeffs[index])
    #A_matrix[rf_wavelength_bin[w] + 1, rf_wavelength_bin[w]] += weight[w] * coeffs[w] * (1 - coeffs[w])
    #A_matrix[rf_wavelength_bin[w], rf_wavelength_bin[w] + 1] += weight[w] * coeffs[w] * (1 - coeffs[w])
    #A_matrix[rf_wavelength_bin[w] + 1, rf_wavelength_bin[w] + 1] += weight[w] * (1 - coeffs[w]) * (1 - coeffs[w])

    return A_matrix, B_matrix