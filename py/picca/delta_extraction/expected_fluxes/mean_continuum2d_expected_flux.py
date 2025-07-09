"""This module defines the class Dr16ExpectedFlux"""
import logging

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from picca.delta_extraction.errors import ExpectedFluxError
from picca.delta_extraction.astronomical_objects.forest import Forest
from picca.delta_extraction.expected_fluxes.dr16_expected_flux import Dr16ExpectedFlux, defaults, accepted_options
from picca.delta_extraction.utils import (update_accepted_options,
                                          update_default_options,
                                          ABSORBER_IGM)

accepted_options = update_accepted_options(accepted_options, [
    "limit z", "num z bins" 
])

defaults = update_default_options(
    defaults, {
        "limit z": (1.94, 4.5),
        "num z bins": 10,
    })


class MeanContinuum2dExpectedFlux(Dr16ExpectedFlux):
    """Class to the expected flux as done in the DR16 SDSS analysys
    The mean expected flux is calculated iteratively as explained in
    du Mas des Bourboux et al. (2020) except that the mean continuum is
    computed in 2D (wavelength and redshift) instead of 1D.

    Methods
    -------
    (see Dr16ExpectedFlux in py/picca/delta_extraction/expected_fluxes/dr16_expected_flux.py)
    
    Attributes
    ----------
    (see Dr16ExpectedFlux in py/picca/delta_extraction/expected_fluxes/dr16_expected_flux.py)

    get_mean_cont: scipy.interpolate.RegularGridInterpolator
    Interpolation function to compute the unabsorbed mean quasar continua.

    get_mean_cont_weight: scipy.interpolate.RegularGridInterpolator
    Interpolation function to compute the weights associated with the unabsorbed
    mean quasar continua.

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
        self.limit_z = None
        self.num_z_bins = None
        self.z_bins = None
        self.z_centers = None
        self.__parse_config(config)

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
                "Missing or invalid argument 'num z bins' required by MeanContinuum2dExpectedFlux")
        self.num_z_bins = num_z_bins

        self.z_bins = np.linspace(self.limit_z[0], self.limit_z[1],
                                  self.num_z_bins + 1)
        self.z_centers = (self.z_bins[:-1] + self.z_bins[1:]) / 2

    def _initialize_mean_continuum_arrays(self):
        """Initialize mean continuum arrays
        The initialized arrays are:
        - self.get_mean_cont
        - self.get_mean_cont_weight
        """
        # initialize the mean quasar continuum
        # TODO: maybe we can drop this and compute first the mean quasar
        # continuum on compute_expected_flux
        mean_cont = np.ones(
            (self.z_bins.size - 1, Forest.log_lambda_rest_frame_grid.size))
        mean_cont_weight = np.zeros(
            (self.z_bins.size - 1, Forest.log_lambda_rest_frame_grid.size))
        
        self.get_mean_cont = RegularGridInterpolator(
            (self.z_centers, Forest.log_lambda_rest_frame_grid), mean_cont, bounds_error=False, fill_value=0.0
        )
        self.get_mean_cont_weight = RegularGridInterpolator(
            (self.z_centers, Forest.log_lambda_rest_frame_grid), mean_cont_weight, bounds_error=False, fill_value=0.0
        )

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
        
        mean_cont = np.zeros(
            (self.z_bins.size - 1, Forest.log_lambda_rest_frame_grid.size))
        mean_cont_weight = np.zeros(
            (self.z_bins.size - 1, Forest.log_lambda_rest_frame_grid.size))


        # first compute <F/C> in bins. C=Cont_old*spectrum_dependent_fitting_fct
        # (and Cont_old is constant for all spectra in a bin), thus we actually
        # compute
        #    1/Cont_old * <F/spectrum_dependent_fitting_function>
        for forest in forests:
            if forest.bad_continuum_reason is not None:
                continue
            # Compute redshifts, use quasar redshift
            forest_z = forest.z * np.ones_like(forest.log_lambda) 

            # Bin indices for each pixel
            lam_bins = Forest.find_bins(
                forest.log_lambda - np.log10(1 + forest.z),
                Forest.log_lambda_rest_frame_grid)
            z_bins_idx = np.digitize(forest_z, self.z_bins) - 1


            weights = self.compute_forest_weights(forest, forest.continuum)
            forest_continuum = which_cont(forest)

            # Accumulate in 2D
            for lam_bin, zbin, continuum, weight in zip(lam_bins, z_bins_idx, forest_continuum, weights):
                if 0 <= lam_bin < Forest.log_lambda_rest_frame_grid.size and 0 <= zbin < self.z_bins.size - 1:
                    mean_cont[zbin, lam_bin] += continuum * weight
                    mean_cont_weight[zbin, lam_bin] += weight
       
        # Normalize
        w = mean_cont_weight > 0
        mean_cont[w] /= mean_cont_weight[w]
        mean_cont /= mean_cont[w].mean()
        
    
        # 2D interpolator for mean continuum
        self.get_mean_cont = RegularGridInterpolator(
            (self.z_centers, Forest.log_lambda_rest_frame_grid), mean_cont, bounds_error=False, fill_value=0.0
        )
        # 2D interpolator for weights
        self.get_mean_cont_weight = RegularGridInterpolator(
            (self.z_centers, Forest.log_lambda_rest_frame_grid), mean_cont_weight, bounds_error=False, fill_value=0.0
        )

    def hdu_cont(self, results):
        """Add to the results file an HDU with the continuum information

        Arguments
        ---------
        results: fitsio.FITS
        The open fits file
        """
        results.write([
            Forest.log_lambda_rest_frame_grid,
            self.get_mean_cont(Forest.log_lambda_rest_frame_grid),
            self.get_mean_cont_weight(Forest.log_lambda_rest_frame_grid),
        ],
                      names=['LOGLAM_REST', 'MEAN_CONT', 'WEIGHT'],
                      units=['log(Angstrom)', Forest.flux_units, ''],
                      extname='CONT')
        results["CONT"].write_comment("Mean quasar continuum")
        results["CONT"].write_checksum()

        # Create meshgrid for evaluation
        z_meshgrid, log_lam_mesh_grid = np.meshgrid(self.z_centers, Forest.log_lambda_rest_frame_grid, indexing='ij')
        points = np.stack([z_meshgrid.ravel(), log_lam_mesh_grid.ravel()], axis=-1)
        mean_cont_2d = self.get_mean_cont(points).reshape(z_meshgrid.shape)
        mean_cont_weight_2d = self.get_mean_cont_weight(points).reshape(z_meshgrid.shape)

        results.write([
            z_meshgrid,
            log_lam_mesh_grid,
            mean_cont_2d,
            mean_cont_weight_2d,
        ],
            names=['Z_CENTERS', 'LOGLAM_REST', 'MEAN_CONT', 'WEIGHT'],
            units=['', 'log(Angstrom)', Forest.flux_units, ''],
            extname='CONT')
        results["CONT"].write_comment("2D mean quasar continuum (z, loglam)")
        results["CONT"].write_checksum()
