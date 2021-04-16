"""This module defines the classes SdssDlaMask and Dla used in the
masking of DLAs"""
import logging
import numpy as np
import fitsio
from astropy.table import Table

from picca.delta_extraction.astronomical_objects.forest import Forest
from picca.delta_extraction.errors import MaskError
from picca.delta_extraction.mask import Mask
from picca.delta_extraction.utils import ABSORBER_IGM

# create logger
module_logger = logging.getLogger(__name__)

defaults = {
    "dla mask limit": 0.8,
}

np.random.seed(0)

class SdssDlaMask(Mask):
    """Class to mask DLAs

    Methods
    -------
    __init__
    apply_mask

    Attributes
    ----------
    los_ids: dict (from Mask)
    A dictionary with the DLAs contained in each line of sight. Keys are the
    identifier for the line of sight and values are lists of (z_abs, nhi)

    dla_mask_limit: float
    Lower limit on the DLA transmission. Transmissions below this number are
    masked

    mask: astropy.Table
    Table containing specific intervals of wavelength to be masked for DLAs
    """
    def __init__(self, config):
        """Initializes class instance.

        Arguments
        ---------
        config: configparser.SectionProxy
        Parsed options to initialize class
        """
        self.logger = logging.getLogger(__name__)
        # first load the dla catalogue
        dla_catalogue = config.get("dla catalogue")
        if dla_catalogue is None:
            raise MaskError("Missing argument 'dla catalogue' required by DlaMask")

        self.logger.progress(f"Reading DLA catalog from: {dla_catalogue}")
        columns_list = ["THING_ID", "Z", "NHI"]
        try:
            hdul = fitsio.FITS(dla_catalogue)
            cat = {col: hdul["DLACAT"][col][:] for col in columns_list}
        except OSError:
            raise MaskError(f"Error loading SdssDlaMask. File {dla_catalogue} does "
                            "not have extension 'DLACAT'")
        except ValueError:
            aux = "', '".join(columns_list)
            raise MaskError(f"Error loading SdssDlaMask. File {dla_catalogue} does "
                            f"not have fields '{aux}' in HDU 'DLACAT'")
        finally:
            hdul.close()

        # group DLAs on the same line of sight together
        self.los_ids = {}
        for thingid in np.unique(cat["THING_ID"]):
            w = (thingid == cat["THING_ID"])
            self.los_ids[thingid] = list(zip(cat["Z"][w], cat['NHI'][w]))
        num_dlas = np.sum([len(thingid) for thingid in self.los_ids.values()])

        self.logger.progress(' In catalog: {} DLAs'.format(num_dlas))
        self.logger.progress(' In catalog: {} forests have a DLA\n'.format(len(self.los_ids)))

        # setup transmission limit
        # transmissions below this number are masked
        self.dla_mask_limit = config.getfloat("dla mask limit")
        if self.dla_mask_limit is None:
            self.dla_mask_limit = defaults.get("dla mask limit")

        # load mask
        mask_file = config.get("mask file")
        if mask_file is not None:
            try:
                self.mask = Table.read(mask_file,
                                       names=('type', 'wave_min', 'wave_max', 'frame'),
                                       format='ascii')
                self.mask['log_wave_min'] = np.log10(self.mask['wave_min'])
                self.mask['log_wave_max'] = np.log10(self.mask['wave_max'])
                self.mask = self.mask['frame'] == 'RF_DLA'
            except (OSError, ValueError):
                raise MaskError("ERROR: Error while reading mask_file file "
                                f"{mask_file}")
        else:
            self.mask = Table(names=('type', 'wave_min', 'wave_max', 'frame',
                                     'log_wave_min', 'log_wave_max'))

    def apply_mask(self, forest):
        """Applies the mask. The mask is done by removing the affected
        pixels from the arrays in Forest.mask_fields

        Arguments
        ---------
        forest: Forest
        A Forest instance to which the correction is applied

        Raises
        ------
        MaskError if forest instance does not have the attribute
        'log_lambda'
        """
        if Forest.wave_solution != "log":
            raise MaskError("SdssDlaMask should only be applied when "
                            "Forest.wave_solution is 'log'. Found: "
                            f"{Forest.wave_solution}")

        # load DLAs
        if self.los_ids.get(forest.los_id) is not None:
            dla_transmission = np.ones(len(forest.log_lambda))
            for (z_abs, nhi) in self.los_ids.get(forest.los_id):
                dla_transmission *= DlaProfile(forest.log_lambda, z_abs, nhi).transmission

            # find out which pixels to mask
            w = dla_transmission > self.dla_mask_limit
            if len(self.mask) > 0:
                for mask_range in self.mask:
                    for (z_abs, nhi) in self.los_ids.get(forest.los_id):
                        w &= ((forest.log_lambda - np.log10(1. + z_abs) <
                               mask_range['log_wave_min']) |
                              (forest.log_lambda - np.log10(1. + z_abs) >
                               mask_range['log_wave_max']))

            # do the actual masking
            forest.transmission_correction *= dla_transmission
            for param in Forest.mask_fields:
                setattr(forest, param, getattr(forest, param)[w])


class DlaProfile:
    """Class to represent Damped Lyman-alpha Absorbers.

    Methods
    -------
    __init__
    profile_lya_absorption
    profile_lyb_absorption
    tau_lya
    tau_lyb
    voigt

    Attributes
    ----------
    log_lambda: array of float
    Logarithm of the wavelength (in Angs)

    nhi: float
    DLA column density in log10(cm^-2)

    transmission: array of floats
    Decrease of the transmitted flux due to the presence of a DLA

    z_abs: float
    Redshift of the absorption
    """
    def __init__(self, log_lambda, z_abs, nhi):
        """Initializes class instance."""
        self.z_abs = z_abs
        self.nhi = nhi

        self.transmission = self.profile_lya_absorption(10**log_lambda,
                                                        z_abs, nhi)
        self.transmission *= self.profile_lyb_absorption(
            10**log_lambda, z_abs, nhi)

    @staticmethod
    def profile_lya_absorption(lambda_, z_abs, nhi):
        """Computes the absorption profile for Lyman-alpha absorption.

        Args:
            lambda_: array of floats
                Wavelength (in Angs)
            z_abs: float
                Redshift of the absorption
            nhi: float
                DLA column density in log10(cm^-2)

        Returns:
            The absorption profile.
        """
        return np.exp(-DlaProfile.tau_lya(lambda_, z_abs, nhi))

    @staticmethod
    def profile_lyb_absorption(lambda_, z_abs, nhi):
        """Computes the absorption profile for Lyman-beta absorption.

        Args:
            lambda_: array of floats
                Wavelength (in Angs)
            z_abs: float
                Redshift of the absorption
            nhi: float
                DLA column density in log10(cm^-2)

        Returns:
            The absorption profile.
        """
        return np.exp(-DlaProfile.tau_lyb(lambda_, z_abs, nhi))

    ### Implementation of Pasquier code,
    ###     also in Rutten 2003 at 3.3.3
    @staticmethod
    def tau_lya(lambda_, z_abs, nhi):
        """Computes the optical depth for Lyman-alpha absorption.

        Args:
            lambda_: array of floats
                Wavelength (in Angs)
            z_abs: float
                Redshift of the absorption
            nhi: float
                DLA column density in log10(cm^-2)

        Returns:
            The optical depth.
        """
        lambda_lya = ABSORBER_IGM["LYA"]  ## Lya wavelength [A]
        gamma = 6.625e8  ## damping constant of the transition [s^-1]
        osc_strength = 0.4164  ## oscillator strength of the atomic transition
        speed_light = 3e8  ## speed of light [m/s]
        thermal_velocity = 30000.  ## sqrt(2*k*T/m_proton) with
        ## T = 5*10^4 ## [m.s^-1]
        nhi_cm2 = 10**nhi  ## column density [cm^-2]
        lambda_rest_frame = lambda_ / (1 + z_abs)
        ## wavelength at DLA restframe [A]

        u_voight = ((speed_light / thermal_velocity) *
                    (lambda_lya / lambda_rest_frame - 1))
        ## dimensionless frequency offset in Doppler widths.
        a_voight = lambda_lya * 1e-10 * gamma / (4 * np.pi * thermal_velocity)
        ## Voigt damping parameter
        voigt = DlaProfile.voigt(a_voight, u_voight)
        thermal_velocity /= 1000.
        ## 1.497e-16 = e**2/(4*sqrt(pi)*epsilon0*m_electron*c)*1e-10
        ## [m^2.s^-1.m/]
        ## we have b/1000 & 1.497e-15 to convert
        ## 1.497e-15*osc_strength*lambda_rest_frame*h/n to cm^2
        tau = (1.497e-15 * nhi_cm2 * osc_strength * lambda_rest_frame * voigt /
               thermal_velocity)
        return tau

    @staticmethod
    def tau_lyb(lambda_, z_abs, nhi):
        """Computes the optical depth for Lyman-beta absorption.

        Args:
            lambda_: array of floats
                Wavelength (in Angs)
            z_abs: float
                Redshift of the absorption
            nhi: float
                DLA column density in log10(cm^-2)

        Returns:
            The optical depth.
        """
        lam_lyb = ABSORBER_IGM["LYB"]
        gamma = 0.079120
        osc_strength = 1.897e8
        speed_light = 3e8  ## speed of light m/s
        thermal_velocity = 30000.
        nhi_cm2 = 10**nhi
        lambda_rest_frame = lambda_ / (1 + z_abs)

        u_voight = ((speed_light / thermal_velocity) *
                    (lam_lyb / lambda_rest_frame - 1))
        a_voight = lam_lyb * 1e-10 * gamma / (4 * np.pi * thermal_velocity)
        voigt = DlaProfile.voigt(a_voight, u_voight)
        thermal_velocity /= 1000.
        tau = (1.497e-15 * nhi_cm2 * osc_strength * lambda_rest_frame * voigt /
               thermal_velocity)
        return tau

    @staticmethod
    def voigt(a_voight, u_voight):
        """Computes the classical Voigt function

        Args:
            a_voight: array of floats
            Voigt damping parameter.

            u_voight: array of floats
            Dimensionless frequency offset in Doppler widths.

        Returns:
            The Voigt function for each element in a, u
        """
        nun_points = 1000
        gaussian_dist = np.random.normal(size=nun_points) * np.sqrt(2)
        unnormalized_voigt = np.mean(
            1 / (a_voight**2 + (gaussian_dist[:, None] - u_voight)**2), axis=0)
        return unnormalized_voigt * a_voight / np.sqrt(np.pi)
