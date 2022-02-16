"""This module defines data structure to deal with Damped Lyman-alpha
Absorbers (DLAs)

This module provides with one class (DLA). See the respective
docstrings for more details
"""
import numpy as np

from . import constants

np.random.seed(0)
num_points = 10000
gaussian_dist = np.random.normal(size=num_points) * np.sqrt(2)

class DLA:
    """Class to represent Damped Lyman-alpha Absorbers.

    Attributes:
        thingid: int
            ThingID of the observation.
        z_abs: float
            Redshift of the absorption
        nhi: float
            DLA column density in log10(cm^-2)
        transmission: array of floats
            Decrease of the transmitted flux due to the presence of a DLA

    Methods:
        __init__: Initialize class instance.
        profile_lya_absorption: Computes the absorption profile for Lyman-alpha
            absorption.
        tau_lya: Computes the optical depth for Lyman-alpha absorption.
        profile_lyb_absorption: Computes the absorption profile for Lyman-beta
            absorption.
        tau_lyb: Computes the optical depth for Lyman-beta absorption.
        voigt: Computes the classical Voigt function
    """

    def __init__(self, data, z_abs, nhi):
        """Initializes class instance."""
        self.thingid = data.thingid
        self.z_abs = z_abs
        self.nhi = nhi

        self.transmission = self.profile_lya_absorption(10**data.log_lambda,
                                                        z_abs, nhi)
        self.transmission *= self.profile_lyb_absorption(
            10**data.log_lambda, z_abs, nhi)

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
        return np.exp(-DLA.tau_lya(lambda_, z_abs, nhi))

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
        lambda_lya = constants.ABSORBER_IGM["LYA"]  ## Lya wavelength [A]
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
        voigt = DLA.voigt(a_voight, u_voight)
        thermal_velocity /= 1000.
        ## 1.497e-16 = e**2/(4*sqrt(pi)*epsilon0*m_electron*c)*1e-10
        ## [m^2.s^-1.m/]
        ## we have b/1000 & 1.497e-15 to convert
        ## 1.497e-15*osc_strength*lambda_rest_frame*h/n to cm^2
        tau = (1.497e-15 * nhi_cm2 * osc_strength * lambda_rest_frame * voigt /
               thermal_velocity)
        return tau

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
        return np.exp(-DLA.tau_lyb(lambda_, z_abs, nhi))

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
        lam_lyb = constants.ABSORBER_IGM["LYB"]
        gamma = 0.079120
        osc_strength = 1.897e8
        speed_light = 3e8  ## speed of light m/s
        thermal_velocity = 30000.
        nhi_cm2 = 10**nhi
        lambda_rest_frame = lambda_ / (1 + z_abs)

        u_voight = ((speed_light / thermal_velocity) *
                    (lam_lyb / lambda_rest_frame - 1))
        a_voight = lam_lyb * 1e-10 * gamma / (4 * np.pi * thermal_velocity)
        voigt = DLA.voigt(a_voight, u_voight)
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
        unnormalized_voigt = np.mean(
            1 / (a_voight**2 + (gaussian_dist[:, None] - u_voight)**2), axis=0)
        return unnormalized_voigt * a_voight / np.sqrt(np.pi)
