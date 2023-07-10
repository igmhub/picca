"""This module defines data structure to deal with Damped Lyman-alpha
Absorbers (DLAs)

This module provides with one class (DLA). See the respective
docstrings for more details
"""
import numpy as np

from . import constants
from scipy.special import voigt_profile

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

    ### Implementation based on Garnett2018
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
        e = 1.6021e-19 #C
        epsilon0 = 8.8541e-12 #C^2.s^2.kg^-1.m^-3
        f = 0.4164
        mp = 1.6726e-27 #kg
        me = 9.109e-31 #kg
        c = 2.9979e8 #m.s^-1
        k = 1.3806e-23 #m^2.kg.s^-2.K-1
        T = 5*1e4 #K
        gamma = 6.2648e+08 #s^-1
        lam_lya = constants.ABSORBER_IGM["LYA"] #A

        lambda_rest_frame = lambda_/(1+z_abs)
        
        v = c *(lambda_rest_frame/lam_lya-1)
        b = np.sqrt(2*k*T/mp)
        small_gamma = gamma*lam_lya/(4*np.pi)*1e-10
        
        nhi_m2 = 10**nhi*1e4
        
        tau = nhi_m2*np.pi*e**2*f*lam_lya*1e-10
        tau /= 4*np.pi*epsilon0*me*c
        tau *= voigt_profile(v, b/np.sqrt(2), small_gamma)
        
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
        e = 1.6021e-19  # C
        epsilon0 = 8.8541e-12  # C^2.s^2.kg^-1.m^-3
        f = 0.079142
        mp = 1.6726e-27  # kg
        me = 9.109e-31  # kg
        c = 2.9979e8  # m.s^-1
        k = 1.3806e-23  # m^2.kg.s^-2.K-1
        T = 5 * 1e4  # K
        gamma = 1.6725e8  # s^-1
        lam_lyb = constants.ABSORBER_IGM["LYB"] #A

        lambda_rest_frame = lambda_/(1+z_abs)
        
        v = c *(lambda_rest_frame/lam_lyb-1)
        b = np.sqrt(2*k*T/mp)
        small_gamma = gamma*lam_lyb/(4*np.pi)*1e-10
        
        nhi_m2 = 10**nhi*1e4
        
        tau = nhi_m2*np.pi*e**2*f*lam_lyb*1e-10
        tau /= 4*np.pi*epsilon0*me*c
        tau *= voigt_profile(v, b/np.sqrt(2), small_gamma)
        
        return tau