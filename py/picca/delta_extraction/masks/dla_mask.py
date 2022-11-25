"""This module defines the classes DlaMask and Dla used in the
masking of DLAs"""
import logging

from astropy.table import Table
import fitsio
import numpy as np

from picca.delta_extraction.astronomical_objects.forest import Forest
from picca.delta_extraction.errors import MaskError
from picca.delta_extraction.mask import Mask
from picca.delta_extraction.utils import ABSORBER_IGM

defaults = {
    "dla mask limit": 0.8,
    "los_id name": "THING_ID",
}

accepted_options = ["dla mask limit", "los_id name", "mask file", "filename", "keep pixels"]

np.random.seed(0)
NUM_POINTS = 10000
GAUSSIAN_DIST = np.random.normal(size=NUM_POINTS) * np.sqrt(2)


class DlaMask(Mask):
    """Class to mask DLAs

    Methods
    -------
    __init__
    apply_mask

    Attributes
    ----------
    (see Mask in py/picca/delta_extraction/mask.py)

    dla_mask_limit: float
    Lower limit on the DLA transmission. Transmissions below this number are
    masked

    logger: logging.Logger
    Logger object

    mask: astropy.Table
    Table containing specific intervals of wavelength to be masked for DLAs
    """
    def __init__(self, config):
        """Initializes class instance.

        Arguments
        ---------
        config: configparser.SectionProxy
        Parsed options to initialize class

        Raise
        -----
        MaskError if there are missing variables
        MaskError if input file does not have extension DLACAT
        MaskError if input file does not have fields THING_ID, Z, NHI in extension
        DLACAT
        MaskError upon OsError when reading the mask file
        """
        self.logger = logging.getLogger(__name__)

        super().__init__(config)

        # first load the dla catalogue
        filename = config.get("filename")
        if filename is None:
            raise MaskError("Missing argument 'filename' required by DlaMask")

        los_id_name = config.get("los_id name")
        if los_id_name is None:
            raise MaskError(
                "Missing argument 'los_id name' required by DlaMask")

        self.logger.progress(f"Reading DLA catalog from: {filename}")

        accepted_zcolnames = ["Z_DLA", "Z"]
        z_colname = accepted_zcolnames[0]
        try:
            with fitsio.FITS(filename) as hdul:
                hdul_colnames = set(hdul["DLACAT"].get_colnames())
                z_colname = hdul_colnames.intersection(accepted_zcolnames)
                if not z_colname:
                    raise ValueError(f"Z colname has to be one of {', '.join(accepted_zcolnames)}")
                z_colname = z_colname.pop()
                columns_list = [los_id_name, z_colname, "NHI"]
                cat = {col: hdul["DLACAT"][col][:] for col in columns_list}
        except OSError as error:
            raise MaskError(
                f"Error loading DlaMask. File {filename} does "
                "not have extension 'DLACAT'"
            ) from error
        except ValueError as error:
            aux = "', '".join(columns_list)
            raise MaskError(
                f"Error loading DlaMask. File {filename} does "
                f"not have fields '{aux}' in HDU 'DLACAT'"
            ) from error

        # group DLAs on the same line of sight together
        self.los_ids = {}
        for los_id in np.unique(cat[los_id_name]):
            w = (los_id == cat[los_id_name])
            self.los_ids[los_id] = list(zip(cat[z_colname][w], cat['NHI'][w]))
        num_dlas = np.sum([len(los_id) for los_id in self.los_ids.values()])

        self.logger.progress(f'In catalog: {num_dlas} DLAs')
        self.logger.progress(f'In catalog: {len(self.los_ids)} forests have a DLA\n')

        # setup transmission limit
        # transmissions below this number are masked
        self.dla_mask_limit = config.getfloat("dla mask limit")
        if self.dla_mask_limit is None:
            raise MaskError("Missing argument 'dla mask limit' "
                            "required by DlaMask")

        # load mask
        mask_file = config.get("mask file")
        if mask_file is not None:
            try:
                self.mask = Table.read(mask_file,
                                       names=('type', 'wave_min', 'wave_max',
                                              'frame'),
                                       format='ascii')
                self.mask = self.mask['frame'] == 'RF_DLA'
            except (OSError, ValueError) as error:
                raise MaskError(
                    f"ERROR: Error while reading mask_file file {mask_file}"
                ) from error
        else:
            self.mask = Table(names=('type', 'wave_min', 'wave_max', 'frame'))

    def apply_mask(self, forest):
        """Apply the mask. The mask is done by removing the affected
        pixels from the arrays in Forest.mask_fields

        Arguments
        ---------
        forest: Forest
        A Forest instance to which the correction is applied

        Raise
        -----
        MaskError if Forest.wave_solution is not 'log'
        """
        lambda_ = 10**forest.log_lambda

        # load DLAs
        if self.los_ids.get(forest.los_id) is not None:
            dla_transmission = np.ones(len(lambda_))
            for (z_abs, nhi) in self.los_ids.get(forest.los_id):
                dla_transmission *= DlaProfile(lambda_, z_abs,
                                               nhi).transmission

            # find out which pixels to mask
            w = dla_transmission > self.dla_mask_limit
            if len(self.mask) > 0:
                for mask_range in self.mask:
                    for (z_abs, nhi) in self.los_ids.get(forest.los_id):
                        w &= ((lambda_ / (1. + z_abs) < mask_range['wave_min'])
                              | (lambda_ /
                                 (1. + z_abs) > mask_range['wave_max']))

            # do the actual masking
            forest.transmission_correction *= dla_transmission
            for param in Forest.mask_fields:
                self._masker(forest, param, w)

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
    def __init__(self, lambda_, z_abs, nhi):
        """Initialize class instance.

        Arguments
        ---------
        lambda_: array of floats
        Wavelength (in Angs)

        z_abs: float
        Redshift of the absorption

        nhi: float
        DLA column density in log10(cm^-2)
        """
        self.z_abs = z_abs
        self.nhi = nhi

        self.transmission = self.profile_lya_absorption(lambda_, z_abs, nhi)
        self.transmission *= self.profile_lyb_absorption(lambda_, z_abs, nhi)

    @staticmethod
    def profile_lya_absorption(lambda_, z_abs, nhi):
        """Compute the absorption profile for Lyman-alpha absorption.

        Arguments
        ---------
        lambda_: array of floats
        Wavelength (in Angs)

        z_abs: float
        Redshift of the absorption

        nhi: float
        DLA column density in log10(cm^-2)

        Return
        ------
        profile: array of floats
        The absorption profile.
        """
        return np.exp(-DlaProfile.tau_lya(lambda_, z_abs, nhi))

    @staticmethod
    def profile_lyb_absorption(lambda_, z_abs, nhi):
        """Computes the absorption profile for Lyman-beta absorption.

        Arguments
        ---------
        lambda_: array of floats
        Wavelength (in Angs)

        z_abs: float
        Redshift of the absorption

        nhi: float
        DLA column density in log10(cm^-2)

        Return
        ------
        profile: array of floats
        The absorption profile.
        """
        return np.exp(-DlaProfile.tau_lyb(lambda_, z_abs, nhi))

    ### Implementation of Pasquier code,
    ###     also in Rutten 2003 at 3.3.3
    @staticmethod
    def tau_lya(lambda_, z_abs, nhi):
        """Compute the optical depth for Lyman-alpha absorption.

        Arguments
        ---------
        lambda_: array of floats
        Wavelength (in Angs)

        z_abs: float
        Redshift of the absorption

        nhi: float
        DLA column density in log10(cm^-2)

        Return
        ------
        tau: array of float
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
        """Compute the optical depth for Lyman-beta absorption.

        Arguments
        ---------
        lambda_: array of floats
        Wavelength (in Angs)

        z_abs: float
        Redshift of the absorption

        nhi: float
        DLA column density in log10(cm^-2)

        Return
        ------
        tau: array of float
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
        """Compute the classical Voigt function

        Arguments
        ---------
        a_voight: array of floats
        Voigt damping parameter.

        u_voight: array of floats
        Dimensionless frequency offset in Doppler widths.

        Return
        ------
        voigt: array of float
        The Voigt function for each element in a, u
        """
        unnormalized_voigt = np.mean(
            1 / (a_voight**2 + (GAUSSIAN_DIST[:, None] - u_voight)**2), axis=0)
        return unnormalized_voigt * a_voight / np.sqrt(np.pi)
