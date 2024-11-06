"""This module defines the classes DlaMask and Dla used in the
masking of DLAs"""
import logging

from astropy.table import Table
import fitsio
import numpy as np
from scipy.constants import (
    speed_of_light as SPEED_LIGHT,
    e as ELEMENTARY_CHARGE,
    epsilon_0 as EPSILON_0,
    m_p as PROTON_MASS,
    m_e as ELECTRON_MASS,
    k as BOLTZMAN_CONSTANT_K,
)
from scipy.special import voigt_profile

from picca.delta_extraction.astronomical_objects.forest import Forest
from picca.delta_extraction.errors import MaskError
from picca.delta_extraction.mask import Mask, accepted_options, defaults
from picca.delta_extraction.utils import (
    ABSORBER_IGM, update_accepted_options, update_default_options)

accepted_options = update_accepted_options(accepted_options, [
    "dla mask limit", "los_id name", "mask file", "filename"
])

defaults = update_default_options(
    defaults, {
        "dla mask limit": 0.8,
        "los_id name": "THING_ID",
    })

LAMBDA_LYA = float(ABSORBER_IGM["LYA"]) ## Lya wavelength [A]
LAMBDA_LYB = float(ABSORBER_IGM["LYB"]) ## Lyb wavelength [A]
OSCILLATOR_STRENGTH_LYA = 0.41641
OSCILLATOR_STRENGTH_LYB = 0.079142
GAMMA_LYA = 6.2648e08  # s^-1 damping constant
GAMMA_LYB = 1.6725e8  # s^-1 damping constant

def dla_profile(lambda_, z_abs, nhi):
    """Compute DLA profile

    Arguments
    ---------
    lambda_: array of floats
    Wavelength (in Angs)

    z_abs: float
    Redshift of the absorption

    nhi: float
    DLA column density in log10(cm^-2)
    """
    transmission = np.exp(
        -compute_tau(lambda_, z_abs, nhi, LAMBDA_LYA, OSCILLATOR_STRENGTH_LYA, GAMMA_LYA)
        -compute_tau(lambda_, z_abs, nhi, LAMBDA_LYB, OSCILLATOR_STRENGTH_LYB, GAMMA_LYB)
    )
    return transmission

# constants to compute the optical depth of the DLA absoprtion
GAS_TEMP = 5 * 1e4  # K

# precomputed factors to save time
# compared to equation 36 of Garnett et al 2017 there is a sqrt(2) missing
# this is because otherwise we need to divide this quantity by sqrt(2)
# when calling scipy.special.voigt
GAUSSIAN_BROADENING_B = np.sqrt(BOLTZMAN_CONSTANT_K * GAS_TEMP / PROTON_MASS)
# the 1e-10 appears as the wavelengths are given in Angstroms
LORENTZIAN_BROADENING_GAMMA_PREFACTOR = 1e-10 / (4 * np.pi)
TAU_PREFACTOR = (
    ELEMENTARY_CHARGE**2 * 1e-10 / ELECTRON_MASS / SPEED_LIGHT / 4 / EPSILON_0)

def compute_tau(lambda_, z_abs, log_nhi, lambda_t, oscillator_strength_f, gamma):
    r"""Compute the optical depth for DLA absorption.

    Tau is computed using equations 34 to 36 of Garnett et al. 2017. We add
    a factor 4pi\epsion_0 in the denominator of their equation 34 so that
    dimensions match. The equations we use are:

    \tau(\lambda, z_{abs}, N_{HI}) = N_{HI} \frac {e^{2} f\lambda_{t} }
        {4 \epsion_0 m_{e} c } \phi{\nu, b, \gamma}

    where e is the elementary charge, \lambda_{t} is the transition wavelength
    and f is the oscillator strength of the transition. The line profile
    function \phi is a Voigt profile, where \nu is ther elative velocity

    \nu = c ( \frac{ \lambda } { \lambda_{t}* (1+z_{DLA}) }  ) ,

    b / \sqrt{2} is the standard deviation of the Gaussian (Maxwellian)
    broadening contribution:

    b = \sqrt{ \frac{ 2kT }{ m_{p} } }

    and \gamma is the width of the Lorenztian broadening contribution:

    \gamma = \frac { \Gamma \lambda_{t} } { 4\pi }

    where \Gamma is a damping constant

    Arguments
    ---------
    lambda_: array of floats
    Wavelength (in Angs)

    z_abs: float
    Redshift of the absorption

    log_nhi: float
    DLA column density in log10(cm^-2)

    lambda_t: float
    Transition wavelength, in Angstroms, e.g. for Lya 1215.67

    oscillator_strength_f: float
    Oscillator strength, e.g. f = 0.41611 for Lya

    gamma: float
    Damping constant (in s^-1)

    Return
    ------
    tau: array of float
    The optical depth.
    """
    # compute broadenings for the voight profile
    relative_velocity_nu = SPEED_LIGHT * (lambda_ / (1 + z_abs) / lambda_t - 1)
    lorentzian_broadening_gamma = (
        LORENTZIAN_BROADENING_GAMMA_PREFACTOR * gamma * lambda_t)

    # convert column density to m^2
    nhi = 10**log_nhi * 1e4

    # the 1e-10 converts the wavelength from Angstroms to meters
    tau = TAU_PREFACTOR * nhi * oscillator_strength_f * lambda_t * voigt_profile(
        relative_velocity_nu, GAUSSIAN_BROADENING_B, lorentzian_broadening_gamma)

    return tau

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
                if len(z_colname)>1 :
                    raise ValueError(
                        "Not clear which column should be used for the DLA redshift among "
                        f"{z_colname}. Please remove or rename one of the columns from the DLA "
                        "fits file.")
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
            w = los_id == cat[los_id_name]
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
                dla_transmission *= dla_profile(lambda_, z_abs,
                                                nhi)

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
