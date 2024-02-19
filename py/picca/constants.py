"""This module defines some constants that are used throughout the package.

It includes the class Cosmo, used to store the fiducial cosmology
"""
import fitsio
import scipy as sp
import numpy as np
from scipy import interpolate
from scipy.integrate import quad
from scipy.constants import speed_of_light as speed_light
from pkg_resources import resource_filename
from picca.utils import userprint

SPEED_LIGHT = speed_light/1000. # [km/s]

# different strategies are explained in
# https://desi.lbl.gov/trac/wiki/keyprojects/y1kp6/Blinding
ACCEPTED_BLINDING_STRATEGIES = ["none", "minimal", "strategyB", "strategyC",
    "strategyBC", "corr_yshift"]

def inv_E_z(z,Om,Or,Ok,wl):
    """ Compute inverse hubble parameter as a function of redshift.
    """
    Ol = 1 - Om - Or
    E_z = np.sqrt(Ol*(1. + z)**(3.*(1. + wl)) +
                    Ok*(1. + z)**2 +
                    Om*(1. + z)**3 +
                    Or*(1. + z)**4)
    return 1./E_z

class Cosmo(object):
    """This class defines the fiducial cosmology

    This class stores useful tabulated functions to transform redshifts
    to distances using the fiducial cosmology. This is done assuming a
    FLRW metric.

    Attributes:
        -

    Methods:
        __init__: Initializes class instances.
        get_r_comov: Interpolates the comoving the redshift array.
        get_hubble: Interpolates the hubble constant on the redshift array.
        distance_to_redshift: Interpolates the redhsift on the comoving distance
            array.
        get_dist_hubble: Interpolates the Hubble distance on the redshfit array.
        get_dist_m: Interpolates the angular diameter distance on the redshfit
            array.
        get_dist_v: Interpolates the geometric mean of the transverse and radial
        distances on the redshfit array.
    """
    # pylint: disable=method-hidden
    # Added here to mark it as function
    def get_dist_v(self, z):
        """Interpolates the geometric mean of the transverse and radial
        distances on the redshfit array.

        Empty function to be loaded at run-time.

        Args:
            z: array of floats
                Array containing the redshifts

        Returns:
            An array with the the geometric mean of the transverse and radial
            distances

        Raises:
            NotImplementedError: Function was not specified
        """
        raise NotImplementedError("Function should be specified at run-time")
    
    def __init__(self,Om,Ok=0.,Or=0.,wl=-1.,H0=67.36,
                 use_mpc=False,blinding=False,verbose=True):
        """Initializes the methods for this instance

        Args:
            Om: float - default: 0.3
                Matter density
            Ok: float - default: 0.0
                Curvature density
            Or: float - default: 0.0
                Radiation density
            wl: float - default: -1.0
                Dark energy equation of state
            H0: float - default: 67.36
                Hubble constant at redshift 0 (in km/s/Mpc)
            use_mpc: bool - default: False
                Return distances in Mpc rather than Mpc/h.
        """
        # Blind data
        if blinding == "none":
            if verbose:
                userprint("ATTENTION: Analysis is not blinded!")
        else:
            userprint(f"ATTENTION: Analysis is blinded with strategy {blinding}")

        if blinding not in  ["strategyB", "strategyBC"]:
            if verbose:
                userprint(f"Om={Om}, Or={Or}, wl={wl}")
        else:
            userprint("The specified cosmology is "
                      f"not used: Om={Om}, Or={Or}, wl={wl}")
            # blind test small
            filename = "DR16_blind_test_small/DR16_blind_test_small.fits"
            # blind test large
            #filename = "DR16_blind_test_small/DR16_blind_test_large.fits"
            # load Om
            filename = resource_filename('picca', 'fitter2')+'/models/{}'.format(filename)
            hdu = fitsio.FITS(filename)
            Om = hdu[1].read_header()['OM']
            Or = hdu[1].read_header()['OR']
            wl = hdu[1].read_header()['W']
            H0 = hdu[1].read_header()['H0']
            hdu.close()

        #new implemenetation
        hubble_distance = SPEED_LIGHT / H0
        if not use_mpc:
            hubble_distance *= H0/100

        self._hubble_distance = hubble_distance
        self.cosmo_params = (Om,Or,Ok,wl)
        self.Ok = Ok

    #new implementation
    def get_dist_hubble(self,z):
        """Get Hubble distance DH as defined in  dMdB (2020) """
        return self._hubble_distance * inv_E_z(z,*self.cosmo_params)

    def get_dist_v(self,z):
        """geometric mean of the transverse and radial
        distances on the redshfit array."""
        return np.power(z * self.get_dist_m(z)**2 * self.get_dist_hubble(z),
                          1./3.)
    
    def get_r_comov(self,z):
        """
        Compute co-moving distance at redshift z in Mpc or Mpc/h (dependent on hubble_distance)
        """
        integral = quad(inv_E_z,0,z,args=self.cosmo_params)[0]
        return self._hubble_distance * integral

    def get_dist_m(self,z):
        """
        Compute transverse comoving distance in Mpc or Mpc/h (dependent on hubble_distance)
        """
        if self.Ok == 0.:
            return self.get_r_comov(z)
        elif self.Ok < 0.:
            return (self._hubble_distance*(1/np.sqrt(abs(self.Ok)))
                    *np.sin(np.sqrt(abs(self.Ok))*self.get_r_comov(z)/self._hubble_distance))
        elif self.Ok > 0.:
            return (self._hubble_distance*(1/np.sqrt(self.Ok))
                    *np.sinh(np.sqrt(self.Ok)*self.get_r_comov(z)/self._hubble_distance))

### Absorber names and wavelengths [Angstrom]
ABSORBER_IGM = {
    "Halpha"      : 6562.8,
    "Hbeta"       : 4862.68,
    "MgI(2853)"   : 2852.96,
    "MgII(2804)"  : 2803.5324,
    "MgII(eff)"   : 2798.75,
    "MgII(2796)"  : 2796.3511,
    "FeII(2600)"  : 2600.1724835,
    "FeII(2587)"  : 2586.6495659,
    "MnII(2577)"  : 2576.877,
    "FeII(2383)"  : 2382.7641781,
    "FeII(2374)"  : 2374.4603294,
    "FeII(2344)"  : 2344.2129601,
    "CIII(1908)"  : 1908.73,
    "AlIII(1863)" : 1862.79113,
    "AlIII(1855)" : 1854.71829,
    "AlII(1671)"  : 1670.7886,
    "FeII(1608)"  : 1608.4511,
    "CIV(1551)"   : 1550.77845,
    "CIV(eff)"    : 1549.06,
    "CIV(1548)"   : 1548.2049,
    "SiII(1527)"  : 1526.70698,
    "NiII(1455)"  : 1454.842,
    "SiIV(1403)"  : 1402.77291,
    "SiIV(eff)"   : 1396.76,
    "SiIV(1394)"  : 1393.76018,
    "NiII(1370)"  : 1370.132,
    "CII(1335)"   : 1334.5323,
    "NiII(1317)"  : 1317.217,
    "SiII(1304)"  : 1304.3702,
    "OI(1302)"    : 1302.1685,
    "SiII(1260)"  : 1260.4221,
    "SII(1254)"   : 1253.811,
    "SII(1251)"   : 1250.584,
    "NV(1243)"    : 1242.804,
    "NV(1239)"    : 1238.821,
    "LYA"         : 1215.67,
    "SiIII(1207)" : 1206.500,
    "NI(1200)"    : 1200.,
    "SiII(1193)"  : 1193.2897,
    "SiII(1190)"  : 1190.4158,
    "PII(1153)"   : 1152.818,
    "FeII(1145)"  : 1144.9379,
    "FeII(1143)"  : 1143.2260,
    "NI(1134)"    : 1134.4149,
    "FeII(1125)"  : 1125.4477,
    "FeIII(1123)" : 1122.526,
    "FeII(1097)"  : 1096.8769,
    "NII(1084)"   : 1083.990,
    "FeII(1082)"  : 1081.8748,
    "FeII(1063)"  : 1063.002,
    "OI(1039)"    : 1039.230,
    "OVI(1038)"   : 1037.613,
    "CII(1037)"   : 1036.7909,
    "OVI(1032)"   : 1031.912,
    "LYB"         : 1025.72,
    "SiII(1021)"  : 1020.6989,
    "SIII(1013)"  : 1012.502,
    "SiII(990)"   : 989.8731,
    "OI(989)"     : 988.7,
    "CIII(977)"   : 977.020,
    "LY3"         : 972.537,
    "LY4"         : 949.7431,
    "LY5"         : 937.8035,
    "LY6"         : 930.7483,
    "LY7"         : 926.2257,
    "LY8"         : 923.1504,
    "LY9"         : 920.9631,
    "LY10"        : 919.3514,
}
