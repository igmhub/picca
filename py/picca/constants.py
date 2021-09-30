"""This module defines some constants that are used throughout the package.

It includes the class Cosmo, used to store the fiducial cosmology
"""
import fitsio
import scipy as sp
import numpy as np
from scipy import interpolate
from scipy.constants import speed_of_light as speed_light
from pkg_resources import resource_filename
from picca.utils import userprint

# TODO: this constant is unused. They should be removed at some point
BOSS_LAMBDA_MIN = 3600. # [Angstrom]

SMALL_ANGLE_CUT_OFF = 2./3600.*np.pi/180. # 2 arcsec

SPEED_LIGHT = speed_light/1000. # [km/s]

# different strategies are explained in
# https://desi.lbl.gov/trac/wiki/keyprojects/y1kp6/Blinding
ACCEPTED_BLINDING_STRATEGIES = ["none", "minimal", "strategyA", "strategyB", "strategyC",
    "strategyBC", "strategyABC"]

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
    def get_r_comov(self, z):
        """Interpolates the comoving distance on the redshift array.

        Empty function to be loaded at run-time.

        Args:
            z: array of floats
                Array containing the redshifts

        Returns:
            An array with the comoving distance

        Raises:
            NotImplementedError: Function was not specified
        """
        raise NotImplementedError("Function should be specified at run-time")

    # pylint: disable=method-hidden
    # Added here to mark it as function
    def get_hubble(self, z):
        """Interpolates the Hubble constant on the redshift array.

        Empty function to be loaded at run-time.

        Args:
            z: array of floats
                Array containing the redshifts

        Returns:
            An array with the Hubble constant

        Raises:
            NotImplementedError: Function was not specified
        """
        raise NotImplementedError("Function should be specified at run-time")

    # pylint: disable=method-hidden
    # Added here to mark it as function
    def distance_to_redshift(self, r_comov):
        """Interpolates the redhsift on the comoving distance array.

        Empty function to be loaded at run-time.

        Args:
            r_comov: array of floats
                Array containing the comoving distances

        Returns:
            An array with the redshfits

        Raises:
            NotImplementedError: Function was not specified
        """
        raise NotImplementedError("Function should be specified at run-time")

    # pylint: disable=method-hidden
    # Added here to mark it as function
    def get_dist_hubble(self, z):
        """Interpolates the Hubble distance on the redshfit array.

        Empty function to be loaded at run-time.

        Args:
            z: array of floats
                Array containing the redshifts

        Returns:
            An array with the Hubble distance

        Raises:
            NotImplementedError: Function was not specified
        """
        raise NotImplementedError("Function should be specified at run-time")

    # pylint: disable=method-hidden
    # Added here to mark it as function
    def get_dist_m(self, z):
        """Interpolates the angular diameter distance on the redshfit array.

        Empty function to be loaded at run-time.

        Args:
            z: array of floats
                Array containing the redshifts

        Returns:
            An array with the angular diameter distance

        Raises:
            NotImplementedError: Function was not specified
        """
        raise NotImplementedError("Function should be specified at run-time")

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

    def __init__(self,Om,Ok=0.,Or=0.,wl=-1.,H0=100.,blinding=False):
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
            H0: float - default: 100.0
                Hubble constant at redshift 0 (in km/s/Mpc)
        """

        # Blind data
        if blinding == "none":
            userprint("ATTENTION: Analysis is not blinded!")
        else:
            userprint(f"ATTENTION: Analysis is blinded with strategy {blinding}")

        if blinding in ["strategyA", "strategyB", "strategyBC", "strategyABC"]:
            userprint("The specified cosmology is "
                      f"not used")
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
        else:
            userprint(f"Om={Om}, Or={Or}, wl={wl}, H0={H0}")

        # Ignore evolution of neutrinos from matter to radiation
        Ol = 1. - Ok - Om - Or

        num_bins = 10000
        z_max = 10.
        delta_z = z_max/num_bins
        z = np.arange(num_bins, dtype=float)*delta_z
        hubble = H0*np.sqrt(Ol*(1. + z)**(3.*(1. + wl)) +
                            Ok*(1. + z)**2 +
                            Om*(1. + z)**3 +
                            Or*(1. + z)**4)

        r_comov = np.zeros(num_bins)
        for index in range(1, num_bins):
            r_comov[index] = (SPEED_LIGHT*(1./hubble[index - 1] +
                                           1./hubble[index])/2.*delta_z +
                              r_comov[index - 1])

        self.get_r_comov = interpolate.interp1d(z, r_comov)

        ### dist_m here is the comoving angular diameter distance
        if Ok == 0.:
            dist_m = r_comov
        elif Ok < 0.:
            dist_m = (np.sin(H0*np.sqrt(-Ok)/SPEED_LIGHT*r_comov)/
                      (H0*np.sqrt(-Ok)/SPEED_LIGHT))
        elif Ok > 0.:
            dist_m = (np.sinh(H0*np.sqrt(Ok)/SPEED_LIGHT*r_comov)/
                      (H0*np.sqrt(Ok)/SPEED_LIGHT))

        self.get_hubble = interpolate.interp1d(z, hubble)
        self.distance_to_redshift = interpolate.interp1d(r_comov, z)

        # D_H
        self.get_dist_hubble = interpolate.interp1d(z, SPEED_LIGHT/hubble)
        # D_M
        self.get_dist_m = interpolate.interp1d(z, dist_m)
        # D_V
        dist_v = np.power(z*self.get_dist_m(z)**2*self.get_dist_hubble(z),
                          1./3.)
        self.get_dist_v = interpolate.interp1d(z, dist_v)

### Absorber names and wavelengths [Angstrom]
ABSORBER_IGM = {
    "Halpha"      : 6562.8,
    "Hbeta"       : 4862.68,
    "MgI(2853)"   : 2852.96,
    "MgII(2804)"  : 2803.5324,
    "MgII(2796)"  : 2796.3511,
    "FeII(2600)"  : 2600.1724835,
    "FeII(2587)"  : 2586.6495659,
    "MnII(2577)"  : 2576.877,
    "FeII(2383)"  : 2382.7641781,
    "FeII(2374)"  : 2374.4603294,
    "FeII(2344)"  : 2344.2129601,
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

# Resample flux function from desispec/interpolation:
# def resample_flux(xout, x, flux, ivar=None, extrapolate=False):


def calcMaps(scale=1.0, Om=0.315):
    """initialize class and compute the conversion maps

    Arguments:
    args: float - default: 1.0
    Scaling of ommega matter with respect to the fiducial cosmology

    Om: float - default: 0.315
    Omega matter in the fiducial cosmology
    """
   ###### Definition of the fiducial cosmological model

   fid_Om = Om
   fid_Or = 0
   fid_Ok = 0
   fid_wl = -1

   cosmo_m = Cosmo(Om=fid_Om,Or=fid_Or,
                   Ok=fid_Ok,wl=fid_wl, blinding=False)
   lambda_abs = ABSORBER_IGM['LYA']

   ###### Definition of blind cosmological model

   fid_Om = Om*scale
   fid_Or = 0
   fid_Ok = 0
   fid_wl = -1

   cosmo_m2 = Cosmo(Om=fid_Om,Or=fid_Or,
                    Ok=fid_Ok,wl=fid_wl, blinding=False)
   lambda_abs = ABSORBER_IGM['LYA']

   #######

   zmax = 10.
   l_max = ( lambda_abs * (zmax + 1) ) - 1.3
   ll = np.log10( np.linspace(lambda_abs + .5, l_max, 10000) )
   z = 10**ll/lambda_abs-1.

   r_comov = cosmo_m.get_r_comov(z)
   r_comov2 = cosmo_m2.get_r_comov(z)

   znew = resample_flux(r_comov, r_comov2, z)
   zmask = ( z <= 10 )

   z = z[zmask]
   znew = znew[zmask]
   r_comov = r_comov[zmask]
   r_comov2 = r_comov2[zmask]
   ll = ll[zmask]

   llnew = np.log10( lambda_abs * ( znew + 1 ) )
   zmz = -znew + z
   lol = 10**ll / (10**llnew)

   return z, zmz, 10**ll, lol

def resample_flux(xout, x, flux, ivar=None, extrapolate=False):
    # Taken from desispec.interpolation
    if ivar is None:
        return _unweighted_resample(xout, x, flux, extrapolate=extrapolate)
    else:
        if extrapolate :
            raise ValueError("Cannot extrapolate ivar. Either set ivar=None and extrapolate=True or the opposite")
        a = _unweighted_resample(xout, x, flux*ivar, extrapolate=False)
        b = _unweighted_resample(xout, x, ivar, extrapolate=False)
        mask = (b>0)
        outflux = np.zeros(a.shape)
        outflux[mask] = a[mask] / b[mask]
        dx = np.gradient(x)
        dxout = np.gradient(xout)
        outivar = _unweighted_resample(xout, x, ivar/dx)*dxout

        return outflux, outivar

def _unweighted_resample(output_x,input_x,input_flux_density, extrapolate=False) :
    # Taken from desispec.interpolation

    ix=input_x
    iy=input_flux_density
    ox=output_x

    # boundary of output bins
    bins=np.zeros(ox.size+1)
    bins[1:-1]=(ox[:-1]+ox[1:])/2.
    bins[0]=1.5*ox[0]-0.5*ox[1]     # = ox[0]-(ox[1]-ox[0])/2
    bins[-1]=1.5*ox[-1]-0.5*ox[-2]  # = ox[-1]+(ox[-1]-ox[-2])/2

    tx=bins.copy()

    if not extrapolate :
        # note we have to keep the array sorted here because we are going to use it for interpolation
        ix = np.append( 2*ix[0]-ix[1] , ix)
        iy = np.append(0.,iy)
        ix = np.append(ix, 2*ix[-1]-ix[-2])
        iy = np.append(iy, 0.)

    ty=np.interp(tx,ix,iy)

    #  add input nodes which are inside the node array
    k=np.where((ix>=tx[0])&(ix<=tx[-1]))[0]
    if k.size :
        tx=np.append(tx,ix[k])
        ty=np.append(ty,iy[k])

    # sort this node array
    p = tx.argsort()
    tx=tx[p]
    ty=ty[p]

    trapeze_integrals=(ty[1:]+ty[:-1])*(tx[1:]-tx[:-1])/2.

    trapeze_centers=(tx[1:]+tx[:-1])/2.
    binsize = bins[1:]-bins[:-1]

    if np.any(binsize<=0)  :
        raise ValueError("Zero or negative bin size")

    return np.histogram(trapeze_centers, bins=bins, weights=trapeze_integrals)[0] / binsize

def blindData(data, z_, zmz_, l_, lol_):
    """Ads AP blinding (strategy A) to the deltas data

    Args:
        data: dict
            A dictionary with the forests in each healpix.
        z, zmz: map to blind the qso redshift value
        l, lol: map to blind the qso lambda values

    Returns:
        The following variables:
            data: dict
            A dictionary with the forests in each healpix after the blinding ha
            been applied
    """
    lpix = []

    for healpix in sorted(list(data.keys())):
        for forest in data[healpix]:
            # Z QSO ap shift
            Za = forest.z_qso
            Z_rebin = np.interp( Za, z_, zmz_ )
            forest.z_qso = Za + Z_rebin

            # QSO forest ap shift with interval conservation
            l = 10**( forest.log_lambda )
            lol_rebin = resample_flux( l, l_, lol_ )

            l_rebin = lol_rebin*l     #this new vector is not linearly spaced in lambda
            # creating new linear loglam interval to save all data
            l2 = 10**( np.arange(np.log10( np.min(l_rebin)), np.log10(np.max(l_rebin)) , forest.delta_log_lambda)   )
            # l2 = l-( l[0]-l_rebin[0] )  #to use original picca interval, just shifting it (more data loss)
            flux, ivar = resample_flux(l2, l_rebin, forest.flux, ivar=forest.ivar )   #rebbining while conserving flux
            lnrest = np.log10(   l2 / (1+forest.z_qso)   )

            # to calc pixel loss
            #lrebinrest = np.log10( l_rebin / ( 1 + forest.z_qso ) )
            #lrebinmask = ( lrebinrest > forest.log_lambda_min_rest_frame) & ( lrebinrest < forest.log_lambda_max_rest_frame) & (np.log10(l_rebin) > forest.log_lambda_min ) & (  (l_rebin) < ( 10**forest.log_lambda_max - 1) )

            # mask: first two for restframe lya range, last two to prep_del obs frame stacks
            lmask = (lnrest > forest.log_lambda_min_rest_frame) & (lnrest < forest.log_lambda_max_rest_frame) & (np.log10(l2) > forest.log_lambda_min ) & (  (l2) < ( 10**forest.log_lambda_max - 1) )
            forest.log_lambda = np.log10( l2[lmask]  )
            forest.flux = flux[lmask]
            forest.ivar = ivar[lmask]

            # pixel loss
            #diff = len(l_rebin) - sum(lrebinmask)
            #lpix.append( np.hstack(( Za, forest.z_qso, diff, len(l_rebin) ))  )
    # pixel loss
    #np.save('/global/homes/s/sfbeltr/respaldo/losspixels.picca', lpix)
    return data
