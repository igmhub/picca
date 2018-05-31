import scipy as sp
from scipy import interpolate

deg = sp.pi/180.

boss_lambda_min = 3600. ## Angstrom

speed_light = 299792458. ## m/s

small_angle_cut_off = 2./3600.*sp.pi/180. ## 2 arcsec

class cosmo:

    def __init__(self,Om,Ok=0):

        ### ignore Orad and neutrinos
        c = speed_light/1000. ## km/s
        H0 = 100. ## km/s/Mpc
        Or = 0.
        Ol = 1.-Ok-Om-Or

        nbins = 10000
        zmax  = 10.
        dz    = zmax/nbins
        z=sp.arange(nbins)*dz
        hubble = H0*sp.sqrt( Ol + Ok*(1.+z)**2 + Om*(1.+z)**3 + Or*(1.+z)**4 )

        chi=sp.zeros(nbins)
        for i in range(1,nbins):
            chi[i]=chi[i-1]+c*(1./hubble[i-1]+1./hubble[i])/2.*dz

        self.r_comoving = interpolate.interp1d(z,chi)

        ## da here is the comoving angular diameter distance
        if Ok==0.:
            da = chi
        elif Ok<0.:
            da = sp.sin(H0*sp.sqrt(-Ok)/c*chi)/(H0*sp.sqrt(-Ok)/c)
        elif Ok>0.:
            da = sp.sinh(H0*sp.sqrt(Ok)/c*chi)/(H0*sp.sqrt(Ok)/c)

        self.da = interpolate.interp1d(z,da)
        self.hubble = interpolate.interp1d(z,hubble)
        self.r_2_z = interpolate.interp1d(chi,z)

### Absorber names and wavelengths [Angstrom]
absorber_IGM = {
    'Halpha'      : 6562.8,
    'Hbeta'       : 4862.68,
    'MgI(2853)'   : 2852.96,
    'MgII(2804)'  : 2803.5324,
    'MgII(2796)'  : 2796.3511,
    'FeII(2600)'  : 2600.1724835,
    'FeII(2587)'  : 2586.6495659,
    'MnII(2577)'  : 2576.877,
    'FeII(2383)'  : 2382.7641781,
    'FeII(2374)'  : 2374.4603294,
    'FeII(2344)'  : 2344.2129601,
    'AlIII(1863)' : 1862.79113,
    'AlIII(1855)' : 1854.71829,
    'AlII(1671)'  : 1670.7886,
    'FeII(1608)'  : 1608.4511,
    'CIV(1551)'   : 1550.77845,
    'CIV(eff)'    : 1549.06,
    'CIV(1548)'   : 1548.2049,
    'SiII(1527)'  : 1526.70698,
    'SiIV(1403)'  : 1402.77291,
    'SiIV(1394)'  : 1393.76018,
    'CII(1335)'   : 1334.5323,
    'SiII(1304)'  : 1304.3702,
    'OI(1302)'    : 1302.1685,
    'SiII(1260)'  : 1260.4221,
    'NV(1243)'    : 1242.804,
    'NV(1239)'    : 1238.821,
    'LYA'         : 1215.67,
    'SiIII(1207)' : 1206.500,
    'NI(1200)'    : 1200.,
    'SiII(1193)'  : 1193.2897,
    'SiII(1190)'  : 1190.4158,
    'OI(1039)'    : 1039.230,
    'OVI(1038)'   : 1037.613,
    'OVI(1032)'   : 1031.912,
    'LYB'         : 1025.72,
}


