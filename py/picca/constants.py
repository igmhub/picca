import scipy as sp
from scipy import interpolate

deg = sp.pi/180.

boss_lambda_min = 3600. ## Angstrom

from scipy.constants import speed_of_light as speed_light

small_angle_cut_off = 2./3600.*sp.pi/180. ## 2 arcsec

class cosmo:

    def __init__(self,Om,Ok=0.,Or=0.,wl=-1.,H0=100.):

        ### Ignore evolution of neutrinos from matter to radiation
        ### H0 in km/s/Mpc
        c = speed_light/1000. ## km/s
        Ol = 1.-Ok-Om-Or

        nbins = 10000
        zmax  = 10.
        dz    = zmax/nbins
        z=sp.arange(nbins)*dz
        hubble = H0*sp.sqrt( Ol*(1.+z)**(3.*(1.+wl)) + Ok*(1.+z)**2 + Om*(1.+z)**3 + Or*(1.+z)**4 )

        chi=sp.zeros(nbins)
        for i in range(1,nbins):
            chi[i]=chi[i-1]+c*(1./hubble[i-1]+1./hubble[i])/2.*dz

        self.r_comoving = interpolate.interp1d(z,chi)

        ### dm here is the comoving angular diameter distance
        if Ok==0.:
            dm = chi
        elif Ok<0.:
            dm = sp.sin(H0*sp.sqrt(-Ok)/c*chi)/(H0*sp.sqrt(-Ok)/c)
        elif Ok>0.:
            dm = sp.sinh(H0*sp.sqrt(Ok)/c*chi)/(H0*sp.sqrt(Ok)/c)

        self.hubble = interpolate.interp1d(z,hubble)
        self.r_2_z = interpolate.interp1d(chi,z)

        ### D_H
        self.dist_hubble = interpolate.interp1d(z,c/hubble)
        ### D_M
        self.dm = interpolate.interp1d(z,dm)
        ### D_V
        y = sp.power(z*self.dm(z)**2*self.dist_hubble(z),1./3.)
        self.dist_v = interpolate.interp1d(z,y)

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
    'NiII(1455)'  : 1454.842,
    'SiIV(1403)'  : 1402.77291,
    'SiIV(1394)'  : 1393.76018,
    'NiII(1370)'  : 1370.132,
    'CII(1335)'   : 1334.5323,
    'NiII(1317)'  : 1317.217,
    'SiII(1304)'  : 1304.3702,
    'OI(1302)'    : 1302.1685,
    'SiII(1260)'  : 1260.4221,
    'SII(1254)'   : 1253.811,
    'SII(1251)'   : 1250.584,
    'NV(1243)'    : 1242.804,
    'NV(1239)'    : 1238.821,
    'LYA'         : 1215.67,
    'SiIII(1207)' : 1206.500,
    'NI(1200)'    : 1200.,
    'SiII(1193)'  : 1193.2897,
    'SiII(1190)'  : 1190.4158,
    'PII(1153)'   : 1152.818,
    'FeII(1145)'  : 1144.9379,
    'FeII(1143)'  : 1143.2260,
    'NI(1134)'    : 1134.4149,
    'FeII(1125)'  : 1125.4477,
    'FeIII(1123)' : 1122.526,
    'FeII(1097)'  : 1096.8769,
    'NII(1084)'   : 1083.990,
    'FeII(1082)'  : 1081.8748,
    'FeII(1063)'  : 1063.002,
    'OI(1039)'    : 1039.230,
    'OVI(1038)'   : 1037.613,
    'CII(1037)'   : 1036.7909,
    'OVI(1032)'   : 1031.912,
    'LYB'         : 1025.72,
    'SiII(1021)'  : 1020.6989,
    'SIII(1013)'  : 1012.502,
    'SiII(990)'   : 989.8731,
    'OI(989)'     : 988.7,
    'CIII(977)'   : 977.020,
    'LY3'         : 972.537,
    'LY4'         : 949.7431,
    'LY5'         : 937.8035,
    'LY6'         : 930.7483,
    'LY7'         : 926.2257,
    'LY8'         : 923.1504,
    'LY9'         : 920.9631,
    'LY10'        : 919.3514,
}
