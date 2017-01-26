import scipy as sp
from scipy import interpolate

lya=1215.67 ## angstrom

deg = sp.pi/180.

boss_lambda_min = 3600.


class cosmo:

    def __init__(self,Om):
        H0 = 100. ## km/s/Mpc
        ## ignore Orad and neutrinos
        nbins=10000
        zmax=10.
        dz = zmax/nbins
        z=sp.array(range(nbins))*dz
        hubble = H0*sp.sqrt(Om*(1+z)**3+1-Om)
        c = 299792.4583

        chi=sp.zeros(nbins)
        for i in range(1,nbins):
            chi[i]=chi[i-1]+c*(1./hubble[i-1]+1/hubble[i])/2.*dz

        self.r_comoving = interpolate.interp1d(z,chi)
        self.hubble = interpolate.interp1d(z,hubble)
        self.r_2_z = interpolate.interp1d(chi,z)

### Absorber names and wavelengths
absorber_IGM = {
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
    'FeII(1609)'  : 1608.4511,
    'CIV(1551)'   : 1550.77845,
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
    'LYB'         : 1025.7223,
}


