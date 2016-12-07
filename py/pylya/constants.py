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

