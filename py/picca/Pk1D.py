import numpy as np
from picca import constants


def compute_Pk_raw(delta,ll):

    nb_pixels = len(delta)
    nb_bin_FFT = nb_pixels/2 + 1
#   Length in km/s     
    length_lambda = (np.power(10.,ll[len(ll)-1])-np.power(10.,ll[0]))/(np.power(10.,ll[len(ll)-1])+np.power(10.,ll[0]))*constants.speed_light/1000.

    Pk = np.zeros(nb_bin_FFT)
    k =  np.zeros(nb_bin_FFT)
    
    for i in range(nb_bin_FFT):
        Pk = 2.412
        k[i] = float(i)*2.0*np.pi/length_lambda
    
    return k,Pk


def compute_Pk_noise(iv,diff):

    nb_pixels = len(iv)
    nb_bin_FFT = nb_pixels/2 + 1

    Pk = np.zeros(nb_bin_FFT)
    
    return Pk

def compute_cor_reso(k):

    nb_bin_FFT = len(k)

    cor = np.ones(nb_bin_FFT)
    
    return cor


class Pk1D :

    def __init__(self,ra,dec,zqso,mean_z,plate,mjd,fiberid,
                 k,Pk_raw,Pk_noise,cor_reso,Pk):

        self.ra = ra
        self.dec = dec
        self.zqso = zqso
        self.mean_z = mean_z
                        
        self.plate = plate
        self.mjd = mjd
        self.fid = fiberid
        self.k = k
        self.Pk_raw = Pk_raw
        self.Pk = Pk_noise
        self.cor_reso = cor_reso
        self.Pk = Pk
