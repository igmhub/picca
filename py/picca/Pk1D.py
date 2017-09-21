import numpy as np
from picca import constants
import pyfftw

def compute_Pk_raw(delta,ll):

#   Length in km/s     
    length_lambda = (np.power(10.,ll[len(ll)-1])-np.power(10.,ll[0]))/(np.power(10.,ll[len(ll)-1])+np.power(10.,ll[0]))*2.0*constants.speed_light/1000.
    
# make 1D FFT        
    nb_pixels = len(delta)
    nb_bin_FFT = nb_pixels/2 + 1
    a = pyfftw.empty_aligned(nb_pixels, dtype='complex128')
    fft = pyfftw.builders.fft(a)
    for i in range(nb_pixels): a[i]=delta[i]
    fft_a = fft() 
    
# compute power spectrum        
    Pk = np.zeros(nb_bin_FFT)
    k =  np.zeros(nb_bin_FFT)        
    for i in range(nb_bin_FFT):
        Pk[i] = float(fft_a[i].real**2 + fft_a[i].imag**2)*length_lambda/float(nb_pixels**2)
        k[i] = float(i)*2.0*np.pi/length_lambda
    
    return k,Pk


def compute_Pk_noise(iv,diff,ll):

    nb_pixels = len(iv)
    nb_bin_FFT = nb_pixels/2 + 1

    nb_noise_exp = 10
    Pk = np.zeros(nb_bin_FFT)
    err = 1.0/np.sqrt(iv)
    
    for iexp in range(nb_noise_exp):
        delta_exp= np.zeros(nb_pixels)
        for i in range(nb_pixels):
            delta_exp[i] = np.random.normal(0.,err[i])
        k_exp,Pk_exp = compute_Pk_raw(delta_exp,ll)
        Pk += Pk_exp 
        
    Pk /= float(nb_noise_exp)

    k_diff,Pk_diff = compute_Pk_raw(diff,ll)
    
    return Pk,Pk_diff

def compute_cor_reso(delta_pixel,mean_reso,k):

    nb_bin_FFT = len(k)
    cor = np.ones(nb_bin_FFT)

    sinc = np.ones(nb_bin_FFT)
    sinc[k>0.] =  (np.sin(k[k>0.]*delta_pixel/2.0)/(k[k>0.]*delta_pixel/2.0))**2
    
    cor *= np.exp(-(k*mean_reso)**2)
    cor *= sinc
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
