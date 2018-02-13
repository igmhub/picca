import numpy as np
from picca import constants
import pyfftw


def split_forest(nb_part,dll,ll,de,diff,iv,first_pixel):

    print 'first_pixel =',first_pixel
    
#    ll_limit=[ll[0]]
#    nb_bin= len(ll)/nb_part

    ll_limit=[ll[first_pixel]]
    nb_bin= (len(ll)-first_pixel)/nb_part
    
    m_z_arr = []
    ll_arr = []
    de_arr = []
    diff_arr = []
    iv_arr = []

    ll_c = ll.copy()
    de_c = de.copy()
    diff_c = diff.copy()
    iv_c = iv.copy()

    for p in range(1,nb_part) :
        ll_limit.append(ll[nb_bin*p+first_pixel])
        
    ll_limit.append(ll[len(ll)-1]+0.1*dll)
    
    lamb_limit =  np.power(10.,ll_limit)
    print lamb_limit

    for p in range(nb_part) : 

        selection = (ll_c>= ll_limit[p]) & (ll_c<ll_limit[p+1])

        ll_part = ll_c[selection]
        de_part = de_c[selection]
        diff_part = diff_c[selection]
        iv_part = iv_c[selection]
             
        lam_lya = constants.absorber_IGM["LYA"]
        m_z = (np.power(10.,ll_part[len(ll_part)-1])+np.power(10.,ll_part[0]))/2./lam_lya -1.0

        m_z_arr.append(m_z)
        ll_arr.append(ll_part)
        de_arr.append(de_part)
        diff_arr.append(diff_part)
        iv_arr.append(iv_part)
  
    return m_z_arr,ll_arr,de_arr,diff_arr,iv_arr


def fill_masked_pixels(dll,ll,delta,diff,iv):

    ll_idx = ll.copy()
    ll_idx -= ll[0]
    ll_idx /= dll
    ll_idx += 0.5
    index =np.array(ll_idx,dtype=int)
    index_all = range(index[-1]+1)
    index_ok = np.in1d(index_all, index)

    delta_new = np.zeros(len(index_all))
    delta_new[index_ok]=delta

    ll_new = np.array(index_all,dtype=float)
    ll_new *= dll
    ll_new += ll[0]

    diff_new = np.zeros(len(index_all))
    diff_new[index_ok]=diff

    iv_new = np.ones(len(index_all))
    iv_new *=0.0
    iv_new[index_ok]=iv

    nb_masked_pixel=len(index_all)-len(index)

    return ll_new,delta_new,diff_new,iv_new,nb_masked_pixel

def compute_Pk_raw(dll,delta,ll):

    #   Length in km/s     
    length_lambda = dll*constants.speed_light/1000.*np.log(10.)*len(delta)
    
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


def compute_Pk_noise(dll,iv,diff,ll,run_noise):

    nb_pixels = len(iv)
    nb_bin_FFT = nb_pixels/2 + 1

    nb_noise_exp = 10
    Pk = np.zeros(nb_bin_FFT)
    err = np.zeros(nb_pixels)
    w = iv>0
    err[w] = 1.0/np.sqrt(iv[w])

    if (run_noise) :
        for iexp in range(nb_noise_exp):
            delta_exp= np.zeros(nb_pixels)
            delta_exp[w] = np.random.normal(0.,err[w])
            k_exp,Pk_exp = compute_Pk_raw(dll,delta_exp,ll)
            Pk += Pk_exp 
        
        Pk /= float(nb_noise_exp)

    k_diff,Pk_diff = compute_Pk_raw(dll,diff,ll)
    
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
