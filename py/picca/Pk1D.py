
import numpy as np
import scipy as sp
from scipy.fftpack import fft

from picca import constants
from picca.utils import userprint


def split_forest(nb_part,delta_log_lambda,log_lambda,delta,exposures_diff,ivar,first_pixel):

    ll_limit=[log_lambda[first_pixel]]
    nb_bin= (len(log_lambda)-first_pixel)//nb_part

    m_z_arr = []
    ll_arr = []
    de_arr = []
    diff_arr = []
    iv_arr = []

    ll_c = log_lambda.copy()
    de_c = delta.copy()
    diff_c = exposures_diff.copy()
    iv_c = ivar.copy()

    for p in range(1,nb_part) :
        ll_limit.append(log_lambda[nb_bin*p+first_pixel])

    ll_limit.append(log_lambda[len(log_lambda)-1]+0.1*delta_log_lambda)

    for p in range(nb_part) :

        selection = (ll_c>= ll_limit[p]) & (ll_c<ll_limit[p+1])

        ll_part = ll_c[selection]
        de_part = de_c[selection]
        diff_part = diff_c[selection]
        iv_part = iv_c[selection]

        lam_lya = constants.absorber_IGM["LYA"]
        m_z = (sp.power(10.,ll_part[len(ll_part)-1])+sp.power(10.,ll_part[0]))/2./lam_lya -1.0

        m_z_arr.append(m_z)
        ll_arr.append(ll_part)
        de_arr.append(de_part)
        diff_arr.append(diff_part)
        iv_arr.append(iv_part)

    return m_z_arr,ll_arr,de_arr,diff_arr,iv_arr

def rebin_diff_noise(delta_log_lambda,log_lambda,exposures_diff):

    crebin = 3
    if (exposures_diff.size < crebin):
        userprint("Warning: exposures_diff.size too small for rebin")
        return exposures_diff
    delta_log_lambda2 = crebin*delta_log_lambda

    # rebin not mixing pixels separated by masks
    bin2 = sp.floor((log_lambda-log_lambda.min())/delta_log_lambda2+0.5).astype(int)

    # rebin regardless of intervening masks
    # nmax = diff.size//crebin
    # bin2 = np.zeros(diff.size)
    # for n in range (1,nmax +1):
    #     bin2[n*crebin:] += sp.ones(diff.size-n*crebin)

    cdiff2 = sp.bincount(bin2.astype(int),weights=exposures_diff)
    civ2 = sp.bincount(bin2.astype(int))
    w = (civ2>0)
    if (len(civ2) == 0) :
        userprint( "Error: exposures_diff size = 0 ",exposures_diff)
    diff2 = cdiff2[w]/civ2[w]*sp.sqrt(civ2[w])
    diffout = np.zeros(exposures_diff.size)
    nmax = len(exposures_diff)//len(diff2)
    for n in range (nmax+1) :
        lengthmax = min(len(exposures_diff),(n+1)*len(diff2))
        diffout[n*len(diff2):lengthmax] = diff2[:lengthmax-n*len(diff2)]
        sp.random.shuffle(diff2)

    return diffout


def fill_masked_pixels(delta_log_lambda,log_lambda,delta,exposures_diff,ivar,no_apply_filling):


    if no_apply_filling : return log_lambda,delta,exposures_diff,ivar,0


    ll_idx = log_lambda.copy()
    ll_idx -= log_lambda[0]
    ll_idx /= delta_log_lambda
    ll_idx += 0.5
    index =sp.array(ll_idx,dtype=int)
    index_all = range(index[-1]+1)
    index_ok = sp.in1d(index_all, index)

    delta_new = np.zeros(len(index_all))
    delta_new[index_ok]=delta

    ll_new = sp.array(index_all,dtype=float)
    ll_new *= delta_log_lambda
    ll_new += log_lambda[0]

    diff_new = np.zeros(len(index_all))
    diff_new[index_ok]=exposures_diff

    iv_new = sp.ones(len(index_all))
    iv_new *=0.0
    iv_new[index_ok]=ivar

    nb_masked_pixel=len(index_all)-len(index)

    return ll_new,delta_new,diff_new,iv_new,nb_masked_pixel

def compute_Pk_raw(delta_log_lambda,delta,log_lambda):

    #   Length in km/s
    length_lambda = delta_log_lambda*constants.speed_light/1000.*sp.log(10.)*len(delta)

    # make 1D FFT
    nb_pixels = len(delta)
    nb_bin_FFT = nb_pixels//2 + 1
    fft_a = fft(delta)

    # compute power spectrum
    fft_a = fft_a[:nb_bin_FFT]
    Pk = (fft_a.real**2+fft_a.imag**2)*length_lambda/nb_pixels**2
    k = np.arange(nb_bin_FFT,dtype=float)*2*sp.pi/length_lambda

    return k,Pk


def compute_Pk_noise(delta_log_lambda,ivar,exposures_diff,log_lambda,run_noise):

    nb_pixels = len(ivar)
    nb_bin_FFT = nb_pixels//2 + 1

    nb_noise_exp = 10
    Pk = np.zeros(nb_bin_FFT)
    err = np.zeros(nb_pixels)
    w = ivar>0
    err[w] = 1.0/sp.sqrt(ivar[w])

    if (run_noise) :
        for _ in range(nb_noise_exp): #iexp unused, but needed
            delta_exp= np.zeros(nb_pixels)
            delta_exp[w] = sp.random.normal(0.,err[w])
            _,Pk_exp = compute_Pk_raw(delta_log_lambda,delta_exp,log_lambda) #k_exp unused, but needed
            Pk += Pk_exp

        Pk /= float(nb_noise_exp)

    _,Pk_diff = compute_Pk_raw(delta_log_lambda,exposures_diff,log_lambda) #k_diff unused, but needed

    return Pk,Pk_diff

def compute_cor_reso(delta_pixel,mean_reso,k):

    nb_bin_FFT = len(k)
    cor = sp.ones(nb_bin_FFT)

    sinc = sp.ones(nb_bin_FFT)
    sinc[k>0.] =  (sp.sin(k[k>0.]*delta_pixel/2.0)/(k[k>0.]*delta_pixel/2.0))**2

    cor *= sp.exp(-(k*mean_reso)**2)
    cor *= sinc
    return cor


class Pk1D :

    def __init__(self,ra,dec,z_qso,mean_z,plate,mjd,fiberid,msnr,mreso,
                 k,Pk_raw,Pk_noise,cor_reso,Pk,nb_mp,Pk_diff=None):

        self.ra = ra
        self.dec = dec
        self.z_qso = z_qso
        self.mean_z = mean_z
        self.mean_snr = msnr
        self.mean_reso = mreso
        self.nb_mp = nb_mp

        self.plate = plate
        self.mjd = mjd
        self.fiberid = fiberid
        self.k = k
        self.Pk_raw = Pk_raw
        self.Pk_noise = Pk_noise
        self.cor_reso = cor_reso
        self.Pk = Pk
        self.Pk_diff = Pk_diff


    @classmethod
    def from_fitsio(cls,hdu):

        """
        read Pk1D from fits file
        """

        hdr = hdu.read_header()

        ra = hdr['RA']
        dec = hdr['DEC']
        z_qso = hdr['Z']
        mean_z = hdr['MEANZ']
        mean_reso = hdr['MEANRESO']
        mean_snr = hdr['MEANSNR']
        plate = hdr['PLATE']
        mjd = hdr['MJD']
        fiberid = hdr['FIBER']
        nb_mp = hdr['NBMASKPIX']

        data = hdu.read()
        k = data['k'][:]
        Pk = data['Pk'][:]
        Pk_raw = data['Pk_raw'][:]
        Pk_noise = data['Pk_noise'][:]
        cor_reso = data['cor_reso'][:]
        Pk_diff = data['Pk_diff'][:]

        return cls(ra,dec,z_qso,mean_z,plate,mjd,fiberid, mean_snr, mean_reso,k,Pk_raw,Pk_noise,cor_reso, Pk,nb_mp,Pk_diff)
