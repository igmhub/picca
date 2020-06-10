"""This module defines functions and variables required for the analysis of
the 1D power spectra

This module provides with one clas (Pk1D) and several functions:
    - split_forest
    - rebin_diff_noise
    - fill_masked_pixels
    - compute_Pk_raw
    - compute_Pk_noise
    - compute_cor_reso
See the respective docstrings for more details
"""
import numpy as np
from scipy.fftpack import fft

from picca import constants
from picca.utils import userprint


def split_forest(num_parts, delta_log_lambda, log_lambda, delta, exposures_diff,
                 ivar, first_pixel_index, abs_igm="LYA"):
    """Splits the forest in n parts

    Args:
        num_parts: int
            Number of parts
        delta_log_lambda: float
            Variation of the logarithm of the wavelength between two pixels
        log_lambda: array of float
            Logarith of the wavelength (in Angs)
        delta: array of float
            Mean transmission fluctuation (delta field)
        exposures_diff: array of float
            Semidifference between two customized coadded spectra obtained from
            weighted averages of the even-number exposures, for the first
            spectrum, and of the odd-number exposures, for the second one
        ivar: array of floats
            Inverse variances
        first_pixel_index: int
            Index of the first pixel in the forest
        abs_igm: string - default: "LYA"
            Name of the absorption in picca.constants defining the
            redshift of the forest pixels

    Returns:
        The following variables:
            mean_z_array: Array with the mean redshift the parts of the forest
            log_lambda_array: Array with logarith of the wavelength for the
                parts of the forest
            delta_array: Array with the deltas for the parts of the forest
            exposures_diff_array: Array with the exposures_diff for the parts of
                the forest
            ivar_array: Array with the ivar for the parts of the forest

    """
    log_lambda_limit = [log_lambda[first_pixel_index]]
    num_bins = (len(log_lambda) - first_pixel_index)//num_parts

    mean_z_array = []
    log_lambda_array = []
    delta_array = []
    exposures_diff_array = []
    ivar_array = []

    for index in range(1, num_parts):
        log_lambda_limit.append(log_lambda[num_bins*index + first_pixel_index])

    log_lambda_limit.append(log_lambda[len(log_lambda) - 1] +
                            0.1*delta_log_lambda)

    for index in range(num_parts):
        selection = ((log_lambda >= log_lambda_limit[index]) &
                     (log_lambda < log_lambda_limit[index + 1]))

        log_lambda_part = log_lambda[selection].copy()
        lambda_abs_igm = constants.ABSORBER_IGM[abs_igm]
        mean_z = (np.power(10., log_lambda_part[len(log_lambda_part) - 1]) +
                  np.power(10., log_lambda_part[0]))/2./lambda_abs_igm -1.0

        mean_z_array.append(mean_z)
        log_lambda_array.append(log_lambda_part)
        delta_array.append(delta[selection].copy())
        exposures_diff_array.append(exposures_diff[selection].copy())
        ivar_array.append(ivar[selection].copy())

    return (mean_z_array, log_lambda_array, delta_array, exposures_diff_array,
            ivar_array)

def rebin_diff_noise(delta_log_lambda, log_lambda, exposures_diff):
    """Rebin the semidifference between two customized coadded spectra to
    construct the noise array

    The rebinning is done by combining 3 of the original pixels into analysis
    pixels.

    Args:
        delta_log_lambda: float
            Variation of the logarithm of the wavelength between two pixels
        log_lambda: array of floats
            Array containing the logarithm of the wavelengths (in Angs)
        exposures_diff: array of floats
            Semidifference between two customized coadded spectra obtained from
            weighted averages of the even-number exposures, for the first
            spectrum, and of the odd-number exposures, for the second one

    Returns:
        The noise array
    """
    rebin = 3
    if exposures_diff.size < rebin:
        userprint("Warning: exposures_diff.size too small for rebin")
        return exposures_diff
    rebin_delta_log_lambda = rebin*delta_log_lambda

    # rebin not mixing pixels separated by masks
    bins = np.floor((log_lambda - log_lambda.min())/
                    rebin_delta_log_lambda + 0.5).astype(int)

    rebin_exposure_diff = np.bincount(bins.astype(int), weights=exposures_diff)
    rebin_counts = np.bincount(bins.astype(int))
    w = (rebin_counts > 0)
    if len(rebin_counts) == 0:
        userprint("Error: exposures_diff size = 0 ", exposures_diff)
    rebin_exposure_diff = rebin_exposure_diff[w]/np.sqrt(rebin_counts[w])

    # now merge the rebinned array into a noise array
    noise = np.zeros(exposures_diff.size)
    for index in range(len(exposures_diff)//len(rebin_exposure_diff) + 1):
        length_max = min(len(exposures_diff),
                         (index + 1)*len(rebin_exposure_diff))
        noise[index*len(rebin_exposure_diff):
              length_max] = rebin_exposure_diff[:(length_max - index*
                                                  len(rebin_exposure_diff))]
        # shuffle the array before the next iteration
        np.random.shuffle(rebin_exposure_diff)

    return noise


def fill_masked_pixels(delta_log_lambda,log_lambda,delta,exposures_diff,ivar,no_apply_filling):


    if no_apply_filling : return log_lambda,delta,exposures_diff,ivar,0


    ll_idx = log_lambda.copy()
    ll_idx -= log_lambda[0]
    ll_idx /= delta_log_lambda
    ll_idx += 0.5
    index =np.array(ll_idx,dtype=int)
    index_all = range(index[-1]+1)
    index_ok = np.in1d(index_all, index)

    delta_new = np.zeros(len(index_all))
    delta_new[index_ok]=delta

    ll_new = np.array(index_all,dtype=float)
    ll_new *= delta_log_lambda
    ll_new += log_lambda[0]

    diff_new = np.zeros(len(index_all))
    diff_new[index_ok]=exposures_diff

    iv_new = np.ones(len(index_all))
    iv_new *=0.0
    iv_new[index_ok]=ivar

    nb_masked_pixel=len(index_all)-len(index)

    return ll_new,delta_new,diff_new,iv_new,nb_masked_pixel

def compute_Pk_raw(delta_log_lambda,delta,log_lambda):

    #   Length in km/s
    length_lambda = delta_log_lambda*constants.SPEED_LIGHT*np.log(10.)*len(delta)

    # make 1D FFT
    nb_pixels = len(delta)
    nb_bin_FFT = nb_pixels//2 + 1
    fft_a = fft(delta)

    # compute power spectrum
    fft_a = fft_a[:nb_bin_FFT]
    Pk = (fft_a.real**2+fft_a.imag**2)*length_lambda/nb_pixels**2
    k = np.arange(nb_bin_FFT,dtype=float)*2*np.pi/length_lambda

    return k,Pk


def compute_Pk_noise(delta_log_lambda,ivar,exposures_diff,log_lambda,run_noise):

    nb_pixels = len(ivar)
    nb_bin_FFT = nb_pixels//2 + 1

    nb_noise_exp = 10
    Pk = np.zeros(nb_bin_FFT)
    err = np.zeros(nb_pixels)
    w = ivar>0
    err[w] = 1.0/np.sqrt(ivar[w])

    if (run_noise) :
        for _ in range(nb_noise_exp): #iexp unused, but needed
            delta_exp= np.zeros(nb_pixels)
            delta_exp[w] = np.random.normal(0.,err[w])
            _,Pk_exp = compute_Pk_raw(delta_log_lambda,delta_exp,log_lambda) #k_exp unused, but needed
            Pk += Pk_exp

        Pk /= float(nb_noise_exp)

    _,Pk_diff = compute_Pk_raw(delta_log_lambda,exposures_diff,log_lambda) #k_diff unused, but needed

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
