"""This module defines functions and variables required for the analysis of
the 1D power spectra

This module provides with one clas (Pk1D) and several functions:
    - split_forest
    - rebin_diff_noise
    - fill_masked_pixels
    - compute_pk_raw
    - compute_pk_noise
    - compute_correction_reso
See the respective docstrings for more details
"""
import numpy as np
from scipy.fftpack import fft

from . import constants
from .utils import userprint


def split_forest(num_parts,
                 delta_log_lambda,
                 log_lambda,
                 delta,
                 exposures_diff,
                 ivar,
                 first_pixel_index,
                 abs_igm="LYA"):
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
    num_bins = (len(log_lambda) - first_pixel_index) // num_parts

    mean_z_array = []
    log_lambda_array = []
    delta_array = []
    exposures_diff_array = []
    ivar_array = []

    for index in range(1, num_parts):
        log_lambda_limit.append(log_lambda[num_bins * index +
                                           first_pixel_index])

    log_lambda_limit.append(log_lambda[len(log_lambda) - 1] +
                            0.1 * delta_log_lambda)

    for index in range(num_parts):
        selection = ((log_lambda >= log_lambda_limit[index]) &
                     (log_lambda < log_lambda_limit[index + 1]))

        log_lambda_part = log_lambda[selection].copy()
        lambda_abs_igm = constants.ABSORBER_IGM[abs_igm]
        mean_z = (np.power(10., log_lambda_part[len(log_lambda_part) - 1]) +
                  np.power(10., log_lambda_part[0])) / 2. / lambda_abs_igm - 1.0

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
    rebin_delta_log_lambda = rebin * delta_log_lambda

    # rebin not mixing pixels separated by masks
    bins = np.floor((log_lambda - log_lambda.min()) / rebin_delta_log_lambda +
                    0.5).astype(int)

    rebin_exposure_diff = np.bincount(bins.astype(int), weights=exposures_diff)
    rebin_counts = np.bincount(bins.astype(int))
    w = (rebin_counts > 0)
    if len(rebin_counts) == 0:
        userprint("Error: exposures_diff size = 0 ", exposures_diff)
    rebin_exposure_diff = rebin_exposure_diff[w] / np.sqrt(rebin_counts[w])

    # now merge the rebinned array into a noise array
    noise = np.zeros(exposures_diff.size)
    for index in range(len(exposures_diff) // len(rebin_exposure_diff) + 1):
        length_max = min(len(exposures_diff),
                         (index + 1) * len(rebin_exposure_diff))
        noise[index *
              len(rebin_exposure_diff):length_max] = rebin_exposure_diff[:(
                  length_max - index * len(rebin_exposure_diff))]
        # shuffle the array before the next iteration
        np.random.shuffle(rebin_exposure_diff)

    return noise


def fill_masked_pixels(delta_log_lambda, log_lambda, delta, exposures_diff,
                       ivar, no_apply_filling):
    """Fills the masked pixels with zeros

    Args:
        delta_log_lambda: float
            Variation of the logarithm of the wavelength between two pixels
        log_lambda: array of floats
            Array containing the logarithm of the wavelengths (in Angs)
        delta: array of floats
            Mean transmission fluctuation (delta field)
        exposures_diff: array of floats
            Semidifference between two customized coadded spectra obtained from
            weighted averages of the even-number exposures, for the first
            spectrum, and of the odd-number exposures, for the second one
        ivar: array of floats
            Array containing the inverse variance
        no_apply_filling: boolean
            If True, then return the original arrays

    Returns:
        The following variables:
            log_lambda_new: Array containing the logarithm of the wavelengths
                (in Angs)
            delta_new: Mean transmission fluctuation (delta field)
            exposures_diff_new: Semidifference between two customized coadded
                spectra obtained from weighted averages of the even-number
                exposures, for the first spectrum, and of the odd-number
                exposures, for the second one
            ivar_new: Array containing the inverse variance
            num_masked_pixels: Number of masked pixels
    """
    if no_apply_filling:
        return log_lambda, delta, exposures_diff, ivar, 0

    log_lambda_index = log_lambda.copy()
    log_lambda_index -= log_lambda[0]
    log_lambda_index /= delta_log_lambda
    log_lambda_index += 0.5
    log_lambda_index = np.array(log_lambda_index, dtype=int)
    index_all = range(log_lambda_index[-1] + 1)
    index_ok = np.in1d(index_all, log_lambda_index)

    delta_new = np.zeros(len(index_all))
    delta_new[index_ok] = delta

    log_lambda_new = np.array(index_all, dtype=float)
    log_lambda_new *= delta_log_lambda
    log_lambda_new += log_lambda[0]

    exposures_diff_new = np.zeros(len(index_all))
    exposures_diff_new[index_ok] = exposures_diff

    ivar_new = np.zeros(len(index_all), dtype=float)
    ivar_new[index_ok] = ivar

    num_masked_pixels = len(index_all) - len(log_lambda_index)

    return (log_lambda_new, delta_new, exposures_diff_new, ivar_new,
            num_masked_pixels)


def compute_pk_raw(delta_log_lambda, delta):
    """Computes the raw power spectrum

    Args:
        delta_log_lambda: float
            Variation of the logarithm of the wavelength between two pixels
        delta: array of floats
            Mean transmission fluctuation (delta field)

    Returns:
        The following variables
            k: the Fourier modes the Power Spectrum is measured on
            pk: the Power Spectrum
    """
    # spectral length in km/s
    length_lambda = (delta_log_lambda * constants.SPEED_LIGHT * np.log(10.) *
                     len(delta))

    # make 1D FFT
    num_pixels = len(delta)
    num_bins_fft = num_pixels // 2 + 1
    fft_delta = fft(delta)

    # compute power spectrum
    fft_delta = fft_delta[:num_bins_fft]
    pk = (fft_delta.real**2 + fft_delta.imag**2) * length_lambda / num_pixels**2
    k = np.arange(num_bins_fft, dtype=float) * 2 * np.pi / length_lambda

    return k, pk


def compute_pk_noise(delta_log_lambda, ivar, exposures_diff, run_noise):
    """Computes the noise power spectrum

    Two noise power spectrum are computed: one using the pipeline noise and
    another one using the noise derived from exposures_diff

    Args:
        delta_log_lambda: float
            Variation of the logarithm of the wavelength between two pixels
        ivar: array of floats
            Array containing the inverse variance
        exposures_diff: array of floats
            Semidifference between two customized coadded spectra obtained from
            weighted averages of the even-number exposures, for the first
            spectrum, and of the odd-number exposures, for the second one
        run_noise: boolean
            If False the noise power spectrum using the pipeline noise is not
            computed and an array filled with zeros is returned instead

    Returns:
        The following variables
            pk_noise: the noise Power Spectrum using the pipeline noise
            pk_diff: the noise Power Spectrum using the noise derived from
                exposures_diff
    """
    num_pixels = len(ivar)
    num_bins_fft = num_pixels // 2 + 1

    num_noise_exposures = 10
    pk_noise = np.zeros(num_bins_fft)
    error = np.zeros(num_pixels)
    w = ivar > 0
    error[w] = 1.0 / np.sqrt(ivar[w])

    if run_noise:
        for _ in range(num_noise_exposures):
            delta_exp = np.zeros(num_pixels)
            delta_exp[w] = np.random.normal(0., error[w])
            _, pk_exp = compute_pk_raw(delta_log_lambda, delta_exp)
            pk_noise += pk_exp

        pk_noise /= float(num_noise_exposures)

    _, pk_diff = compute_pk_raw(delta_log_lambda, exposures_diff)

    return pk_noise, pk_diff


def compute_correction_reso(delta_pixel, mean_reso, k):
    """Computes the resolution correction

    Args:
        delta_pixel: float
            Variation of the logarithm of the wavelength between two pixels
            (in km/s)
        mean_reso: float
            Mean resolution of the forest
        k: array of floats
            Fourier modes

    Returns:
        The resolution correction
    """
    num_bins_fft = len(k)
    correction = np.ones(num_bins_fft)

    sinc = np.ones(num_bins_fft)
    sinc[k > 0.] = (np.sin(k[k > 0.] * delta_pixel / 2.0) /
                    (k[k > 0.] * delta_pixel / 2.0))**2

    correction *= np.exp(-(k * mean_reso)**2)
    correction *= sinc
    return correction


class Pk1D:
    """Class to represent the 1D Power Spectrum for a given forest

    Attributes:
        ra: float
            Right-ascension of the quasar (in radians).
        dec: float
            Declination of the quasar (in radians).
        z_qso: float
            Redshift of the quasar.
        plate: integer
            Plate number of the observation.
        fiberid: integer
            Fiberid of the observation.
        mjd: integer
            Modified Julian Date of the observation.
        mean_snr: float
            Mean signal-to-noise ratio in the forest
        mean_reso: float
            Mean resolution of the forest
        mean_z: float
            Mean redshift of the forest
        num_masked_pixels: int
            Number of masked pixels
        k: array of floats
            Fourier modes
        pk_raw: array of floats
            Raw power spectrum
        pk_noise: array of floats
            Noise power spectrum for the different Fourier modes
        correction_reso: array of floats
            Resolution correction
        pk: array of floats
            Power Spectrum
        pk_diff: array of floats or None
            Power spectrum of exposures_diff for the different Fourier modes

    Methods:
        __init__: Initialize class instance.
        from_fitsio: Initialize instance from a fits file.
    """

    def __init__(self,
                 ra,
                 dec,
                 z_qso,
                 mean_z,
                 plate,
                 mjd,
                 fiberid,
                 mean_snr,
                 mean_reso,
                 k,
                 pk_raw,
                 pk_noise,
                 correction_reso,
                 pk,
                 num_masked_pixels,
                 pk_diff=None):
        """Initializes instance

        Args:
            ra: float
                Right-ascension of the quasar (in radians).
            dec: float
                Declination of the quasar (in radians).
            z_qso: float
                Redshift of the quasar.
            mean_z: float
                Mean redshift of the forest
            plate: integer
                Plate number of the observation.
            mjd: integer
                Modified Julian Date of the observation.
            fiberid: integer
                Fiberid of the observation.
            mean_snr: float
                Mean signal-to-noise ratio in the forest
            mean_reso: float
                Mean resolution of the forest
            k: array of floats
                Fourier modes
            pk_raw: array of floats
                Raw power spectrum
            pk_noise: array of floats
                Noise power spectrum for the different Fourier modes
            correction_reso: array of floats
                Resolution correction
            pk: array of floats
                Power Spectrum
            num_masked_pixels: int
                Number of masked pixels
            pk_diff: array of floats or None - default: None
                Power spectrum of exposures_diff for the different Fourier modes
        """
        self.ra = ra
        self.dec = dec
        self.z_qso = z_qso
        self.mean_z = mean_z
        self.mean_snr = mean_snr
        self.mean_reso = mean_reso
        self.num_masked_pixels = num_masked_pixels

        self.plate = plate
        self.mjd = mjd
        self.fiberid = fiberid
        self.k = k
        self.pk_raw = pk_raw
        self.pk_noise = pk_noise
        self.correction_reso = correction_reso
        self.pk = pk
        self.pk_diff = pk_diff

    @classmethod
    def from_fitsio(cls, hdu):
        """Reads the 1D Power Spectrum from fits file

        Args:
            hdu: Header Data Unit
                Header Data Unit where the 1D Power Spectrum is read

        Returns:
            An intialized instance of Pk1D
        """

        header = hdu.read_header()

        ra = header['RA']
        dec = header['DEC']
        z_qso = header['Z']
        mean_z = header['MEANZ']
        mean_reso = header['MEANRESO']
        mean_snr = header['MEANSNR']
        plate = header['PLATE']
        mjd = header['MJD']
        fiberid = header['FIBER']
        num_masked_pixels = header['NBMASKPIX']

        data = hdu.read()
        k = data['k'][:]
        pk = data['Pk'][:]
        pk_raw = data['Pk_raw'][:]
        pk_noise = data['Pk_noise'][:]
        correction_reso = data['cor_reso'][:]
        pk_diff = data['Pk_diff'][:]

        return cls(ra, dec, z_qso, mean_z, plate, mjd, fiberid, mean_snr,
                   mean_reso, k, pk_raw, pk_noise, correction_reso, pk,
                   num_masked_pixels, pk_diff)
