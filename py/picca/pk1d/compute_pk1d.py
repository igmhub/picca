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
from numpy.fft import rfft, rfftfreq

from picca import constants
from picca.utils import userprint


def split_forest(num_parts,
                 pixel_step,
                 lambda_or_log_lambda,
                 delta,
                 exposures_diff,
                 ivar,
                 first_pixel_index,
                 abs_igm="LYA",
                 reso_matrix=None,
                 linear_binning=False):
    """Split the forest in n parts

    Arguments
    ---------
    num_parts: int
    Number of parts

    pixel_step: float
    Variation of the wavelength (or log(wavelength)) between two pixels

    lambda_or_log_lambda: array of float
    Wavelength (in Angs) (or its log, but needs to be consistent with
    delta_lambda_or_log_lambda)

    delta: array of float
    Mean transmission fluctuation (delta field)

    exposures_diff: array of float
    Semidifference between two customized coadded spectra obtained from
    weighted averages of the even-number exposures, for the first
    spectrum, and of the odd-number exposures, for the second one

    ivar: array of float
    Inverse variances

    first_pixel_index: int
    Index of the first pixel in the forest

    abs_igm: string - Default: "LYA"
    Name of the absorption in picca.constants defining the
    redshift of the forest pixels

    reso_matrix: 2d-array of float
    The resolution matrix used for corrections

    linear_binning: bool
    assume linear wavelength binning, lambda_or_log_lambda vectors will be
    actual lambda (else they will be log_lambda)

    Return
    ------
    mean_z_array: array of float
    Array with the mean redshift the parts of the forest
    lambda_or_log_lambda_array: Array with logarith of the wavelength for the
        parts of the forest

    delta_array: array of float
    Array with the deltas for the parts of the forest
    exposures_diff_array: Array with the exposures_diff for the parts of
    the forest

    ivar_array: array of float
    Array with the ivar for the parts of the forest
    """
    lambda_or_log_lambda_limit = [lambda_or_log_lambda[first_pixel_index]]
    num_bins = (len(lambda_or_log_lambda) - first_pixel_index) // num_parts

    mean_z_array = []
    lambda_or_log_lambda_array = []
    delta_array = []
    exposures_diff_array = []
    ivar_array = []
    if reso_matrix is not None:
        reso_matrix_array = []

    for index in range(1, num_parts):
        lambda_or_log_lambda_limit.append(
            lambda_or_log_lambda[num_bins * index + first_pixel_index])

    lambda_or_log_lambda_limit.append(
        lambda_or_log_lambda[len(lambda_or_log_lambda) - 1] + 0.1 * pixel_step)

    for index in range(num_parts):
        selection = (
            (lambda_or_log_lambda >= lambda_or_log_lambda_limit[index]) &
            (lambda_or_log_lambda < lambda_or_log_lambda_limit[index + 1]))

        lambda_or_log_lambda_part = lambda_or_log_lambda[selection].copy()
        lambda_abs_igm = constants.ABSORBER_IGM[abs_igm]

        if linear_binning:
            mean_z = np.mean(lambda_or_log_lambda_part) / lambda_abs_igm - 1.0
        else:
            mean_z = np.mean(10**
                             lambda_or_log_lambda_part) / lambda_abs_igm - 1.0

        if reso_matrix is not None:
            reso_matrix_part = reso_matrix[:, selection].copy()

        mean_z_array.append(mean_z)
        lambda_or_log_lambda_array.append(lambda_or_log_lambda_part)
        delta_array.append(delta[selection].copy())
        if exposures_diff is not None:
            exposures_diff_array.append(exposures_diff[selection].copy())
        else:  # fill exposures_diff_array with zeros
            userprint('WARNING: exposure_diff is None, filling with zeros.')
            exposures_diff_array.append(np.zeros(delta[selection].shape))
        ivar_array.append(ivar[selection].copy())
        if reso_matrix is not None:
            reso_matrix_array.append(reso_matrix_part)

    out = [
        mean_z_array, lambda_or_log_lambda_array, delta_array,
        exposures_diff_array, ivar_array
    ]
    if reso_matrix is not None:
        out.append(reso_matrix_array)
    return out


def rebin_diff_noise(pixel_step, lambda_or_log_lambda, exposures_diff):
    """Rebin the semidifference between two customized coadded spectra to
    construct the noise array

    Note that inputs can be either linear or log-lambda spaced units (but
    pixel_step and lambda_or_log_lambda need the same unit)

    The rebinning is done by combining 3 of the original pixels into analysis
    pixels.

    Arguments
    ---------
    pixel_step: float
    Variation of the logarithm of the wavelength between two pixels
    for linear binnings this would need to be the wavelength difference

    lambda_or_log_lambda: array of float
    Array containing the logarithm of the wavelengths (in Angs)
    for linear binnings this would need to be just wavelength

    exposures_diff: array of float
    Semidifference between two customized coadded spectra obtained from
    weighted averages of the even-number exposures, for the first
    spectrum, and of the odd-number exposures, for the second one

    Return
    ------
    noise: array of float
    The noise array
    """
    rebin = 3
    if exposures_diff.size < rebin:
        userprint("Warning: exposures_diff.size too small for rebin")
        return exposures_diff
    rebin_delta_lambda_or_log_lambda = rebin * pixel_step

    # rebin not mixing pixels separated by masks
    bins = np.floor((lambda_or_log_lambda - lambda_or_log_lambda.min()) /
                    rebin_delta_lambda_or_log_lambda + 0.5).astype(int)

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


def fill_masked_pixels(delta_lambda_or_log_lambda, lambda_or_log_lambda, delta,
                       exposures_diff, ivar, no_apply_filling):
    """Fills the masked pixels with zeros

    Note that inputs can be either linear or log-lambda spaced units (but
    delta_lambda_or_log_lambda and lambda_or_log_lambda need the same unit);
    spectrum needs to be uniformly binned in that unit

    Arguments
    ---------
    delta_lambda_or_log_lambda: float
    Variation of the logarithm of the wavelength between two pixels

    lambda_or_log_lambda: array of float
    Array containing the logarithm of the wavelengths (in Angs)

    delta: array of float
    Mean transmission fluctuation (delta field)

    exposures_diff: array of float
    Semidifference between two customized coadded spectra obtained from
    weighted averages of the even-number exposures, for the first
    spectrum, and of the odd-number exposures, for the second one

    ivar: array of float
    Array containing the inverse variance

    no_apply_filling: boolean
    If True, then return the original arrays

    Return
    ------
    lambda_or_log_lambda_new: array of float
    Array containing the wavelength (in Angs) or its logarithm depending on the
    wavelength solution used

    delta_new: array of float
    Mean transmission fluctuation (delta field)

    exposures_diff_new: array of float
    Semidifference between two customized coadded spectra obtained from
    weighted averages of the even-number exposures, for the first spectrum,
    and of the odd-number exposures, for the second one

    ivar_new: array of float
    Array containing the inverse variance

    num_masked_pixels: array of int
    Number of masked pixels
    """
    if no_apply_filling:
        return lambda_or_log_lambda, delta, exposures_diff, ivar, 0

    lambda_or_log_lambda_index = lambda_or_log_lambda.copy()
    lambda_or_log_lambda_index -= lambda_or_log_lambda[0]
    lambda_or_log_lambda_index /= delta_lambda_or_log_lambda
    lambda_or_log_lambda_index += 0.5
    lambda_or_log_lambda_index = np.array(lambda_or_log_lambda_index, dtype=int)
    index_all = range(lambda_or_log_lambda_index[-1] + 1)
    index_ok = np.in1d(index_all, lambda_or_log_lambda_index)

    delta_new = np.zeros(len(index_all))
    delta_new[index_ok] = delta

    lambda_or_log_lambda_new = np.array(index_all, dtype=float)
    lambda_or_log_lambda_new *= delta_lambda_or_log_lambda
    lambda_or_log_lambda_new += lambda_or_log_lambda[0]

    exposures_diff_new = np.zeros(len(index_all))
    if exposures_diff is not None:
        exposures_diff_new[index_ok] = exposures_diff

    ivar_new = np.zeros(len(index_all), dtype=float)
    ivar_new[index_ok] = ivar

    num_masked_pixels = len(index_all) - len(lambda_or_log_lambda_index)

    return (lambda_or_log_lambda_new, delta_new, exposures_diff_new, ivar_new,
            num_masked_pixels)


def compute_pk_raw(delta_lambda_or_log_lambda, delta, linear_binning=False):
    """Compute the raw power spectrum

    Arguments
    ---------
    delta_lambda_or_log_lambda: float
    Variation of (the logarithm of) the wavelength between two pixels

    delta: array of float
    Mean transmission fluctuation (delta field)

    linear_binning: bool
    If set then inputs need to be in AA, outputs will be 1/AA else inputs will
    be in log(AA) and outputs in s/km

    Return
    ------
    k: array of float
    The Fourier modes the Power Spectrum is measured on

    pk: array of float
    The Power Spectrum
    """
    if linear_binning:  # spectral length in AA
        length_lambda = (delta_lambda_or_log_lambda * len(delta))
    else:  # spectral length in km/s
        length_lambda = (delta_lambda_or_log_lambda * constants.SPEED_LIGHT *
                         np.log(10.) * len(delta))

    # make 1D FFT
    num_pixels = len(delta)
    fft_delta = rfft(delta)

    # compute power spectrum
    pk = (fft_delta.real**2 + fft_delta.imag**2) * length_lambda / num_pixels**2
    k = 2 * np.pi * rfftfreq(num_pixels, length_lambda / num_pixels)

    return k, pk


def compute_pk_noise(delta_lambda_or_log_lambda,
                     ivar,
                     exposures_diff,
                     run_noise,
                     num_noise_exposures=10,
                     linear_binning=False):
    """Compute the noise power spectrum

    Two noise power spectrum are computed: one using the pipeline noise and
    another one using the noise derived from exposures_diff

    Arguments
    ---------
    delta_lambda_or_log_lambda: float
    Variation of the logarithm of the wavelength between two pixels

    ivar: array of float
    Array containing the inverse variance

    exposures_diff: array of float
    Semidifference between two customized coadded spectra obtained from
    weighted averages of the even-number exposures, for the first
    spectrum, and of the odd-number exposures, for the second one

    run_noise: boolean
    If False the noise power spectrum using the pipeline noise is not
    computed and an array filled with zeros is returned instead

    num_noise_exposures: int
    Number of exposures to average for noise power estimate

    Return
    ------
    pk_noise: array of float
    The noise Power Spectrum using the pipeline noise

    pk_diff: array of float
    The noise Power Spectrum using the noise derived from exposures_diff
    """
    num_pixels = len(ivar)
    num_bins_fft = num_pixels // 2 + 1

    pk_noise = np.zeros(num_bins_fft)
    error = np.zeros(num_pixels)
    w = ivar > 0
    error[w] = 1.0 / np.sqrt(ivar[w])

    if run_noise:
        for _ in range(num_noise_exposures):
            delta_exp = np.zeros(num_pixels)
            delta_exp[w] = np.random.normal(0., error[w])
            _, pk_exp = compute_pk_raw(delta_lambda_or_log_lambda,
                                       delta_exp,
                                       linear_binning=linear_binning)
            pk_noise += pk_exp

        pk_noise /= float(num_noise_exposures)

    _, pk_diff = compute_pk_raw(delta_lambda_or_log_lambda,
                                exposures_diff,
                                linear_binning=linear_binning)

    return pk_noise, pk_diff


def compute_correction_reso(delta_pixel, mean_reso, k):
    """Computes the resolution correction

    Arguments
    ---------
    delta_pixel: float
    Variation of the logarithm of the wavelength between two pixels
    (in km/s or Ang depending on the units of k submitted)

    mean_reso: float
    Mean resolution of the forest

    k: array of float
    Fourier modes

    Return
    ------
    correction: array of float
    The resolution correction
    """
    num_bins_fft = len(k)
    correction = np.ones(num_bins_fft)

    pixelization_factor = np.sinc(k * delta_pixel / (2 * np.pi))**2

    correction *= np.exp(-(k * mean_reso)**2)
    correction *= pixelization_factor
    return correction


def compute_correction_reso_matrix(reso_matrix, k, delta_pixel, num_pixel):
    """Computes the resolution correction based on the resolution matrix using linear binning

    Arguments
    ---------
    delta_pixel: float
    Variation of the logarithm of the wavelength between two pixels
    (in km/s or Ang depending on the units of k submitted)

    num_pixel: int
    Length  of the spectrum in pixels

    mean_reso: float
    Mean resolution of the forest

    k: array of float
    Fourier modes

    Return
    ------
    correction: array of float
    The resolution correction
    """
    #this allows either computing the power for each pixel seperately or for the mean
    reso_matrix = np.atleast_2d(reso_matrix)

    w2_arr = []
    #first compute the power in the resmat for each pixel, then average
    for resmat in reso_matrix:
        r = np.append(resmat, np.zeros(num_pixel - resmat.size))
        k_resmat, w2 = compute_pk_raw(delta_pixel, r, linear_binning=True)
        try:
            assert np.all(k_resmat == k)
        except AssertionError as error:
            raise Exception(
                "for some reason the resolution matrix correction has "
                "different k scaling than the pk") from error
        w2 /= w2[0]
        w2_arr.append(w2)

    w_res2 = np.mean(w2_arr, axis=0)
    pixelization_factor = np.sinc(k * delta_pixel / (2 * np.pi))**2

    # the following assumes that the resolution matrix is storing the actual
    # resolution convolved with the pixelization kernel along each matrix axis
    correction = w_res2
    correction /= pixelization_factor

    return correction


class Pk1D:
    """Class to represent the 1D Power Spectrum for a given forest

    Class Methods
    -------------
    from_fitsio

    Methods
    -------
    __init__

    Attributes
    ----------
    ra: float
    Right-ascension of the quasar (in radians).

    dec: float
    Declination of the quasar (in radians).

    z_qso: float
    Redshift of the quasar.

    plate: int
    Plate number of the observation.

    fiberid: int
    Fiberid of the observation.

    mjd: int
    Modified Julian Date of the observation.

    mean_snr: float
    Mean signal-to-noise ratio in the forest

    mean_reso: float
    Mean resolution of the forest

    mean_z: float
    Mean redshift of the forest

    num_masked_pixels: int
    Number of masked pixels

    k: array of float
    Fourier modes

    pk_raw: array of float
    Raw power spectrum

    pk_noise: array of float
    Noise power spectrum for the different Fourier modes

    correction_reso: array of float
    Resolution correction

    pk: array of float
    Power Spectrum

    pk_diff: array of float or None
    Power spectrum of exposures_diff for the different Fourier modes
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
        """Initialize instance

        Arguments
        ---------
        ra: float
        Right-ascension of the quasar (in radians).

        dec: float
        Declination of the quasar (in radians).

        z_qso: float
        Redshift of the quasar.

        mean_z: float
        Mean redshift of the forest

        plate: int
        Plate number of the observation.

        mjd: int
        Modified Julian Date of the observation.

        fiberid: int
        Fiberid of the observation.

        mean_snr: float
        Mean signal-to-noise ratio in the forest

        mean_reso: float
        Mean resolution of the forest

        k: array of float
        Fourier modes

        pk_raw: array of float
        Raw power spectrum

        pk_noise: array of float
        Noise power spectrum for the different Fourier modes

        correction_reso: array of float
        Resolution correction

        pk: array of float
        Power Spectrum

        num_masked_pixels: int
        Number of masked pixels

        pk_diff: array of float or None - default: None
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
        """Read the 1D Power Spectrum from fits file

        Arguments
        ---------
        hdu: Header Data Unit
        Header Data Unit where the 1D Power Spectrum is read

        Return
        ------
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
