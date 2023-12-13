#!/usr/bin/env python3
import argparse

import numpy as np
from scipy.constants import speed_of_light
from scipy.special import wofz

from picca import constants
from picca.delta_extraction.masks.dla_mask import (
    dla_profile, LAMBDA_LYA, OSCILLATOR_STRENGTH_LYA, GAMMA_LYA)


def profile_wave_to_comov_dist(wave, profile_wave, cosmo, differential=False):
    """Convert profile from a function of wavelength to a function of comoving distance

    Parameters
    ----------
    wave : array
        Observed wavelength
    profile_wave : array
        Profile as a function of wavelength
    cosmo : picca.constants.Cosmo
        Cosmology object
    differential : bool, optional
        Whether we have to account for a dlambda / dr factor, by default False

    Returns
    -------
    (array, array)
        (comoving distance grid, profile as a function of comoving distance)
    """
    z = wave / constants.ABSORBER_IGM["LYA"] - 1
    comov_dist = cosmo.get_r_comov(z)
    lin_spaced_comov_dist = np.linspace(comov_dist[0], comov_dist[-1], comov_dist.size)

    profile_comov_dist = profile_wave
    if differential:
        # We are in the f(lambda)dlambda = f(r)dr case,
        # so need to account for dlambda / dr = H(z) * Lambda_Lya / c
        profile_comov_dist = cosmo.get_hubble(z) * constants.ABSORBER_IGM["LYA"] / speed_of_light

    profile_comov_dist = np.interp(lin_spaced_comov_dist, comov_dist, profile_comov_dist)

    return lin_spaced_comov_dist, profile_comov_dist


def fft_profile(profile, dx):
    """Compute Fourier transform of a voigt profile

    Parameters
    ----------
    profile : array
        Input voigt profile in real space (function of comoving distance)
    dx : float
        Comoving distance bin size

    Returns
    -------
    (array, array)
        (wavenumber grid, voigt profile in Fourier space)
    """
    # not normalized
    size = profile.size
    ft_profile = dx * np.fft.fftshift(np.fft.fft(profile - 1))
    k = np.fft.fftshift(np.fft.fftfreq(size, dx)) * (2 * np.pi)

    return k, np.abs(ft_profile)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description=('Compute FVoigt profile'))

    parser.add_argument('-o', '--output', type=str, required=True,
                        help=('Name and path of file to write FVoigt profile to'))

    parser.add_argument('--weights', type=str, required=True,
                        help=('File with total weight as a function of observed wavelength'))

    parser.add_argument('--cddf', type=str, required=True,
                        help=('File with column density distribution function (CDDF)'))

    parser.add_argument('--dla-prob', type=str, required=True,
                        help=('File with probability of finding DLAs in a forest '
                              'as a function of observed wavelength'))

    parser.add_argument('--z-dla', type=float, required=True,
                        help=('Mean DLA redshift'))

    parser.add_argument('--normalize', action='store_true', required=False,
                        help=('Whether the output FVoigt function is normalized'))

    parser.add_argument('--positive-fvoigt', action='store_true', required=False,
                        help=('Whether the output FVoigt function should be positive'))

    args = parser.parse_args()

    cddf_NHI, dN_NHI = np.loadtxt(args.cddf)
    weights_wave = np.loadtxt(args.weights)
    dla_prob_wave = np.loadtxt(args.dla_prob)

    fidcosmo = constants.Cosmo(Om=0.3147)

    comov_dist_prob, dla_prob_comov_dist = profile_wave_to_comov_dist(
        dla_prob_wave[0], dla_prob_wave[1], fidcosmo, differential=True)
    comov_dist_weights, weights_comov_dist = profile_wave_to_comov_dist(
        weights_wave[:, 0], weights_wave[:, 1], fidcosmo)

    weight_interp = np.interp(
        comov_dist_prob, comov_dist_weights, weights_comov_dist, left=0, right=0)
    mean_density = np.average(dla_prob_comov_dist, weights=weight_interp)

    wave = np.arange(2000, 8000, 1)  # TODO this grid may be too sparse
    integrand = np.empty((dN_NHI.size, wave.size))# // 2 + 1))
    for i, NHI in enumerate(dN_NHI):
        profile_wave = dla_profile(wave, args.z_dla, NHI)
        profile_wave /= np.mean(profile_wave)

        # r is in Mpc h^-1 --> k (from tf) will be in (Mpc h^-1)^-1 = h Mpc^-1 :)
        comov_dist, profile_comov_dist = profile_wave_to_comov_dist(wave, profile_wave, fidcosmo)
        k, fourier_profile = fft_profile(profile_comov_dist, np.abs(comov_dist[1] - comov_dist[0]))

        integrand[i] = fourier_profile * mean_density * cddf_NHI[i]

    Fvoigt = np.zeros(k.size)
    for i in range(k.size):
        Fvoigt[i] = np.trapz(integrand[:, i], dN_NHI)

    Fvoigt = Fvoigt[k > 0]
    k = k[k > 0]

    if args.normalize:
        Fvoigt /= Fvoigt[0]
    if not args.positive_fvoigt:
        Fvoigt = -Fvoigt

    np.savetxt(args.output, np.c_[k, Fvoigt])


if __name__ == '__main__':
    main()
