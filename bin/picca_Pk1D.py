#!/usr/bin/env python
"""Compute the 1D power spectrum
"""
import sys
import argparse
import glob
from array import array
import numpy as np
import fitsio

from picca import constants
from picca.data import Delta
from picca.pk1d import (compute_correction_reso, compute_pk_noise,
                        compute_pk_raw, fill_masked_pixels, rebin_diff_noise,
                        split_forest)
from picca.utils import userprint


def make_tree(tree, max_num_bins):
    """Makes the ROOT tree to save the data

    Args:
        tree: TTree
            The ROOT tree to fill
        max_num_bins: int
            Maximum number of bins allowed

    Returns:
        The following variables:
            z_qso: Quasar redshifts
            mean_z: Mean redshift of the forest
            mean_reso: Mean resolution of the forest
            mean_snr: Mean signal-to-noise ratio in the forest
            lambda_min_tree: Minimum wavelength (in Angs)
            lambda_max_tree: Maximum wavelength (in Angs)
            plate: Plate number of the observation
            mjd: Modified Julian Date of the observation
            fiber: Fiberid of the object
            nb_mask_pix: Number of masked pixels
            num_bins_tree: Number of bins (storage tree version)
            k_tree: Fourier modes (tree version)
            pk_tree: Power spectrum for the different fourier modes (storage
                tree version)
            pk_raw_tree: Raw power spectrum for the different fourier modes
                (storage tree version)
            pk_noise_tree: Noise power spectrum for the different fourier modes
                (storage tree version)
            correction_reso_tree: Resolution for the correlation function
                (storage tree version)
            pk_diff_tree: Power spectrum of exposures_diff for the different
                fourier modes (storage tree version)
    """
    z_qso = array('f', [0.])
    mean_z = array('f', [0.])
    mean_reso = array('f', [0.])
    mean_snr = array('f', [0.])
    num_masked_pixels_tree = array('f', [0.])

    lambda_min_tree = array('f', [0.])
    lambda_max_tree = array('f', [0.])

    plate = array('i', [0])
    mjd = array('i', [0])
    fiber = array('i', [0])

    num_bins_tree = array('i', [0])
    k_tree = array('f', max_num_bins * [0.])
    pk_tree = array('f', max_num_bins * [0.])
    pk_raw_tree = array('f', max_num_bins * [0.])
    pk_noise_tree = array('f', max_num_bins * [0.])
    pk_diff_tree = array('f', max_num_bins * [0.])
    correction_reso_tree = array('f', max_num_bins * [0.])

    tree.Branch("z_qso", z_qso, "z_qso/F")
    tree.Branch("mean_z", mean_z, "mean_z/F")
    tree.Branch("mean_reso", mean_reso, "mean_reso/F")
    tree.Branch("mean_snr", mean_snr, "mean_snr/F")
    tree.Branch("lambda_min", lambda_min_tree, "lambda_min/F")
    tree.Branch("lambda_max", lambda_max_tree, "lambda_max/F")
    tree.Branch("nb_masked_pixel", num_masked_pixels_tree, "nb_mask_pixel/F")

    tree.Branch("plate", plate, "plate/I")
    tree.Branch("mjd", mjd, "mjd/I")
    tree.Branch("fiber", fiber, "fiber/I")

    tree.Branch("NbBin", num_bins_tree, "NbBin/I")
    tree.Branch("k", k_tree, "k[NbBin]/F")
    tree.Branch("Pk_raw", pk_raw_tree, "Pk_raw[NbBin]/F")
    tree.Branch("Pk_noise", pk_noise_tree, "Pk_noise[NbBin]/F")
    tree.Branch("Pk_diff", pk_diff_tree, "Pk_diff[NbBin]/F")
    tree.Branch("cor_reso", correction_reso_tree, "cor_reso[NbBin]/F")
    tree.Branch("Pk", pk_tree, "Pk[NbBin]/F")

    return (z_qso, mean_z, mean_reso, mean_snr, lambda_min_tree,
            lambda_max_tree, plate, mjd, fiber, num_masked_pixels_tree,
            num_bins_tree, k_tree, pk_tree, pk_raw_tree, pk_noise_tree,
            correction_reso_tree, pk_diff_tree)


def compute_mean_delta(log_lambda, delta, ivar, z_qso, hist_delta,
                       hist_delta_rest_frame, hist_delta_obs_frame, hist_ivar,
                       hist_snr, hist_weighted_delta_rest_frame,
                       hist_weighted_delta_obs_frame):
    """Computes the mean delta and stores it in the control histogram variables

    Args:
        log_lambda: array of floats
            Logarithm of the wavelength (in Angs)
        delta: array of floats
            Mean transmission fluctuation (delta field)
        ivar: array of floats
            Inverse variance
        z_qso: float
            Redshift of the quasar
        hist_delta: TProfile2D
            Root Profile2D histogram to store the mean delta as a function of
            lambda-lambda_rest_frame
        hist_delta_rest_frame: TProfile
            Root Profile histogram to store the mean delta as a function of
            lambda_rest_frame
        hist_delta_obs_frame: TProfile
            Root TProfile histogram to store the mean delta as a function of
            observed lambda
        hist_ivar: TH1D
            Root TH1D histogram to store the inverse variance
        hist_snr: TH1D
            Root TH1D histogram to store the signal-to-noise ratio
        hist_weighted_delta_rest_frame: TProfile
            Root TProfile histogram to store the mean weighted delta as a
            function of lambda_rest_frame
        hist_weighted_delta_obs_frame: TProfile
            Root TProfile histogram to store the mean weighted delta as a
            function of observed lambda
    """

    for index, _ in enumerate(log_lambda):
        lambda_ = np.power(10., log_lambda[index])
        lambda_rf = lambda_ / (1. + z_qso)
        hist_delta.Fill(lambda_, lambda_rf, delta[index])
        hist_delta_rest_frame.Fill(lambda_rf, delta[index])
        hist_delta_obs_frame.Fill(lambda_, delta[index])
        hist_ivar.Fill(ivar[index])
        snr_pixel = (delta[index] + 1) * np.sqrt(ivar[index])
        hist_snr.Fill(snr_pixel)
        hist_ivar.Fill(ivar[index])
        if ivar[index] < 1000:
            hist_weighted_delta_rest_frame.Fill(lambda_rf, delta[index],
                                                ivar[index])
            hist_weighted_delta_obs_frame.Fill(lambda_, delta[index],
                                               ivar[index])


def main(cmdargs):
    # pylint: disable-msg=too-many-locals,too-many-branches,too-many-statements
    """Compute the 1D power spectrum"""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Compute the 1D power spectrum')

    parser.add_argument('--out-dir',
                        type=str,
                        default=None,
                        required=True,
                        help='Output directory')

    parser.add_argument(
        '--out-format',
        type=str,
        default='fits',
        required=False,
        help='Output format: root or fits (if root call PyRoot)')

    parser.add_argument('--in-dir',
                        type=str,
                        default=None,
                        required=True,
                        help='Directory to delta files')

    parser.add_argument(
        '--in-format',
        type=str,
        default='fits',
        required=False,
        help=' Input format used for input files: ascii or fits')

    parser.add_argument('--SNR-min',
                        type=float,
                        default=2.,
                        required=False,
                        help='Minimal mean SNR per pixel ')

    parser.add_argument('--reso-max',
                        type=float,
                        default=85.,
                        required=False,
                        help='Maximal resolution in km/s ')

    parser.add_argument('--lambda-obs-min',
                        type=float,
                        default=3600.,
                        required=False,
                        help='Lower limit on observed wavelength [Angstrom]')

    parser.add_argument('--nb-part',
                        type=int,
                        default=3,
                        required=False,
                        help='Number of parts in forest')

    parser.add_argument('--nb-pixel-min',
                        type=int,
                        default=75,
                        required=False,
                        help='Minimal number of pixels in a part of forest')

    parser.add_argument(
        '--nb-pixel-masked-max',
        type=int,
        default=40,
        required=False,
        help='Maximal number of masked pixels in a part of forest')

    parser.add_argument('--no-apply-filling',
                        action='store_true',
                        default=False,
                        required=False,
                        help='Dont fill masked pixels')

    parser.add_argument(
        '--noise-estimate',
        type=str,
        default='mean_diff',
        required=False,
        help=('Estimate of Pk_noise '
              'pipeline/diff/mean_diff/rebin_diff/mean_rebin_diff'))

    parser.add_argument('--forest-type',
                        type=str,
                        default='Lya',
                        required=False,
                        help='Forest used: Lya, SiIV, CIV')

    parser.add_argument('--debug',
                        action='store_true',
                        default=False,
                        required=False,
                        help='Fill root histograms for debugging')

    parser.add_argument(
        '--abs-igm',
        type=str,
        default='LYA',
        required=False,
        help=('Name of the absorption line in picca.constants defining the '
              'redshift of the forest pixels'))

    args = parser.parse_args(cmdargs)

    # Create root file
    if args.out_format == 'root':
        # pylint: disable-msg=import-error,import-outside-toplevel
        # import is done here as ROOT is not a required package for the code
        # to run, except if args.out_format is set to 'root'
        from ROOT import TH1D, TFile, TTree, TProfile2D, TProfile
        store_file = TFile(args.out_dir + "/Testpicca.root", "RECREATE",
                           "PK 1D studies studies")
        max_num_bins = 700
        tree = TTree("Pk1D", "SDSS 1D Power spectrum Ly-a")
        (z_qso, mean_z, mean_reso, mean_snr, lambda_min_tree, lambda_max_tree,
         plate, mjd, fiber, num_masked_pixels_tree, num_bins_tree, k_tree,
         pk_tree, pk_raw_tree, pk_noise_tree, correction_reso_tree,
         pk_diff_tree) = make_tree(tree, max_num_bins)

        # control histograms
        if args.forest_type == 'Lya':
            lambda_min = 1040.
            lambda_max = 1200.
        elif args.forest_type == 'SiIV':
            lambda_min = 1270.
            lambda_max = 1380.
        elif args.forest_type == 'CIV':
            lambda_min = 1410.
            lambda_max = 1520.
        hist_delta = TProfile2D('hdelta',
                                'delta mean as a function of lambda-lambdaRF',
                                36, 3600., 7200., 16, lambda_min, lambda_max,
                                -5.0, 5.0)
        hist_delta_rest_frame = TProfile(
            'hdelta_RF', 'delta mean as a function of lambdaRF', 320,
            lambda_min, lambda_max, -5.0, 5.0)
        hist_delta_obs_frame = TProfile(
            'hdelta_OBS', 'delta mean as a function of lambdaOBS', 1800, 3600.,
            7200., -5.0, 5.0)
        hist_weighted_delta_rest_frame = TProfile(
            'hdelta_RF_we', 'delta mean weighted as a function of lambdaRF',
            320, lambda_min, lambda_max, -5.0, 5.0)
        hist_weighted_delta_obs_frame = TProfile(
            'hdelta_OBS_we', 'delta mean weighted as a function of lambdaOBS',
            1800, 3600., 7200., -5.0, 5.0)
        hist_ivar = TH1D('hivar', '  ivar ', 10000, 0.0, 10000.)
        hist_snr = TH1D('hsnr', '  snr per pixel ', 100, 0.0, 100.)
        hist_weighted_delta_rest_frame.Sumw2()
        hist_weighted_delta_obs_frame.Sumw2()

    # Read deltas
    if args.in_format == 'fits':
        files = sorted(glob.glob(args.in_dir + "/*.fits.gz"))
    elif args.in_format == 'ascii':
        files = sorted(glob.glob(args.in_dir + "/*.txt"))

    num_data = 0

    # initialize randoms
    np.random.seed(4)

    # loop over input files
    for index, file in enumerate(files):
        if index % 1 == 0:
            userprint("\rread {} of {} {}".format(index, len(files), num_data),
                      end="")

        # read fits or ascii file
        if args.in_format == 'fits':
            hdul = fitsio.FITS(file)
            deltas = [
                Delta.from_fitsio(hdu, pk1d_type=True) for hdu in hdul[1:]
            ]
        elif args.in_format == 'ascii':
            ascii_file = open(file, 'r')
            deltas = [Delta.from_ascii(line) for line in ascii_file]

        num_data += len(deltas)
        userprint("\n ndata =  ", num_data)
        results = None

        # loop over deltas
        for delta in deltas:

            # Selection over the SNR and the resolution
            if (delta.mean_snr <= args.SNR_min or
                    delta.mean_reso >= args.reso_max):
                continue

            # first pixel in forest
            selected_pixels = 10**delta.log_lambda > args.lambda_obs_min
            first_pixel_index = (np.argmax(selected_pixels)
                                 if np.any(selected_pixels) else len(selected_pixels))

            # minimum number of pixel in forest
            min_num_pixels = args.nb_pixel_min
            if (len(delta.log_lambda) - first_pixel_index) < min_num_pixels:
                continue

            # Split in n parts the forest
            max_num_parts = (len(delta.log_lambda) -
                             first_pixel_index) // min_num_pixels
            num_parts = min(args.nb_part, max_num_parts)
            (mean_z_array, log_lambda_array, delta_array, exposures_diff_array,
             ivar_array) = split_forest(num_parts, delta.delta_log_lambda,
                                        delta.log_lambda, delta.delta,
                                        delta.exposures_diff, delta.ivar,
                                        first_pixel_index)
            for index2 in range(num_parts):

                # rebin exposures_diff spectrum
                if (args.noise_estimate == 'rebin_diff' or
                        args.noise_estimate == 'mean_rebin_diff'):
                    exposures_diff_array[index2] = rebin_diff_noise(
                        delta.delta_log_lambda, log_lambda_array[index2],
                        exposures_diff_array[index2])

                # Fill masked pixels with 0.
                (log_lambda_new, delta_new, exposures_diff_new, ivar_new,
                 num_masked_pixels) = fill_masked_pixels(
                     delta.delta_log_lambda, log_lambda_array[index2],
                     delta_array[index2], exposures_diff_array[index2],
                     ivar_array[index2], args.no_apply_filling)
                if num_masked_pixels > args.nb_pixel_masked_max:
                    continue
                if args.out_format == 'root' and args.debug:
                    compute_mean_delta(log_lambda_new, delta_new, ivar_new,
                                       delta.z_qso, hist_delta,
                                       hist_delta_rest_frame,
                                       hist_delta_obs_frame, hist_ivar,
                                       hist_snr, hist_weighted_delta_rest_frame,
                                       hist_weighted_delta_obs_frame)

                # Compute pk_raw
                k, pk_raw = compute_pk_raw(delta.delta_log_lambda, delta_new)

                # Compute pk_noise
                run_noise = False
                if args.noise_estimate == 'pipeline':
                    run_noise = True
                pk_noise, pk_diff = compute_pk_noise(delta.delta_log_lambda,
                                                     ivar_new,
                                                     exposures_diff_new,
                                                     run_noise)

                # Compute resolution correction
                delta_pixel = (delta.delta_log_lambda * np.log(10.) *
                               constants.speed_light / 1000.)
                correction_reso = compute_correction_reso(
                    delta_pixel, delta.mean_reso, k)

                # Compute 1D Pk
                if args.noise_estimate == 'pipeline':
                    pk = (pk_raw - pk_noise) / correction_reso
                elif (args.noise_estimate == 'diff' or
                      args.noise_estimate == 'rebin_diff'):
                    pk = (pk_raw - pk_diff) / correction_reso
                elif (args.noise_estimate == 'mean_diff' or
                      args.noise_estimate == 'mean_rebin_diff'):
                    selection = (k > 0) & (k < 0.02)
                    if args.noise_estimate == 'mean_rebin_diff':
                        selection = (k > 0.003) & (k < 0.02)
                    mean_pk_diff = (sum(pk_diff[selection]) /
                                    float(len(pk_diff[selection])))
                    pk = (pk_raw - mean_pk_diff) / correction_reso

                # save in root format
                if args.out_format == 'root':
                    z_qso[0] = delta.z_qso
                    mean_z[0] = mean_z_array[index2]
                    mean_reso[0] = delta.mean_reso
                    mean_snr[0] = delta.mean_snr
                    lambda_min_tree[0] = np.power(10., log_lambda_new[0])
                    lambda_max_tree[0] = np.power(10., log_lambda_new[-1])
                    num_masked_pixels_tree[0] = num_masked_pixels

                    plate[0] = delta.plate
                    mjd[0] = delta.mjd
                    fiber[0] = delta.fiberid

                    num_bins_tree[0] = min(len(k), max_num_bins)
                    for index3 in range(num_bins_tree[0]):
                        k_tree[index3] = k[index3]
                        pk_raw_tree[index3] = pk_raw[index3]
                        pk_noise_tree[index3] = pk_noise[index3]
                        pk_diff_tree[index3] = pk_diff[index3]
                        pk_tree[index3] = pk[index3]
                        correction_reso_tree[index3] = correction_reso[index3]

                    tree.Fill()

                # save in fits format
                if args.out_format == 'fits':
                    header = [{
                        'name': 'RA',
                        'value': delta.ra,
                        'comment': "QSO's Right Ascension [degrees]"
                    }, {
                        'name': 'DEC',
                        'value': delta.dec,
                        'comment': "QSO's Declination [degrees]"
                    }, {
                        'name': 'Z',
                        'value': delta.z_qso,
                        'comment': "QSO's redshift"
                    }, {
                        'name': 'MEANZ',
                        'value': mean_z_array[index2],
                        'comment': "Absorbers mean redshift"
                    }, {
                        'name': 'MEANRESO',
                        'value': delta.mean_reso,
                        'comment': 'Mean resolution [km/s]'
                    }, {
                        'name': 'MEANSNR',
                        'value': delta.mean_snr,
                        'comment': 'Mean signal to noise ratio'
                    }, {
                        'name': 'NBMASKPIX',
                        'value': num_masked_pixels,
                        'comment': 'Number of masked pixels in the section'
                    }, {
                        'name': 'PLATE',
                        'value': delta.plate,
                        'comment': "Spectrum's plate id"
                    }, {
                        'name':
                            'MJD',
                        'value':
                            delta.mjd,
                        'comment': ('Modified Julian Date,date the spectrum '
                                    'was taken')
                    }, {
                        'name': 'FIBER',
                        'value': delta.fiberid,
                        'comment': "Spectrum's fiber number"
                    }]

                    cols = [k, pk_raw, pk_noise, pk_diff, correction_reso, pk]
                    names = [
                        'k', 'Pk_raw', 'Pk_noise', 'Pk_diff', 'cor_reso', 'Pk'
                    ]
                    comments = [
                        'Wavenumber', 'Raw power spectrum',
                        "Noise's power spectrum",
                        'Noise coadd difference power spectrum',
                        'Correction resolution function',
                        'Corrected power spectrum (resolution and noise)'
                    ]
                    units = [
                        '(km/s)^-1', 'km/s', 'km/s', 'km/s', 'km/s', 'km/s'
                    ]

                    try:
                        results.write(cols,
                                      names=names,
                                      header=header,
                                      comments=comments,
                                      units=units)
                    except AttributeError:
                        results = fitsio.FITS(
                            (args.out_dir + '/Pk1D-' + str(index) + '.fits.gz'),
                            'rw',
                            clobber=True)
                        results.write(cols,
                                      names=names,
                                      header=header,
                                      comment=comments,
                                      units=units)
        if (args.out_format == 'fits' and results is not None):
            results.close()

    # Store root file results
    if args.out_format == 'root':
        store_file.Write()

    userprint("all done ")


if __name__ == '__main__':
    cmdargs=sys.argv[1:]
    main(cmdargs)
