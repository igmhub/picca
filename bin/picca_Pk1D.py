#!/usr/bin/env python
"""Compute the 1D power spectrum
"""
import sys
import argparse
import glob
from array import array
import numpy as np
import fitsio
import os

from picca import constants
from picca.data import Delta
from picca.pk1d.compute_pk1d import (compute_correction_reso,
                        compute_correction_reso_matrix, compute_pk_noise,
                        compute_pk_raw, fill_masked_pixels, rebin_diff_noise,
                        split_forest, check_linear_binning, Pk1D)
from picca.utils import userprint
from multiprocessing import Pool


# loop over input files
num_data=0
def process_all_files(index_file_args):
    global num_data
    file_index, file, args = index_file_args
    if file_index % 5 == 0:
        userprint("\rread {} of {} {}".format(file_index, args.len_files,
                                              num_data),
                  end="")

    # read fits or ascii file
    if args.in_format == 'fits':
        hdul = fitsio.FITS(file)
        try:
            deltas = [Delta.from_fitsio(hdu, pk1d_type=True) for hdu in hdul[1:]]
            running_on_raw_transmission = False
        except ValueError:
            print("\nPk1d_type=True didn't work on read in, maybe perfect model? Trying without any noise or resolution corrections!")
            deltas = [Delta.from_fitsio(hdu,pk1d_type=False) for hdu in hdul[1:]]
            for delta in deltas:
                delta.ivar=np.ones(delta.delta.shape)*1e10
                delta.mean_snr=1e5
                delta.mean_reso=1e-3
                delta.mean_reso_pix=1e-3
                delta.exposures_diff = np.zeros(delta.delta.shape)
            running_on_raw_transmission = True
    elif args.in_format == 'ascii':
        ascii_file = open(file, 'r')
        deltas = [Delta.from_ascii(line) for line in ascii_file]
        running_on_raw_transmission = False

    #add the check for linear binning on first spectrum only (assuming homogeneity within the file)
    delta = deltas[0]
    linear_binning, pixel_step = check_linear_binning(delta)
    if linear_binning:
        userprint("\n\nUsing linear binning, results will have units of AA")
        if (args.disable_reso_matrix or not hasattr(delta, 'resolution_matrix')
                or delta.resolution_matrix is None):
            userprint(
                "Resolution matrix not found or disabled, using Gaussian resolution correction\n"
            )
            reso_correction = "Gaussian"
        else:
            userprint("Using Resolution matrix for resolution correction\n")
            reso_correction = "matrix"
    else:
        userprint("\n\nUsing log binning, results will have units of km/s")
        reso_correction = "Gaussian"
        userprint("Using Gaussian resolution correction\n")

    use_exp_diff_cut=False
    #add check if diff has ever been set
    for delta in deltas:
        #use this cut if some spectra have exposures_differences calculated
        if not np.sum(delta.exposures_diff)==0:
            use_exp_diff_cut = True
            break

    num_data += len(deltas)
    userprint("\n ndata =  ", num_data)
    file_out = None

    for delta in deltas:
        if (delta.mean_snr <= args.SNR_min
                or delta.mean_reso >= args.reso_max):
            continue
        if use_exp_diff_cut:
            if np.sum(delta.exposures_diff)==0:
                continue

        # first pixel in forest
        selected_pixels = 10**delta.log_lambda > args.lambda_obs_min
        #this works as selected_pixels returns a bool and argmax points
        #towards the first occurance for equal values
        first_pixel_index = (np.argmax(selected_pixels) if
                             np.any(selected_pixels) else len(selected_pixels))

        # minimum number of pixel in forest
        min_num_pixels = args.nb_pixel_min
        if (len(delta.log_lambda) - first_pixel_index) < min_num_pixels:
            continue

        # Split the forest in n parts
        max_num_parts = (len(delta.log_lambda) -
                         first_pixel_index) // min_num_pixels
        num_parts = min(args.nb_part, max_num_parts)

        #the split_forest function works with either binning, but needs to be uniform
        if linear_binning:
            split_array = split_forest(
                num_parts,
                pixel_step,
                10**delta.log_lambda,
                delta.delta,
                delta.exposures_diff,
                delta.ivar,
                first_pixel_index,
                reso_matrix=(delta.resolution_matrix
                             if reso_correction == 'matrix' else None),
                linear_binning=True)
            if reso_correction == 'matrix':
                (mean_z_array, lambda_array, delta_array, exposures_diff_array,
                 ivar_array, reso_matrix_array) = split_array
            else:
                (mean_z_array, lambda_array, delta_array, exposures_diff_array,
                 ivar_array) = split_array
        else:
            (mean_z_array, log_lambda_array, delta_array, exposures_diff_array,
             ivar_array) = split_forest(num_parts, pixel_step,
                                        delta.log_lambda, delta.delta,
                                        delta.exposures_diff, delta.ivar,
                                        first_pixel_index)

        #the rebin_diff_noise function works with either binning, but needs to be uniform
        for part_index in range(num_parts):
            # rebin exposures_diff spectrum
            if (args.noise_estimate == 'rebin_diff'
                    or args.noise_estimate == 'mean_rebin_diff'):
                if linear_binning:
                    exposures_diff_array[part_index] = rebin_diff_noise(
                        pixel_step, lambda_array[part_index],
                        exposures_diff_array[part_index])
                else:
                    exposures_diff_array[part_index] = rebin_diff_noise(
                        pixel_step, log_lambda_array[part_index],
                        exposures_diff_array[part_index])

            # Fill masked pixels with 0.
            #the fill_masked_pixels function works with either binning, but needs to be uniform
            if linear_binning:
                #the resolution matrix does not need to have pixels filled in any way...
                (lambda_new, delta_new, exposures_diff_new, ivar_new,
                 num_masked_pixels) = fill_masked_pixels(
                     pixel_step, lambda_array[part_index],
                     delta_array[part_index], exposures_diff_array[part_index],
                     ivar_array[part_index], args.no_apply_filling)
            else:
                (log_lambda_new, delta_new, exposures_diff_new, ivar_new,
                 num_masked_pixels) = fill_masked_pixels(
                     pixel_step, log_lambda_array[part_index],
                     delta_array[part_index], exposures_diff_array[part_index],
                     ivar_array[part_index], args.no_apply_filling)
            if num_masked_pixels > args.nb_pixel_masked_max:
                continue

            # Compute pk_raw, needs uniform binning
            k, pk_raw = compute_pk_raw(pixel_step,
                                       delta_new,
                                       linear_binning=linear_binning)

            # Compute pk_noise
            if (args.noise_estimate == 'pipeline')|(args.noise_estimate == 'mean_pipeline'):
                run_gaussian_noise = True
            else:
                run_gaussian_noise = False
            if not running_on_raw_transmission:
                pk_noise, pk_diff = compute_pk_noise(pixel_step,
                                                     ivar_new,
                                                     exposures_diff_new,
                                                     run_gaussian_noise,
                                                     linear_binning=linear_binning,
                                                     num_noise_exposures=args.num_noise_exp)
            else:
                pk_noise=pk_diff=np.zeros(pk_raw.shape)

            # Compute resolution correction, needs uniform binning
            if not running_on_raw_transmission:
                if linear_binning:
                    #in this case all is in AA space
                    if reso_correction == 'matrix':
                        correction_reso = compute_correction_reso_matrix(
                            reso_matrix=np.mean(reso_matrix_array[part_index],
                                                axis=1),
                            k=k,
                            delta_pixel=pixel_step,
                            num_pixel=len(lambda_new),
                            pixelization_correction = args.add_pixelization_correction)
                    elif reso_correction == 'Gaussian':
                        #this is roughly converting the mean resolution estimate back to pixels
                        #and then multiplying with pixel size
                        mean_reso_AA = pixel_step * delta.mean_reso_pix
                        correction_reso = compute_correction_reso(
                            delta_pixel=pixel_step, mean_reso=mean_reso_AA, k=k)
                else:
                    #in this case all is in velocity space
                    delta_pixel = (pixel_step * np.log(10.) *
                                constants.speed_light / 1000.)
                    correction_reso = compute_correction_reso(
                        delta_pixel=delta_pixel, mean_reso=delta.mean_reso, k=k)
            else:
                correction_reso= compute_correction_reso(delta_pixel=pixel_step, mean_reso=0., k=k)

            # Compute 1D Pk
            if args.noise_estimate == 'pipeline' or running_on_raw_transmission:
                pk = (pk_raw - pk_noise) / correction_reso
            elif args.noise_estimate == 'mean_pipeline':
                if args.kmin_noise_avg is None and linear_binning:
                    #this is roughly the same range as eBOSS analyses for z=2.2
                    selection = (k > 0) & (k < 1.5)
                elif args.kmin_noise_avg is None:
                    selection = (k > 0) & (k < 0.02)
                else:
                    selection = (((k > args.kmin_noise_avg) if args.kmax_noise_avg is not None else 1) &
                                 ((k < args.kmax_noise_avg) if args.kmax_noise_avg is not None else 1))
                mean_pk_noise = np.mean(pk_noise[selection])
                pk = (pk_raw - mean_pk_noise) / correction_reso



            elif (args.noise_estimate == 'diff' or args.noise_estimate == 'rebin_diff'):
                pk = (pk_raw - pk_diff) / correction_reso
            elif (args.noise_estimate == 'mean_diff' or 'mean_rebin_diff'):
                if args.kmin_noise_avg is None and linear_binning:
                    #this is roughly the same range as eBOSS analyses for z=2.2
                    selection = (k > 0) & (k < 1.5)
                elif args.kmin_noise_avg is None:
                    selection = (k > 0) & (k < 0.02)
                else:
                    selection = (((k > args.kmin_noise_avg) if args.kmax_noise_avg is not None else 1) &
                                 ((k < args.kmax_noise_avg) if args.kmax_noise_avg is not None else 1))
                mean_pk_diff = np.mean(pk_diff[selection])
                pk = (pk_raw - mean_pk_diff) / correction_reso

            # save in fits format
            if args.out_format == 'fits':
                pk1d_class = Pk1D(
                    ra=delta.ra,
                    dec=delta.dec,
                    z_qso=delta.z_qso,
                    mean_z=mean_z_array[part_index],
                    mean_snr=delta.mean_snr,
                    mean_reso=delta.mean_reso,
                    num_masked_pixels=num_masked_pixels,
                    linear_bining=linear_binning,
                    los_id=delta.los_id,
                    chunk_id=part_index,
                    k=k,
                    pk_raw=pk_raw,
                    pk_noise=pk_noise,
                    pk_diff=pk_diff,
                    correction_reso=correction_reso,
                    pk=pk,
                )
                
                if file_out is None:
                    file_out = fitsio.FITS((args.out_dir + '/Pk1D-' +
                                           str(file_index) + '.fits.gz'),
                                          'rw',
                                          clobber=True)
                                    
                pk1d_class.write_fits(file_out)

    if (args.out_format == 'fits' and file_out is not None):
        file_out.close()

    return 0


def main(cmdargs):
    # pylint: disable-msg=too-many-locals,too-many-branches,too-many-statements
    """Compute the 1D power spectrum
    Uses the resolution matrix correction for DESI data"""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Compute the 1D power spectrum')

    parser.add_argument('--out-dir',
                        type=str,
                        default=None,
                        required=True,
                        help='Output directory')

    parser.add_argument('--out-format',
                        type=str,
                        default='fits',
                        required=False,
                        help='Output format: ascii or fits')

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
              'pipeline/mean_pipeline/diff/mean_diff/rebin_diff/mean_rebin_diff'))

    parser.add_argument('--forest-type',
                        type=str,
                        default='Lya',
                        required=False,
                        help='Forest used: Lya, SiIV, CIV')

    parser.add_argument(
        '--abs-igm',
        type=str,
        default='LYA',
        required=False,
        help=('Name of the absorption line in picca.constants defining the '
              'redshift of the forest pixels'))

    #additional options
    parser.add_argument('--num-processors',
                        type=int,
                        default=1,
                        required=False,
                        help='Number of processors to use for computation')

    parser.add_argument(
        '--num-noise-exp',
        default=100,
        type=int,
        required=False,
        help='number of pipeline noise realizations to generate per spectrum')

    parser.add_argument('--disable-reso-matrix',
                        default=False,
                        action='store_true',
                        required=False,
                        help=('do not use the resolution matrix even '
                              'if it exists and we are on linear binning'))

    parser.add_argument('--add-pixelization-correction',
                        default=False,
                        action='store_true',
                        required=False,
                        help=('Add a pixelization correction, as if the resolution  '
                              'matrix was doubly pixelized. Only use this option in '
                              'quickquasars mocks'))

    parser.add_argument(
        '--seed',
        default=4,
        required=False,
        type=int,
        help=
        ('seed for random number generator, default 4, let system determine seed if set to 0'
         ))


    parser.add_argument(
        '--kmin_noise_avg',
        default=None,
        required=False,
        type=float,
        help=
        ('minimal mode to take into account when computing noise/diff power average'
         ))
    parser.add_argument(
        '--kmax_noise_avg',
        default=None,
        required=False,
        type=float,
        help=
        ('maximal mode to take into account when computing noise/diff power average'
         ))

    args = parser.parse_args(cmdargs)

    if args.seed==0:
        seed=None
    else:
        seed=args.seed
    # Read deltas
    if args.in_format == 'fits':
        files = sorted(glob.glob(args.in_dir + "/*.fits.gz"))
    elif args.in_format == 'ascii':
        files = sorted(glob.glob(args.in_dir + "/*.txt"))

    # initialize randoms
    np.random.seed(seed)
    userprint(f"Computing Pk1d for {args.in_dir}")
    args.len_files = len(files)
    #create output dir if it does not exist
    os.makedirs(args.out_dir, exist_ok=True)

    print([[i, f] for i, f in enumerate(files)])
    if args.num_processors > 1:
        pool = Pool(args.num_processors)
        index_file_args = [(i, f, args) for i, f in enumerate(files)]
        pool.map(process_all_files, index_file_args)
    else:
        [process_all_files((i, f, args)) for i, f in enumerate(files)]
    userprint("all done ")


if __name__ == '__main__':
    cmdargs = sys.argv[1:]
    main(cmdargs)
