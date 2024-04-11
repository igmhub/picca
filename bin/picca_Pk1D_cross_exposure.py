#!/usr/bin/env python
"""Compute the individual cross-exposure 1D power spectra
"""

import sys, os, argparse, glob
import fitsio
import numpy as np
from picca.pk1d.compute_pk1d import compute_pk_cross_exposure, Pk1D
from picca.utils import userprint
import multiprocessing as mp
from functools import partial


def treat_pk_file(out_dir, filename):
    """
    Takes a single file containing the FFT of delta for
    multiple exposures and computes the cross-exposure power spectrum. The function
    returns nothing, but writes to disk a new fits file with all the information needed
    to compute Pk_cross_exposure. This is done by looping over each targetid and chunkid, 
    and computing Pk_cross_exposure for each pair of exposures.
    
    Arguments
    ---------
    out_dir: string
    The directory path where the cross-exposure will be written

    filename: string
    The file for which to compute cross-exposure

    Return
    ------
    None
    """
    fft_delta_list = []
    file_out = None
    file_number = filename.split("-")[-1].split(".fits.gz")[0]
    with fitsio.FITS(filename, "r") as hdus:
        for _, hdu in enumerate(hdus[1:]):
            fft_delta_list.append(Pk1D.from_fitsio(hdu))

    targetid_list = np.array(
        [fft_delta_list[i].los_id for i in range(len(fft_delta_list))]
    )
    chunkid_list = np.array(
        [fft_delta_list[i].chunk_id for i in range(len(fft_delta_list))]
    )

    unique_targetid = np.unique(targetid_list)
    unique_chunkid = np.unique(chunkid_list)
    for los_id in unique_targetid:
        for chunk_id in unique_chunkid:
            index = np.argwhere((targetid_list == los_id) & (chunkid_list == chunk_id))
            if len(index) < 2:
                continue

            index = np.concatenate(index, axis=0)

            len_delta_list = [fft_delta_list[i].fft_delta.size for i in index]
            if len(np.unique(len_delta_list)) != 1:
                userprint(
                    f"""The exposures of the sub-forest with LOS_ID {los_id}, and CHUNK_ID {chunk_id}"""
                    """ have different lenghts, they will not be used."""
                )
                continue

            ra = fft_delta_list[index[0]].ra
            dec = fft_delta_list[index[0]].dec
            z_qso = fft_delta_list[index[0]].z_qso
            mean_z = fft_delta_list[index[0]].mean_z

            num_masked_pixels = fft_delta_list[index[0]].num_masked_pixels
            linear_bining = fft_delta_list[index[0]].linear_bining

            k = fft_delta_list[index[0]].k

            mean_snr = np.sqrt(len(index)) * np.mean(
                [fft_delta_list[i].mean_snr for i in index]
            )
            mean_reso = np.mean([fft_delta_list[i].mean_reso for i in index])
            fft_delta = np.array([fft_delta_list[i].fft_delta for i in index])

            pk_raw_cross_exposure = compute_pk_cross_exposure(
                fft_delta,
                fft_delta,
            )

            # Computation of the noise term with decomposition delta_F = delta_lya + delta_n
            fft_delta_noise = np.array(
                [fft_delta_list[i].fft_delta_noise for i in index]
            )
            pk_noise_cross_exposure_auto = compute_pk_cross_exposure(
                fft_delta_noise,
                fft_delta_noise,
            )
            pk_noise_cross_exposure_cross = compute_pk_cross_exposure(
                fft_delta,
                fft_delta_noise,
            )

            pk_noise_cross_exposure = (
                2 * pk_noise_cross_exposure_cross - pk_noise_cross_exposure_auto
            )

            # Since diff is a method using exposure differences, it cannot be computed
            # in a general matter here.
            pk_diff = np.zeros_like(pk_raw_cross_exposure)

            correction_reso = np.mean(
                [fft_delta_list[i].correction_reso for i in index], axis=0
            )

            # Computing pk by taking exposure dependent resolution correction
            fft_delta_resocorr = np.array(
                [
                    fft_delta_list[i].fft_delta
                    / np.sqrt(fft_delta_list[i].correction_reso)
                    for i in index
                ]
            )

            pk_cross_exposure = compute_pk_cross_exposure(
                fft_delta_resocorr,
                fft_delta_resocorr,
            )

            pk1d_class = Pk1D(
                ra=ra,
                dec=dec,
                z_qso=z_qso,
                mean_z=mean_z,
                mean_snr=mean_snr,
                mean_reso=mean_reso,
                num_masked_pixels=num_masked_pixels,
                linear_bining=linear_bining,
                los_id=los_id,
                chunk_id=chunk_id,
                k=k,
                pk_raw=pk_raw_cross_exposure,
                pk_noise=pk_noise_cross_exposure,
                pk_diff=pk_diff,
                correction_reso=correction_reso,
                pk=pk_cross_exposure,
            )
            if file_out is None:
                file_out = fitsio.FITS(
                    (out_dir + "/Pk1D-" + str(file_number) + ".fits.gz"),
                    "rw",
                    clobber=True,
                )

            pk1d_class.write_fits(file_out)


def main(cmdargs):
    """Compute the averaged 1D power spectrum"""

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Compute the averaged 1D power spectrum",
    )

    parser.add_argument(
        "--in-dir",
        type=str,
        default=None,
        required=True,
        help="Directory to individual fft delta files",
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        required=True,
        help="Directory to individual P1D files",
    )
    parser.add_argument(
        "--num-processors",
        type=int,
        default=1,
        required=False,
        help="Number of processors to use for computation",
    )

    args = parser.parse_args(cmdargs)

    os.makedirs(args.out_dir, exist_ok=True)

    files = glob.glob(os.path.join(args.in_dir, f"Pk1D-*.fits.gz"))

    func = partial(treat_pk_file, args.out_dir)

    if args.num_processors <= 1:
        for filename in files:
            func(filename)
    else:
        with mp.Pool(args.num_processors) as pool:
            pool.map(func, files)


if __name__ == "__main__":
    cmdargs = sys.argv[1:]
    main(cmdargs)
