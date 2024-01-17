#!/usr/bin/env python
"""Compute the individual cross-exposure 1D power spectra
"""

import sys, os, argparse, glob
import fitsio
import numpy as np
from picca.pk1d.compute_pk1d import compute_pk_cross_exposure, Pk1D
import multiprocessing as mp
from functools import partial


def treat_pk_file(out_dir, filename):
    fft_delta_list = []
    file_out = None
    file_number = filename.split("-")[-1].split(".fits.gz")[0]
    with fitsio.FITS(filename, "r") as hdus:
        for _, hdu in enumerate(hdus[1:]):
            fft_delta = Pk1D.from_fitsio(hdu)
            if len(fft_delta.los_id.split("_")) < 2:
                raise ValueError("The format of targetid is not adapted to cross exposure estimate"
                                "Please use use non-coadded spectra and keep single exposures at the"
                                "delta extraction stage to separate exposures")

            fft_delta_list.append(fft_delta)

    targetid_list = np.array(
        [fft_delta_list[i].los_id.split("_")[0] for i in range(len(fft_delta_list))]
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

            ra = fft_delta_list[index[0]].ra
            dec = fft_delta_list[index[0]].dec
            z_qso = fft_delta_list[index[0]].z_qso
            mean_z = fft_delta_list[index[0]].mean_z

            num_masked_pixels = fft_delta_list[index[0]].num_masked_pixels
            linear_bining = fft_delta_list[index[0]].linear_bining

            k = fft_delta_list[index[0]].k

            mean_snr = np.sqrt(len(index)) * np.mean([fft_delta_list[i].mean_snr for i in index])
            mean_reso = np.mean([fft_delta_list[i].mean_reso for i in index])

            fft_delta_real = np.array([fft_delta_list[i].fft_delta_real for i in index])
            fft_delta_imag = np.array([fft_delta_list[i].fft_delta_imag for i in index])

            pk_raw_cross_exposure = compute_pk_cross_exposure(
                fft_delta_real, fft_delta_imag
            )

            fft_delta_noise_real = np.array([fft_delta_list[i].fft_delta_noise_real for i in index])
            fft_delta_noise_imag = np.array([fft_delta_list[i].fft_delta_noise_imag for i in index])

            pk_noise_cross_exposure = compute_pk_cross_exposure(
                fft_delta_noise_real, fft_delta_noise_imag
            )

            # Since diff is a method using exposure differences, it cannot be computed
            # in a general matter here. 
            pk_diff = np.zeros_like(fft_delta_list[0].pk_noise)

            correction_reso = np.mean(
                [fft_delta_list[i].correction_reso for i in index], axis=0
            )


            pk_cross_exposure = (pk_raw_cross_exposure - pk_noise_cross_exposure) / correction_reso

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
