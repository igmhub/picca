#!/usr/bin/env python
"""Plots the 1D Power Spectrum
"""
import argparse
import glob
import sys
import fitsio
import numpy as np
import matplotlib.pyplot as plt

from picca.utils import userprint
from picca.pk1d import Pk1D


def main(cmdargs):
    """Plots the 1D Power Spectrum"""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--in-dir',
                        type=str,
                        default=None,
                        required=True,
                        help='data directory')

    parser.add_argument('--SNR-min',
                        type=float,
                        default=2.,
                        required=False,
                        help='minimal mean SNR per pixel ')

    parser.add_argument('--z-min',
                        type=float,
                        default=2.1,
                        required=False,
                        help='minimal mean absorption redshift ')

    parser.add_argument('--reso-max',
                        type=float,
                        default=85.,
                        required=False,
                        help='maximal resolution in km/s ')

    parser.add_argument('--out-fig',
                        type=str,
                        default='Pk1D.png',
                        required=False,
                        help='data directory')

    args = parser.parse_args(cmdargs)

    # Binning corresponding to BOSS paper
    num_z_bins = 13
    num_k_bins = 35
    k_min = 0.000813
    k_max = k_min + num_k_bins * 0.000542

    # initialize arrays
    sum_pk = np.zeros([num_z_bins, num_k_bins], dtype=np.float64)
    sum_pk_square = np.zeros([num_z_bins, num_k_bins], dtype=np.float64)
    counts = np.zeros([num_z_bins, num_k_bins], dtype=np.float64)
    k = np.zeros([num_k_bins], dtype=np.float64)
    for index_k in range(num_k_bins):
        k[index_k] = k_min + (index_k + 0.5) * 0.000542

    # list of Pk(1D)
    files = sorted(glob.glob(args.in_dir + "/*.fits.gz"))

    num_data = 0

    # loop over input files
    for index, file in enumerate(files):
        if index % 1 == 0:
            sys.stderr.write("\rread {} of {} {}".format(
                index, len(files), num_data))

        # read fits files
        hdul = fitsio.FITS(file)
        pk1ds = [Pk1D.from_fitsio(hdu) for hdu in hdul[1:]]
        num_data += len(pk1ds)
        userprint("\n ndata =  ", num_data)

        # loop over pk1ds
        for pk in pk1ds:

            # Selection over the SNR and the resolution
            if pk.mean_snr <= args.SNR_min or pk.mean_reso >= args.reso_max:
                continue

            if pk.mean_z <= args.z_min:
                continue

            index_z = int((pk.mean_z - args.z_min) / 0.2)
            if index_z >= num_z_bins or index_z < 0:
                continue

            for index2 in range(len(pk.k)):
                index_k = int(
                    (pk.k[index2] - k_min) / (k_max - k_min) * num_k_bins)
                if index_k >= num_k_bins or index_k < 0:
                    continue
                sum_pk[index_z, index_k] += pk.pk[index2] * pk.k[index2] / np.pi
                sum_pk_square[index_z, index_k] += (pk.pk[index2] *
                                                    pk.k[index2] / np.pi)**2
                counts[index_z, index_k] += 1.0

    # compute mean and error on Pk
    mean_pk = np.where(counts != 0, sum_pk / counts, 0.0)
    error_pk = np.where(
        counts != 0, np.sqrt(((sum_pk_square / counts) - mean_pk**2) / counts),
        0.0)

    # plot settings
    colors = [
        'm', 'r', 'b', 'k', 'chartreuse', 'gold', 'aqua', 'slateblue', 'orange',
        'mediumblue', 'darkgrey', 'olive', 'firebrick'
    ]
    markersize = 6
    fontsize = 16
    labelsize = 10
    legendsize = 12

    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_yscale('log')

    for index_z in range(num_z_bins):
        z = 2.2 + index_z * 0.2
        #ax.errorbar(k, mean_pk[index_z,:], yerr=error_pk[index_z,:], fmt='o',
        #            color=colors[index_z], markersize=markersize,
        #            label =r('\bf {:1.1f} $\displaystyle <$ '
        #                     'z $\displaystyle <$ {:1.1f}').format(z - 0.1,
        #                                                           z + 0.1))
        ax.errorbar(k,
                    mean_pk[index_z, :],
                    yerr=error_pk[index_z, :],
                    fmt='o',
                    color=colors[index_z],
                    markersize=markersize,
                    label=r' z = {:1.1f}'.format(z))

    #ax.set_xlabel(r'\bf $\displaystyle k [km.s^{-1}]$', fontsize=fontsize)
    ax.set_xlabel(r' k [km/s]', fontsize=fontsize)
    #ax.set_ylabel(r'\bf $\displaystyle \Delta^2_{\mathrm{1d}}(k) $',
    #              fontsize=fontsize, labelpad=-1)
    ax.set_ylabel(r'P(k)k/pi ', fontsize=fontsize, labelpad=-1)
    ax.xaxis.set_ticks_position('both')
    ax.xaxis.set_tick_params(direction='in')
    ax.yaxis.set_ticks_position('both')
    ax.yaxis.set_tick_params(direction='in')
    ax.xaxis.set_tick_params(labelsize=labelsize)
    ax.yaxis.set_tick_params(labelsize=labelsize)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles,
              labels,
              loc=2,
              bbox_to_anchor=(1.03, 0.98),
              borderaxespad=0.,
              fontsize=legendsize)
    fig.subplots_adjust(top=0.98,
                        bottom=0.114,
                        left=0.078,
                        right=0.758,
                        hspace=0.2,
                        wspace=0.2)

    plt.show()
    fig.savefig(args.out_fig, transparent=False)

    userprint("all done ")


if __name__ == '__main__':
    cmdargs=sys.argv[1:]
    main(cmdargs)
