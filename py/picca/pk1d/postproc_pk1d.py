"""This module defines a set of functions to postprocess files produced by compute_pk1d.py.

This module provides 3 functions:
    - read_pk1d: Reads all HDUs in an individual "P1D" FITS file and stacks all data in one table
    - compute_mean_pk1d: Computes the mean P1D in a given (z,k) grid of bins, from individual "P1Ds" of individual chunks
    - parallelize_p1d_comp: Main function, runs read_pk1d in parallel, then runs compute_mean_pk1d
See the respective documentation for details
"""

import os, glob
from multiprocessing import Pool

import numpy as np
from scipy.stats import binned_statistic
from scipy.optimize import curve_fit
import fitsio
from astropy.table import Table, vstack
import astropy.io.fits

from picca.constants import SPEED_LIGHT
from picca.constants import ABSORBER_IGM
from picca.utils import userprint

def read_pk1d(filename, kbin_edges, snrcut=None, zbins=None):
    """Read Pk1D data from a single file

    Arguments:
    ----------
    filename: string
    Fits file, containing individual "P1D" for each chunk

    kbin_edges: array of floats
    Edges of the wavenumber bins to be later used, in Angstrom^-1

    snrcut: float, or array of floats - Default: None
    Chunks with mean SNR > snrcut are discarded. If snrcut is an array,
    zbins must be set, so that the cut is made redshift dependent.

    zbins: array of floats - Default: None
    Required if snrcut is an array of floats. List of redshifts
    associated to the list of snr cuts.

    Return:
    -------
    p1d_table: Table, one entry per mode(k) per chunk
    z_array: array[Nchunks]
    """

    p1d_table = []
    z_array = []
    with fitsio.FITS(filename) as hdus:
        for i,h in enumerate(hdus[1:]):
            data = h.read()
            chunk_header = h.read_header()
            chunk_table = Table(data)
            try:
                chunk_table.rename_column('K','k')
                chunk_table.rename_column('PK','Pk')
                chunk_table.rename_column('PK_RAW','Pk_raw')
                chunk_table.rename_column('PK_NOISE','Pk_noise')
                chunk_table.rename_column('PK_DIFF','Pk_diff')
                chunk_table.rename_column('COR_RESO','cor_reso')
            except:
                pass
            try:
                chunk_table.rename_column('PK_NOISE_MISS','Pk_noise_miss')
            except:
                pass

            if np.nansum(chunk_table['Pk'])==0:
                chunk_table['Pk'] = (chunk_table['Pk_raw'] - chunk_table['Pk_noise']) / chunk_table['cor_reso']

            chunk_table['forest_z'] = float(chunk_header['MEANZ'])
            chunk_table['forest_snr'] = float(chunk_header['MEANSNR'])

            if snrcut is not None :
                if hasattr(snrcut, "__len__"):
                    if len(snrcut) != len(zbins) :
                        raise ValueError("Please provide same size for zbins and snrcut arrays")
                    zbin_index = np.argmin(np.abs(zbins - chunk_header['MEANZ']))
                    snrcut_chunk = snrcut[zbin_index]
                else:
                    snrcut_chunk = snrcut

                if(chunk_header['MEANSNR'] < snrcut_chunk):
                    continue

            # Empirically remove very noisy chunks
            wk, = np.where(chunk_table['k'] < kbin_edges[-1])
            if (chunk_table['Pk_noise'][wk] > 1000000 * chunk_table['Pk_raw'][wk]).any():
                userprint(f"file {filename} hdu {i+1} has very high noise power: discarded")
                continue

            p1d_table.append(chunk_table)
            z_array.append(float(chunk_header['MEANZ']))

    p1d_table = vstack(p1d_table)
    p1d_table['Delta2'] = p1d_table['k'] * p1d_table['Pk'] / np.pi
    p1d_table['Pk_norescor'] = p1d_table['Pk_raw'] - p1d_table['Pk_noise']
    p1d_table['Pk_nonoise'] = p1d_table['Pk_raw'] / p1d_table['cor_reso']
    p1d_table['Pk_noraw'] = p1d_table['Pk_noise'] / p1d_table['cor_reso']
    try:
        p1d_table['Pk_noraw_miss'] = p1d_table['Pk_noise_miss'] / p1d_table['cor_reso']
    except:
        pass
    # the following is unnecessary - and does not work if noise=0 (eg. true cont analysis):
    #p1d_table['Pk/Pk_noise'] = p1d_table['Pk_raw'] / p1d_table['Pk_noise']
    z_array = np.array(z_array)

    return p1d_table, z_array


def compute_mean_pk1d(p1d_table, z_array, zbin_edges, kbin_edges, weight_method, nomedians=False, velunits=False):
    """Compute mean P1D in a set of given (z,k) bins, from individual chunks P1Ds

    Arguments:
    ----------
    p1d_table: Table
    Individual Pk1Ds of the contributing forest chunks, stacked in one table using "read_pk1d",
    Contain 'k', 'Pk_raw', 'Pk_noise', 'Pk_diff', 'cor_reso', 'Pk', 'forest_z', 'forest_snr',
            'Delta2', 'Pk_norescor', 'Pk_nonoise', 'Pk_noraw', ('Pk/Pk_noise')

    z_array: Array of floats
    Mean z of each contributing chunk, stacked in one array using "read_pk1d"

    zbin_edges: Array of floats, Edges of the redshift bins we want to use

    kbin_edges: Array of floats
    Edges of the wavenumber bins we want to use, either in (Angstrom)-1 or s/km

    weight_method: String, 3 possible options:
        'fit_snr': Compute mean P1D with weights estimated by fitting dispersion vs SNR
        'simple_snr': Compute mean P1D with weights computed directly from SNR values
                    (SNR as given in compute_Pk1D outputs)
        'no_weights': Compute mean P1D without weights

    nomedians: Bool - Default: False
    Skip computation of median quantities

    velunits: Bool - Default: False
    Compute P1D in velocity units by converting k on-the-fly from AA-1 to s/km

    Return:
    -------
    meanP1d_table: Table
    One row per (z,k) bin; one column per statistics (eg. meanPk, errorPk_noise...)
    Other columns: 'N' (nb of chunks used), 'index_zbin' (index of associated row in metadata_table), 'zbin'

    metadata_table: Table
    One row per z bin; column values z_min/max, k_min/max, N_chunks
    """

    # Initializing stats we want to compute on data
    stats_array = ['mean','error','min','max']
    if nomedians==False:
        stats_array += ['median']

    p1d_table_cols = p1d_table.colnames

    # Convert data into velocity units
    if velunits==True:
        conversion_factor = (ABSORBER_IGM["LYA"] * (1. + p1d_table['forest_z'])) / SPEED_LIGHT
        p1d_table['k'] *= conversion_factor
        for c in p1d_table_cols:
            if 'Pk' in c:
                p1d_table[c] /= conversion_factor

    # Initialize meanP1D_table of len = (nzbins * nkbins) corresponding to hdu[1] in final ouput
    meanP1D_table = Table()
    nbins_z, nbins_k = len(zbin_edges)-1, len(kbin_edges)-1
    meanP1D_table['zbin'] = np.zeros(nbins_z*nbins_k)
    meanP1D_table['index_zbin'] = np.zeros(nbins_z*nbins_k, dtype=int)
    meanP1D_table['N'] = np.zeros(nbins_z*nbins_k, dtype='int64')
    for c in p1d_table_cols:
        for stats in stats_array:
            meanP1D_table[stats+c] = np.zeros(nbins_z*nbins_k)

    # Initialize metadata_table of len = nbins_z corresponding to hdu[2] in final output
    metadata_table = Table()
    metadata_table['z_min'] = zbin_edges[:-1]
    metadata_table['z_max'] = zbin_edges[1:]
    metadata_table['k_min'] = kbin_edges[0] * np.ones(nbins_z)
    metadata_table['k_max'] = kbin_edges[-1] * np.ones(nbins_z)

    # Number of chunks in each redshift bin
    N_chunks, zbin_chunks, izbin_chunks = binned_statistic(z_array, z_array, statistic='count', bins=zbin_edges)
    metadata_table['N_chunks'] = N_chunks

    zbin_centers = np.around((zbin_edges[1:] + zbin_edges[:-1])/2, 5)
    
    for izbin, zbin in enumerate(zbin_edges[:-1]):  # Main loop 1) z bins

        if N_chunks[izbin]==0:  # Fill rows with NaNs
            i_min = izbin * nbins_k
            i_max = (izbin+1) * nbins_k
            meanP1D_table['zbin'][i_min:i_max] = zbin_centers[izbin]
            meanP1D_table['index_zbin'][i_min:i_max] = izbin
            for c in p1d_table_cols:
                for stats in stats_array:
                    meanP1D_table[stats+c][i_min:i_max] = np.nan
            continue

        for ikbin, kbin in enumerate(kbin_edges[:-1]):  # Main loop 2) k bins

            select = (p1d_table['forest_z'] < zbin_edges[izbin + 1])&(p1d_table['forest_z'] > zbin_edges[izbin])&(p1d_table['k'] < kbin_edges[ikbin + 1])&(p1d_table['k'] > kbin_edges[ikbin]) # select a specific (z,k) bin

            index = (nbins_k * izbin) + ikbin # index to be filled in table
            meanP1D_table['zbin'][index] = zbin_centers[izbin]
            meanP1D_table['index_zbin'][index] = izbin

            N = np.ma.count(p1d_table['k'][select]) # Counts the number of chunks in each (z,k) bin
            meanP1D_table['N'][index] = N

            for ic, c in enumerate(p1d_table_cols):

                if N==0:
                    print('Warning: 0 chunks found in bin '+str(zbin_edges[izbin])+'<z<'+str(zbin_edges[izbin+1])+
                          ', '+str(kbin_edges[ikbin])+'<k<'+str(kbin_edges[ikbin+1]))
                    for stats in stats_array:
                        meanP1D_table[stats+c][index] = np.nan
                    continue

                if weight_method=='fit_snr':
                    snr_bin_edges = np.arange(1,11,1)
                    snr_bins = (snr_bin_edges[:-1]+snr_bin_edges[1:])/2
                    def variance_function(snr, a, b):
                        return (a/(snr-1)**2) + b
                    data_values = p1d_table[c][select]
                    data_snr = p1d_table['forest_snr'][select]
                    # Fit function to observed dispersion:
                    standard_dev,_,_ = binned_statistic(data_snr, data_values,
                                                                 statistic='std', bins=snr_bin_edges)
                    coef, coef_cov = curve_fit(variance_function, snr_bins, standard_dev**2, bounds=(0,np.inf))
                    # Model variance from fit function
                    data_snr[data_snr>11] = 11
                    data_snr[ data_snr<1.01] = 1.01
                    variance_estimated = variance_function(data_snr, *coef)
                    weights = 1. / variance_estimated
                    mean = np.average((p1d_table[c][select]), weights=weights)
                    error = np.sqrt(1. / np.sum(weights))

                elif weight_method=='simple_snr':
                    # for forests with snr>snr_limit (hardcoded to 4 as of now), 
                    # the weight is fixed to (snr_limit - 1)**2 = 9
                    snr_limit = 4 
                    forest_snr = p1d_table['forest_snr'][select]
                    w, = np.where(forest_snr <= 1)
                    if len(w)>0: raise RuntimeError('Cannot add weights with SNR<=1.')
                    weights = (forest_snr - 1)**2
                    weights[forest_snr>snr_limit] = (snr_limit - 1)**2
                    mean = np.average((p1d_table[c][select]), weights=weights)
                    # Need to rescale the weights to find the error:
                    #   weights_true = weights * (N - 1) / alpha
                    alpha = np.sum(weights * ((p1d_table[c][select] - mean)**2))
                    error = np.sqrt(alpha / (np.sum(weights) * (N - 1)))

                elif weight_method=='no_weights':
                    mean = np.average((p1d_table[c][select]))
                    error = np.std((p1d_table[c][select])) / np.sqrt(N-1)  # unbiased estimate: N-1

                else:
                    raise ValueError("Option for 'weight_method' argument not found")

                minimum = np.min((p1d_table[c][select]))
                maximum = np.max((p1d_table[c][select]))

                meanP1D_table['mean'+c][index] = mean
                meanP1D_table['error'+c][index] = error
                meanP1D_table['min'+c][index] = minimum
                meanP1D_table['max'+c][index] = maximum
                if nomedians==False:
                    median = np.median((p1d_table[c][select]))
                    meanP1D_table['median'+c][index] = median

    return meanP1D_table, metadata_table


def parallelize_p1d_comp(data_dir, zbin_edges, kbin_edges, weight_method, 
                         snrcut=None, zbins=None, output_file=None, nomedians=False,
                         velunits=False, overwrite=False):
    """Read individual Pk1D data from a set of files and compute P1D statistics, stored in a summary FITS file.

    Arguments:
    ----------
    data_dir: string, Directory where individual P1D FITS files are located

    output_file: string - default:False
    Output file name. If set to None, file name is set to data_dir/mean_Pk1d_[weight_method]_[snr_cut]_[vel].fits.gz

    overwrite: Bool - default: False
    Overwrite output file if existing

    other args: As defined in previous functions
    """

    if output_file is None:
        output_file = os.path.join(data_dir,
                f'mean_Pk1d_{weight_method}{"" if nomedians else "_medians"}{"_snr_cut" if snrcut is not None else ""}{"_vel" if velunits else ""}.fits.gz')
    if os.path.exists(output_file) and not overwrite:
        outdir=Table.read(output_file)
        return outdir

    searchstr = '*'
    files = glob.glob(os.path.join(data_dir,f"Pk1D{searchstr}.fits.gz"))
    ncpu = 8
    with Pool(ncpu) as pool:
        if snrcut is not None:
            full_p1d_table = pool.starmap(read_pk1d,[[f, kbin_edges, snrcut, zbins] for f in files])
        else:
            full_p1d_table = pool.starmap(read_pk1d,[[f, kbin_edges] for f in files])

    p1d_table = vstack([full_p1d_table[i][0] for i in range(len(full_p1d_table))])
    z_array = np.concatenate(tuple([full_p1d_table[i][1] for i in range(len(full_p1d_table))]))

    full_meanP1D_table, full_metadata_table = compute_mean_pk1d(p1d_table, z_array, zbin_edges,
                                                             kbin_edges, weight_method, nomedians, velunits)

    full_meanP1D_table.meta['velunits']=velunits
    hdu0 = astropy.io.fits.PrimaryHDU()
    hdu1 = astropy.io.fits.table_to_hdu(full_meanP1D_table)
    hdu2 = astropy.io.fits.table_to_hdu(full_metadata_table)
    hdul = astropy.io.fits.HDUList([hdu0, hdu1, hdu2])
    hdul.writeto(output_file, overwrite=overwrite)
