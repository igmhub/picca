"""This module defines a set of functions to postprocess files produced by pk1d.py.

This module provides 3 functions:
    - read_pk1d: Reads all hdu in an individual P1D fits file and stacks all data in one table
    - compute_mean_pk1d: Takes the individual P1D of each forest chunk and computes the mean P1D with adding weights option
    - parallelize_p1d_comp: Parallelizes all individual P1D fits files in a directory, reads and stacks all data in one table using read_pk1d and then computes the mean using compute_mean_pk1d
See the respective documentation for details
"""

import numpy as np
import fitsio
from astropy.table import Table, vstack
from scipy.stats import binned_statistic
import glob
import os

def read_pk1d(f, kbin_edges, snr_cut_mean=None, zbins=None):
    """Read Pk1D data from file(s)
    
    Args:
        f: Fits file, Individual p1d 
        kbin_edges: Array of floats, Edges of the wavenumber bins we want to use (logsample/not)
        snr_cut_mean: Array of floats, Optional
                      Mean SNR threshold to be applied for each redshift bin, Defaults to None
        zbins: Array of floats, Optional if snr_cut_mean is not None
               Which redshift bins to use
    Output:
        data_array: Table, one entry per mode(k) per chunk
        z_array: array[Nchunks]
    """
  
    data_array = []
    z_array = []
    with fitsio.FITS(f) as hdus:
        for i,h in enumerate(hdus[1:]):
            data = h.read()
            header = h.read_header()
            tab = Table(data)
            try:
                tab.rename_column('K','k')
                tab.rename_column('PK','Pk')
                tab.rename_column('PK_RAW','Pk_raw')
                tab.rename_column('PK_NOISE','Pk_noise')
                tab.rename_column('PK_DIFF','Pk_diff')
                tab.rename_column('COR_RESO','cor_reso')
            except:
                pass
            try:
                tab.rename_column('PK_NOISE_MISS','Pk_noise_miss')
            except:
                pass
            
            if np.nansum(tab['Pk'])==0:
                tab['Pk'] = (tab['Pk_raw'] - tab['Pk_noise']) / tab['cor_reso']
                
            tab['forest_z'] = float(header['MEANZ'])
            tab['forest_snr'] = float(header['MEANSNR'])
            
            if snr_cut_mean is not None :
                if len(snr_cut_mean) != len(zbins) :
                    raise ValueError("Please provide same size for zbins and snr_cut_mean arrays")
                    
                zbin_index = np.argmin(np.abs(zbins - header['MEANZ']))
                
                if(header['MEANSNR'] < snr_cut_mean[zbin_index]):
                    continue
                    
            if (tab['Pk_noise'][tab['k']<kbin_edges[-1]]>tab['Pk_raw'][tab['k']<kbin_edges[-1]]*1000000).any():
                print(f"file {f} hdu {i+1} has very high noise power, ignoring, max value: {(tab['Pk_noise'][tab['k']<kbin_edges[-1]]/tab['Pk_raw'][tab['k']<kbin_edges[-1]]).max()}*Praw")
                continue
                
            data_array.append(tab)
            z_array.append(float(header['MEANZ']))
            
    if len(data_array) > 1:
        data_array = vstack(data_array)
        data_array['Delta2'] = data_array['k'] * data_array['Pk'] / np.pi
        data_array['Pk_norescor'] = data_array['Pk_raw'] - data_array['Pk_noise']
        data_array['Pk_nonoise'] = data_array['Pk_raw'] / data_array['cor_reso']
        data_array['Pk_noraw'] = data_array['Pk_noise'] / data_array['cor_reso']
        try:
            data_array['Pk_noraw_miss'] = data_array['Pk_noise_miss'] / data_array['cor_reso']
        except:
            pass
        data_array['Pk/Pk_noise'] = data_array['Pk_raw'] / data_array['Pk_noise']
    else:
        print(f"only {len(data_array)} spectra in file, ignoring this as it currently messes with analysis")
    z_array = np.array(z_array)

    return data_array, z_array
  
    
def compute_mean_pk1d(data_array, z_array, zbin_edges, kbin_edges, nomedians=False, velunits=False, addweights=False):
    """Takes the individual P1D of each forest chunk and computes the mean P1D with adding weights option
    
    Args: 
        data_array: Table, Individual_pk1d(s) of the contributing forest chunkcs stacked in one table using "read_pk1d", 
                    containing 'k', 'Pk_raw', 'Pk_noise', 'Pk_diff', 'cor_reso', 'Pk',
                    'forest_z', 'forest_snr','Delta2', 'Pk_norescor', 'Pk_nonoise', 'Pk_noraw', 'Pk/Pk_noise'
        z_array: Array of floats, Mean z of each contributing forest chunck stacked in one array done in "read_pk1d"
        zbin_edges: Array of floats, Edges of the redshift bins we want to use
        kbin_edges: Array of floats, Edges of the wavenumber bins we want to use (logsample/not)
        nomedians: Bool, Optional, Skip median computation, Default to False
        velunits: Bool, Optional, Compute P1D in velocity units, Default to False  
        addweights: Bool, Optional, Compute mean P1D with weights, Default to False
    """
    
    meanP1D_table = Table()
    data_array_cols = data_array.colnames
    
    N_chunks, zbin_chunks, izbin_chunks = binned_statistic(z_array, z_array, statistic='count', bins=zbin_edges)
    
    stats_array = ['mean','error','min','max']
    if nomedians==True:
        stats_array+=['median']
    
    for izbin,zbin in enumerate(zbin_edges[:-1]):
        table_data = Table()
        N_array = np.empty(0)
        
        if N_chunks[izbin]==0: 
            table_data['N'] = np.zeros((1,len(kbin_edges)-1))
            for c in data_array_cols:
                for stats in stats_array:
                    table_data[stats+c] = np.ones((1, len(kbin_edges) - 1)) * np.nan
            continue
        
        table_data['N_chunks']=np.array([N_chunks[izbin]],dtype=int) # number of chunks in each redshift bin
        for ic, c in enumerate(data_array_cols):  # initialize table
            index = len(stats_array)*ic
            table_data['mean'+c] = np.zeros((1,len(kbin_edges)-1))
            table_data['error'+c] = np.zeros((1,len(kbin_edges)-1))
            table_data['min'+c] = np.zeros((1,len(kbin_edges)-1))
            table_data['max'+c] = np.zeros((1,len(kbin_edges)-1))
            if nomedians==True:
                table_data['median'+c] = np.zeros((1,len(kbin_edges)-1))

        for ikbin, kbin in enumerate(kbin_edges[:-1]):
            select=(data_array['forest_z'][:] < zbin_edges[izbin + 1])&(data_array['forest_z'][:] > zbin_edges[izbin])&(data_array['k'][:] < kbin_edges[ikbin + 1])&(data_array['k'][:] > kbin_edges[ikbin]) # select a specific zbin and kbin

            if velunits==True: # Convert data into velocity units
                conversion_factor = (1215.67 * (1 + np.mean(data_array['forest_z'][select]))) / 3e5
                data_array['k'][select]*=conversion_factor
                for c in data_array_cols:
                    if 'Pk' in c:
                        data_array[c][select]/=conversion_factor

            N = np.ma.count(data_array['k'][select]) # Counts the number of chunks in each kbin
            N_array = np.append(N_array, N)  

            for ic, c in enumerate(data_array_cols): 
                
                if addweights==True:
                    weights = (data_array['forest_snr'][select])**2
                    mean = np.average((data_array[c][select]), weights=weights)
                    variance = np.var((data_array[c][select]))
                    weights_coeff = np.sqrt((N_array[ikbin] * variance)/(np.sum(1/weights)))
                    error = weights_coeff * np.sqrt(1/np.sum(weights))
                else:
                    mean = np.average((data_array[c][select])) 
                    error = np.std((data_array[c][select])) / np.sqrt(N_array[ikbin]-1)  # unbiased estimate: N-1 

                minimum = np.min((data_array[c][select]))
                maximum = np.max((data_array[c][select]))
                table_data['mean'+c][0,ikbin] = mean
                table_data['error'+c][0,ikbin] = error
                table_data['min'+c][0,ikbin] = minimum
                table_data['max'+c][0,ikbin] = maximum
                if nomedians==True:
                    median = np.median((data_array[c][select]))
                    table_data['median'+c][0,ikbin] = median
                    median = 0
                mean, variance, weights_coeff, error, minimum, maximum = 0, 0, 0, 0, 0, 0

        table_data['N'] = N_array[np.newaxis,:] 

        meanP1D_table=vstack([meanP1D_table,table_data])
                            
    return meanP1D_table


def parallelize_p1d_comp(data_dir, zbin_edges, kbin_edges, snr_cut_mean=None, zbins=None, nomedians=False, 
                         velunits=False, addweights=False, overwrite=False):
    """Read individual Pk1D data from different files and compute the mean P1D
    
    Args:
        data_dir: Directory where the individual P1D fits files are saved
        overwrite: Bool, Optional, Overwrite files if existing, Defaults to False.
        other args: As defined in previous functions
    """
    
    outfilename=os.path.join(data_dir,f'mean_Pk1d{"_vel" if velunits else ""}.fits.gz')
    if os.path.exists(outfilename) and not overwrite:
        outdir=Table.read(outfilename)
        return outdir
     
    searchstr = '*'
    files = glob.glob(os.path.join(data_dir,f"Pk1D{searchstr}.fits.gz"))
    ncpu = 8
    from multiprocessing import Pool
    with Pool(ncpu) as pool:
        if snr_cut_mean is not None:
            full_data_array = pool.starmap(read_pk1d,[[f, kbin_edges, snr_cut_mean, zbins] for f in files])
        else:
            full_data_array = pool.starmap(read_pk1d,[[f, kbin_edges] for f in files])
    
    data_array = vstack([full_data_array[i][0] for i in range(len(full_data_array))])  
    z_array = np.concatenate(tuple([full_data_array[i][1] for i in range(len(full_data_array))]))

    full_meanP1D_table = compute_mean_pk1d(data_array, z_array, zbin_edges, kbin_edges, nomedians, velunits, addweights)
    
    outdir = full_meanP1D_table
    outdir.meta['velunits']=velunits
    outdir.write(outfilename,overwrite=overwrite)
    return outdir