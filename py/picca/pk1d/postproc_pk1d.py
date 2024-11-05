"""Module defining a set of functions to postprocess files produced by compute_pk1d.py.

This module provides 3 main functions:
    - read_pk1d:
        Read all HDUs in an individual "P1D" FITS file and stacks
        all data in one table
    - compute_mean_pk1d:
        Compute the mean P1D in a given (z,k) grid of bins, from individual
        "P1Ds" of individual chunks
    - parallelize_p1d_comp:
        Main function, runs read_pk1d in parallel, then runs compute_mean_pk1d
See the respective documentation for details
"""

import glob
import os
from functools import partial
from multiprocessing import Pool

import fitsio
import numpy as np
from astropy.stats import bootstrap
from astropy.table import Table, vstack
from scipy.optimize import curve_fit
from scipy.stats import binned_statistic
from picca.constants import ABSORBER_IGM, SPEED_LIGHT
from picca.pk1d.utils import MEANPK_FITRANGE_SNR, fitfunc_variance_pk1d
from picca.utils import userprint


def read_pk1d(filename, kbin_edges, snrcut=None, zbins_snrcut=None, skymask_matrices=None):
    """Read Pk1D data from a single file.

    Arguments
    ---------
    filename: string
    Fits file, containing individual "P1D" for each chunk

    kbin_edges: array of floats
    Edges of the wavenumber bins to be later used, in Angstrom^-1

    snrcut: array of floats or None
    Chunks with mean SNR > snrcut are discarded. If len(snrcut)>1,
    zbins_snrcut must be set, so that the cut is made redshift dependent.

    zbins_snrcut: array of floats or None
    Required if len(snrcut)>1. List of redshifts
    associated to the list of snr cuts.

    skymask_matrices: list or None
    if not None, computes a correction associated to skyline masking, for the Pk_raw
    and Pk_noise vectors. This correction makes use of a matrix operation.
    The corrected powers are stored into "Pk_raw_skycorr" and "Pk_noise_skycorr".
    Note that this correction makes sense only after the Pks have been averaged over sightlines.

    Return
    ------
    p1d_table: Table
    One entry per mode(k) per chunk

    z_array: array of floats
    Nchunks entry.
    If no chunk is selected, None will be returned instead
    """
    p1d_table = []
    z_array = []
    if skymask_matrices is not None:
        meanz_skymatrices = np.array([x[0] for x in skymask_matrices])
    with fitsio.FITS(filename, "r") as hdus:
        for i, hdu in enumerate(hdus[1:]):
            data = hdu.read()
            chunk_header = hdu.read_header()
            chunk_table = Table(data)
            for colname in [
                "k",
                "Pk",
                "Pk_raw",
                "Pk_noise",
                "Pk_diff",
                "cor_reso",
                "Pk_noise_miss",
                "Delta_k",
                "Delta_noise_k",
                "Delta_diff_k",
            ]:
                try:
                    chunk_table.rename_column(colname.upper(), colname)
                except KeyError:
                    pass

            if np.nansum(chunk_table["Pk"]) == 0:
                chunk_table["Pk"] = (
                    chunk_table["Pk_raw"] - chunk_table["Pk_noise"]
                ) / chunk_table["cor_reso"]

            if skymask_matrices is not None:
                chunk_table["Pk_raw_skycorr"] = chunk_table["Pk_raw"]
                chunk_table["Pk_noise_skycorr"] = chunk_table["Pk_noise"]
                w, = np.where(np.isclose(meanz_skymatrices, chunk_header["MEANZ"], atol=1.e-2))
                if len(w)==1:
                    correction_matrix = skymask_matrices[w[0]][1]
                    ll = len(chunk_table["Pk_raw"])
                    correction_matrix = np.copy(correction_matrix[0:ll, 0:ll])
                    if len(chunk_table["Pk_raw"])!=correction_matrix.shape[0]:
                        userprint(f"""file {filename} hdu {i+1}:"""
                                  """Pk_raw doesnt match shape of skymatrix.""")
                    else:
                        chunk_table["Pk_raw_skycorr"] = correction_matrix @ chunk_table["Pk_raw"]
                        chunk_table["Pk_noise_skycorr"] = (
                            correction_matrix @ chunk_table["Pk_noise"]
                        )
                chunk_table["Pk_skycorr"] = (
                    chunk_table["Pk_raw_skycorr"] - chunk_table["Pk_noise_skycorr"]
                ) / chunk_table["cor_reso"]

            chunk_table["forest_z"] = float(chunk_header["MEANZ"])
            chunk_table["forest_snr"] = float(chunk_header["MEANSNR"])
            chunk_table["forest_id"] = int(chunk_header["LOS_ID"])
            if "CHUNK_ID" in chunk_header:
                chunk_table["sub_forest_id"] = (
                    f"{chunk_header['LOS_ID']}_{chunk_header['CHUNK_ID']}"
                )

            if snrcut is not None:
                if len(snrcut) > 1:
                    if (zbins_snrcut is None) or (len(zbins_snrcut) != len(snrcut)):
                        raise ValueError(
                            "Please provide same size for zbins_snrcut and snrcut arrays"
                        )
                    zbin_index = np.argmin(np.abs(zbins_snrcut - chunk_header["MEANZ"]))
                    snrcut_chunk = snrcut[zbin_index]
                else:
                    snrcut_chunk = snrcut[0]
                if chunk_header["MEANSNR"] < snrcut_chunk:
                    continue

            # Empirically remove very noisy chunks
            (wk,) = np.where(chunk_table["k"] < kbin_edges[-1])
            if (
                np.abs(chunk_table["Pk_noise"][wk])
                > 1000000 * np.abs(chunk_table["Pk_raw"][wk])
            ).any():
                userprint(
                    f"file {filename} hdu {i+1} has very high noise power: discarded"
                )
                continue

            p1d_table.append(chunk_table)
            z_array.append(float(chunk_header["MEANZ"]))

    if len(p1d_table) == 0:  # No chunk was selected
        return None

    p1d_table = vstack(p1d_table)
    p1d_table["Delta2"] = p1d_table["k"] * p1d_table["Pk"] / np.pi
    p1d_table["Pk_norescor"] = p1d_table["Pk_raw"] - p1d_table["Pk_noise"]
    p1d_table["Pk_nonoise"] = p1d_table["Pk_raw"] / p1d_table["cor_reso"]
    p1d_table["Pk_noraw"] = p1d_table["Pk_noise"] / p1d_table["cor_reso"]
    if skymask_matrices is not None:
        p1d_table["Delta2_skycorr"] = p1d_table["k"] * p1d_table["Pk_skycorr"] / np.pi
    try:
        p1d_table["Pk_noraw_miss"] = p1d_table["Pk_noise_miss"] / p1d_table["cor_reso"]
    except KeyError:
        pass

    z_array = np.array(z_array)

    return p1d_table, z_array


def mean_p1d_table_regular_slice(izbin, nbins_k):
    """Return the arguments of a slice of mean P1D table for a given redshift.

    Arguments
    ---------
    izbin (int):
    Current redshift bin being considered.

    nbins_k (int):
    Number of k bins.

    Return
    ------
    Arguments of a slice of mean P1D table.
    """
    return izbin * nbins_k, (izbin + 1) * nbins_k


def cov_table_regular_slice(izbin, nbins_k):
    """Return the arguments of a slice of covariance table for a given redshift.

    Arguments
    ---------
    izbin (int):
    Current redshift bin being considered.

    nbins_k (int):
    Number of k bins.

    Return
    ------
    Arguments of a slice of covariance table.
    """
    return izbin * nbins_k * nbins_k, (izbin + 1) * nbins_k * nbins_k


def compute_mean_pk1d(
    p1d_table,
    z_array,
    zbin_edges,
    kbin_edges,
    weight_method,
    apply_z_weights=False,
    nomedians=False,
    velunits=False,
    output_snrfit=None,
    compute_covariance=False,
    compute_bootstrap=False,
    number_bootstrap=50,
    number_worker=8,
):
    """Compute mean P1D in a set of given (z,k) bins, from individual chunks P1Ds.

    Arguments
    ---------
    p1d_table: astropy.table.Table
    Individual Pk1Ds of the contributing forest chunks, stacked in one table using "read_pk1d",
    Contains 'k', 'Pk_raw', 'Pk_noise', 'Pk_diff', 'cor_reso', 'Pk', 'forest_z', 'forest_snr',
            'Delta2', 'Pk_norescor', 'Pk_nonoise', 'Pk_noraw'

    z_array: Array of float
    Mean z of each contributing chunk, stacked in one array using "read_pk1d"

    zbin_edges: Array of float
    Edges of the redshift bins we want to use

    kbin_edges: Array of float
    Edges of the wavenumber bins we want to use, either in (Angstrom)-1 or s/km

    weight_method: string
    2 possible options:
        'fit_snr': Compute mean P1D with weights estimated by fitting dispersion vs SNR
        'no_weights': Compute mean P1D without weights

    apply_z_weights: Bool
    If True, each chunk contributes to two nearest redshift bins with a linear weighting scheme.

    nomedians: Bool
    Skip computation of median quantities

    velunits: Bool
    Compute P1D in velocity units by converting k on-the-fly from AA-1 to s/km

    output_snrfit: string
    If weight_method='fit_snr', the results of the fit can be saved to an ASCII file.
    The file contains (z k a b standard_dev_points) for the "Pk" variable, for each (z,k) point

    compute_covariance: Bool
    If True, compute statistical covariance matrix of the mean P1D.
    Requires  p1d_table to contain 'sub_forest_id', since correlations are computed
    within individual forests.

    compute_bootstrap: Bool
    If True, compute statistical covariance using a simple bootstrap method.

    number_bootstrap: int
    Number of bootstrap samples used if compute_bootstrap is True.

    number_worker: int
    Calculations of mean P1Ds and covariances are run parallel over redshift bins.

    Return
    ------
    meanP1d_table: astropy.table.Table
    One row per (z,k) bin; one column per statistics (eg. meanPk, errorPk_noise...)
    Other columns: 'N' (nb of chunks used), 'index_zbin' (index of associated
    row in metadata_table), 'zbin'

    metadata_table: astropy.table.Table
    One row per z bin; column values z_min/max, k_min/max, N_chunks
    """
    # Initializing stats we want to compute on data
    stats_array = ["mean", "error", "min", "max"]
    if not nomedians:
        stats_array += ["median"]

    p1d_table_cols = p1d_table.colnames
    p1d_table_cols.remove("forest_id")
    if "sub_forest_id" in p1d_table_cols:
        p1d_table_cols.remove("sub_forest_id")

    p1d_table_cols = [col for col in p1d_table_cols if "Delta_" not in col]

    # Convert data into velocity units
    if velunits:
        conversion_factor = (
            ABSORBER_IGM["LYA"] * (1.0 + p1d_table["forest_z"])
        ) / SPEED_LIGHT
        p1d_table["k"] *= conversion_factor
        for col in p1d_table_cols:
            if "Pk" in col:
                p1d_table[col] /= conversion_factor

    # Initialize mean_p1d_table of len = (nzbins * nkbins) corresponding to hdu[1] in final ouput
    mean_p1d_table = Table()
    nbins_z, nbins_k = len(zbin_edges) - 1, len(kbin_edges) - 1
    mean_p1d_table["zbin"] = np.zeros(nbins_z * nbins_k)
    mean_p1d_table["index_zbin"] = np.zeros(nbins_z * nbins_k, dtype=int)
    mean_p1d_table["N"] = np.zeros(nbins_z * nbins_k, dtype=int)
    for col in p1d_table_cols:
        for stats in stats_array:
            mean_p1d_table[stats + col] = np.ones(nbins_z * nbins_k) * np.nan

    # Initialize metadata_table of len = nbins_z corresponding to hdu[2] in final output
    metadata_table = Table()
    metadata_table["z_min"] = zbin_edges[:-1]
    metadata_table["z_max"] = zbin_edges[1:]
    metadata_table["k_min"] = kbin_edges[0] * np.ones(nbins_z)
    metadata_table["k_max"] = kbin_edges[-1] * np.ones(nbins_z)

    if compute_covariance or compute_bootstrap:
        if "sub_forest_id" not in p1d_table.columns:
            userprint(
                """sub_forest_id cannot be computed from individual pk files,
                necessary to compute covariance. Skipping calculation"""
            )
            compute_covariance, compute_bootstrap = False, False
            cov_table = None

        elif apply_z_weights:
            userprint(
                """Covariance calculations are not compatible redshift weighting yes.
                Skipping calculation"""
            )
            compute_covariance, compute_bootstrap = False, False
            cov_table = None

        else:
            # Initialize cov_table of len = (nzbins * nkbins * nkbins)
            # corresponding to hdu[3] in final ouput
            cov_table = Table()
            cov_table["zbin"] = np.zeros(nbins_z * nbins_k * nbins_k)
            cov_table["index_zbin"] = np.zeros(nbins_z * nbins_k * nbins_k, dtype=int)
            cov_table["N"] = np.zeros(nbins_z * nbins_k * nbins_k, dtype=int)
            cov_table["covariance"] = np.zeros(nbins_z * nbins_k * nbins_k)
            cov_table["k1"] = np.zeros(nbins_z * nbins_k * nbins_k)
            cov_table["k2"] = np.zeros(nbins_z * nbins_k * nbins_k)

            if compute_bootstrap:
                cov_table["boot_covariance"] = np.zeros(nbins_z * nbins_k * nbins_k)
                cov_table["error_boot_covariance"] = np.zeros(
                    nbins_z * nbins_k * nbins_k
                )

            k_index = np.full(len(p1d_table["k"]), -1, dtype=int)
            for ikbin, _ in enumerate(kbin_edges[:-1]):  # First loop 1) k bins
                select = (p1d_table["k"] < kbin_edges[ikbin + 1]) & (
                    p1d_table["k"] > kbin_edges[ikbin]
                )  # select a specific k bin
                k_index[select] = ikbin
    else:
        cov_table = None

    # Number of chunks in each redshift bin
    n_chunks, _, _ = binned_statistic(
        z_array, z_array, statistic="count", bins=zbin_edges
    )
    metadata_table["N_chunks"] = n_chunks

    zbin_centers = np.around((zbin_edges[1:] + zbin_edges[:-1]) / 2, 5)
    if weight_method == "fit_snr":
        snrfit_table = np.zeros(
            (nbins_z * nbins_k, 13)
        )  # 13 entries: z k a b + 9 SNR bins used for the fit
    else:
        snrfit_table = None

    userprint("Computing average p1d")
    # Main loop 1) z bins
    params_pool = [[izbin] for izbin, _ in enumerate(zbin_edges[:-1])]

    func = partial(
        compute_average_pk_redshift,
        p1d_table,
        p1d_table_cols,
        weight_method,
        apply_z_weights,
        nomedians,
        nbins_z,
        zbin_centers,
        zbin_edges,
        n_chunks,
        nbins_k,
        kbin_edges,
    )
    if number_worker == 1:
        output_mean = [func(p[0]) for p in params_pool]
    else:
        with Pool(number_worker) as pool:
            output_mean = pool.starmap(func, params_pool)

    fill_average_pk(
        nbins_z,
        nbins_k,
        output_mean,
        mean_p1d_table,
        p1d_table_cols,
        weight_method,
        snrfit_table,
        nomedians,
    )

    if compute_covariance or compute_bootstrap:
        userprint("Computation of p1d groups for covariance matrix calculation")

        p1d_groups = []
        for izbin in range(nbins_z):

            zbin_array = np.full(nbins_k * nbins_k, zbin_centers[izbin])
            index_zbin_array = np.full(nbins_k * nbins_k, izbin, dtype=int)
            kbin_centers = (kbin_edges[1:] + kbin_edges[:-1]) / 2
            k1_array, k2_array = np.meshgrid(kbin_centers, kbin_centers, indexing="ij")
            k1_array = np.ravel(k1_array)
            k2_array = np.ravel(k2_array)
            select = mean_p1d_table["index_zbin"] == izbin
            n_array = np.ravel(
                np.outer(mean_p1d_table["N"][select], mean_p1d_table["N"][select])
            )
            index_cov = cov_table_regular_slice(izbin, nbins_k)
            cov_table["zbin"][index_cov[0]:index_cov[1]] = zbin_array
            cov_table["index_zbin"][index_cov[0]:index_cov[1]] = index_zbin_array
            cov_table["k1"][index_cov[0]:index_cov[1]] = k1_array
            cov_table["k2"][index_cov[0]:index_cov[1]] = k2_array
            cov_table["N"][index_cov[0]:index_cov[1]] = n_array
            index_mean = mean_p1d_table_regular_slice(izbin, nbins_k)
            mean_pk = mean_p1d_table["meanPk"][index_mean[0] : index_mean[1]]
            error_pk = mean_p1d_table["errorPk"][index_mean[0] : index_mean[1]]

            if n_chunks[izbin] == 0:
                p1d_weights_z, covariance_weights_z, p1d_groups_z = [], [], []
            else:
                p1d_weights_z, covariance_weights_z, p1d_groups_z = compute_p1d_groups(
                    weight_method,
                    nbins_k,
                    zbin_edges,
                    izbin,
                    p1d_table,
                    k_index,
                    snrfit_table,
                    number_worker,
                )

            p1d_groups.append(
                [
                    mean_pk,
                    error_pk,
                    p1d_weights_z,
                    covariance_weights_z,
                    p1d_groups_z,
                ]
            )

        compute_and_fill_covariance(
            compute_covariance,
            compute_bootstrap,
            weight_method,
            nbins_k,
            nbins_z,
            p1d_groups,
            number_worker,
            number_bootstrap,
            cov_table,
        )

    if output_snrfit is not None:
        np.savetxt(
            output_snrfit,
            snrfit_table,
            fmt="%.5e",
            header="Result of fit: Variance(Pks) vs SNR\n"
            "SNR bin edges used: 1,  2,  3,  4,  5,  6,  7,  8,  9, 10\n"
            "z k a b standard_dev_points",
        )

    return mean_p1d_table, metadata_table, cov_table


def fill_average_pk(
    nbins_z,
    nbins_k,
    output_mean,
    mean_p1d_table,
    p1d_table_cols,
    weight_method,
    snrfit_table,
    nomedians,
):
    """Fill the average P1D table for all redshift and k bins.

    Arguments
    ---------
    nbins_z: int,
    Number of redshift bins.

    nbins_k: int,
    Number of k bins.

    output_mean: tuple of numpy ndarray,
    Result of the mean calculation

    mean_p1d_table: numpy ndarray,
    Table to fill.

    p1d_table_cols: List of str,
    Column names in the input table to be averaged.

    weight_method: string
    2 possible options:
        'fit_snr': Compute mean P1D with weights estimated by fitting dispersion vs SNR
        'no_weights': Compute mean P1D without weights

    snrfit_table: numpy ndarray,
    Table containing SNR fit infos

    nomedians: bool,
    If True, do not use median values in the fit to the SNR.

    Return
    ------
    None

    """
    for izbin in range(nbins_z):  # Main loop 1) z bins
        (
            zbin_array,
            index_zbin_array,
            n_array,
            mean_array,
            error_array,
            min_array,
            max_array,
            median_array,
            snrfit_array,
        ) = (*output_mean[izbin],)
        index_mean = mean_p1d_table_regular_slice(izbin,nbins_k)

        mean_p1d_table["zbin"][index_mean[0]:index_mean[1]] = zbin_array
        mean_p1d_table["index_zbin"][index_mean[0]:index_mean[1]] = index_zbin_array
        mean_p1d_table["N"][index_mean[0]:index_mean[1]] = n_array
        for icol, col in enumerate(p1d_table_cols):
            mean_p1d_table["mean" + col][index_mean[0]:index_mean[1]] = mean_array[icol]
            mean_p1d_table["error" + col][index_mean[0]:index_mean[1]] = error_array[icol]
            mean_p1d_table["min" + col][index_mean[0]:index_mean[1]] = min_array[icol]
            mean_p1d_table["max" + col][index_mean[0]:index_mean[1]] = max_array[icol]
            if not nomedians:
                mean_p1d_table["median" + col][index_mean[0]:index_mean[1]] = median_array[icol]
        if weight_method == "fit_snr":
            snrfit_table[index_mean[0]:index_mean[1], :] = snrfit_array


def compute_average_pk_redshift(
    p1d_table,
    p1d_table_cols,
    weight_method,
    apply_z_weights,
    nomedians,
    nbins_z,
    zbin_centers,
    zbin_edges,
    n_chunks,
    nbins_k,
    kbin_edges,
    izbin,
):
    """Compute the average P1D table for the given redshift and k bins.

    The function computes the mean P1D table for each redshift and k bin.
    If there are no chunks in a given bin, the rows in the
    table for that bin will be filled with NaNs.
    The mean value for each bin is calculated using a weighting method,
    either a fit to the SNR or using weights based on the redshift.

    Arguments
    ---------
    p1d_table: numpy ndarray,
    Table containing the data to be averaged.

    p1d_table_cols: List of str,
    Column names in the input table to be averaged.

    weight_method: str,
    Method to weight the data.

    apply_z_weights: bool,
    If True, apply redshift weights.

    nomedians: bool,
    If True, do not use median values in the fit to the SNR.

    nbins_z: int,
    Number of redshift bins.

    zbin_centers: numpy ndarray,
    Centers of the redshift bins.

    zbin_edges: numpy ndarray,
    Edges of the redshift bins.

    n_chunks: numpy ndarray,
    Number of chunks in each redshift bin.

    nbins_k: int,
    Number of k bins.

    kbin_edges: numpy ndarray,
    Edges of the k bins.

    izbin: int,
    Index of the current redshift bin.

    Return
    ------
    zbin_array
    index_zbin_array
    n_array
    mean_array
    error_array
    min_array
    max_array
    median_array
    snrfit_array
    """
    n_array = np.zeros(nbins_k, dtype=int)
    zbin_array = np.zeros(nbins_k)
    index_zbin_array = np.zeros(nbins_k, dtype=int)
    mean_array = []
    error_array = []
    min_array = []
    max_array = []
    if not nomedians:
        median_array = []
    else:
        median_array = None
    if weight_method == "fit_snr":
        snrfit_array = np.zeros((nbins_k, 13))
    else:
        snrfit_array = None

    for col in p1d_table_cols:
        mean_array.append(np.zeros(nbins_k))
        error_array.append(np.zeros(nbins_k))
        min_array.append(np.zeros(nbins_k))
        max_array.append(np.zeros(nbins_k))
        if not nomedians:
            median_array.append(np.zeros(nbins_k))

    if n_chunks[izbin] == 0:  # Fill rows with NaNs
        zbin_array[:] = zbin_centers[izbin]
        index_zbin_array[:] = izbin
        n_array[:] = 0
        for icol, col in enumerate(p1d_table_cols):
            mean_array[icol][:] = np.nan
            error_array[icol][:] = np.nan
            min_array[icol][:] = np.nan
            max_array[icol][:] = np.nan
            if not nomedians:
                median_array[icol][:] = np.nan
        if weight_method == "fit_snr":
            snrfit_array[:] = np.nan

        return (
            zbin_array,
            index_zbin_array,
            n_array,
            mean_array,
            error_array,
            min_array,
            max_array,
            median_array,
            snrfit_array,
        )

    for ikbin, kbin in enumerate(kbin_edges[:-1]):  # Main loop 2) k bins
        if apply_z_weights:  # special chunk selection in that case
            delta_z = zbin_centers[1:] - zbin_centers[:-1]
            if not np.allclose(delta_z, delta_z[0], atol=1.0e-3):
                raise ValueError(
                    "z bins should have equal widths with apply_z_weights."
                )
            delta_z = delta_z[0]

            select = (p1d_table["k"] < kbin_edges[ikbin + 1]) & (
                p1d_table["k"] > kbin_edges[ikbin]
            )
            if izbin in (0, nbins_z - 1):
                # First and last bin: in order to avoid edge effects,
                #    use only chunks within the bin
                select = (
                    select
                    & (p1d_table["forest_z"] > zbin_edges[izbin])
                    & (p1d_table["forest_z"] < zbin_edges[izbin + 1])
                )
            else:
                select = (
                    select
                    & (p1d_table["forest_z"] < zbin_centers[izbin + 1])
                    & (p1d_table["forest_z"] > zbin_centers[izbin - 1])
                )

            redshift_weights = (
                1.0
                - np.abs(p1d_table["forest_z"][select] - zbin_centers[izbin]) / delta_z
            )

        else:
            select = (
                (p1d_table["forest_z"] < zbin_edges[izbin + 1])
                & (p1d_table["forest_z"] > zbin_edges[izbin])
                & (p1d_table["k"] < kbin_edges[ikbin + 1])
                & (p1d_table["k"] > kbin_edges[ikbin])
            )  # select a specific (z,k) bin

        zbin_array[ikbin] = zbin_centers[izbin]
        index_zbin_array[ikbin] = izbin

        # Counts the number of chunks in each (z,k) bin
        num_chunks = np.ma.count(p1d_table["k"][select])

        n_array[ikbin] = num_chunks

        if weight_method == "fit_snr":
            if num_chunks == 0:
                userprint(
                    "Warning: 0 chunks found in bin "
                    + str(zbin_edges[izbin])
                    + "<z<"
                    + str(zbin_edges[izbin + 1])
                    + ", "
                    + str(kbin_edges[ikbin])
                    + "<k<"
                    + str(kbin_edges[ikbin + 1])
                )
                continue
            snr_bin_edges = np.arange(
                MEANPK_FITRANGE_SNR[0], MEANPK_FITRANGE_SNR[1] + 1, 1
            )
            snr_bins = (snr_bin_edges[:-1] + snr_bin_edges[1:]) / 2

            p1d_values = p1d_table["Pk"][select]
            data_snr = p1d_table["forest_snr"][select]
            mask_nan_p1d_values = (~np.isnan(p1d_values)) & (~np.isnan(data_snr))
            data_snr, p1d_values = (
                data_snr[mask_nan_p1d_values],
                p1d_values[mask_nan_p1d_values],
            )
            if len(p1d_values) == 0:
                continue
            standard_dev, _, _ = binned_statistic(
                data_snr, p1d_values, statistic="std", bins=snr_bin_edges
            )
            standard_dev_full = np.copy(standard_dev)
            standard_dev, snr_bins = (
                standard_dev[~np.isnan(standard_dev)],
                snr_bins[~np.isnan(standard_dev)],
            )
            if len(standard_dev) == 0:
                continue
            coef, *_ = curve_fit(
                fitfunc_variance_pk1d,
                snr_bins,
                standard_dev**2,
                bounds=(0, np.inf),
            )
            data_snr[data_snr > MEANPK_FITRANGE_SNR[1]] = MEANPK_FITRANGE_SNR[1]
            data_snr[data_snr < 1.01] = 1.01
            variance_estimated = fitfunc_variance_pk1d(data_snr, *coef)
            weights = 1.0 / variance_estimated

        for icol, col in enumerate(p1d_table_cols):
            if num_chunks == 0:
                userprint(
                    "Warning: 0 chunks found in bin "
                    + str(zbin_edges[izbin])
                    + "<z<"
                    + str(zbin_edges[izbin + 1])
                    + ", "
                    + str(kbin_edges[ikbin])
                    + "<k<"
                    + str(kbin_edges[ikbin + 1])
                )
                continue

            if weight_method == "fit_snr":
                snr_bin_edges = np.arange(
                    MEANPK_FITRANGE_SNR[0], MEANPK_FITRANGE_SNR[1] + 1, 1
                )
                snr_bins = (snr_bin_edges[:-1] + snr_bin_edges[1:]) / 2

                data_values = p1d_table[col][select]
                data_snr = p1d_table["forest_snr"][select]
                data_snr, data_values = (
                    data_snr[mask_nan_p1d_values],
                    data_values[mask_nan_p1d_values],
                )
                # Fit function to observed dispersion in col:
                standard_dev_col, _, _ = binned_statistic(
                    data_snr, data_values, statistic="std", bins=snr_bin_edges
                )
                standard_dev_col, snr_bins = (
                    standard_dev_col[~np.isnan(standard_dev_col)],
                    snr_bins[~np.isnan(standard_dev_col)],
                )

                coef_col, *_ = curve_fit(
                    fitfunc_variance_pk1d,
                    snr_bins,
                    standard_dev_col**2,
                    bounds=(0, np.inf),
                )

                # Model variance from fit function
                data_snr[data_snr > MEANPK_FITRANGE_SNR[1]] = MEANPK_FITRANGE_SNR[1]
                data_snr[data_snr < 1.01] = 1.01
                variance_estimated_col = fitfunc_variance_pk1d(data_snr, *coef_col)
                weights_col = 1.0 / variance_estimated_col
                if apply_z_weights:
                    mean = np.average(data_values, weights=weights * redshift_weights)
                else:
                    mean = np.average(data_values, weights=weights)
                if apply_z_weights:
                    # Analytic expression for the re-weighted average:
                    error = np.sqrt(np.sum(weights_col * redshift_weights)) / np.sum(
                        weights_col
                    )
                else:
                    error = np.sqrt(1.0 / np.sum(weights_col))
                    # Variance estimator derived by Jean-Marc, we keep the estimated one.
                    # error = np.sqrt(((np.sum(weights)**2 / np.sum(weights**2)) - 1 )**(-1) * (
                    # ( np.sum(weights**2 * data_values**2) / np.sum(weights**2) ) - (
                    # np.sum(weights * data_values)/ np.sum(weights) )**2 ))
                if col == "Pk":
                    standard_dev = np.concatenate(
                        [
                            standard_dev,
                            np.full(len(standard_dev[np.isnan(standard_dev)]), np.nan),
                        ]
                    )
                    snrfit_array[ikbin, 0:4] = [
                        zbin_centers[izbin],
                        (kbin + kbin_edges[ikbin + 1]) / 2.0,
                        coef[0],
                        coef[1],
                    ]
                    snrfit_array[ikbin, 4:] = standard_dev_full  # also save nan values

            elif weight_method == "no_weights":
                if apply_z_weights:
                    mean = np.average(p1d_table[col][select], weights=redshift_weights)
                    # simple analytic expression:
                    error = np.std(p1d_table[col][select]) * (
                        np.sqrt(np.sum(redshift_weights**2)) / np.sum(redshift_weights)
                    )
                else:
                    mean = np.mean(p1d_table[col][select])
                    # unbiased estimate: num_chunks-1
                    error = np.std(p1d_table[col][select]) / np.sqrt(num_chunks - 1)

            else:
                raise ValueError("Option for 'weight_method' argument not found")

            minimum = np.min((p1d_table[col][select]))
            maximum = np.max((p1d_table[col][select]))
            mean_array[icol][ikbin] = mean
            error_array[icol][ikbin] = error
            min_array[icol][ikbin] = minimum
            max_array[icol][ikbin] = maximum
            if not nomedians:
                median = np.median((p1d_table[col][select]))
                median_array[icol][ikbin] = median
    return (
        zbin_array,
        index_zbin_array,
        n_array,
        mean_array,
        error_array,
        min_array,
        max_array,
        median_array,
        snrfit_array,
    )


def compute_and_fill_covariance(
    compute_covariance,
    compute_bootstrap,
    weight_method,
    nbins_k,
    nbins_z,
    p1d_groups,
    number_worker,
    number_bootstrap,
    cov_table,
):
    """Compute the covariance and bootstrap covariance and fill the corresponding
    cov_table variable

    compute_covariance: Bool
    If True, compute statistical covariance matrix of the mean P1D.
    Requires  p1d_table to contain 'sub_forest_id', since correlations are computed
    within individual forests.

    compute_bootstrap: Bool
    If True, compute statistical covariance using a simple bootstrap method.

    weight_method: str,
    Method to weight the data.

    nbins_k (int):
    Number of k bins.

    nbins_z (int):
    Number of z bins.

    p1d_groups (array-like):
    Individual p1d pixels grouped in the same wavenumber binning for all subforest

    number_worker: int
    Calculations of mean P1Ds and covariances are run parallel over redshift bins.

    number_bootstrap: int
    Number of bootstrap samples used if compute_bootstrap is True.

    cov_table (array-like):
    Covariance table to fill.

    Return
    ------
    None
    """

    if compute_covariance:
        userprint("Computation of covariance matrix")

        func = partial(
            compute_cov,
            weight_method,
            nbins_k,
        )
        if number_worker == 1:
            output_cov = [func(*p1d_group) for p1d_group in p1d_groups]
        else:
            with Pool(number_worker) as pool:
                output_cov = pool.starmap(func, p1d_groups)

        for izbin in range(nbins_z):
            covariance_array = output_cov[izbin]
            index_cov = cov_table_regular_slice(izbin, nbins_k)
            cov_table["covariance"][index_cov[0]:index_cov[1]] = covariance_array

    if compute_bootstrap:
        userprint("Computing covariance matrix with bootstrap method")
        p1d_groups_bootstrap = []
        for izbin in range(nbins_z):
            number_sub_forests = len(p1d_groups[izbin][2])
            if number_sub_forests > 0:
                bootid = np.array(
                    bootstrap(np.arange(number_sub_forests), number_bootstrap)
                ).astype(int)
            else:
                bootid = np.full(number_bootstrap, None)

            for iboot in range(number_bootstrap):
                if bootid[iboot] is None:
                    (
                        mean_pk,
                        error_pk,
                        p1d_weights_z,
                        covariance_weights_z,
                        p1d_groups_z,
                    ) = ([], [], [], [], [])
                else:
                    mean_pk = p1d_groups[izbin][0]
                    error_pk = p1d_groups[izbin][1]
                    p1d_weights_z = p1d_groups[izbin][2][bootid[iboot]]
                    covariance_weights_z = p1d_groups[izbin][3][bootid[iboot]]
                    p1d_groups_z = p1d_groups[izbin][4][bootid[iboot]]
                p1d_groups_bootstrap.append(
                    [
                        mean_pk,
                        error_pk,
                        p1d_weights_z,
                        covariance_weights_z,
                        p1d_groups_z,
                    ]
                )

        func = partial(
            compute_cov,
            weight_method,
            nbins_k,
        )
        if number_worker == 1:
            output_cov = [func(*p) for p in p1d_groups_bootstrap]
        else:
            with Pool(number_worker) as pool:
                output_cov = pool.starmap(func, p1d_groups_bootstrap)

        for izbin in range(nbins_z):
            boot_cov = []
            for iboot in range(number_bootstrap):
                covariance_array = (output_cov[izbin * number_bootstrap + iboot],)
                boot_cov.append(covariance_array)

            index_cov = cov_table_regular_slice(izbin, nbins_k)
            cov_table["boot_covariance"][index_cov[0]:index_cov[1]] = np.mean(boot_cov, axis=0)
            cov_table["error_boot_covariance"][index_cov[0]:index_cov[1]] = np.std(boot_cov, axis=0)


def compute_p1d_groups(
    weight_method,
    nbins_k,
    zbin_edges,
    izbin,
    p1d_table,
    k_index,
    snrfit_table,
    number_worker,
):
    """Compute the P1D groups before covariance matrix calculation.
    Put all the P1D in the same k grid.

    Arguments
    ---------
    weight_method: str,
    Method to weight the data.

    nbins_k (int):
    Number of k bins.

    zbin_edges (array-like):
    All redshift bins.

    izbin (int):
    Current redshift bin being considered.

    p1d_table (array-like):
    Table of 1D power spectra, with columns 'Pk' and 'sub_forest_id'.

    k_index (array-like):
    Array of indices for k-values, with -1 indicating values outside of the k bins.

    snrfit_table: numpy ndarray,
    Table containing SNR fit infos

    number_worker: int
    Number of workers for the parallelization

    Return
    ------
    p1d_weights  (array-like):
    Weights associated with p1d pixels for all subforest, used in the calculation of covariance.

    covariance_weights (array-like):
    Weights for all subforest used inside the main covariance sum.

    p1d_groups (array-like):
    Individual p1d pixels grouped in the same wavenumber binning for all subforest
    """

    select_z = (p1d_table["forest_z"] < zbin_edges[izbin + 1]) & (
        p1d_table["forest_z"] > zbin_edges[izbin]
    )
    p1d_sub_table = Table(
        [
            p1d_table["sub_forest_id"][select_z],
            k_index[select_z],
            p1d_table["Pk"][select_z],
        ],
        names=("sub_forest_id", "k_index", "pk"),
    )
    if weight_method == "fit_snr":
        index_slice = mean_p1d_table_regular_slice(izbin, nbins_k)
        snrfit_z = snrfit_table[index_slice[0] : index_slice[1], :]
        p1d_sub_table["weight"] = 1 / fitfunc_variance_pk1d(
            p1d_table["forest_snr"][select_z],
            snrfit_z[k_index[select_z], 2],
            snrfit_z[k_index[select_z], 3],
        )
    else:
        p1d_sub_table["weight"] = 1

    # Remove bins that are not associated with any wavenumber bin.
    p1d_sub_table = p1d_sub_table[p1d_sub_table["k_index"] >= 0]

    grouped_table = p1d_sub_table.group_by("sub_forest_id")
    p1d_los_table = [
        group[["k_index", "pk", "weight"]] for group in grouped_table.groups
    ]

    del grouped_table

    func = partial(
        compute_groups_for_one_forest,
        nbins_k,
    )
    if number_worker == 1:
        output_cov = [func(*p1d_los) for p1d_los in p1d_los_table]
    else:
        with Pool(number_worker) as pool:
            output_cov = np.array(pool.map(func, p1d_los_table))

    del p1d_los_table

    p1d_weights, covariance_weights, p1d_groups = (
        output_cov[:, 0, :],
        output_cov[:, 1, :],
        output_cov[:, 2, :],
    )
    return p1d_weights, covariance_weights, p1d_groups


def compute_groups_for_one_forest(nbins_k, p1d_los):
    """Compute the P1D groups for one subforest.

    Arguments
    ---------
    nbins_k (int):
    Number of k bins.

    p1d_los (array-like):
    Table containing all p1d pixels unordered for one subforest

    Return
    ------
    p1d_weights_id  (array-like):
    Weights associated with p1d pixels for one subforest, used in the calculation of covariance.

    covariance_weights_id (array-like):
    Weights for one subforest used inside the main covariance sum.

    p1d_groups_id (array-like):
    Individual p1d pixels grouped in the same wavenumber binning for one subforest
    """
    p1d_weights_id = np.zeros(nbins_k)
    covariance_weights_id = np.zeros(nbins_k)
    p1d_groups_id = np.zeros(nbins_k)

    for ikbin in range(nbins_k):
        mask_ikbin = p1d_los["k_index"] == ikbin
        number_in_bins = len(mask_ikbin[mask_ikbin])
        if number_in_bins != 0:
            weight = p1d_los["weight"][mask_ikbin][0]
            p1d_weights_id[ikbin] = weight
            covariance_weights_id[ikbin] = weight / number_in_bins
            p1d_groups_id[ikbin] = np.nansum(
                p1d_los["pk"][mask_ikbin] * covariance_weights_id[ikbin]
            )
    return p1d_weights_id, covariance_weights_id, p1d_groups_id


def compute_cov(
    weight_method,
    nbins_k,
    mean_pk,
    error_pk,
    p1d_weights,
    covariance_weights,
    p1d_groups,
):
    """Compute the covariance of a set of 1D power spectra.
    This is a new version of the covariance calculation.
    It needs that the input data are expressed on the same
    wavenumber grid. The calculation is then performed in
    an entire vectorized way.

    Arguments
    ---------
    weight_method: str,
    Method to weight the data.

    nbins_k (int):
    Number of k bins.

    mean_pk (array-like):
    Mean 1D power spectra, for the considered redshift bin.

    error_pk (array-like):
    Standard deviation of the 1D power spectra, for the considered redshift bin.

    p1d_weights  (array-like):
    Weights associated with p1d pixels for all subforest, used in the calculation of covariance.

    covariance_weights (array-like):
    Weights for all subforest used inside the main covariance sum.

    p1d_groups (array-like):
    Individual p1d pixels grouped in the same wavenumber binning for all subforest

    Return
    ------
    covariance_array (array-like):
    Array of covariance coefficients.
    """

    if len(p1d_groups) == 0:
        return np.full(nbins_k * nbins_k, np.nan)

    mean_pk_product = np.outer(mean_pk, mean_pk)

    sum_p1d_weights = np.nansum(p1d_weights, axis=0)
    weights_sum_product = np.outer(sum_p1d_weights, sum_p1d_weights)

    p1d_groups_product_sum = np.zeros((nbins_k, nbins_k))
    covariance_weights_product_sum = np.zeros((nbins_k, nbins_k))
    weights_product_sum = np.zeros((nbins_k, nbins_k))

    for i, p1d_group in enumerate(p1d_groups):
        # The summation is done with np.nansum instead of simple addition to not
        # include the NaN that are present in the individual p1d.
        # The summation is not done at the end, to prevent memory overhead.
        p1d_groups_product_sum = np.nansum(
            [p1d_groups_product_sum, np.outer(p1d_group, p1d_group)], axis=0
        )
        covariance_weights_product_sum = np.nansum(
            [
                covariance_weights_product_sum,
                np.outer(covariance_weights[i], covariance_weights[i]),
            ],
            axis=0,
        )
        weights_product_sum = np.nansum(
            [weights_product_sum, np.outer(p1d_weights[i], p1d_weights[i])], axis=0
        )

    del p1d_groups, covariance_weights, p1d_weights

    covariance_matrix = ((weights_sum_product /weights_product_sum) - 1)**(-1) * (
        (p1d_groups_product_sum / covariance_weights_product_sum) - mean_pk_product
    )

    # For fit_snr method, due to the SNR fitting scheme used for weighting,
    # the diagonal of the weigthed sample covariance matrix is not equal
    # to the error in mean P1D. This is tested on Ohio mocks and data.
    # We choose to renormalize the whole covariance matrix.
    if weight_method == "fit_snr":
        covariance_diag = np.diag(covariance_matrix)
        covariance_matrix = (
            covariance_matrix
            * np.outer(error_pk, error_pk)
            / np.sqrt(np.outer(covariance_diag, covariance_diag))
        )

    covariance_array = np.ravel(covariance_matrix)

    return covariance_array


def compute_cov_not_vectorized(
    p1d_table,
    mean_p1d_table,
    n_chunks,
    k_index,
    nbins_k,
    weight_method,
    snrfit_table,
    izbin,
    select_z,
    sub_forest_ids,
):
    """Compute the covariance of a set of 1D power spectra.
    This is an old implementation which is summing up each individual mode every time.
    This is very slow for large data samples.

    Arguments
    ---------
    p1d_table (array-like):
    Table of 1D power spectra, with columns 'Pk' and 'sub_forest_id'.

    mean_p1d_table (array-like):
    Table of mean 1D power spectra, with column 'meanPk'.

    n_chunks (array-like):
    Array of the number of chunks in each redshift bin.

    k_index (array-like):
    Array of indices for k-values, with -1 indicating values outside of the k bins.

    nbins_k (int):
    Number of k bins.

    weight_method: str,
    Method to weight the data.

    snrfit_table: numpy ndarray,
    Table containing SNR fit infos

    izbin (int):
    Current redshift bin being considered.

    select_z (array-like):
    Boolean array for selecting data points based on redshift.

    sub_forest_ids (array-like):
    Array of chunk ids.

    Return
    ------
    covariance_array (array-like):
    Array of covariance coefficients.
    """
    n_array = np.zeros(nbins_k * nbins_k)
    covariance_array = np.zeros(nbins_k * nbins_k)
    if weight_method == "fit_snr":
        weight_array = np.zeros(nbins_k * nbins_k)

    if n_chunks[izbin] == 0:  # Fill rows with NaNs
        covariance_array[:] = np.nan
        return covariance_array

    # First loop 1) id sub-forest bins
    for sub_forest_id in sub_forest_ids:
        select_id = select_z & (p1d_table["sub_forest_id"] == sub_forest_id)
        selected_pk = p1d_table["Pk"][select_id]
        selected_ikbin = k_index[select_id]

        if weight_method == "fit_snr":
            # Definition of weighted unbiased sample covariance taken from
            # "George R. Price, Ann. Hum. Genet., Lond, pp485-490,
            # Extension of covariance selection mathematics, 1972"
            # adapted with weights w_i = np.sqrt(w_ipk * w_ipk2) for covariance
            # to obtain the right definition of variance on the diagonal
            selected_snr = p1d_table["forest_snr"][select_id]
            index_slice = mean_p1d_table_regular_slice(izbin, nbins_k)
            snrfit_z = snrfit_table[index_slice[0] : index_slice[1], :]
            selected_variance_estimated = fitfunc_variance_pk1d(
                selected_snr, snrfit_z[selected_ikbin, 2], snrfit_z[selected_ikbin, 3]
            )
            # First loop 2) selected pk
            for ipk, _ in enumerate(selected_pk):
                ikbin = selected_ikbin[ipk]
                # First loop 3) selected pk
                for ipk2 in range(ipk, len(selected_pk)):
                    ikbin2 = selected_ikbin[ipk2]
                    if (ikbin2 != -1) & (ikbin != -1):
                        # index of the (ikbin,ikbin2) coefficient on the top of the matrix
                        index = (nbins_k * ikbin) + ikbin2
                        weight = 1 / selected_variance_estimated[ipk]
                        weight2 = 1 / selected_variance_estimated[ipk2]
                        covariance_array[index] = covariance_array[index] + selected_pk[
                            ipk
                        ] * selected_pk[ipk2] * np.sqrt(weight * weight2)
                        n_array[index] = n_array[index] + 1
                        weight_array[index] = weight_array[index] + np.sqrt(
                            weight * weight2
                        )
        else:
            # First loop 2) selected pk
            for ipk, _ in enumerate(selected_pk):
                ikbin = selected_ikbin[ipk]
                # First loop 3) selected pk
                for ipk2 in range(ipk, len(selected_pk)):
                    ikbin2 = selected_ikbin[ipk2]
                    if (ikbin2 != -1) & (ikbin != -1):
                        # index of the (ikbin,ikbin2) coefficient on the top of the matrix
                        index = (nbins_k * ikbin) + ikbin2
                        covariance_array[index] = (
                            covariance_array[index]
                            + selected_pk[ipk] * selected_pk[ipk2]
                        )
                        n_array[index] = n_array[index] + 1
    # Second loop 1) k bins
    for ikbin in range(nbins_k):
        mean_ikbin = mean_p1d_table["meanPk"][(nbins_k * izbin) + ikbin]

        # Second loop 2) k bins
        for ikbin2 in range(ikbin, nbins_k):
            mean_ikbin2 = mean_p1d_table["meanPk"][(nbins_k * izbin) + ikbin2]

            # index of the (ikbin,ikbin2) coefficient on the top of the matrix
            index = (nbins_k * ikbin) + ikbin2
            if weight_method == "fit_snr":
                covariance_array[index] = (
                    covariance_array[index] / weight_array[index]
                ) - mean_ikbin * mean_ikbin2
            else:
                covariance_array[index] = (
                    covariance_array[index] / n_array[index]
                ) - mean_ikbin * mean_ikbin2

            # Same normalization for fit_snr, equivalent to dividing to
            # weight_array[index] if weights are normalized to give
            # sum(weight_array[index]) = n_array[index]
            covariance_array[index] = covariance_array[index] / n_array[index]

            if ikbin2 != ikbin:
                # index of the (ikbin,ikbin2) coefficient on the bottom of the matrix
                index_2 = (nbins_k * ikbin2) + ikbin
                covariance_array[index_2] = covariance_array[index]
                n_array[index_2] = n_array[index]

    # For fit_snr method, due to the SNR fitting scheme used for weighting,
    # the diagonal of the weigthed sample covariance matrix is not equal
    # to the error in mean P1D. This is tested on Ohio mocks.
    # We choose to renormalize the whole covariance matrix.
    if weight_method == "fit_snr":
        # Third loop 1) k bins
        for ikbin in range(nbins_k):
            std_ikbin = mean_p1d_table["errorPk"][(nbins_k * izbin) + ikbin]
            # Third loop 2) k bins
            for ikbin2 in range(ikbin, nbins_k):
                std_ikbin2 = mean_p1d_table["errorPk"][(nbins_k * izbin) + ikbin2]

                # index of the (ikbin,ikbin2) coefficient on the top of the matrix
                index = (nbins_k * ikbin) + ikbin2
                covariance_array[index] = (
                    covariance_array[index]
                    * (std_ikbin * std_ikbin2)
                    / np.sqrt(
                        covariance_array[(nbins_k * ikbin) + ikbin]
                        * covariance_array[(nbins_k * ikbin2) + ikbin2]
                    )
                )

                if ikbin2 != ikbin:
                    # index of the (ikbin,ikbin2) coefficient on the bottom of the matrix
                    index_2 = (nbins_k * ikbin2) + ikbin
                    covariance_array[index_2] = (
                        covariance_array[index_2]
                        * (std_ikbin * std_ikbin2)
                        / np.sqrt(
                            covariance_array[(nbins_k * ikbin) + ikbin]
                            * covariance_array[(nbins_k * ikbin2) + ikbin2]
                        )
                    )

    return covariance_array


def run_postproc_pk1d(
    data_dir,
    output_file,
    zbin_edges,
    kbin_edges,
    weight_method="no_weights",
    apply_z_weights=False,
    snrcut=None,
    skymask_matrices=None,
    zbins_snrcut=None,
    output_snrfit=None,
    nomedians=False,
    velunits=False,
    overwrite=False,
    ncpu=8,
    compute_covariance=False,
    compute_bootstrap=False,
    number_bootstrap=50,
):
    """
    Read individual Pk1D data from a set of files and compute P1D statistics.

    Arguments
    ---------
    data_dir: string
    Directory where individual P1D FITS files are located

    output_file: string
    Output file name

    overwrite: Bool
    Overwrite output file if existing

    ncpu: int
    The I/O function read_pk1d() is run parallel

    Other arguments are as defined
    in compute_mean_pk1d() or read_pk1d()
    """
    if os.path.exists(output_file) and not overwrite:
        raise RuntimeError("Output file already exists: " + output_file)

    searchstr = "*"
    files = glob.glob(os.path.join(data_dir, f"Pk1D{searchstr}.fits.gz"))

    with Pool(ncpu) as pool:
        output_readpk1d = pool.starmap(
            read_pk1d, [[f, kbin_edges, snrcut, zbins_snrcut, skymask_matrices] for f in files]
        )

    output_readpk1d = [x for x in output_readpk1d if x is not None]
    p1d_table = vstack([output_readpk1d[i][0] for i in range(len(output_readpk1d))])
    z_array = np.concatenate(
        tuple(output_readpk1d[i][1] for i in range(len(output_readpk1d)))
    )

    userprint("Individual P1Ds read, now computing statistics.")

    mean_p1d_table, metadata_table, cov_table = compute_mean_pk1d(
        p1d_table,
        z_array,
        zbin_edges,
        kbin_edges,
        weight_method,
        apply_z_weights,
        nomedians=nomedians,
        velunits=velunits,
        output_snrfit=output_snrfit,
        compute_covariance=compute_covariance,
        compute_bootstrap=compute_bootstrap,
        number_bootstrap=number_bootstrap,
        number_worker=ncpu,
    )

    metadata_header = {
        "VELUNITS": velunits,
        "NQSO": len(np.unique(p1d_table["forest_id"])),
    }

    write_mean_pk1d(
        output_file,
        mean_p1d_table,
        metadata_table,
        metadata_header,
        cov_table,
    )


def check_mean_pk1d_compatibility(
    mean_p1d_tables,
    metadata_tables,
    metadata_headers,
    cov_tables,
):
    """
    Check the compatibility between different average p1d

    Arguments
    ---------
    mean_p1d_tables: array-like
    List of p1d tables to average

    metadata_tables: array-like
    List of metadata table of each mean p1d

    metadata_headers: array-like
    List of metadata header of each mean p1d

    mean_p1d_tables: array-like
    List of covariance tables to average
    """

    if len(mean_p1d_tables) <= 1:
        raise ValueError(
            """Not enough mean p1d data to coadd (0 or 1),"""
            """ Please check the input data"""
        )
    if (len(mean_p1d_tables) != len(metadata_tables)) | (
        len(mean_p1d_tables) != len(metadata_headers)
    ):
        raise ValueError(
            """The number of mean p1d data to coadd is different,"""
            """ Please check the input data"""
        )
    if (len(mean_p1d_tables) != len(cov_tables)) & (len(cov_tables) != 0):
        raise ValueError(
            """The input mean p1d data does not all contains covariance,"""
            """ Please check the input data"""
        )

    z_min = metadata_tables[-1]["z_min"]
    z_max = metadata_tables[-1]["z_max"]
    k_min = metadata_tables[-1]["k_min"]
    k_max = metadata_tables[-1]["k_max"]
    k_bin_size = len(mean_p1d_tables[-1]["meank"])
    for i, metadata in enumerate(metadata_tables[:-1]):
        test = np.all(metadata["z_min"] == z_min)
        test &= np.all(metadata["z_max"] == z_max)
        test &= np.all(metadata["k_min"] == k_min)
        test &= np.all(metadata["k_max"] == k_max)
        test &= len(mean_p1d_tables[i]["meank"]) == k_bin_size

        if not test:
            raise ValueError(
                """The input mean p1d data does not have the same k and z binning,"""
                """ Please check the input data"""
            )

    velunits = metadata_headers[-1]["VELUNITS"]
    for _, header in enumerate(metadata_headers[:-1]):
        if header["VELUNITS"] != velunits:
            raise ValueError(
                """The input mean p1d data are not expressed in the same units,"""
                """ Please check the input data"""
            )

    mean_p1d_table_colnames = mean_p1d_tables[-1].colnames
    for i, table in enumerate(mean_p1d_tables[:-1]):
        if not np.all(table.colnames == mean_p1d_table_colnames):
            raise ValueError(
                """The mean p1d tables does not have the same column names,"""
                """ Please check the input data"""
            )
    cov_table_colnames = cov_tables[-1].colnames
    for i, table in enumerate(cov_tables[:-1]):
        if not np.all(table.colnames == cov_table_colnames):
            raise ValueError(
                """The covariance tables does not have the same column names,"""
                """ Please check the input data"""
            )


def average_mean_pk1d_files(
    mean_p1d_names,
    output_path,
    weighted_mean=False,
):
    """
    Compute and write the average of mean p1d

    Arguments
    ---------
    mean_p1d_names: array-like
    List of the name of p1d means to average

    output_path: str
    Output path where to write

    weighted_mean: bool
    If True, compute the weighted average using the errors as weights.
    """

    mean_p1d_tables, metadata_tables, metadata_headers, cov_tables = (
        [],
        [],
        [],
        [],
    )

    for mean_p1d_name in mean_p1d_names:
        hdus = fitsio.FITS(mean_p1d_name)
        metadata_tables.append(Table(hdus[2].read()))
        metadata_headers.append(hdus[2].read_header())
        mean_p1d_tables.append(Table(hdus[1].read()))
        if len(hdus) > 3:
            cov_tables.append(Table(hdus[3].read()))

    check_mean_pk1d_compatibility(
        mean_p1d_tables, metadata_tables, metadata_headers, cov_tables
    )

    combination_metadata_header = {
        "VELUNITS": metadata_headers[0]["VELUNITS"],
        "NQSO": np.sum([header["NQSO"] for header in metadata_headers]),
    }

    combination_metadata_table = metadata_tables[0][
        ["z_min", "z_max", "k_min", "k_max"]
    ]
    combination_metadata_table["N_chunks"] = np.sum(
        np.array([metadata["N_chunks"] for metadata in metadata_tables]), axis=0
    )

    combination_mean_p1d_table = mean_p1d_tables[0][["zbin", "index_zbin"]]
    combination_mean_p1d_table["N"] = np.sum(
        np.array([mean_p1d_table["N"] for mean_p1d_table in mean_p1d_tables]), axis=0
    )
    for colname in mean_p1d_tables[0].colnames:
        all_mean_p1d_colname = np.array(
            [mean_p1d[colname] for mean_p1d in mean_p1d_tables]
        )
        masked_all_mean_p1d_colname = np.ma.masked_array(
            all_mean_p1d_colname, np.isnan(all_mean_p1d_colname)
        )
        if ("mean" in colname) | ("median" in colname):
            if weighted_mean:
                weight_colname = (
                    colname.replace("mean", "error")
                    if "mean" in colname
                    else colname.replace("median", "error")
                )
                weights = (
                    1
                    / (
                        np.array(
                            [mean_p1d[weight_colname] for mean_p1d in mean_p1d_tables]
                        )
                    )
                    ** 2
                )
                new_mean_p1d_colname = np.ma.average(
                    masked_all_mean_p1d_colname, axis=0, weights=weights
                ).filled(np.nan)
            else:
                new_mean_p1d_colname = np.ma.average(
                    masked_all_mean_p1d_colname, axis=0
                ).filled(np.nan)
        elif "error" in colname:
            new_mean_p1d_colname = np.ma.average(
                masked_all_mean_p1d_colname, axis=0
            ).filled(np.nan) / np.sqrt(len(mean_p1d_tables))
        elif "min" in colname:
            new_mean_p1d_colname = np.ma.min(
                masked_all_mean_p1d_colname, axis=0
            ).filled(np.nan)
        elif "max" in colname:
            new_mean_p1d_colname = np.ma.max(
                masked_all_mean_p1d_colname, axis=0
            ).filled(np.nan)
        else:
            continue
        combination_mean_p1d_table[colname] = new_mean_p1d_colname

    combination_cov_table = cov_tables[0][["zbin", "index_zbin"]]
    combination_cov_table["N"] = np.sum(
        np.array([cov_table["N"] for cov_table in cov_tables]),
        axis=0,
    )

    if len(cov_tables) == 0:
        combination_cov_table = None
    else:
        all_covariance = np.array([cov_table["covariance"] for cov_table in cov_tables])
        masked_all_covariance = np.ma.masked_array(
            all_covariance, np.isnan(all_covariance)
        )
        combination_cov_table["covariance"] = np.ma.average(
            masked_all_covariance, axis=0
        ).filled(np.nan)
        combination_cov_table["k1"] = cov_tables[0]["k1"]
        combination_cov_table["k2"] = cov_tables[0]["k2"]

        if "boot_covariance" in cov_tables[0].colnames:
            all_boot_covariance = np.array(
                [cov_table["boot_covariance"] for cov_table in cov_tables]
            )
            all_error_boot_covariance = np.array(
                [cov_table["error_boot_covariance"] for cov_table in cov_tables]
            )

            masked_all_boot_covariance = np.ma.masked_array(
                all_boot_covariance, np.isnan(all_boot_covariance)
            )
            masked_all_error_boot_covariance = np.ma.masked_array(
                all_error_boot_covariance, np.isnan(all_error_boot_covariance)
            )

            if weighted_mean:
                weights = 1 / masked_all_error_boot_covariance**2
                combination_cov_table["boot_covariance"] = np.ma.average(
                    masked_all_boot_covariance, axis=0, weights=weights
                ).filled(np.nan)
            else:
                combination_cov_table["boot_covariance"] = np.ma.average(
                    masked_all_boot_covariance, axis=0
                ).filled(np.nan)
            combination_cov_table["error_boot_covariance"] = np.ma.average(
                masked_all_error_boot_covariance, axis=0
            ).filled(np.nan) / np.sqrt(len(cov_tables))

    output_file = os.path.join(output_path, mean_p1d_names[0].split("/")[-1])
    write_mean_pk1d(
        output_file,
        combination_mean_p1d_table,
        combination_metadata_table,
        combination_metadata_header,
        combination_cov_table,
    )


def write_mean_pk1d(
    output_file,
    mean_p1d_table,
    metadata_table,
    metadata_header,
    cov_table,
):
    """
    Write a mean p1d file

    Arguments
    ---------
    output_file: string
    Output file name

    mean_p1d_table: Table
    Table containing the mean p1d properties

    metadata_table: Table
    Table containing the metadata

    metadata_header: Table
    Table containing the metadata header to write

    cov_table: Table
    Table containing the covariance of the mean p1d
    """
    result = fitsio.FITS(output_file, "rw", clobber=True)
    result.write(mean_p1d_table.as_array())
    result.write(
        metadata_table.as_array(),
        header=metadata_header,
    )
    if cov_table is not None:
        result.write(cov_table.as_array())
    result.close()
