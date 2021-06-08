"""This module defines functions and variables required for the correlation
analysis of two delta fields

This module provides several functions:
    - fill_neighs
    - compute_xi
    - compute_xi_forest_pairs
    - compute_dmat
    - compute_dmat_forest_pairs
    - compute_metal_dmat
    - compute_wick_terms
    - compute_wickT1234_pairs
    - compute_wickT1234_pairs
    - compute_xi_1d
See the respective docstrings for more details
"""
import numpy as np
from healpy import query_disc
from numba import jit, int32

from . import constants
from .utils import userprint

num_bins_r_par = None
num_bins_r_trans = None
num_model_bins_r_par = None
num_model_bins_r_trans = None
r_par_max = None
r_par_min = None
r_trans_max = None
z_cut_max = None
z_cut_min = None
ang_max = None
nside = None

counter = None
num_data = None

z_ref = None
alpha = None
alpha_obj = None
lambda_abs = None
alpha_abs = None

data = None
objs = None

reject = None
lock = None

cosmo = None
ang_correlation = False

# variables used in the wick covariance matrix computation
get_variance_1d = {}
xi_1d = {}
max_diagram = None
xi_wick = None


def fill_neighs(healpixs):
    """Create and store a list of neighbours for each of the healpix.

    Neighbours are added to the delta objects directly

    Args:
        healpixs: array of ints
            List of healpix numbers
    """
    for healpix in healpixs:
        for delta in data[healpix]:
            healpix_neighbours = query_disc(
                nside, [delta.x_cart, delta.y_cart, delta.z_cart],
                ang_max,
                inclusive=True)
            healpix_neighbours = [
                other_healpix for other_healpix in healpix_neighbours
                if other_healpix in objs
            ]
            neighbours = [
                obj for other_healpix in healpix_neighbours
                for obj in objs[other_healpix] if obj.thingid != delta.thingid
            ]
            ang = delta.get_angle_between(neighbours)
            w = ang < ang_max
            if not ang_correlation:
                r_comov = np.array([obj.r_comov for obj in neighbours])
                w &= (delta.r_comov[0] - r_comov) * np.cos(ang / 2.) < r_par_max
                w &= (delta.r_comov[-1] - r_comov) * np.cos(
                    ang / 2.) > r_par_min
            neighbours = np.array(neighbours)[w]
            delta.neighbours = np.array([
                obj for obj in neighbours
                if ((delta.z[-1] + obj.z_qso) / 2. >= z_cut_min and
                    (delta.z[-1] + obj.z_qso) / 2. < z_cut_max)
            ])


def compute_xi(healpixs):
    """Computes the correlation function for each of the healpixs.

    Args:
        healpixs: array of ints
            List of healpix numbers

    Returns:
        The following variables:
            weights: Total weights in the correlation function pixels
            xi: The correlation function
            r_par: Parallel distance of the correlation function pixels
            r_trans: Transverse distance of the correlation function pixels
            z: Redshift of the correlation function pixels
            num_pairs: Number of pairs in the correlation function pixels
    """
    xi = np.zeros(num_bins_r_par * num_bins_r_trans)
    weights = np.zeros(num_bins_r_par * num_bins_r_trans)
    r_par = np.zeros(num_bins_r_par * num_bins_r_trans)
    r_trans = np.zeros(num_bins_r_par * num_bins_r_trans)
    z = np.zeros(num_bins_r_par * num_bins_r_trans)
    num_pairs = np.zeros(num_bins_r_par * num_bins_r_trans, dtype=np.int64)

    for healpix in healpixs:
        for delta in data[healpix]:
            with lock:
                xicounter = round(counter.value * 100. / num_data, 2)
                if (counter.value % 1000 == 0):
                    userprint(("computing xi: {}%").format(xicounter))
                counter.value += 1

            if delta.neighbours.size != 0:
                ang = delta.get_angle_between(delta.neighbours)
                z_qso = np.array([obj.z_qso for obj in delta.neighbours])
                weights_qso = np.array(
                    [obj.weights for obj in delta.neighbours])
                if ang_correlation:
                    lambda_qso = np.array(
                        [10.**obj.log_lambda for obj in delta.neighbours])
                    compute_xi_forest_pairs_fast(delta.z, 10.**delta.log_lambda,
                                                 10.**delta.log_lambda,
                                                 delta.weights, delta.delta,
                                                 z_qso, lambda_qso, lambda_qso,
                                                 weights_qso, ang, weights, xi,
                                                 r_par, r_trans, z, num_pairs)
                else:
                    r_comov_qso = np.array(
                        [obj.r_comov for obj in delta.neighbours])
                    dist_m_qso = np.array(
                        [obj.dist_m for obj in delta.neighbours])
                    compute_xi_forest_pairs_fast(delta.z, delta.r_comov,
                                                 delta.dist_m, delta.weights,
                                                 delta.delta, z_qso,
                                                 r_comov_qso, dist_m_qso,
                                                 weights_qso, ang, weights, xi,
                                                 r_par, r_trans, z, num_pairs)

                #-- The following was used by compute_xi_forest_pairs_fast
                #-- which will be deprecated
                #xi[:len(rebin_xi)] += rebin_xi
                #weights[:len(rebin_weight)] += rebin_weight
                #r_par[:len(rebin_r_par)] += rebin_r_par
                #r_trans[:len(rebin_r_trans)] += rebin_r_trans
                #z[:len(rebin_z)] += rebin_z
                #num_pairs[:len(rebin_num_pairs)] += rebin_num_pairs.astype(int)
            setattr(delta, "neighbours", None)

    w = weights > 0
    xi[w] /= weights[w]
    r_par[w] /= weights[w]
    r_trans[w] /= weights[w]
    z[w] /= weights[w]
    return weights, xi, r_par, r_trans, z, num_pairs


#-- This has been superseeded by compute_xi_forest_pairs_fast
#-- and will be deprecated
@jit
def compute_xi_forest_pairs(z1, r_comov1, dist_m1, weights1, delta1, z2,
                            r_comov2, dist_m2, weights2, ang):
    """Computes the contribution of a given pair of forests to the correlation
    function.

    Args:
        z1: array of float
            Redshift of pixel 1
        r_comov1: array of float
            Comoving distance for forest 1 (in Mpc/h)
        dist_m1: array of float
            Comoving angular distance for forest 1 (in Mpc/h)
        weights1: array of float
            Pixel weights for forest 1
        delta1: array of float
            Delta field for forest 1
        z2: array of float
            Redshift of pixel 2
        r_comov2: array of float
            Comoving distance for forest 2 (in Mpc/h)
        dist_m2: array of float
            Comoving angular distance for forest 2 (in Mpc/h)
        weights2: array of float
            Pixel weights for forest 2
        ang: array of float
            Angular separation between pixels in forests 1 and 2

    Returns:
        The following variables:
            rebin_weight: The weight of the correlation function pixels
                properly rebinned
            rebin_xi: The correlation function properly rebinned
            rebin_r_par: The parallel distance of the correlation function
                pixels properly rebinned
            rebin_r_trans: The transverse distance of the correlation function
                pixels properly rebinned
            rebin_z: The redshift of the correlation function pixels properly
                rebinned
            rebin_num_pairs: The number of pairs of the correlation function
                pixels properly rebinned
    """
    if ang_correlation:
        r_par = r_comov1[:, None] / r_comov2
        r_trans = ang * np.ones_like(r_par)
    else:
        r_par = (r_comov1[:, None] - r_comov2) * np.cos(ang / 2)
        r_trans = (dist_m1[:, None] + dist_m2) * np.sin(ang / 2)
    z = (z1[:, None] + z2) / 2

    weights12 = weights1[:, None] * weights2
    delta_times_weight = (weights1 * delta1)[:, None] * weights2

    w = (r_par > r_par_min) & (r_par < r_par_max) & (r_trans < r_trans_max)
    r_par = r_par[w]
    r_trans = r_trans[w]
    z = z[w]
    weights12 = weights12[w]
    delta_times_weight = delta_times_weight[w]

    bins_r_par = ((r_par - r_par_min) / (r_par_max - r_par_min) *
                  num_bins_r_par).astype(int)
    bins_r_trans = (r_trans / r_trans_max * num_bins_r_trans).astype(int)
    bins = bins_r_trans + num_bins_r_trans * bins_r_par

    rebin_xi = np.bincount(bins, weights=delta_times_weight)
    rebin_weight = np.bincount(bins, weights=weights12)
    rebin_r_par = np.bincount(bins, weights=r_par * weights12)
    rebin_r_trans = np.bincount(bins, weights=r_trans * weights12)
    rebin_z = np.bincount(bins, weights=z * weights12)
    rebin_num_pairs = np.bincount(bins, weights=(weights12 > 0.))

    return (rebin_weight, rebin_xi, rebin_r_par, rebin_r_trans, rebin_z,
            rebin_num_pairs)


@jit(nopython=True)
def compute_xi_forest_pairs_fast(z1, r_comov1, dist_m1, weights1, delta1, z2,
                                 r_comov2, dist_m2, weights2, ang, rebin_weight,
                                 rebin_xi, rebin_r_par, rebin_r_trans, rebin_z,
                                 rebin_num_pairs):
    """Computes the contribution of a given pair of forests to the correlation
    function.

    Args:
        z1: array of float
            Redshift of pixel 1
        r_comov1: array of float
            Comoving distance for forest 1 (in Mpc/h)
        dist_m1: array of float
            Comoving angular distance for forest 1 (in Mpc/h)
        weights1: array of float
            Pixel weights for forest 1
        delta1: array of float
            Delta field for forest 1
        z2: array of float
            Redshift of pixel 2
        r_comov2: array of float
            Comoving distance for forest 2 (in Mpc/h)
        dist_m2: array of float
            Comoving angular distance for forest 2 (in Mpc/h)
        weights2: array of float
            Pixel weights for forest 2
        ang: array of float
            Angular separation between pixels in forests 1 and 2
        rebin_weight: The weight of the correlation function pixels
            properly rebinned
        rebin_xi: The correlation function properly rebinned
        rebin_r_par: The parallel distance of the correlation function
            pixels properly rebinned
        rebin_r_trans: The transverse distance of the correlation function
            pixels properly rebinned
        rebin_z: The redshift of the correlation function pixels properly
            rebinned
        rebin_num_pairs: The number of pairs of the correlation function
            pixels properly rebinned
    """
    for i in range(z1.size):
        if weights1[i] == 0:
            continue

        for j in range(len(z2)):
            if weights2[j] == 0:
                continue

            if ang_correlation:
                r_par = r_comov1[i] / r_comov2[j]
                r_trans = ang[j]
            else:
                r_par = (r_comov1[i] - r_comov2[j]) * np.cos(ang[j] / 2)
                r_trans = (dist_m1[i] + dist_m2[j]) * np.sin(ang[j] / 2)

            if (r_par >= r_par_max or r_trans >= r_trans_max or
                    r_par <= r_par_min):
                continue

            delta_times_weight = delta1[i] * weights1[i] * weights2[j]
            weights12 = weights1[i] * weights2[j]
            z = (z1[i] + z2[j]) / 2

            bins_r_par = np.floor(
                (r_par - r_par_min) / (r_par_max - r_par_min) * num_bins_r_par)
            bins_r_trans = np.floor(r_trans / r_trans_max * num_bins_r_trans)
            bins = np.int(bins_r_trans + num_bins_r_trans * bins_r_par)

            rebin_xi[bins] += delta_times_weight
            rebin_weight[bins] += weights12
            rebin_r_par[bins] += r_par * weights12
            rebin_r_trans[bins] += r_trans * weights12
            rebin_z[bins] += z * weights12
            rebin_num_pairs[bins] += 1


def compute_dmat(healpixs):
    """Computes the distortion matrix for each of the healpixs.

    Args:
        healpixs: array of ints
            List of healpix numbers

    Returns:
        The following variables:
            weights_dmat: Total weight in the distortion matrix pixels
            dmat: The distortion matrix
            r_par_eff: Effective parallel distance of the distortion matrix
                pixels
            r_trans_eff: Effective transverse distance of the distortion matrix
                pixels
            z_eff: Effective redshift of the distortion matrix pixels
            weight_eff: Effective weight of the distortion matrix pixels
            num_pairs: Total number of pairs
            num_pairs_used: Number of used pairs
    """

    dmat = np.zeros(num_bins_r_par * num_bins_r_trans * num_model_bins_r_trans *
                    num_model_bins_r_par)
    weights_dmat = np.zeros(num_bins_r_par * num_bins_r_trans)
    r_par_eff = np.zeros(num_model_bins_r_trans * num_model_bins_r_par)
    r_trans_eff = np.zeros(num_model_bins_r_trans * num_model_bins_r_par)
    z_eff = np.zeros(num_model_bins_r_trans * num_model_bins_r_par)
    weight_eff = np.zeros(num_model_bins_r_trans * num_model_bins_r_par)

    num_pairs = 0
    num_pairs_used = 0
    for healpix in healpixs:
        for delta1 in data[healpix]:
            with lock:
                xicounter = round(counter.value * 100. / num_data, 2)
                if (counter.value % 1000 == 0):
                    userprint(("computing xi: {}%").format(xicounter))
                counter.value += 1
            order1 = delta1.order
            r_comov1 = delta1.r_comov
            dist_m1 = delta1.dist_m
            weights1 = delta1.weights
            log_lambda1 = delta1.log_lambda
            z1 = delta1.z
            w = np.random.rand(len(delta1.neighbours)) > reject
            if w.sum() == 0:
                continue
            num_pairs += len(delta1.neighbours)
            num_pairs_used += w.sum()
            neighbours = delta1.neighbours[w]
            ang = delta1.get_angle_between(neighbours)
            r_comov2 = np.array([obj.r_comov for obj in neighbours])
            dist_m2 = np.array([obj.dist_m for obj in neighbours])
            weights2 = np.array([obj.weights for obj in neighbours])
            z2 = np.array([obj.z_qso for obj in neighbours])
            compute_dmat_forest_pairs_fast(log_lambda1, r_comov1, dist_m1, z1,
                                           weights1, r_comov2, dist_m2, z2,
                                           weights2, ang, weights_dmat, dmat,
                                           r_par_eff, r_trans_eff, z_eff,
                                           weight_eff, order1)
            setattr(delta1, "neighbours", None)

    dmat = dmat.reshape(num_bins_r_par * num_bins_r_trans,
                        num_model_bins_r_par * num_model_bins_r_trans)

    return (weights_dmat, dmat, r_par_eff, r_trans_eff, z_eff, weight_eff,
            num_pairs, num_pairs_used)


#-- This has been superseeded by compute_dmat_forest_pairs_fast
#-- and will be deprecated
@jit
def compute_dmat_forest_pairs(log_lambda1, r_comov1, dist_m1, z1, weights1,
                              r_comov2, dist_m2, z2, weights2, ang,
                              weights_dmat, dmat, r_par_eff, r_trans_eff, z_eff,
                              weight_eff):
    """Computes the contribution of a given pair of forests-quasar to the
    distortion matrix.

    Args:
        log_lambda1: array of float
            Logarithm of the wavelength (in Angs) for forest 1
        r_comov1: array of floats
            Comoving distance (in Mpc/h) for forest 1
        dist_m1: array of floats
            Angular distance for forest 1
        z1: array of floats
            Redshifts for forest 1
        weights1: array of floats
            Weights for forest 1
        r_comov2: array of floats
            Comoving distance (in Mpc/h) for forest 2
        dist_m2: array of floats
            Angular distance for forest 2
        z2: array of floats
            Redshifts for forest 2
        weights2: array of floats
            Weights for forest 2
        ang: array of floats
            Angular separation between pixels in forests 1 and 2
        weights_dmat: array of floats
            Total weight in the distortion matrix pixels
        dmat: array of floats
            The distortion matrix
        r_par_eff: array of floats
            Effective parallel distance for the distortion matrix bins
        r_trans_eff: array of floats
            Effective transverse distance for the distortion matrix bins
        z_eff: array of floats
            Effective redshift for the distortion matrix bins
        weight_eff: array of floats
            Effective weight of the distortion matrix pixels
    """
    # find distances between pixels
    r_par = (r_comov1[:, None] - r_comov2) * np.cos(ang / 2)
    r_trans = (dist_m1[:, None] + dist_m2) * np.sin(ang / 2)
    z = (z1[:, None] + z2) / 2.
    w = (r_par > r_par_min) & (r_par < r_par_max) & (r_trans < r_trans_max)

    # locate bins pixels are contributing to (correlation bins)
    bins_r_par = ((r_par - r_par_min) / (r_par_max - r_par_min) *
                  num_bins_r_par).astype(int)
    bins_r_trans = (r_trans / r_trans_max * num_bins_r_trans).astype(int)
    bins = bins_r_trans + num_bins_r_trans * bins_r_par
    bins = bins[w]

    # locate bins pixels are contributing to (model bins)
    model_bins_r_par = ((r_par - r_par_min) / (r_par_max - r_par_min) *
                        num_model_bins_r_par).astype(int)
    model_bins_r_trans = (r_trans / r_trans_max *
                          num_model_bins_r_trans).astype(int)
    model_bins = model_bins_r_trans + num_model_bins_r_trans * model_bins_r_par
    model_bins = model_bins[w]

    # compute useful auxiliar variables to speed up computation of eta
    # (equation 6 of du Mas des Bourboux et al. 2020)

    # denominator second term in equation 6 of du Mas des Bourboux et al. 2020
    sum_weights1 = weights1.sum()

    # mean of log_lambda
    mean_log_lambda1 = np.average(log_lambda1, weights=weights1)

    # log_lambda minus its mean
    log_lambda_minus_mean1 = log_lambda1 - mean_log_lambda1

    # denominator third term in equation 6 of du Mas des Bourboux et al. 2020
    sum_weights_square_log_lambda_minus_mean1 = (
        weights1 * log_lambda_minus_mean1**2).sum()

    # auxiliar variables to loop over distortion matrix bins
    num_pixels1 = len(log_lambda1)
    num_pixels2 = len(r_comov2)
    ij = np.arange(num_pixels1)[:, None] + num_pixels1 * np.arange(num_pixels2)
    ij = ij[w]

    weights12 = weights1[:, None] * weights2
    weights12 = weights12[w]

    rebin = np.bincount(model_bins, weights=weights12 * r_par[w])
    r_par_eff[:rebin.size] += rebin
    rebin = np.bincount(model_bins, weights=weights12 * r_trans[w])
    r_trans_eff[:rebin.size] += rebin
    rebin = np.bincount(model_bins, weights=weights12 * z[w])
    z_eff[:rebin.size] += rebin
    rebin = np.bincount(model_bins, weights=weights12)
    weight_eff[:rebin.size] += rebin

    rebin = np.bincount(bins, weights=weights12)
    weights_dmat[:len(rebin)] += rebin

    # Combining equation 22 and equation 6 of du Mas des Bourboux et al. 2020
    # we find an equation with 3 terms comming from the product of two eta
    # The variables below stand for 2 of these 3 terms (the first one is
    # pretty trivial) and are named to match those of module cf

    # first eta, second term: weight/sum(weights)
    eta2 = np.zeros(num_model_bins_r_par * num_model_bins_r_trans * num_pixels2)
    # first eta, third term: (non-zero only for order=1)
    #   weight*(Lambda-bar(Lambda))*(Lambda-bar(Lambda))/
    eta4 = np.zeros(num_model_bins_r_par * num_model_bins_r_trans * num_pixels2)

    # compute the contributions to the distortion matrix
    rebin = np.bincount(
        ((ij - ij % num_pixels1) // num_pixels1 + num_pixels2 * model_bins),
        weights=((weights1[:, None] * np.ones(num_pixels2))[w] / sum_weights1))
    eta2[:len(rebin)] += rebin
    rebin = np.bincount(
        ((ij - ij % num_pixels1) // num_pixels1 + num_pixels2 * model_bins),
        weights=(((weights1 * log_lambda_minus_mean1)[:, None] *
                  np.ones(num_pixels2))[w] /
                 sum_weights_square_log_lambda_minus_mean1))
    eta4[:len(rebin)] += rebin

    # Now add all the contributions together
    unique_model_bins = np.unique(model_bins)
    for index, (bin, model_bin) in enumerate(zip(bins, model_bins)):
        # first eta, first term: kronecker delta
        dmat[model_bin + num_model_bins_r_par * num_model_bins_r_trans *
             bin] += weights12[index]
        i = ij[index] % num_pixels1
        j = (ij[index] - i) // num_pixels1
        # rest of the terms
        for unique_model_bin in unique_model_bins:
            dmat[unique_model_bin + num_model_bins_r_par *
                 num_model_bins_r_trans * bin] -= weights12[index] * (
                     eta2[j + num_pixels2 * unique_model_bin] +
                     eta4[j + num_pixels2 * unique_model_bin] *
                     log_lambda_minus_mean1[i])


@jit(nopython=True)
def compute_dmat_forest_pairs_fast(log_lambda1, r_comov1, dist_m1, z1, weights1,
                                   r_comov2, dist_m2, z2, weights2, ang,
                                   weights_dmat, dmat, r_par_eff, r_trans_eff,
                                   z_eff, weight_eff, order1):

    #-- First, determine how many relevant pixel pairs for speed up
    num_pairs = 0
    for i in range(z1.size):
        if weights1[i] == 0:
            continue
        for j in range(z2.size):
            if weights2[j] == 0:
                continue
            r_par = (r_comov1[i] - r_comov2[j]) * np.cos(ang[j] / 2)
            r_trans = (dist_m1[i] + dist_m2[j]) * np.sin(ang[j] / 2)
            if (r_par >= r_par_max or r_trans >= r_trans_max or
                    r_par <= r_par_min):
                continue
            num_pairs += 1

    if num_pairs == 0:
        return

    # Compute useful auxiliar variables to speed up computation of eta
    # (equation 6 of du Mas des Bourboux et al. 2020)

    # denominator second term in equation 6 of du Mas des Bourboux et al. 2020
    sum_weights1 = weights1.sum()

    # mean of log_lambda
    mean_log_lambda1 = np.sum(log_lambda1 * weights1) / sum_weights1

    # log_lambda minus its mean
    log_lambda_minus_mean1 = log_lambda1 - mean_log_lambda1

    # denominator third term in equation 6 of du Mas des Bourboux et al. 2020
    sum_weights_square_log_lambda_minus_mean1 = (
        weights1 * log_lambda_minus_mean1**2).sum()

    # auxiliar variables to loop over distortion matrix bins
    num_pixels1 = len(log_lambda1)
    num_pixels2 = len(r_comov2)

    eta2 = np.zeros(num_model_bins_r_par * num_model_bins_r_trans * num_pixels2)
    eta4 = np.zeros(num_model_bins_r_par * num_model_bins_r_trans * num_pixels2)

    #-- Notice that the dtype is numba.int32
    all_bins = np.zeros(num_pairs, dtype=int32)
    all_bins_model = np.zeros(num_pairs, dtype=int32)
    all_i = np.zeros(num_pairs, dtype=int32)
    all_j = np.zeros(num_pairs, dtype=int32)
    k = 0
    for i in range(z1.size):
        if weights1[i] == 0:
            continue
        for j in range(z2.size):
            if weights2[j] == 0:
                continue
            r_par = (r_comov1[i] - r_comov2[j]) * np.cos(ang[j] / 2)
            r_trans = (dist_m1[i] + dist_m2[j]) * np.sin(ang[j] / 2)
            if (r_par >= r_par_max or r_trans >= r_trans_max or
                    r_par < r_par_min):
                continue

            weights12 = weights1[i] * weights2[j]
            z = (z1[i] + z2[j]) / 2

            bins_r_par = np.floor(
                (r_par - r_par_min) / (r_par_max - r_par_min) * num_bins_r_par)
            bins_r_trans = np.floor(r_trans / r_trans_max * num_bins_r_trans)
            bins = int32(bins_r_trans + num_bins_r_trans * bins_r_par)
            model_bins_r_par = np.floor(
                (r_par - r_par_min) / (r_par_max - r_par_min) *
                num_model_bins_r_par)
            model_bins_r_trans = np.floor(r_trans / r_trans_max *
                                          num_model_bins_r_trans)
            model_bins = int32(model_bins_r_trans +
                               num_model_bins_r_trans * model_bins_r_par)

            #-- This will be used later to fill the distortion matrix
            all_bins_model[k] = model_bins
            all_bins[k] = bins
            all_i[k] = i
            all_j[k] = j
            k += 1

            #-- Fill effective quantities (r_par, r_trans, z_eff, weight_eff)
            r_par_eff[model_bins] += weights12 * r_par
            r_trans_eff[model_bins] += weights12 * r_trans
            z_eff[model_bins] += weights12 * z
            weight_eff[model_bins] += weights12
            weights_dmat[bins] += weights12

            # Combining equation 21 and equation 6 of du Mas des Bourboux et al. 2020
            # we find an equation with 9 terms comming from the product of two eta
            # The variables below stand for 8 of these 9 terms (the first one is
            # pretty trivial)

            # first eta, second term: weight/sum(weights)
            # second eta, first term: kronecker delta
            eta2[j + num_pixels2 * model_bins] += weights1[i] / sum_weights1

            if order1 == 1:
                # first eta, third term: (non-zero only for order=1)
                #   weight*(Lambda-bar(Lambda))*(Lambda-bar(Lambda))/
                #   sum(weight*(Lambda-bar(Lambda)**2))
                # second eta, first term: kronecker delta
                eta4[j + num_pixels2 * model_bins] += (
                    weights1[i] * log_lambda_minus_mean1[i] /
                    sum_weights_square_log_lambda_minus_mean1)

    # Now add all the contributions together
    unique_bins_model = np.unique(all_bins_model)
    for pair in range(num_pairs):
        i = all_i[pair]
        j = all_j[pair]
        bins = all_bins[pair]
        model_bins = all_bins_model[pair]
        weights12 = weights1[i] * weights2[j]
        # first eta, first term: kronecker delta
        # second eta, first term: kronecker delta
        dmat_bin = model_bins + num_model_bins_r_par * num_model_bins_r_trans * bins
        dmat[dmat_bin] += weights12
        # rest of the terms
        for k in unique_bins_model:
            dmat_bin = k + num_model_bins_r_par * num_model_bins_r_trans * bins
            dmat[dmat_bin] += (
                weights12 *
                (-eta2[j + num_pixels2 * k] -
                 eta4[j + num_pixels2 * k] * log_lambda_minus_mean1[i]))


def compute_metal_dmat(healpixs, abs_igm="SiII(1526)"):
    """Computes the metal distortion matrix for each of the healpixs.

    Args:
        healpixs: array of ints
            List of healpix numbers
        abs_igm: string - default: "SiII(1526)"
            Name of the absorption in picca.constants defining the
            redshift of the forest pixels

    Returns:
        The following variables:
            weights_dmat: Total weight in the distortion matrix pixels
            dmat: The distortion matrix
            r_par_eff: Effective parallel distance of the distortion matrix
                pixels
            r_trans_eff: Effective transverse distance of the distortion matrix
                pixels
            z_eff: Effective redshift of the distortion matrix pixels
            weight_eff: Effective weight of the distortion matrix pixels
            num_pairs: Total number of pairs
            num_pairs_used: Number of used pairs
    """
    dmat = np.zeros(num_bins_r_par * num_bins_r_trans * num_model_bins_r_trans *
                    num_model_bins_r_par)
    weights_dmat = np.zeros(num_bins_r_par * num_bins_r_trans)
    r_par_eff = np.zeros(num_model_bins_r_trans * num_model_bins_r_par)
    r_trans_eff = np.zeros(num_model_bins_r_trans * num_model_bins_r_par)
    z_eff = np.zeros(num_model_bins_r_trans * num_model_bins_r_par)
    weight_eff = np.zeros(num_model_bins_r_trans * num_model_bins_r_par)

    num_pairs = 0
    num_pairs_used = 0
    for healpix in healpixs:
        for delta1 in data[healpix]:
            with lock:
                dmatcounter = round(counter.value * 100. / num_data, 2)
                if (counter.value % 1000 == 0):
                    userprint(("computing metal dmat {}:"
                               " {}%").format(abs_igm, dmatcounter))
                counter.value += 1

            r_comov1 = delta1.r_comov
            dist_m1 = delta1.dist_m
            weights1 = delta1.weights
            z1_abs1 = 10**delta1.log_lambda / constants.ABSORBER_IGM[abs_igm] - 1
            r_comov1_abs1 = cosmo.get_r_comov(z1_abs1)
            dist_m1_abs1 = cosmo.get_dist_m(z1_abs1)

            # filter cases where the absorption from the metal is
            # inconsistent with the quasar redshift
            w = z1_abs1 < delta1.z_qso
            r_comov1 = r_comov1[w]
            dist_m1 = dist_m1[w]
            weights1 = weights1[w]
            z1_abs1 = z1_abs1[w]
            r_comov1_abs1 = r_comov1_abs1[w]
            dist_m1_abs1 = dist_m1_abs1[w]
            if r_comov1.size == 0:
                continue

            w = np.random.rand(len(delta1.neighbours)) > reject
            num_pairs += len(delta1.neighbours)
            num_pairs_used += w.sum()

            for obj2 in np.array(delta1.neighbours)[w]:
                ang = delta1.get_angle_between(obj2)

                r_comov2 = obj2.r_comov
                dist_m2 = obj2.dist_m
                weights2 = obj2.weights
                z2 = obj2.z_qso

                # compute bins the pairs contribute to
                r_par = (r_comov1 - r_comov2) * np.cos(ang / 2)
                r_trans = (dist_m1 + dist_m2) * np.sin(ang / 2)
                weights12 = weights1 * weights2

                w = ((r_par > r_par_min) & (r_par < r_par_max) &
                     (r_trans < r_trans_max))
                bins_r_par = ((r_par - r_par_min) / (r_par_max - r_par_min) *
                              num_bins_r_par).astype(int)
                bins_r_trans = (r_trans / r_trans_max *
                                num_bins_r_trans).astype(int)
                bins = bins_r_trans + num_bins_r_trans * bins_r_par
                rebin = np.bincount(bins[w], weights=weights12[w])
                weights_dmat[:len(rebin)] += rebin

                r_par_abs = (r_comov1_abs1 - r_comov2) * np.cos(ang / 2)
                r_trans_abs = (dist_m1_abs1 + dist_m2) * np.sin(ang / 2)
                z_weight_evol = (((1. + z1_abs1) /
                                  (1. + z_ref))**(alpha_abs[abs_igm] - 1.))

                model_bins_r_par = ((r_par_abs - r_par_min) /
                                    (r_par_max - r_par_min) *
                                    num_model_bins_r_par).astype(int)
                model_bins_r_trans = (r_trans_abs / r_trans_max *
                                      num_model_bins_r_trans).astype(int)
                model_bins = (model_bins_r_trans +
                              num_model_bins_r_trans * model_bins_r_par)
                w &= ((r_par_abs > r_par_min) & (r_par_abs < r_par_max) &
                      (r_trans_abs < r_trans_max))

                rebin = np.bincount(
                    (model_bins[w] +
                     num_model_bins_r_par * num_model_bins_r_trans * bins[w]),
                    weights=weights12[w] * z_weight_evol[w])
                dmat[:len(rebin)] += rebin

                rebin = np.bincount(model_bins[w],
                                    weights=(r_par_abs[w] * weights12[w] *
                                             z_weight_evol[w]))
                r_par_eff[:len(rebin)] += rebin
                rebin = np.bincount(model_bins[w],
                                    weights=(r_trans_abs[w] * weights12[w] *
                                             z_weight_evol[w]))
                r_trans_eff[:len(rebin)] += rebin
                rebin = np.bincount(model_bins[w],
                                    weights=((z1_abs1 + z2)[w] / 2 *
                                             weights12[w] * z_weight_evol[w]))
                z_eff[:len(rebin)] += rebin
                rebin = np.bincount(model_bins[w],
                                    weights=weights12[w] * z_weight_evol[w])
                weight_eff[:len(rebin)] += rebin
            setattr(delta1, "neighbours", None)

    dmat = dmat.reshape(num_bins_r_par * num_bins_r_trans,
                        num_model_bins_r_par * num_model_bins_r_trans)

    return (weights_dmat, dmat, r_par_eff, r_trans_eff, z_eff, weight_eff,
            num_pairs, num_pairs_used)


def compute_wick_terms(healpixs):
    """
    Computes the Wick expansion terms of the covariance matrix for the object-
    pixel cross-correlation

    Each of the terms represents the contribution of different type of pairs as
    illustrated in figure A.1 from Delubac et al. 2015

    Args:
        healpixs: array of ints
            List of healpix numbers

    Returns:
        The following variables:
            weights_wick: Total weight in the covariance matrix pixels from all
                terms in the Wick expansion
            num_pairs_wick: Number of pairs in the covariance matrix pixels
            num_pairs: Total number of pairs
            num_pairs_used: Total number of used pairs
            t1: Wick expansion, term 1
            t2: Wick expansion, term 2
            t3: Wick expansion, term 3
            t4: Wick expansion, term 4
            t5: Wick expansion, term 5
            t6: Wick expansion, term 6
    """
    t1 = np.zeros(
        (num_bins_r_par * num_bins_r_trans, num_bins_r_par * num_bins_r_trans))
    t2 = np.zeros(
        (num_bins_r_par * num_bins_r_trans, num_bins_r_par * num_bins_r_trans))
    t3 = np.zeros(
        (num_bins_r_par * num_bins_r_trans, num_bins_r_par * num_bins_r_trans))
    t4 = np.zeros(
        (num_bins_r_par * num_bins_r_trans, num_bins_r_par * num_bins_r_trans))
    t5 = np.zeros(
        (num_bins_r_par * num_bins_r_trans, num_bins_r_par * num_bins_r_trans))
    t6 = np.zeros(
        (num_bins_r_par * num_bins_r_trans, num_bins_r_par * num_bins_r_trans))
    weights_wick = np.zeros(num_bins_r_par * num_bins_r_trans)
    num_pairs_wick = np.zeros(num_bins_r_par * num_bins_r_trans, dtype=np.int64)
    num_pairs = 0
    num_pairs_used = 0

    for healpix in healpixs:

        num_pairs += len(data[healpix])
        w = np.random.rand(len(data[healpix])) > reject
        num_pairs_used += w.sum()
        if w.sum() == 0:
            continue

        for delta1 in [
                delta for index, delta in enumerate(data[healpix]) if w[index]
        ]:
            with lock:
                xicounter = round(
                    (counter.value * 100. / num_data / (1. - reject)), 2)
                if (counter.value % 1000 == 0):
                    userprint(("computing computing xi: "
                               " {}%").format(xicounter))
                counter.value += 1

            if delta1.neighbours.size == 0:
                continue

            variance_1d = get_variance_1d[delta1.fname](delta1.log_lambda)
            weights1 = delta1.weights
            weighted_xi_1d_1 = (
                (weights1 * weights1[:, None]) * xi_1d[delta1.fname](
                    abs(delta1.log_lambda - delta1.log_lambda[:, None])) *
                np.sqrt(variance_1d * variance_1d[:, None]))
            r_comov1 = delta1.r_comov
            z1 = delta1.z

            neighbours = delta1.neighbours
            ang12 = delta1.get_angle_between(neighbours)
            r_comov2 = np.array([obj2.r_comov for obj2 in neighbours])
            z2 = np.array([obj2.z_qso for obj2 in neighbours])
            weights2 = np.array([obj2.weights for obj2 in neighbours])

            compute_wickT1234_pairs(ang12, r_comov1, r_comov2, z1, z2, weights1,
                                    weights2, weighted_xi_1d_1, weights_wick,
                                    num_pairs_wick, t1, t2, t3, t4)

            ### Higher order diagrams
            if (xi_wick is None) or (max_diagram <= 4):
                continue
            thingid2 = np.array([obj2.thingid for obj2 in neighbours])
            for delta3 in np.array(delta1.dneighs):
                if delta3.neighbours.size == 0:
                    continue

                ang13 = delta1.get_angle_between(delta3)

                r_comov3 = delta3.r_comov
                weights3 = delta3.weights

                neighbours = delta3.neighbours
                ang34 = delta3.get_angle_between(neighbours)
                r_comov4 = np.array([obj4.r_comov for obj4 in neighbours])
                weights4 = np.array([obj4.weights for obj4 in neighbours])
                thingid4 = np.array([obj4.thingid for obj4 in neighbours])

                if max_diagram == 5:
                    w = np.in1d(delta1.neighbours, delta3.neighbours)
                    if w.sum() == 0:
                        continue
                    aux_ang12 = ang12[w]
                    aux_r_comov2 = r_comov2[w]
                    aux_weights2 = weights2[w]
                    aux_thingid2 = thingid2[w]

                    w = np.in1d(delta3.neighbours, delta1.neighbours)
                    if w.sum() == 0:
                        continue
                    ang34 = ang34[w]
                    r_comov4 = r_comov4[w]
                    weights4 = weights4[w]
                    thingid4 = thingid4[w]

                compute_wickT56_pairs(aux_ang12, ang34, ang13, r_comov1,
                                      aux_r_comov2, r_comov3, r_comov4,
                                      weights1, aux_weights2, weights3,
                                      weights4, aux_thingid2, thingid4, t5, t6)

    return weights_wick, num_pairs_wick, num_pairs, num_pairs_used, t1, t2, t3, t4, t5, t6


#@jit   #this will be removed in the future, it's an older implementation that does not work properly with current numba
def compute_wickT1234_pairs_slow(ang, r_comov1, r_comov2, z1, z2, weights1,
                                 weights2, weighted_xi_1d_1, weights_wick,
                                 num_pairs_wick, t1, t2, t3, t4):
    """
    Computes the Wick expansion terms 1, 2, and 3 of a given pair of forests

    Each of the terms represents the contribution of different type of pairs as
    illustrated in figure A.1 from Delubac et al. 2015

    Args:
        ang: array of floats
            Angular separation between pixels in forests 1 and 2
        r_comov1: array of floats
            Comoving distance (in Mpc/h) for forest 1
        r_comov2: array of floats
            Comoving distance (in Mpc/h) for forest 2
        z1: array of floats
            Redshifts for forest 1
        z2: array of floats
            Redshifts for forest 2
        weights1: array of floats
            Weights for forest 1
        weights2: array of floats
            Weights for forest 2
        weighted_xi_1d_1: array of floats
            Weighted 1D correlation function for forest 1
        weights_wick: array of floats
            Total weight in the covariance matrix pixels
        num_pairs_wick: array of floats
            Total number of pairs in the covariance matrix pixels
        t1: array of floats
            Wick expansion, term 1
        t2: array of floats
            Wick expansion, term 2
        t3: array of floats
            Wick expansion, term 3
        t4: array of floats
            Wick expansion, term 4
    """
    r_par = (r_comov1[:, None] - r_comov2) * np.cos(ang / 2.)
    r_trans = (r_comov1[:, None] + r_comov2) * np.sin(ang / 2.)
    z_weight_evol1 = ((1. + z1) / (1. + z_ref))**(alpha - 1.)
    z_weight_evol2 = ((1. + z2) / (1. + z_ref))**(alpha_obj - 1.)
    weights12 = weights1[:, None] * weights2
    weight1 = weights1[:, None] * np.ones(len(r_comov2))
    index_delta = np.arange(r_comov1.size)[:, None] * np.ones(len(r_comov2),
                                                              dtype='int')
    index_obj = (np.ones(r_comov1.size, dtype='int')[:, None] *
                 np.arange(len(r_comov2)))

    bins_r_par = ((r_par - r_par_min) / (r_par_max - r_par_min) *
                  num_bins_r_par).astype(int)
    bins_r_trans = (r_trans / r_trans_max * num_bins_r_trans).astype(int)
    bins_forest = bins_r_trans + num_bins_r_trans * bins_r_par

    w = (r_par > r_par_min) & (r_par < r_par_max) & (r_trans < r_trans_max)
    if w.sum() == 0:
        return

    bins_forest = bins_forest[w]
    weights12 = weights12[w]
    weight1 = weight1[w]
    index_delta = index_delta[w]
    index_obj = index_obj[w]

    for index1 in range(bins_forest.size):
        p1 = bins_forest[index1]
        i1 = index_delta[index1]
        j1 = index_obj[index1]
        weights_wick[p1] += weights12[index1]
        num_pairs_wick[p1] += 1
        t1[p1,
           p1] += weights12[index1]**2 / weight1[index1] * z_weight_evol1[i1]

        for index2 in range(index1 + 1, bins_forest.size):
            p2 = bins_forest[index2]
            i2 = index_delta[index2]
            j2 = index_obj[index2]
            if j1 == j2:
                prod = weighted_xi_1d_1[i1, i2] * (z_weight_evol2[j1]**2)
                t2[p1, p2] += prod
                t2[p2, p1] += prod
            elif i1 == i2:
                prod = (weights12[index1] * weights12[index2] /
                        weight1[index1] * z_weight_evol1[i1])
                t3[p1, p2] += prod
                t3[p2, p1] += prod
            else:
                prod = (weighted_xi_1d_1[i1, i2] * z_weight_evol2[j1] *
                        z_weight_evol2[j2])
                t4[p1, p2] += prod
                t4[p2, p1] += prod

    return


@jit
def compute_wickT56_pairs(ang12, ang34, ang13, r_comov1, r_comov2, r_comov3,
                          r_comov4, weights1, weights2, weights3, weights4,
                          thingid2, thingid4, t5, t6):
    """
    Compute the Wick covariance matrix for the object-pixel cross-correlation
    for the T5 and T6 diagrams: i.e. the contribution of the 3D auto-correlation
    to the covariance matrix

    Each of the terms represents the contribution of different type of pairs as
    illustrated in figure A.1 from Delubac et al. 2015

    Args:
        ang12: array of floats
            Angular separation between pixels in forests 1 and object 2
        ang34: array of floats
            Angular separation between pixels in forests 3 and object 4
        ang13: array of floats
            Angular separation between pixels in object 2 and 3
        r_comov1: array of floats
            Comoving distance (in Mpc/h) for forest 1
        r_comov2: array of floats
            Comoving distance (in Mpc/h) for object 2
        r_comov3: array of floats
            Comoving distance (in Mpc/h) for forest 3
        r_comov4: array of floats
            Comoving distance (in Mpc/h) for object 4
        weights1: array of floats
            Weights for forest 1
        weights2: array of floats
            Weights for object 2
        weights3: array of floats
            Weights for forest 3
        weights4: array of floats
            Weights for object 4
        thingid2: array of ints
            ThingID of the observation for object 2
        thingid4: array of ints
            ThingID of the observation for object 4
        t4: array of floats
            Wick expansion, term 4
        t5: array of floats
            Wick expansion, term 5
    """
    ### Pair forest_1 - forest_3
    r_par = np.absolute(r_comov1 - r_comov3[:, None]) * np.cos(ang13 / 2.)
    r_trans = (r_comov1 + r_comov3[:, None]) * np.sin(ang13 / 2.)

    w = (r_par < r_par_max) & (r_trans < r_trans_max) & (r_par >= r_par_min)
    if w.sum() == 0:
        return
    bins_r_par = np.floor((r_par - r_par_min) / (r_par_max - r_par_min) *
                          num_bins_r_par).astype(int)
    bins_r_trans = (r_trans / r_trans_max * num_bins_r_trans).astype(int)
    bins_forest13 = bins_r_trans + num_bins_r_trans * bins_r_par
    bins_forest13[~w] = 0
    xi13 = xi_wick[bins_forest13]
    xi13[~w] = 0.

    ### Pair forest_1 - object_2
    r_par = (r_comov1[:, None] - r_comov2) * np.cos(ang12 / 2.)
    r_trans = (r_comov1[:, None] + r_comov2) * np.sin(ang12 / 2.)
    weights12 = weights1[:, None] * weights2
    bins12 = (np.arange(r_comov1.size)[:, None] *
              np.ones_like(r_comov2)).astype(int)
    thingid_wick12 = np.ones_like(weights1[:, None]).astype(int) * thingid2

    w = (r_par > r_par_min) & (r_par < r_par_max) & (r_trans < r_trans_max)
    if w.sum() == 0:
        return
    r_par = r_par[w]
    r_trans = r_trans[w]
    weights12 = weights12[w]
    bins12 = bins12[w]
    thingid_wick12 = thingid_wick12[w]
    bins_r_par = ((r_par - r_par_min) / (r_par_max - r_par_min) *
                  num_bins_r_par).astype(int)
    bins_r_trans = (r_trans / r_trans_max * num_bins_r_trans).astype(int)
    bins_forest12 = bins_r_trans + num_bins_r_trans * bins_r_par

    ### Pair forest_3 - object_4
    r_par = (r_comov3[:, None] - r_comov4) * np.cos(ang34 / 2.)
    r_trans = (r_comov3[:, None] + r_comov4) * np.sin(ang34 / 2.)
    weights34 = weights3[:, None] * weights4
    bins34 = (np.arange(r_comov3.size)[:, None] *
              np.ones_like(r_comov4)).astype(int)
    thingid_wick34 = np.ones_like(weights3[:, None]).astype(int) * thingid4

    w = (r_par > r_par_min) & (r_par < r_par_max) & (r_trans < r_trans_max)
    if w.sum() == 0:
        return
    r_par = r_par[w]
    r_trans = r_trans[w]
    weights34 = weights34[w]
    bins34 = bins34[w]
    thingid_wick34 = thingid_wick34[w]
    bins_r_par = ((r_par - r_par_min) / (r_par_max - r_par_min) *
                  num_bins_r_par).astype(int)
    bins_r_trans = (r_trans / r_trans_max * num_bins_r_trans).astype(int)
    bins_forest34 = bins_r_trans + num_bins_r_trans * bins_r_par

    ### t5
    for index1, p1 in enumerate(bins_forest12):
        selected_bin12 = bins12[index1]
        weight1 = weights12[index1]

        w = thingid_wick34 == thingid_wick12[index1]
        for index2, p2 in enumerate(bins_forest34[w]):
            selected_bin34 = bins34[w][index2]
            weight2 = weights34[w][index2]
            prod = xi13[selected_bin34, selected_bin12] * weight1 * weight2
            t5[p1, p2] += prod
            t5[p2, p1] += prod

    ### t6
    if max_diagram == 5:
        return
    for index1, p1 in enumerate(bins_forest12):
        selected_bin12 = bins12[index1]
        weight1 = weights12[index1]

        for index2, p2 in enumerate(bins_forest34):
            if thingid_wick34[index2] == thingid_wick12[index1]:
                continue
            selected_bin34 = bins34[index2]
            weight2 = weights34[index2]
            prod = xi13[selected_bin34, selected_bin12] * weight1 * weight2
            t6[p1, p2] += prod
            t6[p2, p1] += prod


def compute_xi_1d(healpixs):
    """Computes the 1D autocorrelation delta and objects on the same LOS

    Args:
        healpixs: array of ints
            List of healpix numbers

    Returns:
        The following variables:
            weights1d: Total weights for the 1d correlation function
            xi1d: The 1d correlation function
            r_par1d: The wavelength ratios
            z1d: Mean redshift of pairs
            num_pairs1d: Number of pairs for the 1d correlation function
    """
    xi_1d = np.zeros(num_bins_r_par)
    weights1d = np.zeros(num_bins_r_par)
    r_par1d = np.zeros(num_bins_r_par)
    z1d = np.zeros(num_bins_r_par)
    num_pairs1d = np.zeros(num_bins_r_par, dtype=np.int64)

    for healpix in healpixs:
        for delta in data[healpix]:

            neighbours = [
                obj for obj in objs[healpix] if obj.thingid == delta.thingid
            ]
            if len(neighbours) == 0:
                continue

            z_qso = [obj.z_qso for obj in neighbours]
            weights_qso = [obj.weights for obj in neighbours]
            lambda_qso = [10.**obj.log_lambda for obj in neighbours]
            ang = np.zeros(len(lambda_qso))

            (rebin_weight, rebin_xi, rebin_r_par, _, rebin_z,
             rebin_num_pairs) = compute_xi_forest_pairs_fast(
                 delta.z, 10.**delta.log_lambda, 10.**delta.log_lambda,
                 delta.weights, delta.delta, z_qso, lambda_qso, lambda_qso,
                 weights_qso, ang)

            xi_1d[:rebin_xi.size] += rebin_xi
            weights1d[:rebin_weight.size] += rebin_weight
            r_par1d[:rebin_r_par.size] += rebin_r_par
            z1d[:rebin_z.size] += rebin_z
            num_pairs1d[:rebin_num_pairs.size] += rebin_num_pairs.astype(int)

    w = weights1d > 0.
    xi_1d[w] /= weights1d[w]
    r_par1d[w] /= weights1d[w]
    z[w] /= weights1d[w]

    return weights1d, xi_1d, r_par1d, z, num_pairs1d


@jit(nopython=True)
def compute_wickT1234_pairs(ang, r_comov1, r_comov2, z1, z2, weights1, weights2,
                            weighted_xi_1d_1, weights_wick, num_pairs_wick, t1,
                            t2, t3, t4):
    """
    Computes the Wick expansion terms 1, 2, and 3 of a given pair of forests

    Each of the terms represents the contribution of different type of pairs as
    illustrated in figure A.1 from Delubac et al. 2015

    Args:
        ang: array of floats
            Angular separation between pixels in forests 1 and 2
        r_comov1: array of floats
            Comoving distance (in Mpc/h) for forest 1
        r_comov2: array of floats
            Comoving distance (in Mpc/h) for forest 2
        z1: array of floats
            Redshifts for forest 1
        z2: array of floats
            Redshifts for forest 2
        weights1: array of floats
            Weights for forest 1
        weights2: array of floats
            Weights for forest 2
        weighted_xi_1d_1: array of floats
            Weighted 1D correlation function for forest 1
        weights_wick: array of floats
            Total weight in the covariance matrix pixels
        num_pairs_wick: array of floats
            Total number of pairs in the covariance matrix pixels
        t1: array of floats
            Wick expansion, term 1
        t2: array of floats
            Wick expansion, term 2
        t3: array of floats
            Wick expansion, term 3
        t4: array of floats
            Wick expansion, term 4
    """
    num_pixels1 = len(r_comov1)
    num_pixels2 = len(r_comov2)
    i1 = np.arange(num_pixels1)
    i2 = np.arange(num_pixels2)
    z_weight_evol1 = ((1 + z1) / (1 + z_ref))**(alpha - 1)
    z_weight_evol2 = ((1 + z2) / (1 + z_ref))**(alpha_obj - 1)
    w = np.zeros((num_pixels1, num_pixels2))

    wsum = 0
    for ind2 in i2:  #first figure out how many elements there are
        for ind1 in i1:
            r_par = (r_comov1[ind1] - r_comov2[ind2]) * np.cos(ang[ind2] / 2)
            r_trans = (r_comov1[ind1] + r_comov2[ind2]) * np.sin(ang[ind2] / 2)
            w[ind1,
              ind2] = (r_par < r_par_max) & (r_trans <
                                             r_trans_max) & (r_par >= r_par_min)
            if w[ind1, ind2] > 0:
                wsum += 1
    if wsum == 0:
        return

    bins = np.zeros((wsum), dtype=np.int64)
    bins_forest = np.zeros((wsum), dtype=np.int64)
    weights12 = np.zeros((wsum))
    weight1 = np.zeros((wsum))
    index_obj = np.zeros((wsum), dtype=np.int64)
    index_delta = np.zeros((wsum), dtype=np.int64)
    ind = 0
    for ind2 in i2:
        for ind1 in i1:
            if w[ind1, ind2] == 0:
                continue
            bins[ind] = ind1 + num_pixels1 * ind2
            r_par = (r_comov1[ind1] - r_comov2[ind2]) * np.cos(ang[ind2] / 2)
            r_trans = (r_comov1[ind1] + r_comov2[ind2]) * np.sin(ang[ind2] / 2)
            bin_r_par = int(
                (r_par - r_par_min) / (r_par_max - r_par_min) * num_bins_r_par)
            bin_r_trans = int(r_trans / r_trans_max * num_bins_r_trans)
            bins_forest[ind] = bin_r_trans + num_bins_r_trans * bin_r_par
            weights12[ind] = weights1[ind1] * weights2[ind2]
            weight1[ind] = weights1[ind1]
            index_delta[ind] = ind1
            index_obj[ind] = ind2
            ind += 1

    for index1 in range(bins_forest.size):

        p1 = bins_forest[index1]
        i1 = index_delta[index1]
        j1 = index_obj[index1]
        weights_wick[p1] += weights12[index1]
        num_pairs_wick[p1] += 1
        t1[p1,
           p1] += weights12[index1]**2 / weight1[index1] * z_weight_evol1[i1]

        for index2 in range(index1 + 1, bins_forest.size):
            p2 = bins_forest[index2]
            i2 = index_delta[index2]
            j2 = index_obj[index2]
            if j1 == j2:
                prod = weighted_xi_1d_1[i1, i2] * (z_weight_evol2[j1]**2)
                t2[p1, p2] += prod
                t2[p2, p1] += prod
            elif i1 == i2:
                prod = (weights12[index1] * weights12[index2] /
                        weight1[index1] * z_weight_evol1[i1])
                t3[p1, p2] += prod
                t3[p2, p1] += prod
            else:
                prod = (weighted_xi_1d_1[i1, i2] * z_weight_evol2[j1] *
                        z_weight_evol2[j2])
                t4[p1, p2] += prod
                t4[p2, p1] += prod

    return
