"""This module defines functions and variables required for the correlation
analysis of two delta fields

This module provides several functions:
    - fill_neighs
    - fill_neighs_x_correlation
    - compute_xi
    - compute_xi_forest_pairs
    - compute_dmat
    - compute_dmat_forest_pairs
    - metal_dmat
    - cf1d
    - x_forest_cf1d
    - wickT
    - fill_wickT123
    - fill_wickT45
See the respective docstrings for more details
"""
import numpy as np
import scipy as sp
from healpy import query_disc
from numba import jit

from picca import constants
from picca.utils import userprint

num_bins_r_par = None
num_bins_r_trans = None
num_model_bins_r_trans = None
num_model_bins_r_par = None
r_par_max = None
r_par_min = None
z_cut_max = None
z_cut_min = None
r_trans_max = None
ang_max = None
nside = None

counter = None
num_data = None
num_data2 = None

z_ref = None
alpha = None
alpha2 = None
alpha_abs = None
lambda_abs = None
lambda_abs2 = None

data = None
data2 = None

cosmo = None

reject = None
lock = None
x_correlation = False
ang_correlation = False
remove_same_half_plate_close_pairs = False


def fill_neighs(healpixs):
    """Create and store a list of neighbours for each of the healpix.

    Neighbours

    Args:
        healpixs: array of ints
            List of healpix numbers
    """
    for healpix in healpixs:
        for delta in data[healpix]:
            healpix_neighbours = query_disc(nside,
                                            [delta.x_cart,
                                             delta.y_cart,
                                             delta.z_cart],
                                            ang_max,
                                            inclusive=True)
            if x_correlation:
                healpix_neighbours = [other_healpix
                                      for other_healpix in healpix_neighbours
                                      if other_healpix in data2]
                neighbours = [other_delta
                              for other_healpix in healpix_neighbours
                              for other_delta in data2[other_healpix]
                              if delta.thingid != other_delta.thingid]
            else:
                healpix_neighbours = [other_healpix
                                      for other_healpix in healpix_neighbours
                                      if other_healpix in data]
                neighbours = [other_delta
                              for other_healpix in healpix_neighbours
                              for other_delta in data[other_healpix]
                              if delta.thingid != other_delta.thingid]
            ang = delta^neighbours
            w = ang < ang_max
            neighbours = np.array(neighbours)[w]
            if x_correlation:
                delta.neighbours = [
                    other_delta
                    for other_delta in neighbours
                    if ((other_delta.z[-1] + delta.z[-1])/2. >= z_cut_min and
                        (other_delta.z[-1] + delta.z[-1])/2. < z_cut_max)
                ]
            else:
                delta.neighbours = [
                    other_delta
                    for other_delta in neighbours
                    if (delta.ra > other_delta.ra and
                        (other_delta.z[-1] + delta.z[-1])/2. >= z_cut_min and
                        (other_delta.z[-1] + delta.z[-1])/2. < z_cut_max)
                ]

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
    xi = np.zeros(num_bins_r_par*num_bins_r_trans)
    weights = np.zeros(num_bins_r_par*num_bins_r_trans)
    r_par = np.zeros(num_bins_r_par*num_bins_r_trans)
    r_trans = np.zeros(num_bins_r_par*num_bins_r_trans)
    z = np.zeros(num_bins_r_par*num_bins_r_trans)
    num_pairs = np.zeros(num_bins_r_par*num_bins_r_trans, dtype=np.int64)

    for healpix in healpixs:
        for delta1 in data[healpix]:
            userprint(("\rcomputing xi: "
                       "{}%").format(round(counter.value*100./num_data, 2)),
                      end="")
            with lock:
                counter.value += 1
            for delta2 in delta1.neighbours:
                ang = delta1^delta2
                same_half_plate = ((delta1.plate == delta2.plate) and
                                   ((delta1.fiberid <= 500 and
                                     delta2.fiberid <= 500) or
                                    (delta1.fiberid > 500 and
                                     delta2.fiberid > 500)
                                   )
                                  )
                if ang_correlation:
                    (rebin_weight,
                     rebin_xi,
                     rebin_r_par,
                     rebin_r_trans,
                     rebin_z,
                     rebin_num_pairs) = compute_xi_forest_pairs(delta1.z,
                                                                10.**delta1.log_lambda,
                                                                10.**delta1.log_lambda,
                                                                delta1.weights,
                                                                delta1.delta,
                                                                delta2.z,
                                                                10.**delta2.
                                                                log_lambda,
                                                                10.**delta2.log_lambda,
                                                                delta2.weights,
                                                                delta2.delta,
                                                                ang,
                                                                same_half_plate)
                else:
                    (rebin_weight,
                     rebin_xi,
                     rebin_r_par,
                     rebin_r_trans,
                     rebin_z,
                     rebin_num_pairs) = compute_xi_forest_pairs(delta1.z,
                                                                delta1.r_comov,
                                                                delta1.dist_m,
                                                                delta1.weights,
                                                                delta1.delta,
                                                                delta2.z,
                                                                delta2.r_comov,
                                                                delta2.dist_m,
                                                                delta2.weights,
                                                                delta2.delta,
                                                                ang,
                                                                same_half_plate)

                xi[:len(rebin_xi)] += rebin_xi
                weights[:len(rebin_weight)] += rebin_weight
                r_par[:len(rebin_r_par)] += rebin_r_par
                r_trans[:len(rebin_r_trans)] += rebin_r_trans
                z[:len(rebin_z)] += rebin_z
                num_pairs[:len(rebin_num_pairs)] += rebin_num_pairs.astype(int)
            setattr(delta1, "neighbours", None)

    w = weights > 0
    xi[w] /= weights[w]
    r_par[w] /= weights[w]
    r_trans[w] /= weights[w]
    z[w] /= weights[w]
    return weights, xi, r_par, r_trans, z, num_pairs


@jit
def compute_xi_forest_pairs(z1, r_comov1, dist_m1, weights1, delta1, z2,
                            r_comov2, dist_m2, weights2, delta2, ang,
                            same_half_plate):
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
        delta2: array of float
            Delta field for forest 2
        ang: array of float
            Angular separation between pixels in forests 1 and 2
        same_half_plate: bool
            Flag to determine if the two forests are on the same half plate

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
    delta_times_weight1 = delta1*weights1
    delta_times_weight2 = delta2*weights2
    if ang_correlation:
        r_par = r_comov1/r_comov2[:, None]
        if not x_correlation:
            r_par[r_par < 1.] = 1./r_par[r_par < 1.]
        r_trans = ang*np.ones_like(r_par)
    else:
        r_par = (r_comov1 - r_comov2[:, None])*np.cos(ang/2)
        if not x_correlation:
            r_par = abs(r_par)
        r_trans = (dist_m1 + dist_m2[:, None])*np.sin(ang/2)
    delta_times_weight12 = delta_times_weight1*delta_times_weight2[:, None]
    weights12 = weights1*weights2[:, None]
    z = (z1 + z2[:, None])/2

    w = (r_par < r_par_max) & (r_trans < r_trans_max) & (r_par >= r_par_min)

    r_par = r_par[w]
    r_trans = r_trans[w]
    z = z[w]
    delta_times_weight12 = delta_times_weight12[w]
    weights12 = weights12[w]
    bins_r_par = np.floor((r_par - r_par_min)/
                          (r_par_max - r_par_min)*num_bins_r_par).astype(int)
    bins_r_trans = (r_trans/r_trans_max*num_bins_r_trans).astype(int)
    bins = bins_r_trans + num_bins_r_trans*bins_r_par

    if remove_same_half_plate_close_pairs and same_half_plate:
        w = abs(r_par) < (r_par_max - r_par_min)/num_bins_r_par
        delta_times_weight12[w] = 0.
        weights12[w] = 0.

    rebin_xi = np.bincount(bins, weights=delta_times_weight12)
    rebin_weight = np.bincount(bins, weights=weights12)
    rebin_r_par = np.bincount(bins, weights=r_par*weights12)
    rebin_r_trans = np.bincount(bins, weights=r_trans*weights12)
    rebin_z = np.bincount(bins, weights=z*weights12)
    rebin_num_pairs = np.bincount(bins, weights=(weights12 > 0.))

    return (rebin_weight, rebin_xi, rebin_r_par, rebin_r_trans, rebin_z,
            rebin_num_pairs)


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
    dmat = np.zeros(num_bins_r_par*num_bins_r_trans*
                    num_model_bins_r_trans*num_model_bins_r_par)
    weights_dmat = np.zeros(num_bins_r_par*num_bins_r_trans)
    r_par_eff = np.zeros(num_model_bins_r_trans*num_model_bins_r_par)
    r_trans_eff = np.zeros(num_model_bins_r_trans*num_model_bins_r_par)
    z_eff = np.zeros(num_model_bins_r_trans*num_model_bins_r_par)
    weight_eff = np.zeros(num_model_bins_r_trans*num_model_bins_r_par)

    num_pairs = 0
    num_pairs_used = 0
    for healpix in healpixs:
        for delta1 in data[healpix]:
            userprint(("\rcomputing xi: "
                       "{}%").format(round(counter.value*100./num_data, 3)),
                      end="")
            with lock:
                counter.value += 1
            order1 = delta1.order
            r_comov1 = delta1.r_comov
            dist_m1 = delta1.dist_m
            weights1 = delta1.weights
            log_lambda1 = delta1.log_lambda
            z1 = delta1.z
            w = np.random.rand(len(delta1.neighbours)) > reject
            num_pairs += len(delta1.neighbours)
            num_pairs_used += w.sum()
            for delta2 in np.array(delta1.neighbours)[w]:
                same_half_plate = ((delta1.plate == delta2.plate) and
                                   ((delta1.fiberid <= 500 and
                                     delta2.fiberid <= 500) or
                                    (delta1.fiberid > 500 and
                                     delta2.fiberid > 500)
                                    )
                                   )
                order2 = delta2.order
                ang = delta1^delta2
                r_comov2 = delta2.r_comov
                dist_m2 = delta2.dist_m
                weights2 = delta2.weights
                log_lambda2 = delta2.log_lambda
                z2 = delta2.z
                compute_dmat_forest_pairs(log_lambda1, log_lambda2, r_comov1,
                                          r_comov2, dist_m1, dist_m2, z1, z2,
                                          weights1, weights2, ang, weights_dmat,
                                          dmat, r_par_eff, r_trans_eff, z_eff,
                                          weight_eff, same_half_plate, order1,
                                          order2)
            setattr(delta1, "neighbours", None)

    dmat = dmat.reshape(num_bins_r_par*num_bins_r_trans,
                        num_model_bins_r_par*num_model_bins_r_trans)

    return (weights_dmat, dmat, r_par_eff, r_trans_eff, z_eff, weight_eff,
            num_pairs, num_pairs_used)
@jit
def compute_dmat_forest_pairs(log_lambda1, log_lambda2, r_comov1, r_comov2,
                              dist_m1, dist_m2, z1, z2, weights1, weights2, ang,
                              weights_dmat, dmat, r_par_eff, r_trans_eff, z_eff,
                              weight_eff, same_half_plate, order1, order2):
    """Computes the contribution of a given pair of forests to the distortion
    matrix.

    Args:
        log_lambda1: array of float
            Logarithm of the wavelength (in Angs) for forest 1
        log_lambda2: array of float
            Logarithm of the wavelength (in Angs) for forest 2
        r_comov1: array of floats
            Comoving distance (in Mpc/h) for forest 1
        r_comov2: array of floats
            Comoving distance (in Mpc/h) for forest 2
        dist_m1: array of floats
            Angular distance for forest 1
        dist_m2: array of floats
            Angular distance for forest 2
        z1: array of floats
            Redshifts for forest 1
        z2: array of floats
            Redshifts for forest 2
        weights1: array of floats
            Weights for forest 1
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
        same_half_plate: bool
            Flag to determine if the two forests are on the same half plate
        order1: 0 or 1
            Order of the log10(lambda) polynomial for the continuum fit in
            forest 1
        order 2: 0 or 1
            Order of the log10(lambda) polynomial for the continuum fit in
            forest 2
    """
    # find distances between pixels
    r_par = (r_comov1[:, None] - r_comov2)*np.cos(ang/2)
    if  not x_correlation:
        r_par = abs(r_par)
    r_trans = (dist_m1[:, None] + dist_m2)*np.sin(ang/2)
    z = (z1[:, None] + z2)/2.

    w = (r_par < r_par_max) & (r_trans < r_trans_max) & (r_par >= r_par_min)

    # locate bins pixels are contributing to (correlation bins)
    bins_r_par = np.floor((r_par - r_par_min)/(r_par_max - r_par_min)*
                          num_bins_r_par).astype(int)
    bins_r_trans = (r_trans/r_trans_max*num_bins_r_trans).astype(int)
    bins = bins_r_trans + num_bins_r_trans*bins_r_par
    bins = bins[w]

    # locate bins pixels are contributing to (model bins)
    model_bins_r_par = np.floor((r_par - r_par_min)/(r_par_max - r_par_min)*
                                num_model_bins_r_par).astype(int)
    model_bins_r_trans = (r_trans/r_trans_max*
                          num_model_bins_r_trans).astype(int)
    model_bins = model_bins_r_trans + num_model_bins_r_trans*model_bins_r_par
    model_bins = model_bins[w]

    # compute useful auxiliar variables to speed up computation of eta
    # (equation 6 of du Mas des Bourboux et al. 2020)

    # denominator second term in equation 6 of du Mas des Bourboux et al. 2020
    sum_weights1 = weights1.sum()
    sum_weights2 = weights2.sum()

    # mean of log_lambda
    mean_log_lambda1 = np.average(log_lambda1, weights=weights1)
    mean_log_lambda2 = np.average(log_lambda2, weights=weights2)

    # log_lambda minus its mean
    log_lambda_minus_mean1 = log_lambda1 - mean_log_lambda1
    log_lambda_minus_mean2 = log_lambda2 - mean_log_lambda2

    # denominator third term in equation 6 of du Mas des Bourboux et al. 2020
    sum_weights_square_log_lambda_minus_mean1 = (weights1*
                                                 log_lambda_minus_mean1**2).sum()
    sum_weights_square_log_lambda_minus_mean2 = (weights2*
                                                 log_lambda_minus_mean2**2).sum()

    # auxiliar variables to loop over distortion matrix bins
    num_pixels1 = len(log_lambda1)
    num_pixels2 = len(log_lambda2)
    ij = np.arange(num_pixels1)[:, None] + num_pixels1*np.arange(num_pixels2)
    ij = ij[w]

    weights12 = weights1[:, None]*weights2
    weights12 = weights12[w]

    if remove_same_half_plate_close_pairs and same_half_plate:
        weights12[abs(r_par[w]) < (r_par_max - r_par_min)/num_bins_r_par] = 0.

    rebin = np.bincount(model_bins, weights=weights12*r_par[w])
    r_par_eff[:rebin.size] += rebin
    rebin = np.bincount(model_bins, weights=weights12*r_trans[w])
    r_trans_eff[:rebin.size] += rebin
    rebin = np.bincount(model_bins, weights=weights12*z[w])
    z_eff[:rebin.size] += rebin
    rebin = np.bincount(model_bins, weights=weights12)
    weight_eff[:rebin.size] += rebin

    rebin = np.bincount(bins, weights=weights12)
    weights_dmat[:len(rebin)] += rebin

    # Combining equation 21 and equation 6 of du Mas des Bourboux et al. 2020
    # we find an equation with 9 terms comming from the product of two eta
    # The variables below stand for 8 of these 9 terms (the first one is
    # pretty trivial)

    # first eta, first term: kronecker delta
    # second eta, second term: weight/sum(weights)
    eta1 = np.zeros(num_model_bins_r_par*num_model_bins_r_trans*num_pixels1)

    # first eta, second term: weight/sum(weights)
    # second eta, first term: kronecker delta
    eta2 = np.zeros(num_model_bins_r_par*num_model_bins_r_trans*num_pixels2)

    # first eta, first term: kronecker delta
    # second eta, third term: (non-zero only for order=1)
    #   weight*(Lambda-bar(Lambda))*(Lambda-bar(Lambda))/
    #   sum(weight*(Lambda-bar(Lambda)**2))
    eta3 = np.zeros(num_model_bins_r_par*num_model_bins_r_trans*num_pixels1)

    # first eta, third term: (non-zero only for order=1)
    #   weight*(Lambda-bar(Lambda))*(Lambda-bar(Lambda))/
    #   sum(weight*(Lambda-bar(Lambda)**2))
    # second eta, first term: kronecker delta
    eta4 = np.zeros(num_model_bins_r_par*num_model_bins_r_trans*num_pixels2)

    # first eta, second term: weight/sum(weights)
    # second eta, second term: weight/sum(weights)
    eta5 = np.zeros(num_model_bins_r_par*num_model_bins_r_trans)

    # first eta, second term: weight/sum(weights)
    # second eta, third term: (non-zero only for order=1)
    #   weight*(Lambda-bar(Lambda))*(Lambda-bar(Lambda))/
    #   sum(weight*(Lambda-bar(Lambda)**2))
    eta6 = np.zeros(num_model_bins_r_par*num_model_bins_r_trans)

    # first eta, third term: (non-zero only for order=1)
    #   weight*(Lambda-bar(Lambda))*(Lambda-bar(Lambda))/
    #   sum(weight*(Lambda-bar(Lambda)**2))
    # second eta, second term: weight/sum(weights)
    eta7 = np.zeros(num_model_bins_r_par*num_model_bins_r_trans)

    # first eta, third term: (non-zero only for order=1)
    #   weight*(Lambda-bar(Lambda))*(Lambda-bar(Lambda))/
    #   sum(weight*(Lambda-bar(Lambda)**2))
    # second eta, third term: (non-zero only for order=1)
    #   weight*(Lambda-bar(Lambda))*(Lambda-bar(Lambda))/
    #   sum(weight*(Lambda-bar(Lambda)**2))
    eta8 = np.zeros(num_model_bins_r_par*num_model_bins_r_trans)

    # compute the contributions to the distortion matrix
    rebin = np.bincount(ij%num_pixels1 + num_pixels1*model_bins,
                        weights=((np.ones(num_pixels1)[:, None]*weights2)[w]/
                                 sum_weights2)
                       )
    eta1[:len(rebin)] += rebin
    rebin = np.bincount(((ij - ij%num_pixels1)//num_pixels1 +
                         num_pixels2*model_bins),
                        weights=((weights1[:, None]*np.ones(num_pixels2))[w]/
                                 sum_weights1)
                       )
    eta2[:len(rebin)] += rebin
    rebin = np.bincount(model_bins,
                        weights=((weights1[:, None]*weights2)[w]/sum_weights1/
                                 sum_weights2))
    eta5[:len(rebin)] += rebin

    if order2 == 1:
        rebin = np.bincount(ij%num_pixels1 + num_pixels1*model_bins,
                            weights=((np.ones(num_pixels1)[:, None]*weights2*
                                      log_lambda_minus_mean2)[w]/
                                     sum_weights_square_log_lambda_minus_mean2))
        eta3[:len(rebin)] += rebin
        rebin = np.bincount(model_bins,
                            weights=(((weights1[:, None]*
                                       (weights2*log_lambda_minus_mean2))[w]/
                                      sum_weights1/
                                      sum_weights_square_log_lambda_minus_mean2)))
        eta6[:len(rebin)] += rebin
    if order1 == 1:
        rebin = np.bincount(((ij - ij%num_pixels1)//num_pixels1 +
                             num_pixels2*model_bins),
                            weights=(((weights1*log_lambda_minus_mean1)[:, None]*
                                      np.ones(num_pixels2))[w]/
                                     sum_weights_square_log_lambda_minus_mean1))
        eta4[:len(rebin)] += rebin
        rebin = np.bincount(model_bins,
                            weights=(((weights1*log_lambda_minus_mean1)[:, None]*
                                      weights2)[w]/
                                     sum_weights_square_log_lambda_minus_mean1/
                                     sum_weights2))
        eta7[:len(rebin)] += rebin
        if order2 == 1:
            rebin = np.bincount(model_bins,
                                weights=(((weights1*log_lambda_minus_mean1)[:, None]*
                                          (weights2*log_lambda_minus_mean2))[w]/
                                         sum_weights_square_log_lambda_minus_mean1/
                                         sum_weights_square_log_lambda_minus_mean2))
            eta8[:len(rebin)] += rebin

    # Now add all the contributions together
    unique_model_bins = np.unique(model_bins)
    for index, (bin, model_bin) in enumerate(zip(bins, model_bins)):
        # first eta, first term: kronecker delta
        # second eta, first term: kronecker delta
        dmat[model_bin +
             num_model_bins_r_par*num_model_bins_r_trans*bin] += weights12[index]
        i = ij[index]%num_pixels1
        j = (ij[index]-i)//num_pixels1
        # rest of the terms
        for unique_model_bin in unique_model_bins:
            dmat[unique_model_bin + num_model_bins_r_par*num_model_bins_r_trans*
                 bin] += (weights12[index]*
                          (eta5[unique_model_bin] +
                           eta6[unique_model_bin]*log_lambda_minus_mean2[j] +
                           eta7[unique_model_bin]*log_lambda_minus_mean1[i] +
                           eta8[unique_model_bin]*log_lambda_minus_mean1[i]*
                           log_lambda_minus_mean2[j]) -
                          (weights12[index]*
                           (eta1[i+num_pixels1*unique_model_bin] +
                            eta2[j+num_pixels2*unique_model_bin] +
                            eta3[i+num_pixels1*unique_model_bin]*
                            log_lambda_minus_mean2[j] +
                            eta4[j+num_pixels2*unique_model_bin]*
                            log_lambda_minus_mean1[i]))
                         )

def metal_dmat(healpixs, igm_absorption1="LYA", igm_absorption2="SiIII(1207)"):
    """Computes the metal distortion matrix for each of the healpixs.

    Args:
        healpixs: array of ints
            List of healpix numbers
        igm_absorption1: string - default: "LYA"
            Name of the absorption in picca.constants defining the
            redshift of the forest pixels
        igm_absorption2: string - default: "SiIII(1207)"
            Name of the second absorption in picca.constants defining the
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
    dmat = np.zeros(num_bins_r_par*num_bins_r_trans*num_model_bins_r_trans*
                    num_model_bins_r_par)
    weights_dmat = np.zeros(num_bins_r_par*num_bins_r_trans)
    r_par_eff = np.zeros(num_model_bins_r_trans*num_model_bins_r_par)
    r_trans_eff = np.zeros(num_model_bins_r_trans*num_model_bins_r_par)
    z_eff = np.zeros(num_model_bins_r_trans*num_model_bins_r_par)
    weight_eff = np.zeros(num_model_bins_r_trans*num_model_bins_r_par)

    num_pairs = 0
    num_pairs_used = 0
    for healpix in healpixs:
        for delta1 in data[healpix]:
            userprint(("\rcomputing metal dmat {} {}: "
                       "{}%").format(igm_absorption1,
                                     igm_absorption2,
                                     round(counter.value*100./num_data, 3)),
                      end="")
            with lock:
                counter.value += 1

            w= np.random.rand(len(delta1.neighbours)) > reject
            num_pairs += len(delta1.neighbours)
            num_pairs_used += w.sum()
            for delta2 in np.array(delta1.neighbours)[w]:
                r_comov1 = delta1.r_comov
                dist_m1 = delta1.dist_m
                z1_abs1 = (10**delta1.log_lambda/
                           constants.ABSORBER_IGM[igm_absorption1] - 1)
                r1_abs1 = cosmo.get_r_comov(z1_abs1)
                rdm1_abs1 = cosmo.get_dist_m(z1_abs1)
                weights1 = delta1.weights

                wzcut = z1_abs1<delta1.z_qso
                r_comov1 = r_comov1[wzcut]
                dist_m1 = dist_m1[wzcut]
                weights1 = weights1[wzcut]
                r1_abs1 = r1_abs1[wzcut]
                rdm1_abs1 = rdm1_abs1[wzcut]
                z1_abs1 = z1_abs1[wzcut]

                same_half_plate = (delta1.plate == delta2.plate) and\
                        ( (delta1.fiberid<=500 and delta2.fiberid<=500) or (delta1.fiberid>500 and delta2.fiberid>500) )
                ang = delta1^delta2
                r_comov2 = delta2.r_comov
                dist_m2 = delta2.dist_m
                z2_abs2 = 10**delta2.log_lambda/constants.ABSORBER_IGM[igm_absorption2]-1
                r2_abs2 = cosmo.get_r_comov(z2_abs2)
                rdm2_abs2 = cosmo.get_dist_m(z2_abs2)
                weights2 = delta2.weights

                wzcut = z2_abs2<delta2.z_qso
                r_comov2 = r_comov2[wzcut]
                dist_m2 = dist_m2[wzcut]
                weights2 = weights2[wzcut]
                r2_abs2 = r2_abs2[wzcut]
                rdm2_abs2 = rdm2_abs2[wzcut]
                z2_abs2 = z2_abs2[wzcut]

                r_par = (r_comov1[:,None]-r_comov2)*sp.cos(ang/2)
                if not x_correlation:
                    r_par = abs(r_par)

                r_trans = (dist_m1[:,None]+dist_m2)*sp.sin(ang/2)
                w12 = weights1[:,None]*weights2

                bins_r_par = sp.floor((r_par-r_par_min)/(r_par_max-r_par_min)*num_bins_r_par).astype(int)
                bins_r_trans = (r_trans/r_trans_max*num_bins_r_trans).astype(int)

                if remove_same_half_plate_close_pairs and same_half_plate:
                    wp = abs(r_par) < (r_par_max-r_par_min)/num_bins_r_par
                    w12[wp] = 0.

                bA = bins_r_trans + num_bins_r_trans*bins_r_par
                wA = (bins_r_par<num_bins_r_par) & (bins_r_trans<num_bins_r_trans) & (bins_r_par >=0)
                c = sp.bincount(bA[wA],weights=w12[wA])
                weights_dmat[:len(c)]+=c

                rp_abs1_abs2 = (r1_abs1[:,None]-r2_abs2)*sp.cos(ang/2)

                if not x_correlation:
                    rp_abs1_abs2 = abs(rp_abs1_abs2)

                rt_abs1_abs2 = (rdm1_abs1[:,None]+rdm2_abs2)*sp.sin(ang/2)
                zwe12 = (1+z1_abs1[:,None])**(alpha_abs[igm_absorption1]-1)*(1+z2_abs2)**(alpha_abs[igm_absorption2]-1)/(1+z_ref)**(alpha_abs[igm_absorption1]+alpha_abs[igm_absorption2]-2)

                bp_abs1_abs2 = sp.floor((rp_abs1_abs2-r_par_min)/(r_par_max-r_par_min)*num_model_bins_r_par).astype(int)
                bt_abs1_abs2 = (rt_abs1_abs2/r_trans_max*num_model_bins_r_trans).astype(int)
                bBma = bt_abs1_abs2 + num_model_bins_r_trans*bp_abs1_abs2
                wBma = (bp_abs1_abs2<num_model_bins_r_par) & (bt_abs1_abs2<num_model_bins_r_trans) & (bp_abs1_abs2>=0)
                wAB = wA & wBma
                c = sp.bincount(bBma[wAB]+num_model_bins_r_par*num_model_bins_r_trans*bA[wAB],weights=w12[wAB]*zwe12[wAB])
                dm[:len(c)]+=c
                c = sp.bincount(bBma[wAB],weights=rp_abs1_abs2[wAB]*w12[wAB]*zwe12[wAB])
                r_par_eff[:len(c)]+=c
                c = sp.bincount(bBma[wAB],weights=rt_abs1_abs2[wAB]*w12[wAB]*zwe12[wAB])
                r_trans_eff[:len(c)]+=c
                c = sp.bincount(bBma[wAB],weights=(z1_abs1[:,None]+z2_abs2)[wAB]/2*w12[wAB]*zwe12[wAB])
                z_eff[:len(c)]+=c
                c = sp.bincount(bBma[wAB],weights=w12[wAB]*zwe12[wAB])
                weight_eff[:len(c)]+=c

                if ((not x_correlation) and (igm_absorption1 != igm_absorption2)) or (x_correlation and (lambda_abs == lambda_abs2)):
                    r_comov1 = delta1.r_comov
                    dist_m1 = delta1.dist_m
                    weights1 = delta1.weights
                    z1_abs2 = 10**delta1.log_lambda/constants.ABSORBER_IGM[igm_absorption2]-1
                    r1_abs2 = cosmo.get_r_comov(z1_abs2)
                    rdm1_abs2 = cosmo.get_dist_m(z1_abs2)

                    wzcut = z1_abs2<delta1.z_qso
                    r_comov1 = r_comov1[wzcut]
                    dist_m1 = dist_m1[wzcut]
                    weights1 = weights1[wzcut]
                    z1_abs2 = z1_abs2[wzcut]
                    r1_abs2 = r1_abs2[wzcut]
                    rdm1_abs2 = rdm1_abs2[wzcut]

                    r_comov2 = delta2.r_comov
                    dist_m2 = delta2.dist_m
                    weights2 = delta2.weights
                    z2_abs1 = 10**delta2.log_lambda/constants.ABSORBER_IGM[igm_absorption1]-1
                    r2_abs1 = cosmo.get_r_comov(z2_abs1)
                    rdm2_abs1 = cosmo.get_dist_m(z2_abs1)

                    wzcut = z2_abs1<delta2.z_qso
                    r_comov2 = r_comov2[wzcut]
                    dist_m2 = dist_m2[wzcut]
                    weights2 = weights2[wzcut]
                    z2_abs1 = z2_abs1[wzcut]
                    r2_abs1 = r2_abs1[wzcut]
                    rdm2_abs1 = rdm2_abs1[wzcut]

                    r_par = (r_comov1[:,None]-r_comov2)*sp.cos(ang/2)
                    if not x_correlation:
                        r_par = abs(r_par)

                    r_trans = (dist_m1[:,None]+dist_m2)*sp.sin(ang/2)
                    w12 = weights1[:,None]*weights2

                    bins_r_par = sp.floor((r_par-r_par_min)/(r_par_max-r_par_min)*num_bins_r_par).astype(int)
                    bins_r_trans = (r_trans/r_trans_max*num_bins_r_trans).astype(int)
                    if remove_same_half_plate_close_pairs and same_half_plate:
                        wp = abs(r_par) < (r_par_max-r_par_min)/num_bins_r_par
                        w12[wp] = 0.
                    bA = bins_r_trans + num_bins_r_trans*bins_r_par
                    wA = (bins_r_par<num_bins_r_par) & (bins_r_trans<num_bins_r_trans) & (bins_r_par >=0)
                    c = sp.bincount(bA[wA],weights=w12[wA])
                    weights_dmat[:len(c)]+=c
                    rp_abs2_abs1 = (r1_abs2[:,None]-r2_abs1)*sp.cos(ang/2)
                    if not x_correlation:
                        rp_abs2_abs1 = abs(rp_abs2_abs1)

                    rt_abs2_abs1 = (rdm1_abs2[:,None]+rdm2_abs1)*sp.sin(ang/2)
                    zwe21 = (1+z1_abs2[:,None])**(alpha_abs[igm_absorption2]-1)*(1+z2_abs1)**(alpha_abs[igm_absorption1]-1)/(1+z_ref)**(alpha_abs[igm_absorption1]+alpha_abs[igm_absorption2]-2)

                    bp_abs2_abs1 = sp.floor((rp_abs2_abs1-r_par_min)/(r_par_max-r_par_min)*num_model_bins_r_par).astype(int)
                    bt_abs2_abs1 = (rt_abs2_abs1/r_trans_max*num_model_bins_r_trans).astype(int)
                    bBam = bt_abs2_abs1 + num_model_bins_r_trans*bp_abs2_abs1
                    wBam = (bp_abs2_abs1<num_model_bins_r_par) & (bt_abs2_abs1<num_model_bins_r_trans) & (bp_abs2_abs1>=0)
                    wAB = wA & wBam

                    c = sp.bincount(bBam[wAB],weights=rp_abs2_abs1[wAB]*w12[wAB]*zwe21[wAB])
                    r_par_eff[:len(c)]+=c
                    c = sp.bincount(bBam[wAB],weights=rt_abs2_abs1[wAB]*w12[wAB]*zwe21[wAB])
                    r_trans_eff[:len(c)]+=c
                    c = sp.bincount(bBam[wAB],weights=(z1_abs2[:,None]+z2_abs1)[wAB]/2*w12[wAB]*zwe21[wAB])
                    z_eff[:len(c)]+=c
                    c = sp.bincount(bBam[wAB],weights=w12[wAB]*zwe21[wAB])
                    weight_eff[:len(c)]+=c

                    c = sp.bincount(bBam[wAB]+num_model_bins_r_par*num_model_bins_r_trans*bA[wAB],weights=w12[wAB]*zwe21[wAB])
                    dm[:len(c)]+=c
            setattr(delta1, "neighbours", None)

    return weights_dmat,dm.reshape(num_bins_r_par*num_bins_r_trans,num_model_bins_r_par*num_model_bins_r_trans),r_par_eff,r_trans_eff,z_eff,weight_eff,num_pairs,num_pairs_used



n1d = None
log_lambda_min = None
log_lambda_max = None
delta_log_lambda = None
def cf1d(pix):
    xi1d = np.zeros(n1d**2)
    we1d = np.zeros(n1d**2)
    nb1d = np.zeros(n1d**2,dtype=sp.int64)

    for d in data[pix]:
        bins = ((d.log_lambda-log_lambda_min)/delta_log_lambda+0.5).astype(int)
        bins = bins + n1d*bins[:,None]
        wde = d.weights*d.delta
        weights = d.weights
        xi1d[bins] += wde * wde[:,None]
        we1d[bins] += weights*weights[:,None]
        nb1d[bins] += (weights*weights[:,None]>0.).astype(int)

    w = we1d>0
    xi1d[w]/=we1d[w]
    return we1d,xi1d,nb1d

def x_forest_cf1d(pix):
    xi1d = np.zeros(n1d**2)
    we1d = np.zeros(n1d**2)
    nb1d = np.zeros(n1d**2,dtype=sp.int64)

    for delta1 in data[pix]:
        bins1 = ((delta1.log_lambda-log_lambda_min)/delta_log_lambda+0.5).astype(int)
        wde1 = delta1.weights*delta1.delta
        we1 = delta1.weights

        d2thingid = [delta2.thingid for delta2 in data2[pix]]
        neighs = data2[pix][sp.in1d(d2thingid,[delta1.thingid])]
        for delta2 in neighs:
            bins2 = ((delta2.log_lambda-log_lambda_min)/delta_log_lambda+0.5).astype(int)
            bins = bins1 + n1d*bins2[:,None]
            wde2 = delta2.weights*delta2.delta
            we2 = delta2.weights
            xi1d[bins] += wde1 * wde2[:,None]
            we1d[bins] += we1*we2[:,None]
            nb1d[bins] += (we1*we2[:,None]>0.).astype(int)

    w = we1d>0
    xi1d[w]/=we1d[w]
    return we1d,xi1d,nb1d

v1d = {}
c1d = {}
max_diagram = None
cfWick = {}

## auto
def wickT(pix):

    T1 = np.zeros((num_bins_r_par*num_bins_r_trans,num_bins_r_par*num_bins_r_trans))
    T2 = np.zeros((num_bins_r_par*num_bins_r_trans,num_bins_r_par*num_bins_r_trans))
    T3 = np.zeros((num_bins_r_par*num_bins_r_trans,num_bins_r_par*num_bins_r_trans))
    T4 = np.zeros((num_bins_r_par*num_bins_r_trans,num_bins_r_par*num_bins_r_trans))
    T5 = np.zeros((num_bins_r_par*num_bins_r_trans,num_bins_r_par*num_bins_r_trans))
    T6 = np.zeros((num_bins_r_par*num_bins_r_trans,num_bins_r_par*num_bins_r_trans))
    wAll = np.zeros(num_bins_r_par*num_bins_r_trans)
    nb = np.zeros(num_bins_r_par*num_bins_r_trans,dtype=sp.int64)
    num_pairs = 0
    num_pairs_used = 0

    for ipix in pix:

        r = sp.random.rand(len(data[ipix]))
        w = r>reject
        num_pairs += len(data[ipix])
        num_pairs_used += w.sum()
        if w.sum()==0: continue

        for delta1 in [ td for ti,td in enumerate(data[ipix]) if w[ti] ]:
            userprint("\rcomputing xi: {}%".format(round(counter.value*100./num_data/(1.-reject),3)),end="")
            with lock:
                counter.value += 1
            if len(delta1.neighbours)==0: continue

            v1 = v1d[delta1.fname](delta1.log_lambda)
            weights1 = delta1.weights
            c1d_1 = (weights1*weights1[:,None])*c1d[delta1.fname](abs(delta1.log_lambda-delta1.log_lambda[:,None]))*sp.sqrt(v1*v1[:,None])
            r_comov1 = delta1.r_comov
            z1 = delta1.z

            for i2,delta2 in enumerate(delta1.neighbours):
                ang12 = delta1^delta2

                v2 = v1d[delta2.fname](delta2.log_lambda)
                weights2 = delta2.weights
                c1d_2 = (weights2*weights2[:,None])*c1d[delta2.fname](abs(delta2.log_lambda-delta2.log_lambda[:,None]))*sp.sqrt(v2*v2[:,None])
                r_comov2 = delta2.r_comov
                z2 = delta2.z

                fill_wickT123(r_comov1,r_comov2,ang12,weights1,delta2.weights,z1,z2,c1d_1,c1d_2,wAll,nb,T1,T2,T3)
                if max_diagram<=3: continue

                ### d3 and delta2 have the same 'fname'
                for d3 in delta1.neighbours[:i2]:
                    ang13 = delta1^d3
                    ang23 = delta2^d3

                    v3 = v1d[d3.fname](d3.log_lambda)
                    w3 = d3.weights
                    c1d_3 = (w3*w3[:,None])*c1d[d3.fname](abs(d3.log_lambda-d3.log_lambda[:,None]))*sp.sqrt(v3*v3[:,None])
                    r3 = d3.r_comov
                    z3 = d3.z

                    fill_wickT45(r_comov1,r_comov2,r3, ang12,ang13,ang23, weights1,weights2,w3,
                        z1,z2,z3, c1d_1,c1d_2,c1d_3,
                        delta1.fname,delta2.fname,d3.fname,
                        T4,T5)

                ### TODO: when there is two different catalogs
                ### d3 and delta1 have the same 'fname'

    return wAll, nb, num_pairs, num_pairs_used, T1, T2, T3, T4, T5, T6
@jit
def fill_wickT123(r_comov1,r_comov2,ang,weights1,weights2,z1,z2,c1d_1,c1d_2,wAll,nb,T1,T2,T3):

    n1 = len(r_comov1)
    n2 = len(r_comov2)
    i1 = np.arange(n1)
    i2 = np.arange(n2)
    zw1 = ((1+z1)/(1+z_ref))**(alpha-1)
    zw2 = ((1+z2)/(1+z_ref))**(alpha2-1)

    bins = i1[:,None]+n1*i2
    r_par = (r_comov1[:,None]-r_comov2)*sp.cos(ang/2)
    if not x_correlation:
        r_par = abs(r_par)
    r_trans = (r_comov1[:,None]+r_comov2)*sp.sin(ang/2)
    bins_r_par = sp.floor((r_par-r_par_min)/(r_par_max-r_par_min)*num_bins_r_par).astype(int)
    bins_r_trans = (r_trans/r_trans_max*num_bins_r_trans).astype(int)
    ba = bins_r_trans + num_bins_r_trans*bins_r_par
    weights = weights1[:,None]*weights2
    we1 = weights1[:,None]*sp.ones(weights2.size)
    we2 = sp.ones(weights1.size)[:,None]*weights2
    zw = zw1[:,None]*zw2

    w = (r_par<r_par_max) & (r_trans<r_trans_max) & (r_par>=r_par_min)
    if w.sum()==0: return

    bins = bins[w]
    ba = ba[w]
    weights = weights[w]
    we1 = we1[w]
    we2 = we2[w]
    zw = zw[w]

    for k1 in range(ba.size):
        p1 = ba[k1]
        i1 = bins[k1]%n1
        j1 = (bins[k1]-i1)//n1
        wAll[p1] += weights[k1]
        nb[p1] += 1
        T1[p1,p1] += weights[k1]*zw[k1]

        for k2 in range(k1+1,ba.size):
            p2 = ba[k2]
            i2 = bins[k2]%n1
            j2 = (bins[k2]-i2)//n1
            if i1==i2:
                prod = c1d_2[j1,j2]*we1[k1]*zw1[i1]
                T2[p1,p2] += prod
                T2[p2,p1] += prod
            elif j1==j2:
                prod = c1d_1[i1,i2]*we2[k2]*zw2[j1]
                T2[p1,p2] += prod
                T2[p2,p1] += prod
            else:
                prod = c1d_1[i1,i2]*c1d_2[j1,j2]
                T3[p1,p2] += prod
                T3[p2,p1] += prod

    return
@jit
def fill_wickT45(r_comov1,r_comov2,r3, ang12,ang13,ang23, weights1,weights2,w3, z1,z2,z3, c1d_1,c1d_2,c1d_3, fname1,fname2,fname3, T4,T5):
    """

    """

    ### forest-1 x forest-2
    r_par = (r_comov1[:,None]-r_comov2)*sp.cos(ang12/2.)
    if not x_correlation:
        r_par = np.absolute(r_par)
    r_trans = (r_comov1[:,None]+r_comov2)*sp.sin(ang12/2.)
    pix1_12 = (np.arange(r_comov1.size)[:,None]*sp.ones(r_comov2.size)).astype(int)
    pix2_12 = (sp.ones(r_comov1.size)[:,None]*np.arange(r_comov2.size)).astype(int)
    w = (r_par<r_par_max) & (r_trans<r_trans_max) & (r_par>=r_par_min)
    if w.sum()==0: return
    bins_r_par = sp.floor((r_par-r_par_min)/(r_par_max-r_par_min)*num_bins_r_par).astype(int)
    bins_r_trans = (r_trans/r_trans_max*num_bins_r_trans).astype(int)
    ba12 = bins_r_trans + num_bins_r_trans*bins_r_par
    ba12[~w] = 0
    cf12 = cfWick['{}_{}'.format(fname1,fname2)][ba12]
    cf12[~w] = 0.

    ba12 = ba12[w]
    pix1_12 = pix1_12[w]
    pix2_12 = pix2_12[w]

    ### forest-1 x forest-3
    r_par = (r_comov1[:,None]-r3)*sp.cos(ang13/2.)
    if not x_correlation:
        r_par = np.absolute(r_par)
    r_trans = (r_comov1[:,None]+r3)*sp.sin(ang13/2.)
    pix1_13 = (np.arange(r_comov1.size)[:,None]*sp.ones(r3.size)).astype(int)
    pix3_13 = (sp.ones(r_comov1.size)[:,None]*np.arange(r3.size)).astype(int)
    w = (r_par<r_par_max) & (r_trans<r_trans_max) & (r_par>=r_par_min)
    if w.sum()==0: return
    bins_r_par = sp.floor((r_par-r_par_min)/(r_par_max-r_par_min)*num_bins_r_par).astype(int)
    bins_r_trans = (r_trans/r_trans_max*num_bins_r_trans).astype(int)
    ba13 = bins_r_trans + num_bins_r_trans*bins_r_par
    ba13[~w] = 0
    cf13 = cfWick['{}_{}'.format(fname1,fname3)][ba13]
    cf13[~w] = 0.

    ba13 = ba13[w]
    pix1_13 = pix1_13[w]
    pix3_13 = pix3_13[w]

    ### forest-2 x forest-3
    r_par = (r_comov2[:,None]-r3)*sp.cos(ang23/2.)
    if not x_correlation:
        r_par = np.absolute(r_par)
    r_trans = (r_comov2[:,None]+r3)*sp.sin(ang23/2.)
    pix2_23 = (np.arange(r_comov2.size)[:,None]*sp.ones(r3.size)).astype(int)
    pix3_23 = (sp.ones(r_comov2.size)[:,None]*np.arange(r3.size)).astype(int)
    w = (r_par<r_par_max) & (r_trans<r_trans_max) & (r_par>=r_par_min)
    if w.sum()==0: return
    bins_r_par = sp.floor((r_par-r_par_min)/(r_par_max-r_par_min)*num_bins_r_par).astype(int)
    bins_r_trans = (r_trans/r_trans_max*num_bins_r_trans).astype(int)
    ba23 = bins_r_trans + num_bins_r_trans*bins_r_par
    ba23[~w] = 0
    cf23 = cfWick['{}_{}'.format(fname2,fname3)][ba23]
    cf23[~w] = 0.

    ba23 = ba23[w]
    pix2_23 = pix2_23[w]
    pix3_23 = pix3_23[w]

    ### Wick T4 and T5
    for k1,p1 in enumerate(ba12):
        tpix1_12 = pix1_12[k1]
        tpix2_12 = pix2_12[k1]

        for k2,p2 in enumerate(ba13):
            tpix1_13 = pix1_13[k2]
            tpix3_13 = pix3_13[k2]

            tcf23 = cf23[tpix2_12,tpix3_13]
            if tpix1_12==tpix1_13:
                wcorr = weights1[tpix1_12]*tcf23 ### TODO work on the good formula
                T4[p1,p2] += wcorr
                T4[p2,p1] += wcorr
            else:
                wcorr = c1d_1[tpix1_12,tpix1_13]*tcf23 ### TODO work on the good formula
                T5[p1,p2] += wcorr
                T5[p2,p1] += wcorr

    return
