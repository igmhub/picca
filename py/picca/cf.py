"""This module defines functions and variables required for the correlation
analysis of two delta fields

This module provides several functions:
    - fill_neighs
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

# variables used in the 1D correlation function analysis
num_pixels = None
log_lambda_min = None
log_lambda_max = None
delta_log_lambda = None

# variables used in the wick covariance matrix computation
get_variance_1d = {}
xi_1d = {}
max_diagram = None
xi_wick = {}

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

def compute_metal_dmat(healpixs, abs_igm1="LYA", abs_igm2="SiIII(1207)"):
    """Computes the metal distortion matrix for each of the healpixs.

    Args:
        healpixs: array of ints
            List of healpix numbers
        abs_igm1: string - default: "LYA"
            Name of the absorption in picca.constants defining the
            redshift of the forest pixels
        abs_igm2: string - default: "SiIII(1207)"
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
                       "{}%").format(abs_igm1,
                                     abs_igm2,
                                     round(counter.value*100./num_data, 3)),
                      end="")
            with lock:
                counter.value += 1

            w = np.random.rand(len(delta1.neighbours)) > reject
            num_pairs += len(delta1.neighbours)
            num_pairs_used += w.sum()
            for delta2 in np.array(delta1.neighbours)[w]:
                r_comov1 = delta1.r_comov
                dist_m1 = delta1.dist_m
                z1_abs1 = (10**delta1.log_lambda/
                           constants.ABSORBER_IGM[abs_igm1] - 1)
                r_comov1_abs1 = cosmo.get_r_comov(z1_abs1)
                dist_m1_abs1 = cosmo.get_dist_m(z1_abs1)
                weights1 = delta1.weights

                # filter cases where the absorption from the metal is
                # inconsistent with the quasar redshift
                w = z1_abs1 < delta1.z_qso
                r_comov1 = r_comov1[w]
                dist_m1 = dist_m1[w]
                weights1 = weights1[w]
                r_comov1_abs1 = r_comov1_abs1[w]
                dist_m1_abs1 = dist_m1_abs1[w]
                z1_abs1 = z1_abs1[w]

                same_half_plate = ((delta1.plate == delta2.plate) and
                                   ((delta1.fiberid <= 500 and
                                     delta2.fiberid <= 500) or
                                    (delta1.fiberid > 500 and
                                     delta2.fiberid > 500)))
                ang = delta1^delta2
                r_comov2 = delta2.r_comov
                dist_m2 = delta2.dist_m
                z2_abs2 = (10**delta2.log_lambda/
                           constants.ABSORBER_IGM[abs_igm2]-1)
                r_comov2_abs2 = cosmo.get_r_comov(z2_abs2)
                dist_m2_abs2 = cosmo.get_dist_m(z2_abs2)
                weights2 = delta2.weights

                # filter cases where the absorption from the metal is
                # inconsistent with the quasar redshift
                w = z2_abs2 < delta2.z_qso
                r_comov2 = r_comov2[w]
                dist_m2 = dist_m2[w]
                weights2 = weights2[w]
                r_comov2_abs2 = r_comov2_abs2[w]
                dist_m2_abs2 = dist_m2_abs2[w]
                z2_abs2 = z2_abs2[w]

                # compute bins the pairs contribute to
                r_par = (r_comov1[:, None] - r_comov2)*np.cos(ang/2)
                if not x_correlation:
                    r_par = abs(r_par)

                r_trans = (dist_m1[:, None] + dist_m2)*np.sin(ang/2)
                weights12 = weights1[:, None]*weights2

                bins_r_par = np.floor((r_par - r_par_min)/
                                      (r_par_max - r_par_min)*
                                      num_bins_r_par).astype(int)
                bins_r_trans = (r_trans/r_trans_max*
                                num_bins_r_trans).astype(int)

                if remove_same_half_plate_close_pairs and same_half_plate:
                    weights12[abs(r_par) <
                              (r_par_max - r_par_min)/num_bins_r_par] = 0.

                bins = bins_r_trans + num_bins_r_trans*bins_r_par
                w = ((bins_r_par < num_bins_r_par) &
                     (bins_r_trans < num_bins_r_trans) &
                     (bins_r_par >= 0))
                rebin = np.bincount(bins[w], weights=weights12[w])
                weights_dmat[:len(rebin)] += rebin

                r_par_abs1_abs2 = (r_comov1_abs1[:, None] -
                                   r_comov2_abs2)*np.cos(ang/2)

                if not x_correlation:
                    r_par_abs1_abs2 = abs(r_par_abs1_abs2)

                r_trans_abs1_abs2 = (dist_m1_abs1[:, None] +
                                     dist_m2_abs2)*np.sin(ang/2)
                z_weight_evol = ((1 + z1_abs1[:, None])**(alpha_abs[abs_igm1]
                                                          - 1)*
                                 (1 + z2_abs2)**(alpha_abs[abs_igm2] - 1)/
                                 (1 + z_ref)**(alpha_abs[abs_igm1] +
                                               alpha_abs[abs_igm2] - 2))

                model_bins_r_par = np.floor((r_par_abs1_abs2 - r_par_min)/
                                            (r_par_max - r_par_min)*
                                            num_model_bins_r_par).astype(int)
                model_bins_r_trans = (r_trans_abs1_abs2/r_trans_max*
                                      num_model_bins_r_trans).astype(int)
                model_bins = (model_bins_r_trans +
                              num_model_bins_r_trans*model_bins_r_par)
                w &= ((model_bins_r_par < num_model_bins_r_par) &
                      (model_bins_r_trans < num_model_bins_r_trans) &
                      (model_bins_r_par >= 0))
                rebin = np.bincount((model_bins[w] + num_model_bins_r_par*
                                     num_model_bins_r_trans*bins[w]),
                                    weights=weights12[w]*z_weight_evol[w])
                dmat[:len(rebin)] += rebin
                rebin = np.bincount(model_bins[w],
                                    weights=(r_par_abs1_abs2[w]*weights12[w]*
                                             z_weight_evol[w]))
                r_par_eff[:len(rebin)] += rebin
                rebin = np.bincount(model_bins[w],
                                    weights=(r_trans_abs1_abs2[w]*weights12[w]*
                                             z_weight_evol[w]))
                r_trans_eff[:len(rebin)] += rebin
                rebin = np.bincount(model_bins[w],
                                    weights=((z1_abs1[:, None] + z2_abs2)[w]/2*
                                             weights12[w]*z_weight_evol[w]))
                z_eff[:len(rebin)] += rebin
                rebin = np.bincount(model_bins[w],
                                    weights=weights12[w]*z_weight_evol[w])
                weight_eff[:len(rebin)] += rebin

                if (((not x_correlation) and (abs_igm1 != abs_igm2)) or
                        (x_correlation and (lambda_abs == lambda_abs2))):
                    r_comov1 = delta1.r_comov
                    dist_m1 = delta1.dist_m
                    weights1 = delta1.weights
                    z1_abs2 = (10**delta1.log_lambda/
                               constants.ABSORBER_IGM[abs_igm2] - 1)
                    r_comov1_abs2 = cosmo.get_r_comov(z1_abs2)
                    dist_m1_abs2 = cosmo.get_dist_m(z1_abs2)

                    w = z1_abs2 < delta1.z_qso
                    r_comov1 = r_comov1[w]
                    dist_m1 = dist_m1[w]
                    weights1 = weights1[w]
                    z1_abs2 = z1_abs2[w]
                    r_comov1_abs2 = r_comov1_abs2[w]
                    dist_m1_abs2 = dist_m1_abs2[w]

                    r_comov2 = delta2.r_comov
                    dist_m2 = delta2.dist_m
                    weights2 = delta2.weights
                    z2_abs1 = (10**delta2.log_lambda/
                               constants.ABSORBER_IGM[abs_igm1] - 1)
                    r_comov2_abs1 = cosmo.get_r_comov(z2_abs1)
                    dist_m2_abs1 = cosmo.get_dist_m(z2_abs1)

                    w = z2_abs1 < delta2.z_qso
                    r_comov2 = r_comov2[w]
                    dist_m2 = dist_m2[w]
                    weights2 = weights2[w]
                    z2_abs1 = z2_abs1[w]
                    r_comov2_abs1 = r_comov2_abs1[w]
                    dist_m2_abs1 = dist_m2_abs1[w]

                    r_par = (r_comov1[:, None] - r_comov2)*np.cos(ang/2)
                    if not x_correlation:
                        r_par = abs(r_par)

                    r_trans = (dist_m1[:, None] + dist_m2)*np.sin(ang/2)
                    weights12 = weights1[:, None]*weights2

                    bins_r_par = np.floor((r_par - r_par_min)/
                                          (r_par_max - r_par_min)*
                                          num_bins_r_par).astype(int)
                    bins_r_trans = (r_trans/r_trans_max*
                                    num_bins_r_trans).astype(int)
                    if remove_same_half_plate_close_pairs and same_half_plate:
                        weights12[abs(r_par) <
                                  (r_par_max-r_par_min)/num_bins_r_par] = 0.
                    bins = bins_r_trans + num_bins_r_trans*bins_r_par
                    w = ((bins_r_par < num_bins_r_par) &
                         (bins_r_trans < num_bins_r_trans) &
                         (bins_r_par >= 0))
                    rebin = np.bincount(bins[w], weights=weights12[w])
                    weights_dmat[:len(rebin)] += rebin
                    r_par_abs2_abs1 = (r_comov1_abs2[:, None]-
                                       r_comov2_abs1)*np.cos(ang/2)
                    if not x_correlation:
                        r_par_abs2_abs1 = abs(r_par_abs2_abs1)

                    r_trans_abs2_abs1 = (dist_m1_abs2[:, None] +
                                         dist_m2_abs1)*np.sin(ang/2)
                    z_weight_evol = ((1 + z1_abs2[:, None])**
                                     (alpha_abs[abs_igm2] - 1)*
                                     (1 + z2_abs1)**(alpha_abs[abs_igm1] - 1)/
                                     (1 + z_ref)**(alpha_abs[abs_igm1] +
                                                   alpha_abs[abs_igm2] - 2))

                    model_bins_r_par = np.floor((r_par_abs2_abs1 - r_par_min)/
                                                (r_par_max-r_par_min)*
                                                num_model_bins_r_par).astype(int)
                    model_bins_r_trans = (r_trans_abs2_abs1/r_trans_max*
                                          num_model_bins_r_trans).astype(int)
                    model_bins = (model_bins_r_trans +
                                  num_model_bins_r_trans*model_bins_r_par)
                    w &= ((model_bins_r_par < num_model_bins_r_par) &
                          (model_bins_r_trans < num_model_bins_r_trans) &
                          (model_bins_r_par >= 0))

                    rebin = np.bincount(model_bins[w],
                                        weights=(r_par_abs2_abs1[w]*
                                                 weights12[w]*z_weight_evol[w]))
                    r_par_eff[:len(rebin)] += rebin
                    rebin = np.bincount(model_bins[w],
                                        weights=(r_trans_abs2_abs1[w]*
                                                 weights12[w]*z_weight_evol[w]))
                    r_trans_eff[:len(rebin)] += rebin
                    rebin = np.bincount(model_bins[w],
                                        weights=((z1_abs2[:, None] +
                                                  z2_abs1)[w]/2*
                                                 weights12[w]*z_weight_evol[w]))
                    z_eff[:len(rebin)] += rebin
                    rebin = np.bincount(model_bins[w],
                                        weights=weights12[w]*z_weight_evol[w])
                    weight_eff[:len(rebin)] += rebin

                    rebin = np.bincount((model_bins[w] + num_model_bins_r_par*
                                         num_model_bins_r_trans*bins[w]),
                                        weights=weights12[w]*z_weight_evol[w])
                    dmat[:len(rebin)] += rebin
            setattr(delta1, "neighbours", None)

    dmat = dmat.reshape(num_bins_r_par*num_bins_r_trans,
                        num_model_bins_r_par*num_model_bins_r_trans)
    return (weights_dmat, dmat, r_par_eff, r_trans_eff, z_eff, weight_eff,
            num_pairs, num_pairs_used)

def compute_xi_1d(healpix):
    """Computes the 1D autocorrelation from deltas from the same forest

    Args:
        healpix: ints
            A healpix number

    Returns:
        The following variables:
            weights1d: Total weights for the 1d correlation function
            xi1d: The 1d correlation function
            num_pairs1d: Number of pairs for the 1d correlation function
    """
    xi1d = np.zeros(num_pixels**2)
    weights1d = np.zeros(num_pixels**2)
    num_pairs1d = np.zeros(num_pixels**2, dtype=np.int64)

    for delta in data[healpix]:
        userprint(("\rcomputing xi: "
                   "{}%").format(round(counter.value*100./num_data, 2)),
                  end="")
        bins = ((delta.log_lambda - log_lambda_min)/
                delta_log_lambda + 0.5).astype(int)
        bins = bins + num_pixels*bins[:, None]
        delta_times_weight = delta.weights*delta.delta
        weights = delta.weights
        xi1d[bins] += delta_times_weight * delta_times_weight[:, None]
        weights1d[bins] += weights*weights[:, None]
        num_pairs1d[bins] += (weights*weights[:, None] > 0.).astype(int)

    w = weights1d > 0
    xi1d[w] /= weights1d[w]
    return weights1d, xi1d, num_pairs1d

def compute_xi_1d_cross(healpix):
    """Computes the 1D cross-correlation from deltas from the same forest

    Args:
        healpix: ints
            A healpix number

    Returns:
        The following variables:
            weights1d: Total weights for the 1d correlation function
            xi1d: The 1d correlation function
            num_pairs1d: Number of pairs for the 1d correlation function
    """
    xi1d = np.zeros(num_pixels**2)
    weights1d = np.zeros(num_pixels**2)
    num_pairs1d = np.zeros(num_pixels**2, dtype=np.int64)

    for delta1 in data[healpix]:
        userprint(("\rcomputing xi: "
                   "{}%").format(round(counter.value*100./num_data, 2)),
                  end="")
        bins1 = ((delta1.log_lambda - log_lambda_min)/
                 delta_log_lambda + 0.5).astype(int)
        delta_times_weight1 = delta1.weights*delta1.delta
        weights1 = delta1.weights

        thingids = [delta2.thingid for delta2 in data2[healpix]]
        neighbours = data2[healpix][np.in1d(thingids, [delta1.thingid])]
        for delta2 in neighbours:
            bins2 = ((delta2.log_lambda - log_lambda_min)/
                     delta_log_lambda + 0.5).astype(int)
            bins = bins1 + num_pixels*bins2[:, None]
            delta_times_weight2 = delta2.weights*delta2.delta
            weights2 = delta2.weights
            xi1d[bins] += delta_times_weight1 * delta_times_weight2[:, None]
            weights1d[bins] += weights1*weights2[:, None]
            num_pairs1d[bins] += (weights1*weights2[:, None] > 0.).astype(int)

    w = weights1d > 0
    xi1d[w] /= weights1d[w]
    return weights1d, xi1d, num_pairs1d

def wickT(healpixs):
    """
    Computes the Wick expansion terms of the covariance matrix for the auto-
    correlation analysis

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
    t1 = np.zeros((num_bins_r_par*num_bins_r_trans,
                   num_bins_r_par*num_bins_r_trans))
    t2 = np.zeros((num_bins_r_par*num_bins_r_trans,
                   num_bins_r_par*num_bins_r_trans))
    t3 = np.zeros((num_bins_r_par*num_bins_r_trans,
                   num_bins_r_par*num_bins_r_trans))
    t4 = np.zeros((num_bins_r_par*num_bins_r_trans,
                   num_bins_r_par*num_bins_r_trans))
    t5 = np.zeros((num_bins_r_par*num_bins_r_trans,
                   num_bins_r_par*num_bins_r_trans))
    t6 = np.zeros((num_bins_r_par*num_bins_r_trans,
                   num_bins_r_par*num_bins_r_trans))
    weights_wick = np.zeros(num_bins_r_par*num_bins_r_trans)
    num_pairs_wick = np.zeros(num_bins_r_par*num_bins_r_trans, dtype=np.int64)
    num_pairs = 0
    num_pairs_used = 0

    for healpix in healpixs:
        w = np.random.rand(len(data[healpix])) > reject
        num_pairs += len(data[healpix])
        num_pairs_used += w.sum()
        if w.sum() == 0:
            continue

        for delta1 in [delta
                       for index, delta in enumerate(data[healpix])
                       if w[index]]:
            userprint(("\rcomputing xi: "
                       "{}%").format(round(counter.value*100./num_data/(1. -
                                                                        reject),
                                           3)),
                      end="")
            with lock:
                counter.value += 1
            if len(delta1.neighbours) == 0:
                continue

            variance_1d_1 = get_variance_1d[delta1.fname](delta1.log_lambda)
            weights1 = delta1.weights
            weighted_xi_1d_1 = ((weights1*weights1[:, None])*
                                xi_1d[delta1.fname](abs(delta1.log_lambda -
                                                        delta1.log_lambda[:, None]))*
                                np.sqrt(variance_1d_1*variance_1d_1[:, None]))
            r_comov1 = delta1.r_comov
            z1 = delta1.z

            for index2, delta2 in enumerate(delta1.neighbours):
                ang12 = delta1^delta2

                variance_1d_2 = get_variance_1d[delta2.fname](delta2.log_lambda)
                weights2 = delta2.weights
                weighted_xi_1d_2 = ((weights2*weights2[:, None])*
                                    xi_1d[delta2.fname](abs(delta2.log_lambda -
                                                            delta2.log_lambda[:, None]))*
                                    np.sqrt(variance_1d_2*variance_1d_2[:, None]))
                r_comov2 = delta2.r_comov
                z2 = delta2.z

                fill_wickT123(r_comov1, r_comov2, ang12, weights1,
                              delta2.weights, z1, z2, weighted_xi_1d_1,
                              weighted_xi_1d_2, weights_wick, num_pairs_wick,
                              t1, t2, t3)
                if max_diagram <= 3:
                    continue

                ### delta3 and delta2 have the same 'fname'
                for delta3 in delta1.neighbours[:index2]:
                    ang13 = delta1^delta3
                    ang23 = delta2^delta3

                    variance_1d_3 = get_variance_1d[delta3.fname](delta3.log_lambda)
                    weights3 = delta3.weights
                    weighted_xi_1d_3 = ((weights3*weights3[:, None])*
                                        xi_1d[delta3.fname](abs(delta3.log_lambda -
                                                                delta3.log_lambda[:, None]))*
                                        np.sqrt(variance_1d_3*variance_1d_3[:, None]))
                    r_comov3 = delta3.r_comov
                    z3 = delta3.z

                    fill_wickT45(r_comov1, r_comov2, r_comov3, ang12, ang13,
                                 ang23, weights1, weights2, weights3, z1, z2,
                                 z3, weighted_xi_1d_1, weighted_xi_1d_2, weighted_xi_1d_3,
                                 delta1.fname, delta2.fname, delta3.fname, t4,
                                 t5)

                ### TODO: when there is two different catalogs
                ### delta3 and delta1 have the same 'fname'

    return (weights_wick, num_pairs_wick, num_pairs, num_pairs_used, t1, t2,
            t3, t4, t5, t6)

@jit
def fill_wickT123(r_comov1, r_comov2, ang, weights1, weights2, z1, z2,
                  weighted_xi_1d_1, weighted_xi_1d_2, weights_wick,
                  num_pairs_wick, t1, t2, t3):
    """
    Computes the Wick expansion terms 1, 2, and 3 of a given pair of forests

    Each of the terms represents the contribution of different type of pairs as
    illustrated in figure A.1 from Delubac et al. 2015

    Args:
        r_comov1: array of floats
            Comoving distance (in Mpc/h) for forest 1
        r_comov2: array of floats
            Comoving distance (in Mpc/h) for forest 2
        ang: array of floats
            Angular separation between pixels in forests 1 and 2
        weights1: array of floats
            Weights for forest 1
        weights2: array of floats
            Weights for forest 2
        z1: array of floats
            Redshifts for forest 1
        z2: array of floats
            Redshifts for forest 2
        weighted_xi_1d_1: array of floats
            Weighted 1D correlation function for forest 1
        weighted_xi_1d_2: array of floats
            Weighted 1D correlation function for forest 2
        weights_wick: array of floats
            Total weight in the covariance matrix pixels
        num_pairs_wick: array of floats
            Total number of pairs in the covariance matrix pixels
        t1: Wick expansion, term 1
        t2: Wick expansion, term 2
        t3: Wick expansion, term 3
    """
    num_pixels1 = len(r_comov1)
    num_pixels2 = len(r_comov2)
    i1 = np.arange(num_pixels1)
    i2 = np.arange(num_pixels2)
    z_weight_evol1 = ((1 + z1)/(1 + z_ref))**(alpha - 1)
    z_weight_evol2 = ((1 + z2)/(1 + z_ref))**(alpha2 - 1)

    bins = i1[:, None] + num_pixels1*i2
    r_par = (r_comov1[:, None] - r_comov2)*np.cos(ang/2)
    if not x_correlation:
        r_par = abs(r_par)
    r_trans = (r_comov1[:, None] + r_comov2)*np.sin(ang/2)
    bins_r_par = np.floor((r_par - r_par_min)/(r_par_max - r_par_min)*
                          num_bins_r_par).astype(int)
    bins_r_trans = (r_trans/r_trans_max*num_bins_r_trans).astype(int)
    bins_forest = bins_r_trans + num_bins_r_trans*bins_r_par
    weights12 = weights1[:, None]*weights2
    weight1 = weights1[:, None]*np.ones(weights2.size)
    weight2 = np.ones(weights1.size)[:, None]*weights2
    z_weight_evol = z_weight_evol1[:, None]*z_weight_evol2

    w = (r_par < r_par_max) & (r_trans < r_trans_max) & (r_par >= r_par_min)
    if w.sum() == 0:
        return

    bins = bins[w]
    bins_forest = bins_forest[w]
    weights12 = weights12[w]
    weight1 = weight1[w]
    weight2 = weight2[w]
    z_weight_evol = z_weight_evol[w]

    for index1 in range(bins_forest.size):
        p1 = bins_forest[index1]
        i1 = bins[index1]%num_pixels1
        j1 = (bins[index1]-i1)//num_pixels1
        weights_wick[p1] += weights12[index1]
        num_pairs_wick[p1] += 1
        t1[p1, p1] += weights12[index1]*z_weight_evol[index1]

        for index2 in range(index1 + 1, bins_forest.size):
            p2 = bins_forest[index2]
            i2 = bins[index2]%num_pixels1
            j2 = (bins[index2] - i2)//num_pixels1
            if i1 == i2:
                prod = weighted_xi_1d_2[j1, j2]*weight1[index1]*z_weight_evol1[i1]
                t2[p1, p2] += prod
                t2[p2, p1] += prod
            elif j1 == j2:
                prod = weighted_xi_1d_1[i1, i2]*weight2[index2]*z_weight_evol2[j1]
                t2[p1, p2] += prod
                t2[p2, p1] += prod
            else:
                prod = weighted_xi_1d_1[i1, i2]*weighted_xi_1d_2[j1, j2]
                t3[p1, p2] += prod
                t3[p2, p1] += prod

    return


@jit
def fill_wickT45(r_comov1, r_comov2, r_comov3, ang12, ang13, ang23, weights1,
                 weights2, weights3, weighted_xi_1d_1, weighted_xi_1d_2,
                 weighted_xi_1d_3, fname1, fname2, fname3, t4, t5):
    """
    Computes the Wick expansion terms 4 and 5 of a given set of 3 forests

    Each of the terms represents the contribution of different type of pairs as
    illustrated in figure A.1 from Delubac et al. 2015

    Args:
        r_comov1: array of floats
            Comoving distance (in Mpc/h) for forest 1
        r_comov2: array of floats
            Comoving distance (in Mpc/h) for forest 2
        r_comov3: array of floats
            Comoving distance (in Mpc/h) for forest 3
        ang12: array of floats
            Angular separation between pixels in forests 1 and 2
        ang13: array of floats
            Angular separation between pixels in forests 1 and 3
        ang23: array of floats
            Angular separation between pixels in forests 2 and 3
        weights1: array of floats
            Weights for forest 1
        weights2: array of floats
            Weights for forest 2
        weights3: array of floats
            Weights for forest 3
        weighted_xi_1d_1: array of floats
            Weighted 1D correlation function for forest 1
        weighted_xi_1d_2: array of floats
            Weighted 1D correlation function for forest 2
        weighted_xi_1d_3: array of floats
            Weighted 1D correlation function for forest 3
        fname1: string
            Flag name identifying the group of deltas for forest 1
        fname2: string
            Flag name identifying the group of deltas for forest 2
        fname3: string
            Flag name identifying the group of deltas for forest 3
        t4: Wick expansion, term 4
        t5: Wick expansion, term 5
    """
    ### forest-1 x forest-2
    r_par = (r_comov1[:, None] - r_comov2)*np.cos(ang12/2.)
    if not x_correlation:
        r_par = np.absolute(r_par)
    r_trans = (r_comov1[:, None] + r_comov2)*np.sin(ang12/2.)
    bins1_12 = (np.arange(r_comov1.size)[:, None]*
                np.ones(r_comov2.size)).astype(int)
    bins2_12 = (np.ones(r_comov1.size)[:, None]*
                np.arange(r_comov2.size)).astype(int)
    w = (r_par < r_par_max) & (r_trans < r_trans_max) & (r_par >= r_par_min)
    if w.sum() == 0:
        return
    bins_r_par = np.floor((r_par - r_par_min)/(r_par_max - r_par_min)*
                          num_bins_r_par).astype(int)
    bins_r_trans = (r_trans/r_trans_max*num_bins_r_trans).astype(int)
    bins_forest12 = bins_r_trans + num_bins_r_trans*bins_r_par
    bins_forest12[~w] = 0
    xi12 = xi_wick['{}_{}'.format(fname1, fname2)][bins_forest12]
    xi12[~w] = 0.

    bins_forest12 = bins_forest12[w]
    bins1_12 = bins1_12[w]
    bins2_12 = bins2_12[w]

    ### forest-1 x forest-3
    r_par = (r_comov1[:, None] - r_comov3)*np.cos(ang13/2.)
    if not x_correlation:
        r_par = np.absolute(r_par)
    r_trans = (r_comov1[:, None] + r_comov3)*np.sin(ang13/2.)
    bins1_13 = (np.arange(r_comov1.size)[:, None]*
                np.ones(r_comov3.size)).astype(int)
    bins3_13 = (np.ones(r_comov1.size)[:, None]*
                np.arange(r_comov3.size)).astype(int)
    w = (r_par < r_par_max) & (r_trans < r_trans_max) & (r_par >= r_par_min)
    if w.sum() == 0:
        return
    bins_r_par = np.floor((r_par - r_par_min)/(r_par_max - r_par_min)*
                          num_bins_r_par).astype(int)
    bins_r_trans = (r_trans/r_trans_max*num_bins_r_trans).astype(int)
    bins_forest13 = bins_r_trans + num_bins_r_trans*bins_r_par
    bins_forest13[~w] = 0
    xi13 = xi_wick['{}_{}'.format(fname1, fname3)][bins_forest12]
    xi13[~w] = 0.

    bins_forest12 = bins_forest12[w]
    bins1_13 = bins1_13[w]
    bins3_13 = bins3_13[w]

    ### forest-2 x forest-3
    r_par = (r_comov2[:, None] - r_comov3)*np.cos(ang23/2.)
    if not x_correlation:
        r_par = np.absolute(r_par)
    r_trans = (r_comov2[:, None] + r_comov3)*np.sin(ang23/2.)
    bins2_23 = (np.arange(r_comov2.size)[:, None]*
                np.ones(r_comov3.size)).astype(int)
    bins3_23 = (np.ones(r_comov2.size)[:, None]*
                np.arange(r_comov3.size)).astype(int)
    w = (r_par < r_par_max) & (r_trans < r_trans_max) & (r_par >= r_par_min)
    if w.sum() == 0:
        return
    bins_r_par = np.floor((r_par - r_par_min)/(r_par_max - r_par_min)*
                          num_bins_r_par).astype(int)
    bins_r_trans = (r_trans/r_trans_max*num_bins_r_trans).astype(int)
    bins_forest23 = bins_r_trans + num_bins_r_trans*bins_r_par
    bins_forest23[~w] = 0
    xi23 = xi_wick['{}_{}'.format(fname2, fname3)][bins_forest23]
    xi23[~w] = 0.

    bins_forest23 = bins_forest23[w]
    bins2_23 = bins2_23[w]
    bins3_23 = bins3_23[w]

    ### Wick t4 and t5
    for index1, p1 in enumerate(bins_forest12):
        tpix1_12 = bins1_12[index1]
        tpix2_12 = bins2_12[index1]

        for index2, p2 in enumerate(bins_forest12):
            tpix1_13 = bins1_13[index2]
            tpix3_13 = bins3_13[index2]

            tcf23 = xi23[tpix2_12, tpix3_13]
            if tpix1_12 == tpix1_13:
                # TODO: work on the good formula
                wcorr = weights1[tpix1_12]*tcf23
                t4[p1, p2] += wcorr
                t4[p2, p1] += wcorr
            else:
                # TODO: work on the good formula
                wcorr = weighted_xi_1d_1[tpix1_12, tpix1_13]*tcf23
                t5[p1, p2] += wcorr
                t5[p2, p1] += wcorr

    return
