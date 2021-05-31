"""This module defines functions and variables required for the correlation
analysis of two object catalogues

This module provides several functions:
    - fill_neighs
    - compute_xi
    - compute_xi_forest_pairs
See the respective docstrings for more details
"""
import numpy as np
from healpy import query_disc
from numba import jit

from .utils import userprint

num_bins_r_par = None
num_bins_r_trans = None
r_par_min = None
r_par_max = None
r_trans_max = None
ang_max = None
nside = None

objs = None
objs2 = None

type_corr = None
x_correlation = False

counter = None
lock = None


def fill_neighs(healpixs):
    """Create and store a list of neighbours for each of the healpix.

    Neighbours are added to the delta objects directly

    Args:
        healpixs: array of ints
            List of healpix numbers
    """
    for healpix in healpixs:
        for obj1 in objs[healpix]:
            healpix_neighbours = query_disc(
                nside, [obj1.x_cart, obj1.y_cart, obj1.z_cart],
                ang_max,
                inclusive=True)
            if objs2 is not None:
                healpix_neighbours = [
                    healpix for healpix in healpix_neighbours
                    if healpix in objs2
                ]
                neighbours = [
                    obj2 for healpix in healpix_neighbours
                    for obj2 in objs2[healpix] if obj1.thingid != obj2.thingid
                ]
            else:
                healpix_neighbours = [
                    healpix for healpix in healpix_neighbours if healpix in objs
                ]
                neighbours = [
                    obj2 for healpix in healpix_neighbours
                    for obj2 in objs[healpix] if obj1.thingid != obj2.thingid
                ]
            ang = obj1.get_angle_between(neighbours)
            w = ang < ang_max
            neighbours = np.array(neighbours)[w]
            obj1.neighbours = np.array([
                obj2 for obj2 in neighbours
                if ((obj2.z_qso + obj1.z_qso) / 2. >= z_cut_min and
                    (obj2.z_qso + obj1.z_qso) / 2. < z_cut_max)
            ])


def compute_xi(healpixs):
    """Computes the correlation function for each of the healpixs.

    Args:
        healpixs: array of ints
            List of healpix numbers

    Returns:
        The following variables:
            weights: Total weights in the correlation function pixels
            r_par: Parallel distance of the correlation function pixels
            r_trans: Transverse distance of the correlation function pixels
            z: Redshift of the correlation function pixels
            num_pairs: Number of pairs in the correlation function pixels
    """
    weights = np.zeros(num_bins_r_par * num_bins_r_trans)
    r_par = np.zeros(num_bins_r_par * num_bins_r_trans)
    r_trans = np.zeros(num_bins_r_par * num_bins_r_trans)
    z = np.zeros(num_bins_r_par * num_bins_r_trans)
    num_pairs = np.zeros(num_bins_r_par * num_bins_r_trans, dtype=np.int64)

    for healpix in healpixs:
        for obj1 in objs[healpix]:

            with lock:
                xicounter = round(counter.value * 100. / num_data, 2)
                if (counter.value % 1000 == 0):
                    userprint(f"computing xi: {xicounter}%")
                counter.value += 1

            if obj1.neighbours.size == 0:
                continue

            ang = obj1.get_angle_between(obj1.neighbours)
            z2 = np.array([obj2.z_qso for obj2 in obj1.neighbours])
            r_comov2 = np.array([obj2.r_comov for obj2 in obj1.neighbours])
            dist_m2 = np.array([obj2.dist_m for obj2 in obj1.neighbours])
            weights2 = np.array([obj2.weights for obj2 in obj1.neighbours])

            (rebin_weight, rebin_r_par, rebin_r_trans,
             rebin_z, rebin_num_pairs) = compute_xi_forest_pairs(
                 obj1.z_qso, obj1.r_comov, obj1.dist_m, obj1.weights, z2,
                 r_comov2, dist_m2, weights2, ang)

            weights[:len(rebin_weight)] += rebin_weight
            r_par[:len(rebin_r_par)] += rebin_r_par
            r_trans[:len(rebin_r_trans)] += rebin_r_trans
            z[:len(rebin_z)] += rebin_z
            num_pairs[:len(rebin_num_pairs)] += rebin_num_pairs
            setattr(obj1, "neighbours", None)

    w = weights > 0.
    r_par[w] /= weights[w]
    r_trans[w] /= weights[w]
    z[w] /= weights[w]
    return weights, r_par, r_trans, z, num_pairs


@jit  #will be deprecated
def compute_xi_forest_pairs_slow(z1, r_comov1, dist_m1, weights1, z2, r_comov2,
                                 dist_m2, weights2, ang):
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
            rebin_r_par: The parallel distance of the correlation function
                pixels properly rebinned
            rebin_r_trans: The transverse distance of the correlation function
                pixels properly rebinned
            rebin_z: The redshift of the correlation function pixels properly
                rebinned
            rebin_num_pairs: The number of pairs of the correlation function
                pixels properly rebinned
    """
    r_par = (r_comov1 - r_comov2) * np.cos(ang / 2.)
    if not x_correlation or type_corr in ['DR', 'RD']:
        r_par = np.absolute(r_par)
    r_trans = (dist_m1 + dist_m2) * np.sin(ang / 2.)
    z = (z1 + z2) / 2.
    weights12 = weights1 * weights2

    w = (r_par >= r_par_min) & (r_par < r_par_max) & (r_trans < r_trans_max)
    w &= (weights12 > 0.)
    r_par = r_par[w]
    r_trans = r_trans[w]
    z = z[w]
    weights12 = weights12[w]

    bins_r_par = np.floor((r_par - r_par_min) / (r_par_max - r_par_min) *
                          num_bins_r_par).astype(int)
    bins_r_trans = (r_trans / r_trans_max * num_bins_r_trans).astype(int)
    bins = bins_r_trans + num_bins_r_trans * bins_r_par

    rebin_weight = np.bincount(bins, weights=weights12)
    rebin_r_par = np.bincount(bins, weights=r_par * weights12)
    rebin_r_trans = np.bincount(bins, weights=r_trans * weights12)
    rebin_z = np.bincount(bins, weights=z * weights12)
    rebin_num_pairs = np.bincount(bins)

    return rebin_weight, rebin_r_par, rebin_r_trans, rebin_z, rebin_num_pairs


@jit(nopython=True)
def compute_xi_forest_pairs(z1, r_comov1, dist_m1, weights1, z2, r_comov2,
                            dist_m2, weights2, ang):
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
            rebin_r_par: The parallel distance of the correlation function
                pixels properly rebinned
            rebin_r_trans: The transverse distance of the correlation function
                pixels properly rebinned
            rebin_z: The redshift of the correlation function pixels properly
                rebinned
            rebin_num_pairs: The number of pairs of the correlation function
                pixels properly rebinned
    """
    r_par = (r_comov1 - r_comov2) * np.cos(ang / 2.)
    if not x_correlation or type_corr in ['DR', 'RD']:
        r_par = np.absolute(r_par)
    r_trans = (dist_m1 + dist_m2) * np.sin(ang / 2.)
    z = (z1 + z2) / 2.
    weights12 = weights1 * weights2

    w = ((r_par >= r_par_min) & (r_par < r_par_max) & (r_trans < r_trans_max) &
         (weights12 > 0.))
    r_par = r_par[w]
    r_trans = r_trans[w]
    z = z[w]
    weights12 = weights12[w]

    num_bins = len(r_par)
    bins = np.zeros(num_bins, dtype=np.int64)
    for ind in range(num_bins):
        bin_r_par = int(
            (r_par[ind] - r_par_min) / (r_par_max - r_par_min) * num_bins_r_par)
        bin_r_trans = int(r_trans[ind] / r_trans_max * num_bins_r_trans)
        bins[ind] = bin_r_trans + num_bins_r_trans * bin_r_par

    rebin_weight = numba_bincount(bins, weights=weights12)
    rebin_r_par = numba_bincount(bins, weights=r_par * weights12)
    rebin_r_trans = numba_bincount(bins, weights=r_trans * weights12)
    rebin_z = numba_bincount(bins, weights=z * weights12)
    rebin_num_pairs = numba_bincount_noweights(bins)

    return rebin_weight, rebin_r_par, rebin_r_trans, rebin_z, rebin_num_pairs


@jit(nopython=True)
def numba_bincount_noweights(bins):
    if len(bins) == 0:
        return np.zeros(0, dtype=np.int64)
    maxbins = bins.max() + 1
    num_bins = len(bins)
    out = np.zeros(maxbins, dtype=np.int64)
    for ind in range(num_bins):
        out[bins[ind]] += 1
    return out


@jit(nopython=True)
def numba_bincount(bins, weights):
    if len(bins) == 0:
        return np.zeros(0, dtype=weights.dtype)
    maxbins = bins.max() + 1
    num_bins = len(bins)
    out = np.zeros(maxbins, dtype=weights.dtype)
    for ind in range(num_bins):
        out[bins[ind]] += weights[ind]
    return out
