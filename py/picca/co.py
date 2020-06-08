"""This module defines functions and variables required for the correlation
analysis of two object catalogues

This module provides several functions:
    - fill_neighs
    - compute_xi
    - compute_xi_forest_pairs
See the respective docstrings for more details
"""
import numpy as np
import scipy as sp
from healpy import query_disc
from numba import jit

from picca.utils import userprint

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
            healpix_neighbours = query_disc(nside,
                                            [obj1.x_cart, obj1.y_cart, obj1.z_cart],
                                            ang_max,
                                            inclusive=True)
            if x_correlation:
                healpix_neighbours = [healpix
                                      for healpix in healpix_neighbours
                                      if healpix in objs2]
                neighbours = [obj2
                              for healpix in healpix_neighbours
                              for obj2 in objs2[healpix]
                              if obj1.thingid != obj2.thingid]
            else:
                healpix_neighbours = [healpix
                                      for healpix in healpix_neighbours
                                      if healpix in objs]
                neighbours = [obj2
                              for healpix in healpix_neighbours
                              for obj2 in objs[healpix]
                              if obj1.thingid != obj2.thingid]
            ang = obj1^neighbours
            w = ang < ang_max
            neighbours = np.array(neighbours)[w]
            obj1.neighbours = np.array([obj2
                                        for obj2 in neighbours
                                        if ((obj2.z_qso + obj1.z_qso)/2. >= z_cut_min and
                                            (obj2.z_qso + obj1.z_qso)/2. < z_cut_max)])


def co(healpixs):

    weights = np.zeros(num_bins_r_par*num_bins_r_trans)
    r_par = np.zeros(num_bins_r_par*num_bins_r_trans)
    r_trans = np.zeros(num_bins_r_par*num_bins_r_trans)
    z  = np.zeros(num_bins_r_par*num_bins_r_trans)
    nb = np.zeros(num_bins_r_par*num_bins_r_trans,dtype=sp.int64)

    for healpix in healpixs:
        for obj1 in objs[healpix]:

            userprint("\rcomputing xi: {}%".format(round(counter.value*100./num_data,2)),end="")
            with lock:
                counter.value += 1

            if (obj1.neighbours.size == 0): continue

            ang      = obj1^obj1.neighbours
            zo2      = sp.array([obj2.z_qso    for obj2 in obj1.neighbours])
            r_comov2 = sp.array([obj2.r_comov for obj2 in obj1.neighbours])
            dist_m2 = sp.array([obj2.dist_m for obj2 in obj1.neighbours])
            weo2     = sp.array([obj2.weights      for obj2 in obj1.neighbours])

            cw,crp,crt,cz,cnb = fast_co(obj1.z_qso,obj1.r_comov,obj1.dist_m,obj1.weights,zo2,r_comov2,dist_m2,weo2,ang)

            weights[:len(cw)]  += cw
            r_par[:len(crp)] += crp
            r_trans[:len(crp)] += crt
            z[:len(crp)]  += cz
            nb[:len(cnb)] += cnb
            setattr(obj1,"neighbours",None)

    w = weights>0.
    r_par[w] /= weights[w]
    r_trans[w] /= weights[w]
    z[w]  /= weights[w]
    return weights,r_par,r_trans,z,nb
@jit
def fast_co(z1,r1,rdm1,w1,z2,r2,rdm2,w2,ang):

    r_par  = (r1-r2)*sp.cos(ang/2.)
    if not x_correlation or type_corr in ['DR','RD']:
        r_par = np.absolute(r_par)
    r_trans  = (rdm1+rdm2)*sp.sin(ang/2.)
    z   = (z1+z2)/2.
    w12 = w1*w2

    w   = (r_par>=r_par_min) & (r_par<r_par_max) & (r_trans<r_trans_max) & (w12>0.)
    r_par  = r_par[w]
    r_trans  = r_trans[w]
    z   = z[w]
    w12 = w12[w]

    bp   = sp.floor((r_par-r_par_min)/(r_par_max-r_par_min)*num_bins_r_par).astype(int)
    bt   = (r_trans/r_trans_max*num_bins_r_trans).astype(int)
    bins = bt + num_bins_r_trans*bp

    cw  = sp.bincount(bins,weights=w12)
    crp = sp.bincount(bins,weights=r_par*w12)
    crt = sp.bincount(bins,weights=r_trans*w12)
    cz  = sp.bincount(bins,weights=z*w12)
    cnb = sp.bincount(bins)

    return cw,crp,crt,cz,cnb
