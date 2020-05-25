"""This module defines functions and variables required for the correlation
analysis of two delta fields

This module provides several functions:
    - fill_neighs
    - fill_neighs_x_correlation
    - cf
    - fast_cf
    - dmat
    - fill_dmat
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
ntm = None
npm = None
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

def cf(healpixs):
    """Computes the correlation function for each of the healpix.

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
            nb:
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
                     rebin_num_pairs) = fast_cf(delta1.z,
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
                     rebin_num_pairs) = fast_cf(delta1.z,
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
def fast_cf(z1, r_comov1, dist_m1, weights1, delta1, z2, r_comov2, dist_m2, w2,
            delta2, ang, same_half_plate):
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
    delta_times_weight12 = delta_times_weight*delta_times_weight[:, None]
    weights12 = weights1*weights2[:, None]
    z = (z1 + z2[:, None])/2

    w = (r_par < r_par_max) & (r_trans < r_trans_max) & (r_par >= r_par_min)

    r_par = r_par[w]
    r_trans = r_trans[w]
    z = z[w]
    delta_times_weight12 = delta_times_weight12[w]
    weights12 = weights12[w]
    bin_r_par = np.floor((r_par - r_par_min)/
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

def dmat(pix):

    dm = np.zeros(num_bins_r_par*num_bins_r_trans*ntm*npm)
    wdm = np.zeros(num_bins_r_par*num_bins_r_trans)
    rpeff = np.zeros(ntm*npm)
    rteff = np.zeros(ntm*npm)
    zeff = np.zeros(ntm*npm)
    weff = np.zeros(ntm*npm)

    npairs = 0
    npairs_used = 0
    for p in pix:
        for d1 in data[p]:
            userprint("\rcomputing xi: {}%".format(round(counter.value*100./num_data,3)),end="")
            with lock:
                counter.value += 1
            order1 = d1.order
            r1 = d1.r_comov
            rdm1 = d1.dist_m
            w1 = d1.weights
            l1 = d1.log_lambda
            z1 = d1.z
            r = sp.random.rand(len(d1.neighbours))
            w=r>reject
            npairs += len(d1.neighbours)
            npairs_used += w.sum()
            for d2 in sp.array(d1.neighbours)[w]:
                same_half_plate = (d1.plate == d2.plate) and\
                        ( (d1.fiberid<=500 and d2.fiberid<=500) or (d1.fiberid>500 and d2.fiberid>500) )
                order2 = d2.order
                ang = d1^d2
                r2 = d2.r_comov
                rdm2 = d2.dist_m
                w2 = d2.weights
                l2 = d2.log_lambda
                z2 = d2.z
                fill_dmat(l1,l2,r1,r2,rdm1,rdm2,z1,z2,w1,w2,ang,wdm,dm,rpeff,rteff,zeff,weff,same_half_plate,order1,order2)
            setattr(d1,"neighs",None)

    return wdm,dm.reshape(num_bins_r_par*num_bins_r_trans,npm*ntm),rpeff,rteff,zeff,weff,npairs,npairs_used
@jit
def fill_dmat(l1,l2,r1,r2,rdm1,rdm2,z1,z2,w1,w2,ang,wdm,dm,rpeff,rteff,zeff,weff,same_half_plate,order1,order2):

    r_par = (r1[:,None]-r2)*sp.cos(ang/2)
    if  not x_correlation:
        r_par = abs(r_par)
    r_trans = (rdm1[:,None]+rdm2)*sp.sin(ang/2)
    z = (z1[:,None]+z2)/2.

    w = (r_par<r_par_max) & (r_trans<r_trans_max) & (r_par>=r_par_min)

    bp = sp.floor((r_par-r_par_min)/(r_par_max-r_par_min)*num_bins_r_par).astype(int)
    bt = (r_trans/r_trans_max*num_bins_r_trans).astype(int)
    bins = bt + num_bins_r_trans*bp
    bins = bins[w]

    m_bp = sp.floor((r_par-r_par_min)/(r_par_max-r_par_min)*npm).astype(int)
    m_bt = (r_trans/r_trans_max*ntm).astype(int)
    m_bins = m_bt + ntm*m_bp
    m_bins = m_bins[w]

    sw1 = w1.sum()
    sw2 = w2.sum()

    ml1 = sp.average(l1,weights=w1)
    ml2 = sp.average(l2,weights=w2)

    dl1 = l1-ml1
    dl2 = l2-ml2

    slw1 = (w1*dl1**2).sum()
    slw2 = (w2*dl2**2).sum()

    n1 = len(l1)
    n2 = len(l2)
    ij = np.arange(n1)[:,None]+n1*np.arange(n2)
    ij = ij[w]

    weights = w1[:,None]*w2
    weights = weights[w]

    if remove_same_half_plate_close_pairs and same_half_plate:
        wsame = abs(r_par[w])<(r_par_max-r_par_min)/num_bins_r_par
        weights[wsame] = 0.

    c = sp.bincount(m_bins,weights=weights*r_par[w])
    rpeff[:c.size] += c
    c = sp.bincount(m_bins,weights=weights*r_trans[w])
    rteff[:c.size] += c
    c = sp.bincount(m_bins,weights=weights*z[w])
    zeff[:c.size] += c
    c = sp.bincount(m_bins,weights=weights)
    weff[:c.size] += c

    c = sp.bincount(bins,weights=weights)
    wdm[:len(c)] += c
    eta1 = np.zeros(npm*ntm*n1)
    eta2 = np.zeros(npm*ntm*n2)
    eta3 = np.zeros(npm*ntm*n1)
    eta4 = np.zeros(npm*ntm*n2)
    eta5 = np.zeros(npm*ntm)
    eta6 = np.zeros(npm*ntm)
    eta7 = np.zeros(npm*ntm)
    eta8 = np.zeros(npm*ntm)

    c = sp.bincount(ij%n1+n1*m_bins,weights=(sp.ones(n1)[:,None]*w2)[w]/sw2)
    eta1[:len(c)]+=c
    c = sp.bincount((ij-ij%n1)//n1+n2*m_bins,weights = (w1[:,None]*sp.ones(n2))[w]/sw1)
    eta2[:len(c)]+=c
    c = sp.bincount(m_bins,weights=(w1[:,None]*w2)[w]/sw1/sw2)
    eta5[:len(c)]+=c

    if order2==1:
        c = sp.bincount(ij%n1+n1*m_bins,weights=(sp.ones(n1)[:,None]*w2*dl2)[w]/slw2)
        eta3[:len(c)]+=c
        c = sp.bincount(m_bins,weights=(w1[:,None]*(w2*dl2))[w]/sw1/slw2)
        eta6[:len(c)]+=c
    if order1==1:
        c = sp.bincount((ij-ij%n1)//n1+n2*m_bins,weights = ((w1*dl1)[:,None]*sp.ones(n2))[w]/slw1)
        eta4[:len(c)]+=c
        c = sp.bincount(m_bins,weights=((w1*dl1)[:,None]*w2)[w]/slw1/sw2)
        eta7[:len(c)]+=c
        if order2==1:
            c = sp.bincount(m_bins,weights=((w1*dl1)[:,None]*(w2*dl2))[w]/slw1/slw2)
            eta8[:len(c)]+=c

    ubb = np.unique(m_bins)
    for k, (ba,m_ba) in enumerate(zip(bins,m_bins)):
        dm[m_ba+npm*ntm*ba]+=weights[k]
        i = ij[k]%n1
        j = (ij[k]-i)//n1
        for bb in ubb:
            dm[bb+npm*ntm*ba] += weights[k]*(eta5[bb]+eta6[bb]*dl2[j]+eta7[bb]*dl1[i]+eta8[bb]*dl1[i]*dl2[j])\
             - weights[k]*(eta1[i+n1*bb]+eta3[i+n1*bb]*dl2[j]+eta2[j+n2*bb]+eta4[j+n2*bb]*dl1[i])

def metal_dmat(pix,abs_igm1="LYA",abs_igm2="SiIII(1207)"):

    dm = np.zeros(num_bins_r_par*num_bins_r_trans*ntm*npm)
    wdm = np.zeros(num_bins_r_par*num_bins_r_trans)
    rpeff = np.zeros(ntm*npm)
    rteff = np.zeros(ntm*npm)
    zeff = np.zeros(ntm*npm)
    weff = np.zeros(ntm*npm)

    npairs = 0
    npairs_used = 0
    for p in pix:
        for d1 in data[p]:
            userprint("\rcomputing metal dmat {} {}: {}%".format(abs_igm1,abs_igm2,round(counter.value*100./num_data,3)),end="")
            with lock:
                counter.value += 1

            r = sp.random.rand(len(d1.neighbours))
            w=r>reject
            npairs += len(d1.neighbours)
            npairs_used += w.sum()
            for d2 in sp.array(d1.neighbours)[w]:
                r1 = d1.r_comov
                rdm1 = d1.dist_m
                z1_abs1 = 10**d1.log_lambda/constants.ABSORBER_IGM[abs_igm1]-1
                r1_abs1 = cosmo.get_r_comov(z1_abs1)
                rdm1_abs1 = cosmo.get_dist_m(z1_abs1)
                w1 = d1.weights

                wzcut = z1_abs1<d1.z_qso
                r1 = r1[wzcut]
                rdm1 = rdm1[wzcut]
                w1 = w1[wzcut]
                r1_abs1 = r1_abs1[wzcut]
                rdm1_abs1 = rdm1_abs1[wzcut]
                z1_abs1 = z1_abs1[wzcut]

                same_half_plate = (d1.plate == d2.plate) and\
                        ( (d1.fiberid<=500 and d2.fiberid<=500) or (d1.fiberid>500 and d2.fiberid>500) )
                ang = d1^d2
                r2 = d2.r_comov
                rdm2 = d2.dist_m
                z2_abs2 = 10**d2.log_lambda/constants.ABSORBER_IGM[abs_igm2]-1
                r2_abs2 = cosmo.get_r_comov(z2_abs2)
                rdm2_abs2 = cosmo.get_dist_m(z2_abs2)
                w2 = d2.weights

                wzcut = z2_abs2<d2.z_qso
                r2 = r2[wzcut]
                rdm2 = rdm2[wzcut]
                w2 = w2[wzcut]
                r2_abs2 = r2_abs2[wzcut]
                rdm2_abs2 = rdm2_abs2[wzcut]
                z2_abs2 = z2_abs2[wzcut]

                r_par = (r1[:,None]-r2)*sp.cos(ang/2)
                if not x_correlation:
                    r_par = abs(r_par)

                r_trans = (rdm1[:,None]+rdm2)*sp.sin(ang/2)
                w12 = w1[:,None]*w2

                bp = sp.floor((r_par-r_par_min)/(r_par_max-r_par_min)*num_bins_r_par).astype(int)
                bt = (r_trans/r_trans_max*num_bins_r_trans).astype(int)

                if remove_same_half_plate_close_pairs and same_half_plate:
                    wp = abs(r_par) < (r_par_max-r_par_min)/num_bins_r_par
                    w12[wp] = 0.

                bA = bt + num_bins_r_trans*bp
                wA = (bp<num_bins_r_par) & (bt<num_bins_r_trans) & (bp >=0)
                c = sp.bincount(bA[wA],weights=w12[wA])
                wdm[:len(c)]+=c

                rp_abs1_abs2 = (r1_abs1[:,None]-r2_abs2)*sp.cos(ang/2)

                if not x_correlation:
                    rp_abs1_abs2 = abs(rp_abs1_abs2)

                rt_abs1_abs2 = (rdm1_abs1[:,None]+rdm2_abs2)*sp.sin(ang/2)
                zwe12 = (1+z1_abs1[:,None])**(alpha_abs[abs_igm1]-1)*(1+z2_abs2)**(alpha_abs[abs_igm2]-1)/(1+z_ref)**(alpha_abs[abs_igm1]+alpha_abs[abs_igm2]-2)

                bp_abs1_abs2 = sp.floor((rp_abs1_abs2-r_par_min)/(r_par_max-r_par_min)*npm).astype(int)
                bt_abs1_abs2 = (rt_abs1_abs2/r_trans_max*ntm).astype(int)
                bBma = bt_abs1_abs2 + ntm*bp_abs1_abs2
                wBma = (bp_abs1_abs2<npm) & (bt_abs1_abs2<ntm) & (bp_abs1_abs2>=0)
                wAB = wA & wBma
                c = sp.bincount(bBma[wAB]+npm*ntm*bA[wAB],weights=w12[wAB]*zwe12[wAB])
                dm[:len(c)]+=c
                c = sp.bincount(bBma[wAB],weights=rp_abs1_abs2[wAB]*w12[wAB]*zwe12[wAB])
                rpeff[:len(c)]+=c
                c = sp.bincount(bBma[wAB],weights=rt_abs1_abs2[wAB]*w12[wAB]*zwe12[wAB])
                rteff[:len(c)]+=c
                c = sp.bincount(bBma[wAB],weights=(z1_abs1[:,None]+z2_abs2)[wAB]/2*w12[wAB]*zwe12[wAB])
                zeff[:len(c)]+=c
                c = sp.bincount(bBma[wAB],weights=w12[wAB]*zwe12[wAB])
                weff[:len(c)]+=c

                if ((not x_correlation) and (abs_igm1 != abs_igm2)) or (x_correlation and (lambda_abs == lambda_abs2)):
                    r1 = d1.r_comov
                    rdm1 = d1.dist_m
                    w1 = d1.weights
                    z1_abs2 = 10**d1.log_lambda/constants.ABSORBER_IGM[abs_igm2]-1
                    r1_abs2 = cosmo.get_r_comov(z1_abs2)
                    rdm1_abs2 = cosmo.get_dist_m(z1_abs2)

                    wzcut = z1_abs2<d1.z_qso
                    r1 = r1[wzcut]
                    rdm1 = rdm1[wzcut]
                    w1 = w1[wzcut]
                    z1_abs2 = z1_abs2[wzcut]
                    r1_abs2 = r1_abs2[wzcut]
                    rdm1_abs2 = rdm1_abs2[wzcut]

                    r2 = d2.r_comov
                    rdm2 = d2.dist_m
                    w2 = d2.weights
                    z2_abs1 = 10**d2.log_lambda/constants.ABSORBER_IGM[abs_igm1]-1
                    r2_abs1 = cosmo.get_r_comov(z2_abs1)
                    rdm2_abs1 = cosmo.get_dist_m(z2_abs1)

                    wzcut = z2_abs1<d2.z_qso
                    r2 = r2[wzcut]
                    rdm2 = rdm2[wzcut]
                    w2 = w2[wzcut]
                    z2_abs1 = z2_abs1[wzcut]
                    r2_abs1 = r2_abs1[wzcut]
                    rdm2_abs1 = rdm2_abs1[wzcut]

                    r_par = (r1[:,None]-r2)*sp.cos(ang/2)
                    if not x_correlation:
                        r_par = abs(r_par)

                    r_trans = (rdm1[:,None]+rdm2)*sp.sin(ang/2)
                    w12 = w1[:,None]*w2

                    bp = sp.floor((r_par-r_par_min)/(r_par_max-r_par_min)*num_bins_r_par).astype(int)
                    bt = (r_trans/r_trans_max*num_bins_r_trans).astype(int)
                    if remove_same_half_plate_close_pairs and same_half_plate:
                        wp = abs(r_par) < (r_par_max-r_par_min)/num_bins_r_par
                        w12[wp] = 0.
                    bA = bt + num_bins_r_trans*bp
                    wA = (bp<num_bins_r_par) & (bt<num_bins_r_trans) & (bp >=0)
                    c = sp.bincount(bA[wA],weights=w12[wA])
                    wdm[:len(c)]+=c
                    rp_abs2_abs1 = (r1_abs2[:,None]-r2_abs1)*sp.cos(ang/2)
                    if not x_correlation:
                        rp_abs2_abs1 = abs(rp_abs2_abs1)

                    rt_abs2_abs1 = (rdm1_abs2[:,None]+rdm2_abs1)*sp.sin(ang/2)
                    zwe21 = (1+z1_abs2[:,None])**(alpha_abs[abs_igm2]-1)*(1+z2_abs1)**(alpha_abs[abs_igm1]-1)/(1+z_ref)**(alpha_abs[abs_igm1]+alpha_abs[abs_igm2]-2)

                    bp_abs2_abs1 = sp.floor((rp_abs2_abs1-r_par_min)/(r_par_max-r_par_min)*npm).astype(int)
                    bt_abs2_abs1 = (rt_abs2_abs1/r_trans_max*ntm).astype(int)
                    bBam = bt_abs2_abs1 + ntm*bp_abs2_abs1
                    wBam = (bp_abs2_abs1<npm) & (bt_abs2_abs1<ntm) & (bp_abs2_abs1>=0)
                    wAB = wA & wBam

                    c = sp.bincount(bBam[wAB],weights=rp_abs2_abs1[wAB]*w12[wAB]*zwe21[wAB])
                    rpeff[:len(c)]+=c
                    c = sp.bincount(bBam[wAB],weights=rt_abs2_abs1[wAB]*w12[wAB]*zwe21[wAB])
                    rteff[:len(c)]+=c
                    c = sp.bincount(bBam[wAB],weights=(z1_abs2[:,None]+z2_abs1)[wAB]/2*w12[wAB]*zwe21[wAB])
                    zeff[:len(c)]+=c
                    c = sp.bincount(bBam[wAB],weights=w12[wAB]*zwe21[wAB])
                    weff[:len(c)]+=c

                    c = sp.bincount(bBam[wAB]+npm*ntm*bA[wAB],weights=w12[wAB]*zwe21[wAB])
                    dm[:len(c)]+=c
            setattr(d1,"neighs",None)

    return wdm,dm.reshape(num_bins_r_par*num_bins_r_trans,npm*ntm),rpeff,rteff,zeff,weff,npairs,npairs_used



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

    for d1 in data[pix]:
        bins1 = ((d1.log_lambda-log_lambda_min)/delta_log_lambda+0.5).astype(int)
        wde1 = d1.weights*d1.delta
        we1 = d1.weights

        d2thingid = [d2.thingid for d2 in data2[pix]]
        neighs = data2[pix][sp.in1d(d2thingid,[d1.thingid])]
        for d2 in neighs:
            bins2 = ((d2.log_lambda-log_lambda_min)/delta_log_lambda+0.5).astype(int)
            bins = bins1 + n1d*bins2[:,None]
            wde2 = d2.weights*d2.delta
            we2 = d2.weights
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
    npairs = 0
    npairs_used = 0

    for ipix in pix:

        r = sp.random.rand(len(data[ipix]))
        w = r>reject
        npairs += len(data[ipix])
        npairs_used += w.sum()
        if w.sum()==0: continue

        for d1 in [ td for ti,td in enumerate(data[ipix]) if w[ti] ]:
            userprint("\rcomputing xi: {}%".format(round(counter.value*100./num_data/(1.-reject),3)),end="")
            with lock:
                counter.value += 1
            if len(d1.neighbours)==0: continue

            v1 = v1d[d1.fname](d1.log_lambda)
            w1 = d1.weights
            c1d_1 = (w1*w1[:,None])*c1d[d1.fname](abs(d1.log_lambda-d1.log_lambda[:,None]))*sp.sqrt(v1*v1[:,None])
            r1 = d1.r_comov
            z1 = d1.z

            for i2,d2 in enumerate(d1.neighbours):
                ang12 = d1^d2

                v2 = v1d[d2.fname](d2.log_lambda)
                w2 = d2.weights
                c1d_2 = (w2*w2[:,None])*c1d[d2.fname](abs(d2.log_lambda-d2.log_lambda[:,None]))*sp.sqrt(v2*v2[:,None])
                r2 = d2.r_comov
                z2 = d2.z

                fill_wickT123(r1,r2,ang12,w1,d2.weights,z1,z2,c1d_1,c1d_2,wAll,nb,T1,T2,T3)
                if max_diagram<=3: continue

                ### d3 and d2 have the same 'fname'
                for d3 in d1.neighbours[:i2]:
                    ang13 = d1^d3
                    ang23 = d2^d3

                    v3 = v1d[d3.fname](d3.log_lambda)
                    w3 = d3.weights
                    c1d_3 = (w3*w3[:,None])*c1d[d3.fname](abs(d3.log_lambda-d3.log_lambda[:,None]))*sp.sqrt(v3*v3[:,None])
                    r3 = d3.r_comov
                    z3 = d3.z

                    fill_wickT45(r1,r2,r3, ang12,ang13,ang23, w1,w2,w3,
                        z1,z2,z3, c1d_1,c1d_2,c1d_3,
                        d1.fname,d2.fname,d3.fname,
                        T4,T5)

                ### TODO: when there is two different catalogs
                ### d3 and d1 have the same 'fname'

    return wAll, nb, npairs, npairs_used, T1, T2, T3, T4, T5, T6
@jit
def fill_wickT123(r1,r2,ang,w1,w2,z1,z2,c1d_1,c1d_2,wAll,nb,T1,T2,T3):

    n1 = len(r1)
    n2 = len(r2)
    i1 = np.arange(n1)
    i2 = np.arange(n2)
    zw1 = ((1+z1)/(1+z_ref))**(alpha-1)
    zw2 = ((1+z2)/(1+z_ref))**(alpha2-1)

    bins = i1[:,None]+n1*i2
    r_par = (r1[:,None]-r2)*sp.cos(ang/2)
    if not x_correlation:
        r_par = abs(r_par)
    r_trans = (r1[:,None]+r2)*sp.sin(ang/2)
    bp = sp.floor((r_par-r_par_min)/(r_par_max-r_par_min)*num_bins_r_par).astype(int)
    bt = (r_trans/r_trans_max*num_bins_r_trans).astype(int)
    ba = bt + num_bins_r_trans*bp
    weights = w1[:,None]*w2
    we1 = w1[:,None]*sp.ones(w2.size)
    we2 = sp.ones(w1.size)[:,None]*w2
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
def fill_wickT45(r1,r2,r3, ang12,ang13,ang23, w1,w2,w3, z1,z2,z3, c1d_1,c1d_2,c1d_3, fname1,fname2,fname3, T4,T5):
    """

    """

    ### forest-1 x forest-2
    r_par = (r1[:,None]-r2)*sp.cos(ang12/2.)
    if not x_correlation:
        r_par = np.absolute(r_par)
    r_trans = (r1[:,None]+r2)*sp.sin(ang12/2.)
    pix1_12 = (np.arange(r1.size)[:,None]*sp.ones(r2.size)).astype(int)
    pix2_12 = (sp.ones(r1.size)[:,None]*np.arange(r2.size)).astype(int)
    w = (r_par<r_par_max) & (r_trans<r_trans_max) & (r_par>=r_par_min)
    if w.sum()==0: return
    bp = sp.floor((r_par-r_par_min)/(r_par_max-r_par_min)*num_bins_r_par).astype(int)
    bt = (r_trans/r_trans_max*num_bins_r_trans).astype(int)
    ba12 = bt + num_bins_r_trans*bp
    ba12[~w] = 0
    cf12 = cfWick['{}_{}'.format(fname1,fname2)][ba12]
    cf12[~w] = 0.

    ba12 = ba12[w]
    pix1_12 = pix1_12[w]
    pix2_12 = pix2_12[w]

    ### forest-1 x forest-3
    r_par = (r1[:,None]-r3)*sp.cos(ang13/2.)
    if not x_correlation:
        r_par = np.absolute(r_par)
    r_trans = (r1[:,None]+r3)*sp.sin(ang13/2.)
    pix1_13 = (np.arange(r1.size)[:,None]*sp.ones(r3.size)).astype(int)
    pix3_13 = (sp.ones(r1.size)[:,None]*np.arange(r3.size)).astype(int)
    w = (r_par<r_par_max) & (r_trans<r_trans_max) & (r_par>=r_par_min)
    if w.sum()==0: return
    bp = sp.floor((r_par-r_par_min)/(r_par_max-r_par_min)*num_bins_r_par).astype(int)
    bt = (r_trans/r_trans_max*num_bins_r_trans).astype(int)
    ba13 = bt + num_bins_r_trans*bp
    ba13[~w] = 0
    cf13 = cfWick['{}_{}'.format(fname1,fname3)][ba13]
    cf13[~w] = 0.

    ba13 = ba13[w]
    pix1_13 = pix1_13[w]
    pix3_13 = pix3_13[w]

    ### forest-2 x forest-3
    r_par = (r2[:,None]-r3)*sp.cos(ang23/2.)
    if not x_correlation:
        r_par = np.absolute(r_par)
    r_trans = (r2[:,None]+r3)*sp.sin(ang23/2.)
    pix2_23 = (np.arange(r2.size)[:,None]*sp.ones(r3.size)).astype(int)
    pix3_23 = (sp.ones(r2.size)[:,None]*np.arange(r3.size)).astype(int)
    w = (r_par<r_par_max) & (r_trans<r_trans_max) & (r_par>=r_par_min)
    if w.sum()==0: return
    bp = sp.floor((r_par-r_par_min)/(r_par_max-r_par_min)*num_bins_r_par).astype(int)
    bt = (r_trans/r_trans_max*num_bins_r_trans).astype(int)
    ba23 = bt + num_bins_r_trans*bp
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
                wcorr = w1[tpix1_12]*tcf23 ### TODO work on the good formula
                T4[p1,p2] += wcorr
                T4[p2,p1] += wcorr
            else:
                wcorr = c1d_1[tpix1_12,tpix1_13]*tcf23 ### TODO work on the good formula
                T5[p1,p2] += wcorr
                T5[p2,p1] += wcorr

    return
