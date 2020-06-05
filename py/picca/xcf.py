import numpy as np
import scipy as sp
from healpy import query_disc
from numba import jit

from picca import constants
from picca.utils import userprint

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
z_evol_del = None
z_evol_obj = None
lambda_abs = None
alpha_abs = None

data = None
objs = None

reject = None
lock = None

cosmo = None
ang_correlation = False

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
                other_healpix
                for other_healpix in healpix_neighbours
                if other_healpix in objs
            ]
            neighbours = [
                obj for other_healpix in healpix_neighbours
                for obj in objs[other_healpix]
                if obj.thingid != delta.thingid
            ]
            ang = delta^neighbours
            w = ang < ang_max
            if not ang_correlation:
                r_comov = np.array([obj.r_comov for obj in neighbours])
                w &= (delta.r_comov[0] - r_comov)*np.cos(ang/2.) < r_par_max
                w &= (delta.r_comov[-1] - r_comov)*np.cos(ang/2.) > r_par_min
            neighbours = np.array(neighbours)[w]
            delta.neighbours = np.array([
                obj
                for obj in neighbours
                if ((delta.z[-1] + obj.z_qso)/2. >= z_cut_min and
                    (delta.z[-1] + obj.z_qso)/2. < z_cut_max)
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
    xi = np.zeros(num_bins_r_par*num_bins_r_trans)
    weights = np.zeros(num_bins_r_par*num_bins_r_trans)
    r_par = np.zeros(num_bins_r_par*num_bins_r_trans)
    r_trans = np.zeros(num_bins_r_par*num_bins_r_trans)
    z = np.zeros(num_bins_r_par*num_bins_r_trans)
    num_pairs = np.zeros(num_bins_r_par*num_bins_r_trans, dtype=np.int64)

    for healpix in healpixs:
        for delta in data[healpix]:
            with lock:
                counter.value += 1
            userprint(("\rcomputing xi: "
                       "{}%").format(round(counter.value * 100. / num_data, 3)),
                      end="")
            if delta.neighbours.size != 0:
                ang = delta^delta.neighbours
                z_qso = [obj.z_qso for obj in delta.neighbours]
                weights_qso = [obj.weights for obj in delta.neighbours]
                if ang_correlation:
                    lambda_qso = [10.**obj.log_lambda
                                  for obj in delta.neighbours]
                    (rebin_weight, rebin_xi, rebin_r_par, rebin_r_trans,
                     rebin_z, rebin_num_pairs) = compute_xi_forest_pairs(
                         delta.z, 10.**delta.log_lambda, 10.**delta.log_lambda,
                         delta.weights, delta.delta, z_qso, lambda_qso,
                         lambda_qso, weights_qso, ang)
                else:
                    r_comov_qso = [obj.r_comov for obj in delta.neighbours]
                    dist_m_qso = [obj.dist_m for obj in delta.neighbours]
                    (rebin_weight, rebin_xi, rebin_r_par, rebin_r_trans,
                     rebin_z, rebin_num_pairs) = compute_xi_forest_pairs(
                         delta.z, delta.r_comov, delta.dist_m, delta.weights,
                         delta.delta, z_qso, r_comov_qso, dist_m_qso,
                         weights_qso, ang)

                xi[:len(rebin_xi)] += rebin_xi
                weights[:len(rebin_weight)] += rebin_weight
                r_par[:len(rebin_r_par)] += rebin_r_par
                r_trans[:len(rebin_r_trans)] += rebin_r_trans
                z[:len(rebin_z)] += rebin_z
                num_pairs[:len(rebin_num_pairs)] += rebin_num_pairs.astype(int)
            setattr(delta, "neighbours", None)

    w = weights > 0
    xi[w] /= weights[w]
    r_par[w] /= weights[w]
    r_trans[w] /= weights[w]
    z[w] /= weights[w]
    return weights, xi, r_par, r_trans, z, num_pairs

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
        r_par = r_comov1[:, None]/r_comov2
        r_trans = ang*np.ones_like(r_par)
    else:
        r_par = (r_comov1[:, None] - r_comov2)*np.cos(ang/2)
        r_trans = (dist_m1[:, None] + dist_m2)*np.sin(ang/2)
    z = (z1[:, None] + z2)/2

    weights12 = weights1[:, None]*weights2
    delta_times_weight = (weights1*delta1)[:, None]*weights2

    w = (r_par > r_par_min) & (r_par < r_par_max) & (r_trans < r_trans_max)
    r_par = r_par[w]
    r_trans = r_trans[w]
    z = z[w]
    weights12 = weights12[w]
    delta_times_weight = delta_times_weight[w]

    bins_r_par = ((r_par - r_par_min)/(r_par_max - r_par_min)*
                  num_bins_r_par).astype(int)
    bins_r_trans = (r_trans/r_trans_max*num_bins_r_trans).astype(int)
    bins = bins_r_trans + num_bins_r_trans*bins_r_par

    rebin_xi = np.bincount(bins, weights=delta_times_weight)
    rebin_weight = np.bincount(bins, weights=weights12)
    rebin_r_par = np.bincount(bins, weights=r_par*weights12)
    rebin_r_trans = np.bincount(bins, weights=r_trans*weights12)
    rebin_z = np.bincount(bins, weights=z*weights12)
    rebin_num_pairs = np.bincount(bins, weights=(weights12 > 0.))

    return (rebin_weight, rebin_xi, rebin_r_par, rebin_r_trans, rebin_z,
            rebin_num_pairs)

def compute_dmat(pix):
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

    dmat = np.zeros(num_bins_r_par*num_bins_r_trans*num_model_bins_r_trans*
                    num_model_bins_r_par)
    weights_dmat = np.zeros(num_bins_r_par*num_bins_r_trans)
    r_par_eff = np.zeros(num_model_bins_r_trans*num_model_bins_r_par)
    r_trans_eff = np.zeros(num_model_bins_r_trans*num_model_bins_r_par)
    z_eff = np.zeros(num_model_bins_r_trans*num_model_bins_r_par)
    weight_eff = np.zeros(num_model_bins_r_trans*num_model_bins_r_par)

    num_pairs = 0
    num_pairs_used = 0
    for p in pix:
        for d1 in data[p]:
            userprint("\rcomputing xi: {}%".format(round(counter.value*100./num_data,3)),end="")
            with lock:
                counter.value += 1
            r1 = d1.r_comov
            rdm1 = d1.dist_m
            w1 = d1.weights
            l1 = d1.log_lambda
            z1 = d1.z
            r = sp.random.rand(len(d1.neighbours))
            w=r>reject
            if w.sum()==0:continue
            num_pairs += len(d1.neighbours)
            num_pairs_used += w.sum()
            neighbours = d1.neighbours[w]
            ang = d1^neighbours
            r2 = [q.r_comov for q in neighbours]
            rdm2 = [q.dist_m for q in neighbours]
            w2 = [q.weights for q in neighbours]
            z2 = [q.z_qso for q in neighbours]
            fill_dmat(l1,r1,rdm1,z1,w1,r2,rdm2,z2,w2,ang,weights_dmat,dmat,r_par_eff,r_trans_eff,z_eff,weight_eff)
            for el in list(d1.__dict__.keys()):
                setattr(d1,el,None)

    return weights_dmat,dmat.reshape(num_bins_r_par*num_bins_r_trans,num_model_bins_r_par*num_model_bins_r_trans),r_par_eff,r_trans_eff,z_eff,weight_eff,num_pairs,num_pairs_used
@jit
def fill_dmat(l1,r1,rdm1,z1,w1,r2,rdm2,z2,w2,ang,weights_dmat,dmat,r_par_eff,r_trans_eff,z_eff,weight_eff):
    r_par = (r1[:,None]-r2)*sp.cos(ang/2)
    r_trans = (rdm1[:,None]+rdm2)*sp.sin(ang/2)
    z = (z1[:,None]+z2)/2.
    w = (r_par>r_par_min) & (r_par<r_par_max) & (r_trans<r_trans_max)

    bp = ((r_par-r_par_min)/(r_par_max-r_par_min)*num_bins_r_par).astype(int)
    bt = (r_trans/r_trans_max*num_bins_r_trans).astype(int)
    bins = bt + num_bins_r_trans*bp
    bins = bins[w]

    m_bp = ((r_par-r_par_min)/(r_par_max-r_par_min)*num_model_bins_r_par).astype(int)
    m_bt = (r_trans/r_trans_max*num_model_bins_r_trans).astype(int)
    m_bins = m_bt + num_model_bins_r_trans*m_bp
    m_bins = m_bins[w]

    sw1 = w1.sum()
    ml1 = sp.average(l1,weights=w1)

    dl1 = l1-ml1

    slw1 = (w1*dl1**2).sum()

    n1 = len(l1)
    n2 = len(r2)
    ij = np.arange(n1)[:,None]+n1*np.arange(n2)
    ij = ij[w]

    weights = w1[:,None]*w2
    weights = weights[w]
    c = sp.bincount(bins,weights=weights)
    weights_dmat[:len(c)] += c
    eta2 = np.zeros(num_model_bins_r_par*num_model_bins_r_trans*n2)
    eta4 = np.zeros(num_model_bins_r_par*num_model_bins_r_trans*n2)

    c = sp.bincount(m_bins,weights=weights*r_par[w])
    r_par_eff[:c.size] += c
    c = sp.bincount(m_bins,weights=weights*r_trans[w])
    r_trans_eff[:c.size] += c
    c = sp.bincount(m_bins,weights=weights*z[w])
    z_eff[:c.size] += c
    c = sp.bincount(m_bins,weights=weights)
    weight_eff[:c.size] += c

    c = sp.bincount((ij-ij%n1)//n1+n2*m_bins,weights = (w1[:,None]*sp.ones(n2))[w]/sw1)
    eta2[:len(c)]+=c
    c = sp.bincount((ij-ij%n1)//n1+n2*m_bins,weights = ((w1*dl1)[:,None]*sp.ones(n2))[w]/slw1)
    eta4[:len(c)]+=c

    ubb = np.unique(m_bins)
    for k, (ba,m_ba) in enumerate(zip(bins,m_bins)):
        dmat[m_ba+num_model_bins_r_par*num_model_bins_r_trans*ba]+=weights[k]
        i = ij[k]%n1
        j = (ij[k]-i)//n1
        for bb in ubb:
            dmat[bb+num_model_bins_r_par*num_model_bins_r_trans*ba] -= weights[k]*(eta2[j+n2*bb]+eta4[j+n2*bb]*dl1[i])


def metal_dmat(pix,abs_igm="SiII(1526)"):

    dmat = np.zeros(num_bins_r_par*num_bins_r_trans*num_model_bins_r_trans*num_model_bins_r_par)
    weights_dmat = np.zeros(num_bins_r_par*num_bins_r_trans)
    r_par_eff = np.zeros(num_model_bins_r_trans*num_model_bins_r_par)
    r_trans_eff = np.zeros(num_model_bins_r_trans*num_model_bins_r_par)
    z_eff = np.zeros(num_model_bins_r_trans*num_model_bins_r_par)
    weight_eff = np.zeros(num_model_bins_r_trans*num_model_bins_r_par)

    num_pairs = 0
    num_pairs_used = 0
    for p in pix:
        for d in data[p]:
            with lock:
                userprint("\rcomputing metal dmat {}: {}%".format(abs_igm,round(counter.value*100./num_data,3)),end="")
                counter.value += 1

            r = sp.random.rand(len(d.neighbours))
            w=r>reject
            num_pairs += len(d.neighbours)
            num_pairs_used += w.sum()

            rd = d.r_comov
            rdm = d.dist_m
            wd = d.weights
            zd_abs = 10**d.log_lambda/constants.ABSORBER_IGM[abs_igm]-1
            rd_abs = cosmo.get_r_comov(zd_abs)
            rdm_abs = cosmo.get_dist_m(zd_abs)

            wzcut = zd_abs<d.z_qso
            rd = rd[wzcut]
            rdm = rdm[wzcut]
            wd = wd[wzcut]
            zd_abs = zd_abs[wzcut]
            rd_abs = rd_abs[wzcut]
            rdm_abs = rdm_abs[wzcut]
            if rd.size==0: continue

            for q in sp.array(d.neighbours)[w]:
                ang = d^q

                rq = q.r_comov
                rqm = q.dist_m
                wq = q.weights
                zq = q.z_qso
                r_par = (rd-rq)*sp.cos(ang/2)
                r_trans = (rdm+rqm)*sp.sin(ang/2)
                wdq = wd*wq

                wA = (r_par>r_par_min) & (r_par<r_par_max) & (r_trans<r_trans_max)
                bp = ((r_par-r_par_min)/(r_par_max-r_par_min)*num_bins_r_par).astype(int)
                bt = (r_trans/r_trans_max*num_bins_r_trans).astype(int)
                bA = bt + num_bins_r_trans*bp
                c = sp.bincount(bA[wA],weights=wdq[wA])
                weights_dmat[:len(c)]+=c

                rp_abs = (rd_abs-rq)*sp.cos(ang/2)
                rt_abs = (rdm_abs+rqm)*sp.sin(ang/2)
                zwe = ((1.+zd_abs)/(1.+z_ref))**(alpha_abs[abs_igm]-1.)

                bp_abs = ((rp_abs-r_par_min)/(r_par_max-r_par_min)*num_model_bins_r_par).astype(int)
                bt_abs = (rt_abs/r_trans_max*num_model_bins_r_trans).astype(int)
                bBma = bt_abs + num_model_bins_r_trans*bp_abs
                wBma = (rp_abs>r_par_min) & (rp_abs<r_par_max) & (rt_abs<r_trans_max)
                wAB = wA&wBma
                c = sp.bincount(bBma[wAB]+num_model_bins_r_par*num_model_bins_r_trans*bA[wAB],weights=wdq[wAB]*zwe[wAB])
                dmat[:len(c)]+=c

                c = sp.bincount(bBma[wAB],weights=rp_abs[wAB]*wdq[wAB]*zwe[wAB])
                r_par_eff[:len(c)]+=c
                c = sp.bincount(bBma[wAB],weights=rt_abs[wAB]*wdq[wAB]*zwe[wAB])
                r_trans_eff[:len(c)]+=c
                c = sp.bincount(bBma[wAB],weights=(zd_abs+zq)[wAB]/2*wdq[wAB]*zwe[wAB])
                z_eff[:len(c)]+=c
                c = sp.bincount(bBma[wAB],weights=wdq[wAB]*zwe[wAB])
                weight_eff[:len(c)]+=c
            setattr(d,"neighbours",None)

    return weights_dmat,dmat.reshape(num_bins_r_par*num_bins_r_trans,num_model_bins_r_par*num_model_bins_r_trans),r_par_eff,r_trans_eff,z_eff,weight_eff,num_pairs,num_pairs_used

v1d = {}
c1d = {}
max_diagram = None
cfWick = None
cfWick_np = None
cfWick_nt = None
cfWick_rp_min = None
cfWick_rp_max = None
cfWick_rt_max = None
cfWick_angmax = None

def wickT(pix):
    """Compute the Wick covariance matrix for the object-pixel
        cross-correlation

    Args:
        pix (lst): list of HEALpix pixels

    Returns:
        (tuple): results of the Wick computation

    """
    T1 = np.zeros((num_bins_r_par*num_bins_r_trans,num_bins_r_par*num_bins_r_trans))
    T2 = np.zeros((num_bins_r_par*num_bins_r_trans,num_bins_r_par*num_bins_r_trans))
    T3 = np.zeros((num_bins_r_par*num_bins_r_trans,num_bins_r_par*num_bins_r_trans))
    T4 = np.zeros((num_bins_r_par*num_bins_r_trans,num_bins_r_par*num_bins_r_trans))
    T5 = np.zeros((num_bins_r_par*num_bins_r_trans,num_bins_r_par*num_bins_r_trans))
    T6 = np.zeros((num_bins_r_par*num_bins_r_trans,num_bins_r_par*num_bins_r_trans))
    wAll = np.zeros(num_bins_r_par*num_bins_r_trans)
    num_pairs = np.zeros(num_bins_r_par*num_bins_r_trans,dtype=sp.int64)
    num_pairs = 0
    num_pairs_used = 0

    for healpix in pix:

        num_pairs += len(data[healpix])
        r = sp.random.rand(len(data[healpix]))
        w = r>reject
        num_pairs_used += w.sum()
        if w.sum()==0: continue

        for d1 in [ td for ti,td in enumerate(data[healpix]) if w[ti] ]:
            userprint("\rcomputing xi: {}%".format(round(counter.value*100./num_data/(1.-reject),3)),end="")
            with lock:
                counter.value += 1
            if d1.neighbours.size==0: continue

            v1 = v1d[d1.fname](d1.log_lambda)
            w1 = d1.weights
            c1d_1 = (w1*w1[:,None])*c1d[d1.fname](abs(d1.log_lambda-d1.log_lambda[:,None]))*sp.sqrt(v1*v1[:,None])
            r1 = d1.r_comov
            z1 = d1.z

            neighbours = d1.neighbours
            ang12 = d1^neighbours
            r2 = sp.array([q2.r_comov for q2 in neighbours])
            z2 = sp.array([q2.z_qso for q2 in neighbours])
            w2 = sp.array([q2.weights for q2 in neighbours])

            fill_wickT1234(ang12,r1,r2,z1,z2,w1,w2,c1d_1,wAll,num_pairs,T1,T2,T3,T4)

            ### Higher order diagrams
            if (cfWick is None) or (max_diagram<=4): continue
            thingid2 = sp.array([q2.thingid for q2 in neighbours])
            for d3 in sp.array(d1.dneighs):
                if d3.neighbours.size==0: continue

                ang13 = d1^d3

                r3 = d3.r_comov
                w3 = d3.weights

                neighbours = d3.neighbours
                ang34 = d3^neighbours
                r4 = sp.array([q4.r_comov for q4 in neighbours])
                w4 = sp.array([q4.weights for q4 in neighbours])
                thingid4 = sp.array([q4.thingid for q4 in neighbours])

                if max_diagram==5:
                    w = sp.in1d(d1.neighbours,d3.neighbours)
                    if w.sum()==0: continue
                    t_ang12 = ang12[w]
                    t_r2 = r2[w]
                    t_w2 = w2[w]
                    t_thingid2 = thingid2[w]

                    w = sp.in1d(d3.neighbours,d1.neighbours)
                    if w.sum()==0: continue
                    ang34 = ang34[w]
                    r4 = r4[w]
                    w4 = w4[w]
                    thingid4 = thingid4[w]

                fill_wickT56(t_ang12,ang34,ang13,r1,t_r2,r3,r4,w1,t_w2,w3,w4,t_thingid2,thingid4,T5,T6)

    return wAll, num_pairs, num_pairs, num_pairs_used, T1, T2, T3, T4, T5, T6
@jit
def fill_wickT1234(ang,r1,r2,z1,z2,w1,w2,c1d_1,wAll,num_pairs,T1,T2,T3,T4):
    """Compute the Wick covariance matrix for the object-pixel
        cross-correlation for the T1, T2, T3 and T4 diagrams:
        i.e. the contribution of the 1D auto-correlation to the
        covariance matrix

    Args:
        ang (float array): angle between forest and array of objects
        r1 (float array): comoving distance to each pixel of the forest [Mpc/h]
        r2 (float array): comoving distance to each object [Mpc/h]
        z1 (float array): redshift of each pixel of the forest
        z2 (float array): redshift of each object
        w1 (float array): weight of each pixel of the forest
        w2 (float array): weight of each object
        c1d_1 (float array): covariance between two pixels of the same forest
        wAll (float array): Sum of weight
        num_pairs (int64 array): Number of pairs
        T1 (float 2d array): Contribution of diagram T1
        T2 (float 2d array): Contribution of diagram T2
        T3 (float 2d array): Contribution of diagram T3
        T4 (float 2d array): Contribution of diagram T4

    Returns:

    """
    r_par = (r1[:,None]-r2)*sp.cos(ang/2.)
    r_trans = (r1[:,None]+r2)*sp.sin(ang/2.)
    zw1 = ((1.+z1)/(1.+z_ref))**(z_evol_del-1.)
    zw2 = ((1.+z2)/(1.+z_ref))**(z_evol_obj-1.)
    weights = w1[:,None]*w2
    we1 = w1[:,None]*sp.ones(len(r2))
    idxPix = np.arange(r1.size)[:,None]*sp.ones(len(r2),dtype='int')
    idxQso = sp.ones(r1.size,dtype='int')[:,None]*np.arange(len(r2))

    bp = ((r_par-r_par_min)/(r_par_max-r_par_min)*num_bins_r_par).astype(int)
    bt = (r_trans/r_trans_max*num_bins_r_trans).astype(int)
    ba = bt + num_bins_r_trans*bp

    w = (r_par>r_par_min) & (r_par<r_par_max) & (r_trans<r_trans_max)
    if w.sum()==0: return

    ba = ba[w]
    weights = weights[w]
    we1 = we1[w]
    idxPix = idxPix[w]
    idxQso = idxQso[w]

    for k1 in range(ba.size):
        p1 = ba[k1]
        i1 = idxPix[k1]
        q1 = idxQso[k1]
        wAll[p1] += weights[k1]
        num_pairs[p1] += 1
        T1[p1,p1] += (weights[k1]**2)/we1[k1]*zw1[i1]

        for k2 in range(k1+1,ba.size):
            p2 = ba[k2]
            i2 = idxPix[k2]
            q2 = idxQso[k2]
            if q1==q2:
                wcorr = c1d_1[i1,i2]*(zw2[q1]**2)
                T2[p1,p2] += wcorr
                T2[p2,p1] += wcorr
            elif i1==i2:
                wcorr = (weights[k1]*weights[k2])/we1[k1]*zw1[i1]
                T3[p1,p2] += wcorr
                T3[p2,p1] += wcorr
            else:
                wcorr = c1d_1[i1,i2]*zw2[q1]*zw2[q2]
                T4[p1,p2] += wcorr
                T4[p2,p1] += wcorr

    return
@jit
def fill_wickT56(ang12,ang34,ang13,r1,r2,r3,r4,w1,w2,w3,w4,thingid2,thingid4,T5,T6):
    """Compute the Wick covariance matrix for the object-pixel
        cross-correlation for the T5 and T6 diagrams:
        i.e. the contribution of the 3D auto-correlation to the
        covariance matrix

    Args:
        ang12 (float array): angle between forest and array of objects
        ang34 (float array): angle between another forest and another array of objects
        ang13 (float array): angle between the two forests
        r1 (float array): comoving distance to each pixel of the forest [Mpc/h]
        r2 (float array): comoving distance to each object [Mpc/h]
        r3 (float array): comoving distance to each pixel of another forests [Mpc/h]
        r4 (float array): comoving distance to each object paired to the other forest [Mpc/h]
        w1 (float array): weight of each pixel of the forest
        w2 (float array): weight of each object
        w3 (float array): weight of each pixel of another forest
        w4 (float array): weight of each object paired to the other forest
        thingid2 (float array): THING_ID of each object
        thingid4 (float array): THING_ID of each object paired to the other forest
        T5 (float 2d array): Contribution of diagram T5
        T6 (float 2d array): Contribution of diagram T6

    Returns:

    """

    ### Pair forest_1 - forest_3
    r_par = np.absolute(r1-r3[:,None])*sp.cos(ang13/2.)
    r_trans = (r1+r3[:,None])*sp.sin(ang13/2.)

    w = (r_par<cfWick_rp_max) & (r_trans<cfWick_rt_max) & (r_par>=cfWick_rp_min)
    if w.sum()==0: return
    bp = sp.floor((r_par-cfWick_rp_min)/(cfWick_rp_max-cfWick_rp_min)*cfWick_np).astype(int)
    bt = (r_trans/cfWick_rt_max*cfWick_nt).astype(int)
    ba13 = bt + cfWick_nt*bp
    ba13[~w] = 0
    cf13 = cfWick[ba13]
    cf13[~w] = 0.

    ### Pair forest_1 - object_2
    r_par = (r1[:,None]-r2)*sp.cos(ang12/2.)
    r_trans = (r1[:,None]+r2)*sp.sin(ang12/2.)
    weights = w1[:,None]*w2
    pix = (np.arange(r1.size)[:,None]*sp.ones_like(r2)).astype(int)
    thingid = sp.ones_like(w1[:,None]).astype(int)*thingid2

    w = (r_par>r_par_min) & (r_par<r_par_max) & (r_trans<r_trans_max)
    if w.sum()==0: return
    r_par = r_par[w]
    r_trans = r_trans[w]
    we12 = weights[w]
    pix12 = pix[w]
    thingid12 = thingid[w]
    bp = ((r_par-r_par_min)/(r_par_max-r_par_min)*num_bins_r_par).astype(int)
    bt = (r_trans/r_trans_max*num_bins_r_trans).astype(int)
    ba12 = bt + num_bins_r_trans*bp

    ### Pair forest_3 - object_4
    r_par = (r3[:,None]-r4)*sp.cos(ang34/2.)
    r_trans = (r3[:,None]+r4)*sp.sin(ang34/2.)
    weights = w3[:,None]*w4
    pix = (np.arange(r3.size)[:,None]*sp.ones_like(r4)).astype(int)
    thingid = sp.ones_like(w3[:,None]).astype(int)*thingid4

    w = (r_par>r_par_min) & (r_par<r_par_max) & (r_trans<r_trans_max)
    if w.sum()==0: return
    r_par = r_par[w]
    r_trans = r_trans[w]
    we34 = weights[w]
    pix34 = pix[w]
    thingid34 = thingid[w]
    bp = ((r_par-r_par_min)/(r_par_max-r_par_min)*num_bins_r_par).astype(int)
    bt = (r_trans/r_trans_max*num_bins_r_trans).astype(int)
    ba34 = bt + num_bins_r_trans*bp

    ### T5
    for k1, p1 in enumerate(ba12):
        pix1 = pix12[k1]
        t1 = thingid12[k1]
        w1 = we12[k1]

        w = thingid34==t1
        for k2, p2 in enumerate(ba34[w]):
            pix2 = pix34[w][k2]
            t2 = thingid34[w][k2]
            w2 = we34[w][k2]
            wcorr = cf13[pix2,pix1]*w1*w2
            T5[p1,p2] += wcorr
            T5[p2,p1] += wcorr

    ### T6
    if max_diagram==5: return
    for k1, p1 in enumerate(ba12):
        pix1 = pix12[k1]
        t1 = thingid12[k1]
        w1 = we12[k1]

        for k2, p2 in enumerate(ba34):
            pix2 = pix34[k2]
            t2 = thingid34[k2]
            w2 = we34[k2]
            wcorr = cf13[pix2,pix1]*w1*w2
            if t2==t1: continue
            T6[p1,p2] += wcorr
            T6[p2,p1] += wcorr

    return
def xcf1d(pix):
    """Compute the 1D cross-correlation between delta and objects on the same LOS

    Args:
        pix (list): List of HEALpix to compute

    Returns:
        weights (float array): weights
        xi (float array): correlation
        r_par (float array): wavelenght ratio
        z (float array): Mean redshift of pairs
        num_pairs (int array): Number of pairs
    """
    xi = np.zeros(num_bins_r_par)
    weights = np.zeros(num_bins_r_par)
    r_par = np.zeros(num_bins_r_par)
    z = np.zeros(num_bins_r_par)
    num_pairs = np.zeros(num_bins_r_par,dtype=sp.int64)

    for healpix in pix:
        for d in data[healpix]:

            neighbours = [q for q in objs[healpix] if q.thingid==d.thingid]
            if len(neighbours)==0: continue

            z_qso = [ q.z_qso for q in neighbours ]
            weights_qso = [ q.weights for q in neighbours ]
            lambda_qso = [ 10.**q.log_lambda for q in neighbours ]
            ang = np.zeros(len(lambda_qso))

            cw,cd,crp,_,cz,cnb = fast_xcf(d.z,10.**d.log_lambda,10.**d.log_lambda,d.weights,d.delta,z_qso,lambda_qso,lambda_qso,weights_qso,ang)

            xi[:cd.size] += cd
            weights[:cw.size] += cw
            r_par[:crp.size] += crp
            z[:cz.size] += cz
            num_pairs[:cnb.size] += cnb.astype(int)

            for el in list(d.__dict__.keys()):
                setattr(d,el,None)

    w = weights>0.
    xi[w] /= weights[w]
    r_par[w] /= weights[w]
    z[w] /= weights[w]

    return weights,xi,r_par,z,num_pairs
