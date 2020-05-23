
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
x_correlation = None

counter = None
lock = None

def fill_neighs(pix):
    for ipix in pix:
        for o1 in objs[ipix]:
            npix = query_disc(nside,[o1.x_cart,o1.y_cart,o1.z_cart],ang_max,inclusive = True)
            npix = [p for p in npix if p in objs]
            neighs = [o2 for p in npix for o2 in objs[p] if o1.thingid != o2.thingid]
            ang = o1^neighs
            w = ang<ang_max
            neighs = sp.array(neighs)[w]
            o1.neighs = sp.array([o2 for o2 in neighs if (o2.z_qso+o1.z_qso)/2.>=z_cut_min and (o2.z_qso+o1.z_qso)/2.<z_cut_max])


def fill_neighs_x_correlation(pix):
    for ipix in pix:
        for o1 in objs[ipix]:
            npix = query_disc(nside,[o1.x_cart,o1.y_cart,o1.z_cart],ang_max,inclusive = True)
            npix = [p for p in npix if p in objs2]
            neighs = [o2 for p in npix for o2 in objs2[p] if o1.thingid != o2.thingid]
            ang = o1^neighs
            w = ang<ang_max
            neighs = sp.array(neighs)[w]
            o1.neighs = sp.array([o2 for o2 in neighs if (o2.z_qso+o1.z_qso)/2.>=z_cut_min and (o2.z_qso+o1.z_qso)/2.<z_cut_max])

def co(pix):

    weights = np.zeros(num_bins_r_par*num_bins_r_trans)
    rp = np.zeros(num_bins_r_par*num_bins_r_trans)
    rt = np.zeros(num_bins_r_par*num_bins_r_trans)
    z  = np.zeros(num_bins_r_par*num_bins_r_trans)
    nb = np.zeros(num_bins_r_par*num_bins_r_trans,dtype=sp.int64)

    for ipix in pix:
        for o1 in objs[ipix]:

            userprint("\rcomputing xi: {}%".format(round(counter.value*100./ndata,2)),end="")
            with lock:
                counter.value += 1

            if (o1.neighs.size == 0): continue

            ang      = o1^o1.neighs
            zo2      = sp.array([o2.z_qso    for o2 in o1.neighs])
            r_comov2 = sp.array([o2.r_comov for o2 in o1.neighs])
            dist_m2 = sp.array([o2.dist_m for o2 in o1.neighs])
            weo2     = sp.array([o2.weights      for o2 in o1.neighs])

            cw,crp,crt,cz,cnb = fast_co(o1.z_qso,o1.r_comov,o1.dist_m,o1.weights,zo2,r_comov2,dist_m2,weo2,ang)

            weights[:len(cw)]  += cw
            rp[:len(crp)] += crp
            rt[:len(crp)] += crt
            z[:len(crp)]  += cz
            nb[:len(cnb)] += cnb
            setattr(o1,"neighs",None)

    w = weights>0.
    rp[w] /= weights[w]
    rt[w] /= weights[w]
    z[w]  /= weights[w]
    return weights,rp,rt,z,nb
@jit
def fast_co(z1,r1,rdm1,w1,z2,r2,rdm2,w2,ang):

    rp  = (r1-r2)*sp.cos(ang/2.)
    if not x_correlation or type_corr in ['DR','RD']:
        rp = np.absolute(rp)
    rt  = (rdm1+rdm2)*sp.sin(ang/2.)
    z   = (z1+z2)/2.
    w12 = w1*w2

    w   = (rp>=r_par_min) & (rp<r_par_max) & (rt<r_trans_max) & (w12>0.)
    rp  = rp[w]
    rt  = rt[w]
    z   = z[w]
    w12 = w12[w]

    bp   = sp.floor((rp-r_par_min)/(r_par_max-r_par_min)*num_bins_r_par).astype(int)
    bt   = (rt/r_trans_max*num_bins_r_trans).astype(int)
    bins = bt + num_bins_r_trans*bp

    cw  = sp.bincount(bins,weights=w12)
    crp = sp.bincount(bins,weights=rp*w12)
    crt = sp.bincount(bins,weights=rt*w12)
    cz  = sp.bincount(bins,weights=z*w12)
    cnb = sp.bincount(bins)

    return cw,crp,crt,cz,cnb
