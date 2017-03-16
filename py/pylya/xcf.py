from __future__ import print_function
import scipy as sp
import sys
from healpy import query_disc
from multiprocessing import Pool
from numba import jit
from data import forest
from scipy import random

np = None
nt = None 
rp_max = None
rp_min = None
rt_max = None
angmax = None
nside = None

counter = None
ndels = None

lambda_abs = None

dels = None
objs = None

rej = None
lock = None

def fill_neighs(pix):
    for ipix in pix:
        for d in dels[ipix]:
            npix = query_disc(nside,[d.xcart,d.ycart,d.zcart],angmax,inclusive = True)
            npix = [p for p in npix if p in objs]
            neighs = [q for p in npix for q in objs[p] if q.thid != d.thid]
            ang = d^neighs
            w = ang<angmax
            low_dist = ( d.r_comov[0]  - sp.array([q.r_comov for q in neighs]) )*sp.cos(ang/2) <rp_max
            hig_dist = ( d.r_comov[-1] - sp.array([q.r_comov for q in neighs]) )*sp.cos(ang/2) >rp_min
            w = w & low_dist & hig_dist
            neighs = sp.array(neighs)[w]
            d.neighs = neighs

def xcf(pix):
    xi = sp.zeros(np*nt)
    we = sp.zeros(np*nt)
    rp = sp.zeros(np*nt)
    rt = sp.zeros(np*nt)
    z = sp.zeros(np*nt)
    nb = sp.zeros(np*nt,dtype=sp.int64)

    for ipix in pix:
        for i,d in enumerate(dels[ipix]):
            with lock:
                counter.value +=1
            sys.stderr.write("\r{}%".format(round(counter.value*100./ndels,3)))
            ang = d^d.neighs
            rc_qso = [q.r_comov for q in d.neighs]
            zqso = [q.zqso for q in d.neighs]
            we_qso = [q.we for q in d.neighs]

            if (d.neighs.size != 0):
                cw,cd,crp,crt,cz,cnb = fast_xcf(d.z,d.r_comov,d.we,d.de,zqso,rc_qso,we_qso,ang)
            
                xi[:len(cd)]+=cd
                we[:len(cw)]+=cw
                rp[:len(crp)]+=crp
                rt[:len(crt)]+=crt
                z[:len(cz)]+=cz
                nb[:len(cnb)]+=cnb

    w = we>0
    xi[w]/=we[w]
    rp[w]/=we[w]
    rt[w]/=we[w]
    z[w]/=we[w]
    return we,xi,rp,rt,z,nb
@jit 
def fast_xcf(z1,r1,w1,d1,z2,r2,w2,ang):
    rp = (r1[:,None]-r2)*sp.cos(ang/2)
    rt = (r1[:,None]+r2)*sp.sin(ang/2)
    z = (z1[:,None]+z2)/2

    we = w1[:,None]*w2
    wde = (w1*d1)[:,None]*w2

    w = (rp>rp_min) & (rp<rp_max) & (rt<rt_max)
    rp = rp[w]
    rt = rt[w]
    z  = z[w]
    we = we[w]
    wde = wde[w]

    bp = ((rp-rp_min)/(rp_max-rp_min)*np).astype(int)
    bt = (rt/rt_max*nt).astype(int)
    bins = bt + nt*bp

    cd = sp.bincount(bins,weights=wde)
    cw = sp.bincount(bins,weights=we)
    crp = sp.bincount(bins,weights=rp*we)
    crt = sp.bincount(bins,weights=rt*we)
    cz = sp.bincount(bins,weights=z*we)
    cnb = sp.bincount(bins)

    return cw,cd,crp,crt,cz,cnb



def metal_grid(pix):

    we = sp.zeros(np*nt)
    rp = sp.zeros(np*nt)
    rt = sp.zeros(np*nt)
    z  = sp.zeros(np*nt)
    nb = sp.zeros(np*nt,dtype=sp.int64)

    for ipix in pix:
        for i,d in enumerate(dels[ipix]):
            with lock:
                counter.value +=1
            sys.stderr.write("\r{}%".format(round(counter.value*100./ndels,3)))
            ang = d^d.neighs
            rc_qso = [q.r_comov for q in d.neighs]
            zqso   = [q.zqso for q in d.neighs]
            we_qso = [q.we for q in d.neighs]

            if (d.neighs.size != 0):
                cw,crp,crt,cz,cnb = fast_metal_grid(d.r_comov,d.we,zqso,rc_qso,we_qso,ang,d.z_metal,d.r_comov_metal)
            
                we[:len(cw)]  += cw
                rp[:len(crp)] += crp
                rt[:len(crt)] += crt
                z[:len(cz)]   += cz
                nb[:len(cnb)] += cnb

    w = we>0
    rp[w] /= we[w]
    rt[w] /= we[w]
    z[w]  /= we[w]

    return we,rp,rt,z,nb
@jit 
def fast_metal_grid(r1,w1,z2,r2,w2,ang,z1_metal,r1_metal):

    rp = (r1[:,None]-r2)*sp.cos(ang/2.)
    rt = (r1[:,None]+r2)*sp.sin(ang/2.)

    w   = (rp>rp_min) & (rp<rp_max) & (rt<rt_max)
    rp  = rp[w]
    rt  = rt[w]

    bp   = ((rp-rp_min)/(rp_max-rp_min)*np).astype(int)
    bt   = (rt/rt_max*nt).astype(int)
    bins = bt + nt*bp

    rp_metal = (r1_metal[:,None]-r2)*sp.cos(ang/2.)
    rt_metal = (r1_metal[:,None]+r2)*sp.sin(ang/2.)
    z_metal  = (z1_metal[:,None]+z2)/2.
    we       = w1[:,None]*w2

    rp_metal = rp_metal[w]
    rt_metal = rt_metal[w]
    z_metal  = z_metal[w]
    we       = we[w]

    cw  = sp.bincount(bins,weights=we)
    crp = sp.bincount(bins,weights=rp_metal*we)
    crt = sp.bincount(bins,weights=rt_metal*we)
    cz  = sp.bincount(bins,weights=z_metal*we)
    cnb = sp.bincount(bins)

    return cw,crp,crt,cz,cnb


def dmat(pix):

    dm = sp.zeros(np*nt*nt*np)
    wdm = sp.zeros(np*nt)

    npairs = 0L
    npairs_used = 0L
    for p in pix:
        for d1 in dels[p]:
            sys.stderr.write("\rcomputing xi: {}%".format(round(counter.value*100./ndels,3)))
            with lock:
                counter.value += 1
            r1 = d1.r_comov
            w1 = d1.we
            l1 = d1.ll
            r = random.rand(len(d1.neighs))
            w=r>rej
            if w.sum()==0:continue
            npairs += len(d1.neighs)
            npairs_used += w.sum()
            neighs = d1.neighs[w]
            ang = d1^neighs
            r2 = [q.r_comov for q in neighs]
            w2 = [q.we for q in neighs]
            fill_dmat(l1,r1,w1,r2,w2,ang,wdm,dm)

    return wdm,dm.reshape(np*nt,np*nt),npairs,npairs_used
    
@jit
def fill_dmat(l1,r1,w1,r2,w2,ang,wdm,dm):
    rp = (r1[:,None]-r2)*sp.cos(ang/2)
    rt = (r1[:,None]+r2)*sp.sin(ang/2)
    bp = ((rp-rp_min)/(rp_max-rp_min)*np).astype(int)
    bt = (rt/rt_max*nt).astype(int)
    bins = bt + nt*bp
    
    sw1 = w1.sum()

    ml1 = sp.average(l1,weights=w1)

    dl1 = l1-ml1

    slw1 = (w1*dl1**2).sum()

    w = (rp>rp_min) & (rp<rp_max) & (rt<rt_max)
    bins = bins[w]

    n1 = len(l1)
    n2 = len(r2)
    ij = sp.arange(n1)[:,None]+n1*sp.arange(n2)
    ij = ij[w]

    we = w1[:,None]*w2
    we = we[w]
    c = sp.bincount(bins,weights=we)
    wdm[:len(c)] += c
    eta2 = sp.zeros(np*nt*n2)
    eta4 = sp.zeros(np*nt*n2)

    c = sp.bincount((ij-ij%n1)/n1+n2*bins,weights = (w1[:,None]*sp.ones(n2))[w]/sw1)
    eta2[:len(c)]+=c
    c = sp.bincount((ij-ij%n1)/n1+n2*bins,weights = ((w1*dl1)[:,None]*sp.ones(n2))[w]/slw1)
    eta4[:len(c)]+=c

    ubb = sp.unique(bins)
    for k,ba in enumerate(bins):
        dm[ba+np*nt*ba]+=we[k]
        i = ij[k]%n1
        j = (ij[k]-i)/n1
        for bb in ubb:
            dm[ba+np*nt*bb] -= we[k]*(eta2[j+n2*bb]+eta4[j+n2*bb]*dl1[i])
