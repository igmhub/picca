import scipy as sp
import sys
from healpy import query_disc
from multiprocessing import Pool
from numba import jit
from .data import forest
from scipy import random

from picca import constants

np = None
nt = None
rp_max = None
rp_min = None
rt_max = None
z_cut_max = None
z_cut_min = None
angmax = None
nside = None

counter = None
ndels = None

lambda_abs = None

dels = None
objs = None

rej = None
lock = None

cosmo=None
ang_correlation = None

def fill_neighs(pix):
    for ipix in pix:
        for d in dels[ipix]:
            npix = query_disc(nside,[d.xcart,d.ycart,d.zcart],angmax,inclusive = True)
            npix = [p for p in npix if p in objs]
            neighs = [q for p in npix for q in objs[p] if q.thid != d.thid and (10**(d.ll[-1]- sp.log10(lambda_abs))-1 + q.zqso)/2. >= z_cut_min and (10**(d.ll[-1]- sp.log10(lambda_abs))-1 + q.zqso)/2. < z_cut_max]
            ang = d^neighs
            w = ang<angmax
            if not ang_correlation:
                low_dist = ( d.r_comov[0]  - sp.array([q.r_comov for q in neighs]) )*sp.cos(ang/2) <rp_max
                hig_dist = ( d.r_comov[-1] - sp.array([q.r_comov for q in neighs]) )*sp.cos(ang/2) >rp_min
                w &= low_dist & hig_dist
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
            if (d.neighs.size != 0):
                ang = d^d.neighs
                zqso = [q.zqso for q in d.neighs]
                we_qso = [q.we for q in d.neighs]

                if ang_correlation:
                    l_qso = [10.**q.ll for q in d.neighs]
                    cw,cd,crp,crt,cz,cnb = fast_xcf(d.z,10.**d.ll,d.we,d.de,zqso,l_qso,we_qso,ang)
                else:
                    rc_qso = [q.r_comov for q in d.neighs]
                    cw,cd,crp,crt,cz,cnb = fast_xcf(d.z,d.r_comov,d.we,d.de,zqso,rc_qso,we_qso,ang)

                xi[:len(cd)]+=cd
                we[:len(cw)]+=cw
                rp[:len(crp)]+=crp
                rt[:len(crt)]+=crt
                z[:len(cz)]+=cz
                nb[:len(cnb)]+=cnb.astype(int)
            for el in list(d.__dict__.keys()):
                setattr(d,el,None)

    w = we>0
    xi[w]/=we[w]
    rp[w]/=we[w]
    rt[w]/=we[w]
    z[w]/=we[w]
    return we,xi,rp,rt,z,nb
@jit
def fast_xcf(z1,r1,w1,d1,z2,r2,w2,ang):
    if ang_correlation:
        rp = r1[:,None]/r2
        rt = ang*sp.ones_like(rp)
    else:
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
    cnb = sp.bincount(bins,weights=(we>0.))

    return cw,cd,crp,crt,cz,cnb

def dmat(pix):

    dm = sp.zeros(np*nt*nt*np)
    wdm = sp.zeros(np*nt)

    npairs = 0
    npairs_used = 0
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
            for el in list(d1.__dict__.keys()):
                setattr(d1,el,None)

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

    c = sp.bincount((ij-ij%n1)//n1+n2*bins,weights = (w1[:,None]*sp.ones(n2))[w]/sw1)
    eta2[:len(c)]+=c
    c = sp.bincount((ij-ij%n1)//n1+n2*bins,weights = ((w1*dl1)[:,None]*sp.ones(n2))[w]/slw1)
    eta4[:len(c)]+=c

    ubb = sp.unique(bins)
    for k,ba in enumerate(bins):
        dm[ba+np*nt*ba]+=we[k]
        i = ij[k]%n1
        j = (ij[k]-i)//n1
        for bb in ubb:
            dm[bb+np*nt*ba] -= we[k]*(eta2[j+n2*bb]+eta4[j+n2*bb]*dl1[i])


def metal_dmat(pix,abs_igm="SiII(1526)"):

    dm = sp.zeros(np*nt*ntm*npm)
    wdm = sp.zeros(np*nt)
    rpeff = sp.zeros(ntm*npm)
    rteff = sp.zeros(ntm*npm)
    zeff = sp.zeros(ntm*npm)
    weff = sp.zeros(ntm*npm)

    npairs = 0
    npairs_used = 0
    for p in pix:
        for d in dels[p]:
            with lock:
                sys.stderr.write("\rcomputing metal dmat {}: {}%".format(abs_igm,round(counter.value*100./ndels,3)))
                counter.value += 1
            rd = d.r_comov
            zd_abs = 10**d.ll/constants.absorber_IGM[abs_igm]-1
            rd_abs = cosmo.r_comoving(zd_abs)
            wd = d.we
            r = random.rand(len(d.neighs))
            w=r>rej
            npairs += len(d.neighs)
            npairs_used += w.sum()
            for q in sp.array(d.neighs)[w]:
                ang = d^q

                rq = q.r_comov
                wq = q.we
                zq = q.zqso
                rp = (rd-rq)*sp.cos(ang/2)
                rt = (rd+rq)*sp.sin(ang/2)
                wdq = wd*wq

                wA = (rp>rp_min) & (rp<rp_max) & (rt<rt_max)
                bp = ((rp-rp_min)/(rp_max-rp_min)*np).astype(int)
                bt = (rt/rt_max*nt).astype(int)
                bA = bt + nt*bp
                c = sp.bincount(bA[wA],weights=wdq[wA])
                wdm[:len(c)]+=c

                rp_abs = (rd_abs-rq)*sp.cos(ang/2)
                rt_abs = (rd_abs+rq)*sp.sin(ang/2)

                bp_abs = ((rp_abs-rp_min)/(rp_max-rp_min)*npm).astype(int)
                bt_abs = (rt_abs/rt_max*ntm).astype(int)
                bBma = bt_abs + ntm*bp_abs
                wBma = (rp_abs>rp_min) & (rp_abs<rp_max) & (rt_abs<rt_max)
                wAB = wA&wBma
                c = sp.bincount(bBma[wAB]+npm*ntm*bA[wAB],weights=wdq[wAB])
                dm[:len(c)]+=c

                c = sp.bincount(bBma[wAB],weights=rp_abs[wAB]*wdq[wAB])
                rpeff[:len(c)]+=c
                c = sp.bincount(bBma[wAB],weights=rt_abs[wAB]*wdq[wAB])
                rteff[:len(c)]+=c
                c = sp.bincount(bBma[wAB],weights=(zd_abs+zq)[wAB]/2*wdq[wAB])
                zeff[:len(c)]+=c
                c = sp.bincount(bBma[wAB],weights=wdq[wAB])
                weff[:len(c)]+=c
            setattr(d,"neighs",None)

    return wdm,dm.reshape(np*nt,npm*ntm),rpeff,rteff,zeff,weff,npairs,npairs_used



cf1d = None
lmin = None
dll  = None

cf = None
cf_np = None
cf_nt = None
cf_rp_min = None
cf_rp_max = None
cf_rt_max = None
cf_angmax = None

def wickT(pix):
    T1   = sp.zeros([np*nt,np*nt])
    T2   = sp.zeros([np*nt,np*nt])
    T3   = sp.zeros([np*nt,np*nt])
    T4   = sp.zeros([np*nt,np*nt])
    T5   = sp.zeros([np*nt,np*nt])
    T6   = sp.zeros([np*nt,np*nt])
    wAll = sp.zeros(np*nt)
    nb   = sp.zeros(np*nt,dtype=sp.int64)
    npairs = 0
    npairs_used = 0

    for ipix in pix:
        for i1, d1 in enumerate(dels[ipix]):
            with lock:
                counter.value += 1
            sys.stderr.write("\r{}%".format(round(counter.value*100./ndels,3)))

            if d1.neighs.size==0: continue

            npairs += d1.neighs.size
            r = random.rand(d1.neighs.size)
            w = r>rej
            npairs_used += w.sum()

            if w.sum()==0: continue

            ll1 = d1.ll
            r1 = d1.r_comov
            w1 = d1.we

            neighs = d1.neighs[w]
            ang = d1^neighs
            r2 = [q2.r_comov for q2 in neighs]
            w2 = [q2.we for q2 in neighs]

            fill_wickT1234(ang,ll1,r1,r2,w1,w2,wAll,nb,T1,T2,T3,T4)

            if cf is None: continue

            thid2 = [q2.thid for q2 in neighs]

            #-TODO: Correlation with other nside pixels
            for d3 in dels[ipix][i1+1:]:
                if d3.neighs.size==0: continue

                ang13 = d1^d3
                if ang13>=cf_angmax: continue

                r3 = d3.r_comov
                w3 = d3.we

                neighs = d3.neighs
                ang34 = d3^neighs
                r4 = [q4.r_comov for q4 in neighs]
                w4 = [q4.we for q4 in neighs]
                thid4 = [q4.thid for q4 in neighs]

                fill_wickT56(ang,ang34,ang13,r1,r2,r3,r4,w1,w2,w3,w4,thid2,thid4,T5,T6)

    return wAll, nb, npairs, npairs_used, T1, T2, T3, T4, T5, T6
@jit
def fill_wickT1234(ang,ll1,r1,r2,w1,w2,wAll,nb,T1,T2,T3,T4):

    rp = (r1[:,None]-r2)*sp.cos(ang/2.)
    rt = (r1[:,None]+r2)*sp.sin(ang/2.)
    we = w1[:,None]*w2
    idxPix = ((ll1-lmin)/dll+0.5).astype(int)
    idxPix = idxPix[:,None]*sp.ones_like(r2).astype(int)
    idxQso = sp.ones_like(w1[:,None]).astype(int)*sp.arange(len(r2))

    w = (rp>rp_min) & (rp<rp_max) & (rt<rt_max)
    if w.sum()==0: return

    rp = rp[w]
    rt = rt[w]
    we = we[w]
    idxPix = idxPix[w]
    idxQso = idxQso[w]
    bp = ((rp-rp_min)/(rp_max-rp_min)*np).astype(int)
    bt = (rt/rt_max*nt).astype(int)
    ba = bt + nt*bp

    for k1 in range(rp.size):
        p1 = ba[k1]
        i1 = idxPix[k1]
        q1 = idxQso[k1]
        wAll[p1]  += we[k1]
        nb[p1]    += 1
        T1[p1,p1] += cf1d[i1,i1]*(we[k1]**2)

        for k2 in range(k1+1,rp.size):
            p2 = ba[k2]
            i2 = idxPix[k2]
            q2 = idxQso[k2]
            wcorr = cf1d[i1,i2]*we[k1]*we[k2]
            if q1==q2:
                T2[p1,p2] += wcorr
                T2[p2,p1] += wcorr
            elif i1==i2:
                T3[p1,p2] += wcorr
                T3[p2,p1] += wcorr
            else:
                T4[p1,p2] += wcorr
                T4[p2,p1] += wcorr

    return
@jit
def fill_wickT56(ang12,ang34,ang13,r1,r2,r3,r4,w1,w2,w3,w4,thid2,thid4,T5,T6):

    ### Pair forest_1 - forest_3
    rp = sp.absolute(r1-r3[:,None])*sp.cos(ang13/2.)
    rt = (r1+r3[:,None])*sp.sin(ang13/2.)

    w = (rp<cf_rp_max) & (rt<cf_rt_max) & (rp>=cf_rp_min)
    if w.sum()==0: return
    w = sp.logical_not(w)
    bp = sp.floor((rp-cf_rp_min)/(cf_rp_max-cf_rp_min)*cf_np).astype(int)
    bt = (rt/cf_rt_max*cf_nt).astype(int)
    ba13 = bt + cf_nt*bp
    ba13[w] = 0
    cf13 = cf[ba13]
    cf13[w] = 0.

    ### Pair forest_1 - object_2
    rp = (r1[:,None]-r2)*sp.cos(ang12/2.)
    rt = (r1[:,None]+r2)*sp.sin(ang12/2.)
    we = w1[:,None]*w2
    pix = (sp.arange(r1.size)[:,None]*sp.ones_like(r2)).astype(int)
    thid = sp.ones_like(w1[:,None]).astype(int)*thid2

    w = (rp>rp_min) & (rp<rp_max) & (rt<rt_max)
    if w.sum()==0: return
    rp = rp[w]
    rt = rt[w]
    we12 = we[w]
    pix12 = pix[w]
    thid12 = thid[w]
    bp = ((rp-rp_min)/(rp_max-rp_min)*np).astype(int)
    bt = (rt/rt_max*nt).astype(int)
    ba12 = bt + nt*bp

    ### Pair forest_3 - object_4
    rp = (r3[:,None]-r4)*sp.cos(ang34/2.)
    rt = (r3[:,None]+r4)*sp.sin(ang34/2.)
    we = w3[:,None]*w4
    pix = (sp.arange(r3.size)[:,None]*sp.ones_like(r4)).astype(int)
    thid = sp.ones_like(w3[:,None]).astype(int)*thid4

    w = (rp>rp_min) & (rp<rp_max) & (rt<rt_max)
    if w.sum()==0: return
    rp = rp[w]
    rt = rt[w]
    we34 = we[w]
    pix34 = pix[w]
    thid34 = thid[w]
    bp = ((rp-rp_min)/(rp_max-rp_min)*np).astype(int)
    bt = (rt/rt_max*nt).astype(int)
    ba34 = bt + nt*bp

    ### Fill
    for k1, p1 in enumerate(ba12):
        pix1 = pix12[k1]
        t1 = thid12[k1]
        w1 = we12[k1]

        for k2, p2 in enumerate(ba34):
            pix2 = pix34[k2]
            t2 = thid34[k2]
            w2 = we34[k2]
            wcorr = cf13[pix2,pix1]*w1*w2
            if wcorr==0.: continue

            if t1==t2:
                T5[p1,p2] += wcorr
                T5[p2,p1] += wcorr
            else:
                T6[p1,p2] += wcorr
                T6[p2,p1] += wcorr

    return
