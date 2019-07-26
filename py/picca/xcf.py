from __future__ import print_function
import scipy as sp
from healpy import query_disc
from numba import jit

from picca import constants
from picca.utils import print

np = None
nt = None
npm = None
ntm = None
rp_max = None
rp_min = None
rt_max = None
z_cut_max = None
z_cut_min = None
angmax = None
nside = None

counter = None
ndels = None

zref = None
z_evol_del = None
z_evol_obj = None
lambda_abs = None
alpha_abs= None

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
            neighs = [q for p in npix for q in objs[p] if q.thid != d.thid]
            ang = d^neighs
            w = ang<angmax
            if not ang_correlation:
                r_comov = sp.array([q.r_comov for q in neighs])
                w &= (d.r_comov[0] - r_comov)*sp.cos(ang/2.) < rp_max
                w &= (d.r_comov[-1] - r_comov)*sp.cos(ang/2.) > rp_min
            neighs = sp.array(neighs)[w]
            d.qneighs = sp.array([q for q in neighs if (d.z[-1]+q.zqso)/2.>=z_cut_min and (d.z[-1]+q.zqso)/2.<z_cut_max])

def xcf(pix):
    xi = sp.zeros(np*nt)
    we = sp.zeros(np*nt)
    rp = sp.zeros(np*nt)
    rt = sp.zeros(np*nt)
    z = sp.zeros(np*nt)
    nb = sp.zeros(np*nt,dtype=sp.int64)

    for ipix in pix:
        for d in dels[ipix]:
            with lock:
                counter.value +=1
            print("\rcomputing xi: {}%".format(round(counter.value*100./ndels,3)),end="")
            if (d.qneighs.size != 0):
                ang = d^d.qneighs
                zqso = [q.zqso for q in d.qneighs]
                we_qso = [q.we for q in d.qneighs]
                if ang_correlation:
                    l_qso = [10.**q.ll for q in d.qneighs]
                    cw,cd,crp,crt,cz,cnb = fast_xcf(d.z,10.**d.ll,10.**d.ll,d.we,d.de,zqso,l_qso,l_qso,we_qso,ang)
                else:
                    rc_qso = [q.r_comov for q in d.qneighs]
                    rdm_qso = [q.rdm_comov for q in d.qneighs]
                    cw,cd,crp,crt,cz,cnb = fast_xcf(d.z,d.r_comov,d.rdm_comov,d.we,d.de,zqso,rc_qso,rdm_qso,we_qso,ang)

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
def fast_xcf(z1,r1,rdm1,w1,d1,z2,r2,rdm2,w2,ang):
    if ang_correlation:
        rp = r1[:,None]/r2
        rt = ang*sp.ones_like(rp)
    else:
        rp = (r1[:,None]-r2)*sp.cos(ang/2)
        rt = (rdm1[:,None]+rdm2)*sp.sin(ang/2)
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

    dm = sp.zeros(np*nt*ntm*npm)
    wdm = sp.zeros(np*nt)
    rpeff = sp.zeros(ntm*npm)
    rteff = sp.zeros(ntm*npm)
    zeff = sp.zeros(ntm*npm)
    weff = sp.zeros(ntm*npm)

    npairs = 0
    npairs_used = 0
    for p in pix:
        for d1 in dels[p]:
            print("\rcomputing xi: {}%".format(round(counter.value*100./ndels,3)),end="")
            with lock:
                counter.value += 1
            r1 = d1.r_comov
            rdm1 = d1.rdm_comov
            w1 = d1.we
            l1 = d1.ll
            z1 = d1.z
            r = sp.random.rand(len(d1.qneighs))
            w=r>rej
            if w.sum()==0:continue
            npairs += len(d1.qneighs)
            npairs_used += w.sum()
            neighs = d1.qneighs[w]
            ang = d1^neighs
            r2 = [q.r_comov for q in neighs]
            rdm2 = [q.rdm_comov for q in neighs]
            w2 = [q.we for q in neighs]
            z2 = [q.zqso for q in neighs]
            fill_dmat(l1,r1,rdm1,z1,w1,r2,rdm2,z2,w2,ang,wdm,dm,rpeff,rteff,zeff,weff)
            for el in list(d1.__dict__.keys()):
                setattr(d1,el,None)

    return wdm,dm.reshape(np*nt,npm*ntm),rpeff,rteff,zeff,weff,npairs,npairs_used
@jit
def fill_dmat(l1,r1,rdm1,z1,w1,r2,rdm2,z2,w2,ang,wdm,dm,rpeff,rteff,zeff,weff):
    rp = (r1[:,None]-r2)*sp.cos(ang/2)
    rt = (rdm1[:,None]+rdm2)*sp.sin(ang/2)
    z = (z1[:,None]+z2)/2.
    w = (rp>rp_min) & (rp<rp_max) & (rt<rt_max)

    bp = ((rp-rp_min)/(rp_max-rp_min)*np).astype(int)
    bt = (rt/rt_max*nt).astype(int)
    bins = bt + nt*bp
    bins = bins[w]

    m_bp = ((rp-rp_min)/(rp_max-rp_min)*npm).astype(int)
    m_bt = (rt/rt_max*ntm).astype(int)
    m_bins = m_bt + ntm*m_bp
    m_bins = m_bins[w]

    sw1 = w1.sum()
    ml1 = sp.average(l1,weights=w1)

    dl1 = l1-ml1

    slw1 = (w1*dl1**2).sum()

    n1 = len(l1)
    n2 = len(r2)
    ij = sp.arange(n1)[:,None]+n1*sp.arange(n2)
    ij = ij[w]

    we = w1[:,None]*w2
    we = we[w]
    c = sp.bincount(bins,weights=we)
    wdm[:len(c)] += c
    eta2 = sp.zeros(npm*ntm*n2)
    eta4 = sp.zeros(npm*ntm*n2)

    c = sp.bincount(m_bins,weights=we*rp[w])
    rpeff[:c.size] += c
    c = sp.bincount(m_bins,weights=we*rt[w])
    rteff[:c.size] += c
    c = sp.bincount(m_bins,weights=we*z[w])
    zeff[:c.size] += c
    c = sp.bincount(m_bins,weights=we)
    weff[:c.size] += c

    c = sp.bincount((ij-ij%n1)//n1+n2*m_bins,weights = (w1[:,None]*sp.ones(n2))[w]/sw1)
    eta2[:len(c)]+=c
    c = sp.bincount((ij-ij%n1)//n1+n2*m_bins,weights = ((w1*dl1)[:,None]*sp.ones(n2))[w]/slw1)
    eta4[:len(c)]+=c

    ubb = sp.unique(m_bins)
    for k, (ba,m_ba) in enumerate(zip(bins,m_bins)):
        dm[m_ba+npm*ntm*ba]+=we[k]
        i = ij[k]%n1
        j = (ij[k]-i)//n1
        for bb in ubb:
            dm[bb+npm*ntm*ba] -= we[k]*(eta2[j+n2*bb]+eta4[j+n2*bb]*dl1[i])


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
                print("\rcomputing metal dmat {}: {}%".format(abs_igm,round(counter.value*100./ndels,3)),end="")
                counter.value += 1

            r = sp.random.rand(len(d.qneighs))
            w=r>rej
            npairs += len(d.qneighs)
            npairs_used += w.sum()

            rd = d.r_comov
            rdm = d.rdm_comov
            wd = d.we
            zd_abs = 10**d.ll/constants.absorber_IGM[abs_igm]-1
            rd_abs = cosmo.r_comoving(zd_abs)
            rdm_abs = cosmo.dm(zd_abs)

            wzcut = zd_abs<d.zqso
            rd = rd[wzcut]
            rdm = rdm[wzcut]
            wd = wd[wzcut]
            zd_abs = zd_abs[wzcut]
            rd_abs = rd_abs[wzcut]
            rdm_abs = rdm_abs[wzcut]
            if rd.size==0: continue

            for q in sp.array(d.qneighs)[w]:
                ang = d^q

                rq = q.r_comov
                rqm = q.rdm_comov
                wq = q.we
                zq = q.zqso
                rp = (rd-rq)*sp.cos(ang/2)
                rt = (rdm+rqm)*sp.sin(ang/2)
                wdq = wd*wq

                wA = (rp>rp_min) & (rp<rp_max) & (rt<rt_max)
                bp = ((rp-rp_min)/(rp_max-rp_min)*np).astype(int)
                bt = (rt/rt_max*nt).astype(int)
                bA = bt + nt*bp
                c = sp.bincount(bA[wA],weights=wdq[wA])
                wdm[:len(c)]+=c

                rp_abs = (rd_abs-rq)*sp.cos(ang/2)
                rt_abs = (rdm_abs+rqm)*sp.sin(ang/2)
                zwe = ((1.+zd_abs)/(1.+zref))**(alpha_abs[abs_igm]-1.)

                bp_abs = ((rp_abs-rp_min)/(rp_max-rp_min)*npm).astype(int)
                bt_abs = (rt_abs/rt_max*ntm).astype(int)
                bBma = bt_abs + ntm*bp_abs
                wBma = (rp_abs>rp_min) & (rp_abs<rp_max) & (rt_abs<rt_max)
                wAB = wA&wBma
                c = sp.bincount(bBma[wAB]+npm*ntm*bA[wAB],weights=wdq[wAB]*zwe[wAB])
                dm[:len(c)]+=c

                c = sp.bincount(bBma[wAB],weights=rp_abs[wAB]*wdq[wAB]*zwe[wAB])
                rpeff[:len(c)]+=c
                c = sp.bincount(bBma[wAB],weights=rt_abs[wAB]*wdq[wAB]*zwe[wAB])
                rteff[:len(c)]+=c
                c = sp.bincount(bBma[wAB],weights=(zd_abs+zq)[wAB]/2*wdq[wAB]*zwe[wAB])
                zeff[:len(c)]+=c
                c = sp.bincount(bBma[wAB],weights=wdq[wAB]*zwe[wAB])
                weff[:len(c)]+=c
            setattr(d,"qneighs",None)

    return wdm,dm.reshape(np*nt,npm*ntm),rpeff,rteff,zeff,weff,npairs,npairs_used

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
    T1 = sp.zeros((np*nt,np*nt))
    T2 = sp.zeros((np*nt,np*nt))
    T3 = sp.zeros((np*nt,np*nt))
    T4 = sp.zeros((np*nt,np*nt))
    T5 = sp.zeros((np*nt,np*nt))
    T6 = sp.zeros((np*nt,np*nt))
    wAll = sp.zeros(np*nt)
    nb = sp.zeros(np*nt,dtype=sp.int64)
    npairs = 0
    npairs_used = 0

    for ipix in pix:

        npairs += len(dels[ipix])
        r = sp.random.rand(len(dels[ipix]))
        w = r>rej
        npairs_used += w.sum()
        if w.sum()==0: continue

        for d1 in [ td for ti,td in enumerate(dels[ipix]) if w[ti] ]:
            print("\rcomputing xi: {}%".format(round(counter.value*100./ndels/(1.-rej),3)),end="")
            with lock:
                counter.value += 1
            if d1.qneighs.size==0: continue

            v1 = v1d[d1.fname](d1.ll)
            w1 = d1.we
            c1d_1 = (w1*w1[:,None])*c1d[d1.fname](abs(d1.ll-d1.ll[:,None]))*sp.sqrt(v1*v1[:,None])
            r1 = d1.r_comov
            z1 = d1.z

            neighs = d1.qneighs
            ang12 = d1^neighs
            r2 = sp.array([q2.r_comov for q2 in neighs])
            z2 = sp.array([q2.zqso for q2 in neighs])
            w2 = sp.array([q2.we for q2 in neighs])

            fill_wickT1234(ang12,r1,r2,z1,z2,w1,w2,c1d_1,wAll,nb,T1,T2,T3,T4)

            ### Higher order diagrams
            if (cfWick is None) or (max_diagram<=4): continue
            thid2 = sp.array([q2.thid for q2 in neighs])
            for d3 in sp.array(d1.dneighs):
                if d3.qneighs.size==0: continue

                ang13 = d1^d3

                r3 = d3.r_comov
                w3 = d3.we

                neighs = d3.qneighs
                ang34 = d3^neighs
                r4 = sp.array([q4.r_comov for q4 in neighs])
                w4 = sp.array([q4.we for q4 in neighs])
                thid4 = sp.array([q4.thid for q4 in neighs])

                if max_diagram==5:
                    w = sp.in1d(d1.qneighs,d3.qneighs)
                    if w.sum()==0: continue
                    t_ang12 = ang12[w]
                    t_r2 = r2[w]
                    t_w2 = w2[w]
                    t_thid2 = thid2[w]

                    w = sp.in1d(d3.qneighs,d1.qneighs)
                    if w.sum()==0: continue
                    ang34 = ang34[w]
                    r4 = r4[w]
                    w4 = w4[w]
                    thid4 = thid4[w]

                fill_wickT56(t_ang12,ang34,ang13,r1,t_r2,r3,r4,w1,t_w2,w3,w4,t_thid2,thid4,T5,T6)

    return wAll, nb, npairs, npairs_used, T1, T2, T3, T4, T5, T6
@jit
def fill_wickT1234(ang,r1,r2,z1,z2,w1,w2,c1d_1,wAll,nb,T1,T2,T3,T4):
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
        nb (int64 array): Number of pairs
        T1 (float 2d array): Contribution of diagram T1
        T2 (float 2d array): Contribution of diagram T2
        T3 (float 2d array): Contribution of diagram T3
        T4 (float 2d array): Contribution of diagram T4

    Returns:

    """
    rp = (r1[:,None]-r2)*sp.cos(ang/2.)
    rt = (r1[:,None]+r2)*sp.sin(ang/2.)
    zw1 = ((1.+z1)/(1.+zref))**(z_evol_del-1.)
    zw2 = ((1.+z2)/(1.+zref))**(z_evol_obj-1.)
    we = w1[:,None]*w2
    we1 = w1[:,None]*sp.ones(len(r2))
    idxPix = sp.arange(r1.size)[:,None]*sp.ones(len(r2),dtype='int')
    idxQso = sp.ones(r1.size,dtype='int')[:,None]*sp.arange(len(r2))

    bp = ((rp-rp_min)/(rp_max-rp_min)*np).astype(int)
    bt = (rt/rt_max*nt).astype(int)
    ba = bt + nt*bp

    w = (rp>rp_min) & (rp<rp_max) & (rt<rt_max)
    if w.sum()==0: return

    ba = ba[w]
    we = we[w]
    we1 = we1[w]
    idxPix = idxPix[w]
    idxQso = idxQso[w]

    for k1 in range(ba.size):
        p1 = ba[k1]
        i1 = idxPix[k1]
        q1 = idxQso[k1]
        wAll[p1] += we[k1]
        nb[p1] += 1
        T1[p1,p1] += (we[k1]**2)/we1[k1]*zw1[i1]

        for k2 in range(k1+1,ba.size):
            p2 = ba[k2]
            i2 = idxPix[k2]
            q2 = idxQso[k2]
            if q1==q2:
                wcorr = c1d_1[i1,i2]*(zw2[q1]**2)
                T2[p1,p2] += wcorr
                T2[p2,p1] += wcorr
            elif i1==i2:
                wcorr = (we[k1]*we[k2])/we1[k1]*zw1[i1]
                T3[p1,p2] += wcorr
                T3[p2,p1] += wcorr
            else:
                wcorr = c1d_1[i1,i2]*zw2[q1]*zw2[q2]
                T4[p1,p2] += wcorr
                T4[p2,p1] += wcorr

    return
@jit
def fill_wickT56(ang12,ang34,ang13,r1,r2,r3,r4,w1,w2,w3,w4,thid2,thid4,T5,T6):
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
        thid2 (float array): THING_ID of each object
        thid4 (float array): THING_ID of each object paired to the other forest
        T5 (float 2d array): Contribution of diagram T5
        T6 (float 2d array): Contribution of diagram T6

    Returns:

    """

    ### Pair forest_1 - forest_3
    rp = sp.absolute(r1-r3[:,None])*sp.cos(ang13/2.)
    rt = (r1+r3[:,None])*sp.sin(ang13/2.)

    w = (rp<cfWick_rp_max) & (rt<cfWick_rt_max) & (rp>=cfWick_rp_min)
    if w.sum()==0: return
    bp = sp.floor((rp-cfWick_rp_min)/(cfWick_rp_max-cfWick_rp_min)*cfWick_np).astype(int)
    bt = (rt/cfWick_rt_max*cfWick_nt).astype(int)
    ba13 = bt + cfWick_nt*bp
    ba13[~w] = 0
    cf13 = cfWick[ba13]
    cf13[~w] = 0.

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

    ### T5
    for k1, p1 in enumerate(ba12):
        pix1 = pix12[k1]
        t1 = thid12[k1]
        w1 = we12[k1]

        w = thid34==t1
        for k2, p2 in enumerate(ba34[w]):
            pix2 = pix34[w][k2]
            t2 = thid34[w][k2]
            w2 = we34[w][k2]
            wcorr = cf13[pix2,pix1]*w1*w2
            T5[p1,p2] += wcorr
            T5[p2,p1] += wcorr

    ### T6
    if max_diagram==5: return
    for k1, p1 in enumerate(ba12):
        pix1 = pix12[k1]
        t1 = thid12[k1]
        w1 = we12[k1]

        for k2, p2 in enumerate(ba34):
            pix2 = pix34[k2]
            t2 = thid34[k2]
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
        we (float array): weight
        xi (float array): correlation
        rp (float array): wavelenght ratio
        z (float array): Mean redshift of pairs
        nb (int array): Number of pairs
    """
    xi = sp.zeros(np)
    we = sp.zeros(np)
    rp = sp.zeros(np)
    z = sp.zeros(np)
    nb = sp.zeros(np,dtype=sp.int64)

    for ipix in pix:
        for d in dels[ipix]:

            neighs = [q for q in objs[ipix] if q.thid==d.thid]
            if len(neighs)==0: continue

            zqso = [ q.zqso for q in neighs ]
            we_qso = [ q.we for q in neighs ]
            l_qso = [ 10.**q.ll for q in neighs ]
            ang = sp.zeros(len(l_qso))

            cw,cd,crp,_,cz,cnb = fast_xcf(d.z,10.**d.ll,10.**d.ll,d.we,d.de,zqso,l_qso,l_qso,we_qso,ang)

            xi[:cd.size] += cd
            we[:cw.size] += cw
            rp[:crp.size] += crp
            z[:cz.size] += cz
            nb[:cnb.size] += cnb.astype(int)

            for el in list(d.__dict__.keys()):
                setattr(d,el,None)

    w = we>0.
    xi[w] /= we[w]
    rp[w] /= we[w]
    z[w] /= we[w]

    return we,xi,rp,z,nb
