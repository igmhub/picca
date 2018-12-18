import sys
import scipy as sp
from scipy import random
from healpy import query_disc
from numba import jit

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
            neighs = [q for p in npix for q in objs[p] if q.thid != d.thid]
            ang = d^neighs
            w = ang<angmax
            if not ang_correlation:
                r_comov = sp.array([q.r_comov for q in neighs])
                w &= (d.r_comov[0] - r_comov)*sp.cos(ang/2.) < rp_max
                w &= (d.r_comov[-1] - r_comov)*sp.cos(ang/2.) > rp_min
            neighs = sp.array(neighs)[w]
            d.neighs = sp.array([q for q in neighs if (10**(d.ll[-1]- sp.log10(lambda_abs))-1 + q.zqso)/2. >= z_cut_min and (10**(d.ll[-1]- sp.log10(lambda_abs))-1 + q.zqso)/2. < z_cut_max])

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



def metal_grid(pix):

    we = sp.zeros(np*nt)
    rp = sp.zeros(np*nt)
    rt = sp.zeros(np*nt)
    z  = sp.zeros(np*nt)
    nb = sp.zeros(np*nt,dtype=sp.int64)

    for ipix in pix:
        for d in dels[ipix]:
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
                nb[:len(cnb)] += cnb.astype(int)
            for el in list(d.__dict__.keys()):
                setattr(d,el,None)

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
    cnb = sp.bincount(bins,weights=(we>0.))

    return cw,crp,crt,cz,cnb


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

            r = random.rand(len(d.neighs))
            w=r>rej
            npairs += len(d.neighs)
            npairs_used += w.sum()

            rd = d.r_comov
            wd = d.we
            zd_abs = 10**d.ll/constants.absorber_IGM[abs_igm]-1
            rd_abs = cosmo.r_comoving(zd_abs)

            wzcut = zd_abs<d.zqso
            rd = rd[wzcut]
            wd = wd[wzcut]
            zd_abs = zd_abs[wzcut]
            rd_abs = rd_abs[wzcut]
            if rd.size==0: continue

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

            cw,cd,crp,_,cz,cnb = fast_xcf(d.z,10.**d.ll,d.we,d.de,zqso,l_qso,we_qso,ang)

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
