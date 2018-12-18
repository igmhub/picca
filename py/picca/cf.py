from __future__ import print_function
import scipy as sp
from healpy import query_disc
from numba import jit
from scipy import random

from picca import constants
from picca.utils import print

np = None
nt = None
ntm= None
npm= None
rp_max = None
rp_min = None
z_cut_max = None
z_cut_min = None
rt_max = None
angmax = None
nside = None

counter = None
ndata = None
ndata2 = None

zref = None
alpha= None
alpha2= None
alpha_abs= None
lambda_abs = None
lambda_abs2 = None

data = None
data2 = None

cosmo=None

rej = None
lock = None
x_correlation = None
ang_correlation = None
no_same_wavelength_pairs = None

def fill_neighs(pix):
    for ipix in pix:
        for d1 in data[ipix]:
            npix = query_disc(nside,[d1.xcart,d1.ycart,d1.zcart],angmax,inclusive = True)
            npix = [p for p in npix if p in data]
            neighs = [d for p in npix for d in data[p] if d1.thid != d.thid]
            ang = d1^neighs
            w = ang<angmax
            neighs = sp.array(neighs)[w]
            d1.neighs = [d for d in neighs if d1.ra > d.ra and (10**(d.ll[-1]- sp.log10(lambda_abs)) + 10**(d1.ll[-1] - sp.log10(lambda_abs)))/2. >= z_cut_min+1 and (10**(d.ll[-1]- sp.log10(lambda_abs)) + 10**(d1.ll[-1] - sp.log10(lambda_abs)))/2. < z_cut_max+1 ]

def fill_neighs_x_correlation(pix):
    for ipix in pix:
        for d1 in data[ipix]:
            npix = query_disc(nside,[d1.xcart,d1.ycart,d1.zcart],angmax,inclusive = True)
            npix = [p for p in npix if p in data2]
            neighs = [d for p in npix for d in data2[p] if d1.thid != d.thid]
            ang = d1^neighs
            w = (ang<angmax)
            neighs = sp.array(neighs)[w]
            d1.neighs = [d for d in neighs if (10**(d.ll[-1]- sp.log10(lambda_abs2)) + 10**(d1.ll[-1] - sp.log10(lambda_abs)))/2. >= z_cut_min+1 and (10**(d.ll[-1]- sp.log10(lambda_abs2)) + 10**(d1.ll[-1] - sp.log10(lambda_abs)))/2. < z_cut_max+1 ]

def cf(pix):
    xi = sp.zeros(np*nt)
    we = sp.zeros(np*nt)
    rp = sp.zeros(np*nt)
    rt = sp.zeros(np*nt)
    z = sp.zeros(np*nt)
    nb = sp.zeros(np*nt,dtype=sp.int64)

    for ipix in pix:
        for d1 in data[ipix]:
            print("\rcomputing xi: {}%".format(round(counter.value*100./ndata,2)),end="")
            with lock:
                counter.value += 1
            for d2 in d1.neighs:
                ang = d1^d2
                same_half_plate = (d1.plate == d2.plate) and\
                        ( (d1.fid<=500 and d2.fid<=500) or (d1.fid>500 and d2.fid>500) )
                if ang_correlation:
                    cw,cd,crp,crt,cz,cnb = fast_cf(d1.z,10.**d1.ll,d1.we,d1.de,d2.z,10.**d2.ll,d2.we,d2.de,ang,same_half_plate)
                else:
                    cw,cd,crp,crt,cz,cnb = fast_cf(d1.z,d1.r_comov,d1.we,d1.de,d2.z,d2.r_comov,d2.we,d2.de,ang,same_half_plate)

                xi[:len(cd)]+=cd
                we[:len(cw)]+=cw
                rp[:len(crp)]+=crp
                rt[:len(crp)]+=crt
                z[:len(crp)]+=cz
                nb[:len(cnb)]+=cnb.astype(int)
            setattr(d1,"neighs",None)

    w = we>0
    xi[w]/=we[w]
    rp[w]/=we[w]
    rt[w]/=we[w]
    z[w]/=we[w]
    return we,xi,rp,rt,z,nb
@jit
def fast_cf(z1,r1,w1,d1,z2,r2,w2,d2,ang,same_half_plate):
    wd1 = d1*w1
    wd2 = d2*w2
    if ang_correlation:
        rp = r1/r2[:,None]
        if not x_correlation:
            rp[(rp<1.)] = 1./rp[(rp<1.)]
        rt = ang*sp.ones_like(rp)
    else:
        if x_correlation : rp = (r1-r2[:,None])*sp.cos(ang/2)
        else : rp = abs(r1-r2[:,None])*sp.cos(ang/2)
        rt = (r1+r2[:,None])*sp.sin(ang/2)
    wd12 = wd1*wd2[:,None]
    w12 = w1*w2[:,None]
    z = (z1+z2[:,None])/2

    w = (rp<rp_max) & (rt<rt_max) & (rp>=rp_min)

    rp = rp[w]
    rt = rt[w]
    z  = z[w]
    wd12 = wd12[w]
    w12 = w12[w]
    bp = sp.floor((rp-rp_min)/(rp_max-rp_min)*np).astype(int)
    bt = (rt/rt_max*nt).astype(int)
    bins = bt + nt*bp
    if same_half_plate:
        w = abs(rp)<(rp_max-rp_min)/np
        wd12[w] = 0
        w12[w]=0
    if no_same_wavelength_pairs:
        if ang_correlation:
            w = rp==1.
        else:
            w = rp==0.
        wd12[w] = 0.
        w12[w] = 0.

    cd = sp.bincount(bins,weights=wd12)
    cw = sp.bincount(bins,weights=w12)
    crp = sp.bincount(bins,weights=rp*w12)
    crt = sp.bincount(bins,weights=rt*w12)
    cz = sp.bincount(bins,weights=z*w12)
    cnb = sp.bincount(bins,weights=(w12>0.))

    return cw,cd,crp,crt,cz,cnb

def dmat(pix):

    dm = sp.zeros(np*nt*nt*np)
    wdm = sp.zeros(np*nt)

    npairs = 0
    npairs_used = 0
    for p in pix:
        for d1 in data[p]:
            print("\rcomputing xi: {}%".format(round(counter.value*100./ndata,3)),end="")
            with lock:
                counter.value += 1
            order1 = d1.order
            r1 = d1.r_comov
            w1 = d1.we
            l1 = d1.ll
            r = random.rand(len(d1.neighs))
            w=r>rej
            npairs += len(d1.neighs)
            npairs_used += w.sum()
            for d2 in sp.array(d1.neighs)[w]:
                same_half_plate = (d1.plate == d2.plate) and\
                        ( (d1.fid<=500 and d2.fid<=500) or (d1.fid>500 and d2.fid>500) )
                order2 = d2.order
                ang = d1^d2
                r2 = d2.r_comov
                w2 = d2.we
                l2 = d2.ll
                fill_dmat(l1,l2,r1,r2,w1,w2,ang,wdm,dm,same_half_plate,order1,order2)
            setattr(d1,"neighs",None)

    return wdm,dm.reshape(np*nt,np*nt),npairs,npairs_used

@jit
def fill_dmat(l1,l2,r1,r2,w1,w2,ang,wdm,dm,same_half_plate,order1,order2):

    if x_correlation:
        rp = (r1[:,None]-r2)*sp.cos(ang/2)
    else:
        rp = abs(r1[:,None]-r2)*sp.cos(ang/2)
    rt = (r1[:,None]+r2)*sp.sin(ang/2)
    bp = sp.floor((rp-rp_min)/(rp_max-rp_min)*np).astype(int)
    bt = (rt/rt_max*nt).astype(int)
    bins = bt + nt*bp

    sw1 = w1.sum()
    sw2 = w2.sum()

    ml1 = sp.average(l1,weights=w1)
    ml2 = sp.average(l2,weights=w2)

    dl1 = l1-ml1
    dl2 = l2-ml2

    slw1 = (w1*dl1**2).sum()
    slw2 = (w2*dl2**2).sum()

    w = (rp<rp_max) & (rt<rt_max)& (rp>=rp_min)

    bins = bins[w]

    n1 = len(l1)
    n2 = len(l2)
    ij = sp.arange(n1)[:,None]+n1*sp.arange(n2)
    ij = ij[w]

    we = w1[:,None]*w2
    we = we[w]
    if same_half_plate:
        wsame = abs(rp[w])<(rp_max-rp_min)/np
        we[wsame]=0
    if no_same_wavelength_pairs:
        if ang_correlation:
            wsame = rp==1.
        else:
            wsame = rp==0.
        we[wsame] = 0.

    c = sp.bincount(bins,weights=we)
    wdm[:len(c)] += c
    eta1 = sp.zeros(np*nt*n1)
    eta2 = sp.zeros(np*nt*n2)
    eta3 = sp.zeros(np*nt*n1)
    eta4 = sp.zeros(np*nt*n2)
    eta5 = sp.zeros(np*nt)
    eta6 = sp.zeros(np*nt)
    eta7 = sp.zeros(np*nt)
    eta8 = sp.zeros(np*nt)

    c = sp.bincount(ij%n1+n1*bins,weights=(sp.ones(n1)[:,None]*w2)[w]/sw2)
    eta1[:len(c)]+=c
    c = sp.bincount((ij-ij%n1)//n1+n2*bins,weights = (w1[:,None]*sp.ones(n2))[w]/sw1)
    eta2[:len(c)]+=c
    c = sp.bincount(bins,weights=(w1[:,None]*w2)[w]/sw1/sw2)
    eta5[:len(c)]+=c

    if order2==1:
        c = sp.bincount(ij%n1+n1*bins,weights=(sp.ones(n1)[:,None]*w2*dl2)[w]/slw2)
        eta3[:len(c)]+=c
        c = sp.bincount(bins,weights=(w1[:,None]*(w2*dl2))[w]/sw1/slw2)
        eta6[:len(c)]+=c
    if order1==1:
        c = sp.bincount((ij-ij%n1)//n1+n2*bins,weights = ((w1*dl1)[:,None]*sp.ones(n2))[w]/slw1)
        eta4[:len(c)]+=c
        c = sp.bincount(bins,weights=((w1*dl1)[:,None]*w2)[w]/slw1/sw2)
        eta7[:len(c)]+=c
        if order2==1:
            c = sp.bincount(bins,weights=((w1*dl1)[:,None]*(w2*dl2))[w]/slw1/slw2)
            eta8[:len(c)]+=c

    ubb = sp.unique(bins)
    for k,ba in enumerate(bins):
        dm[ba+np*nt*ba]+=we[k]
        i = ij[k]%n1
        j = (ij[k]-i)//n1
        for bb in ubb:
            dm[bb+np*nt*ba] += we[k]*(eta5[bb]+eta6[bb]*dl2[j]+eta7[bb]*dl1[i]+eta8[bb]*dl1[i]*dl2[j])\
             - we[k]*(eta1[i+n1*bb]+eta3[i+n1*bb]*dl2[j]+eta2[j+n2*bb]+eta4[j+n2*bb]*dl1[i])

def metal_dmat(pix,abs_igm1="LYA",abs_igm2="SiIII(1207)"):

    dm = sp.zeros(np*nt*ntm*npm)
    wdm = sp.zeros(np*nt)
    rpeff = sp.zeros(ntm*npm)
    rteff = sp.zeros(ntm*npm)
    zeff = sp.zeros(ntm*npm)
    weff = sp.zeros(ntm*npm)

    npairs = 0
    npairs_used = 0
    for p in pix:
        for d1 in data[p]:
            with lock:
                print("\rcomputing metal dmat {} {}: {}%".format(abs_igm1,abs_igm2,round(counter.value*100./ndata,3)),end="")
                counter.value += 1

            r = random.rand(len(d1.neighs))
            w=r>rej
            npairs += len(d1.neighs)
            npairs_used += w.sum()
            for d2 in sp.array(d1.neighs)[w]:
                r1 = d1.r_comov
                z1_abs1 = 10**d1.ll/constants.absorber_IGM[abs_igm1]-1
                r1_abs1 = cosmo.r_comoving(z1_abs1)
                w1 = d1.we

                wzcut = z1_abs1<d1.zqso
                r1 = r1[wzcut]
                w1 = w1[wzcut]
                r1_abs1 = r1_abs1[wzcut]
                z1_abs1 = z1_abs1[wzcut]

                same_half_plate = (d1.plate == d2.plate) and\
                        ( (d1.fid<=500 and d2.fid<=500) or (d1.fid>500 and d2.fid>500) )
                ang = d1^d2
                r2 = d2.r_comov
                z2_abs2 = 10**d2.ll/constants.absorber_IGM[abs_igm2]-1
                r2_abs2 = cosmo.r_comoving(z2_abs2)
                w2 = d2.we

                wzcut = z2_abs2<d2.zqso
                r2 = r2[wzcut]
                w2 = w2[wzcut]
                r2_abs2 = r2_abs2[wzcut]
                z2_abs2 = z2_abs2[wzcut]

                rp = (r1[:,None]-r2)*sp.cos(ang/2)
                if not x_correlation:
                    rp = abs(rp)

                rt = (r1[:,None]+r2)*sp.sin(ang/2)
                w12 = w1[:,None]*w2

                bp = sp.floor((rp-rp_min)/(rp_max-rp_min)*np).astype(int)
                bt = (rt/rt_max*nt).astype(int)

                if same_half_plate:
                    wp = abs(rp) < (rp_max-rp_min)/np
                    w12[wp]=0

                if no_same_wavelength_pairs:
                    if ang_correlation:
                        wp = rp==1.
                    else:
                        wp = rp==0.
                    w12[wp] = 0.

                bA = bt + nt*bp
                wA = (bp<np) & (bt<nt) & (bp >=0)
                c = sp.bincount(bA[wA],weights=w12[wA])
                wdm[:len(c)]+=c

                rp_abs1_abs2 = (r1_abs1[:,None]-r2_abs2)*sp.cos(ang/2)

                if not x_correlation:
                    rp_abs1_abs2 = abs(rp_abs1_abs2)

                rt_abs1_abs2 = (r1_abs1[:,None]+r2_abs2)*sp.sin(ang/2)
                zwe12 = (1+z1_abs1[:,None])**(alpha_abs[abs_igm1]-1)*(1+z2_abs2)**(alpha_abs[abs_igm2]-1)/(1+zref)**(alpha_abs[abs_igm1]+alpha_abs[abs_igm2]-2)

                bp_abs1_abs2 = sp.floor((rp_abs1_abs2-rp_min)/(rp_max-rp_min)*npm).astype(int)
                bt_abs1_abs2 = (rt_abs1_abs2/rt_max*ntm).astype(int)
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
                    w1 = d1.we
                    z1_abs2 = 10**d1.ll/constants.absorber_IGM[abs_igm2]-1
                    r1_abs2 = cosmo.r_comoving(z1_abs2)

                    wzcut = z1_abs2<d1.zqso
                    r1 = r1[wzcut]
                    w1 = w1[wzcut]
                    z1_abs2 = z1_abs2[wzcut]
                    r1_abs2 = r1_abs2[wzcut]

                    r2 = d2.r_comov
                    w2 = d2.we
                    z2_abs1 = 10**d2.ll/constants.absorber_IGM[abs_igm1]-1
                    r2_abs1 = cosmo.r_comoving(z2_abs1)

                    wzcut = z2_abs1<d2.zqso
                    r2 = r2[wzcut]
                    w2 = w2[wzcut]
                    z2_abs1 = z2_abs1[wzcut]
                    r2_abs1 = r2_abs1[wzcut]

                    rp = (r1[:,None]-r2)*sp.cos(ang/2)
                    if not x_correlation:
                        rp = abs(rp)

                    rt = (r1[:,None]+r2)*sp.sin(ang/2)
                    w12 = w1[:,None]*w2

                    bp = sp.floor((rp-rp_min)/(rp_max-rp_min)*np).astype(int)
                    bt = (rt/rt_max*nt).astype(int)
                    if same_half_plate:
                        wp = abs(rp) < (rp_max-rp_min)/np
                        w12[wp]=0
                    if no_same_wavelength_pairs:
                        if ang_correlation:
                            wp = rp==1.
                        else:
                            wp = rp==0.
                        w12[wp] = 0.
                    bA = bt + nt*bp
                    wA = (bp<np) & (bt<nt) & (bp >=0)
                    c = sp.bincount(bA[wA],weights=w12[wA])
                    wdm[:len(c)]+=c
                    rp_abs2_abs1 = (r1_abs2[:,None]-r2_abs1)*sp.cos(ang/2)
                    if not x_correlation:
                        rp_abs2_abs1 = abs(rp_abs2_abs1)

                    rt_abs2_abs1 = (r1_abs2[:,None]+r2_abs1)*sp.sin(ang/2)
                    zwe21 = (1+z1_abs2[:,None])**(alpha_abs[abs_igm2]-1)*(1+z2_abs1)**(alpha_abs[abs_igm1]-1)/(1+zref)**(alpha_abs[abs_igm1]+alpha_abs[abs_igm2]-2)

                    bp_abs2_abs1 = sp.floor((rp_abs2_abs1-rp_min)/(rp_max-rp_min)*npm).astype(int)
                    bt_abs2_abs1 = (rt_abs2_abs1/rt_max*ntm).astype(int)
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

    return wdm,dm.reshape(np*nt,npm*ntm),rpeff,rteff,zeff,weff,npairs,npairs_used



n1d = None
lmin = None
lmax = None
dll = None
def cf1d(pix):
    xi1d = sp.zeros(n1d**2)
    we1d = sp.zeros(n1d**2)
    nb1d = sp.zeros(n1d**2,dtype=sp.int64)

    for d in data[pix]:
        bins = ((d.ll-lmin)/dll+0.5).astype(int)
        bins = bins + n1d*bins[:,None]
        wde = d.we*d.de
        we = d.we
        xi1d[bins] += wde * wde[:,None]
        we1d[bins] += we*we[:,None]
        nb1d[bins] += (we*we[:,None]>0.).astype(int)

    w = we1d>0
    xi1d[w]/=we1d[w]
    return we1d,xi1d,nb1d

def x_forest_cf1d(pix):
    xi1d = sp.zeros(n1d**2)
    we1d = sp.zeros(n1d**2)
    nb1d = sp.zeros(n1d**2,dtype=sp.int64)

    for d1 in data[pix]:
        bins1 = ((d1.ll-lmin)/dll+0.5).astype(int)
        wde1 = d1.we*d1.de
        we1 = d1.we

        d2thingid = [d2.thid for d2 in data2[pix]]
        neighs = data2[pix][sp.in1d(d2thingid,[d1.thid])]
        for d2 in neighs:
            bins2 = ((d2.ll-lmin)/dll+0.5).astype(int)
            bins = bins1 + n1d*bins2[:,None]
            wde2 = d2.we*d2.de
            we2 = d2.we
            xi1d[bins] += wde1 * wde2[:,None]
            we1d[bins] += we1*we2[:,None]
            nb1d[bins] += (we1*we2[:,None]>0.).astype(int)

    w = we1d>0
    xi1d[w]/=we1d[w]
    return we1d,xi1d,nb1d

v1d = None
c1d = None
## auto
def t123(pix):
    t123_loc = sp.zeros([np*nt,np*nt])
    w123 = sp.zeros(np*nt)
    npairs = 0
    npairs_used = 0
    for ipix in pix:
        for d1 in data[ipix]:
            print("\rcomputing xi: {}%".format(round(counter.value*100./ndata,3)),end="")
            with lock:
                counter.value += 1
            v1 = v1d(d1.ll)
            w1 = d1.we
            c1d_1 = (w1*w1[:,None])*c1d(abs(d1.ll-d1.ll[:,None]))*sp.sqrt(v1*v1[:,None])
            r1 = d1.r_comov
            z1 = 10**d1.ll/lambda_abs-1
            r = random.rand(len(d1.neighs))
            w = r>rej
            npairs+=len(d1.neighs)
            npairs_used += w.sum()
            for d2 in sp.array(d1.neighs)[w]:
                ang = d1^d2

                same_half_plate = (d1.plate == d2.plate) and\
                        ( (d1.fid<=500 and d2.fid<=500) or (d1.fid>500 and d2.fid>500) )
                v2 = v1d(d2.ll)
                w2 = d2.we
                c1d_2 = (w2*w2[:,None])*c1d(abs(d2.ll-d2.ll[:,None]))*sp.sqrt(v2*v2[:,None])
                r2 = d2.r_comov
                z2 = 10**d2.ll/lambda_abs-1

                fill_t123(r1,r2,ang,w1,w2,z1,z2,c1d_1,c1d_2,w123,t123_loc,same_half_plate)
            setattr(d1,"neighs",None)

    return w123,t123_loc,npairs,npairs_used


@jit
def fill_t123(r1,r2,ang,w1,w2,z1,z2,c1d_1,c1d_2,w123,t123_loc,same_half_plate):

    n1 = len(r1)
    n2 = len(r2)
    i1 = sp.arange(n1)
    i2 = sp.arange(n2)
    zw1 = ((1+z1)/(1+zref))**(alpha-1)
    zw2 = ((1+z2)/(1+zref))**(alpha-1)

    bins = i1[:,None]+n1*i2
    if x_correlation:
        rp = (r1[:,None]-r2)*sp.cos(ang/2)
    else :
        rp = abs(r1[:,None]-r2)*sp.cos(ang/2)
    rt = (r1[:,None]+r2)*sp.sin(ang/2)
    bp = sp.floor((rp-rp_min)/(rp_max-rp_min)*np).astype(int)
    bt = (rt/rt_max*nt).astype(int)
    ba = bt + nt*bp
    we = w1[:,None]*w2
    zw = zw1[:,None]*zw2

    w = (rp<rp_max) & (rt<rt_max) & (rp>=rp_min)

    if same_half_plate:
        w = w & (abs(rp)>(rp_max-rp_min)/np)

    bins = bins[w]
    ba = ba[w]
    we = we[w]
    zw = zw[w]

    for k in range(w.sum()):
        i1 = bins[k]%n1
        j1 = (bins[k]-i1)//n1
        w123[ba[k]]+=we[k]
        t123_loc[ba[k],ba[k]]+=we[k]/zw[k]
        for l in range(k+1,w.sum()):
            i2 = bins[l]%n1
            j2 = (bins[l]-i2)//n1
            prod = c1d_1[i1,i2]*c1d_2[j1,j2]
            t123_loc[ba[k],ba[l]]+=prod
            t123_loc[ba[l],ba[k]]+=prod

