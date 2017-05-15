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
rt_max = None
angmax = None
nside = None

counter = None
ndata = None

zref = None
alpha= None
lambda_abs = None

data = None

rej = None
lock = None

def fill_neighs(pix):
    for ipix in pix:
        for d1 in data[ipix]:
            npix = query_disc(nside,[d1.xcart,d1.ycart,d1.zcart],angmax,inclusive = True)
            npix = [p for p in npix if p in data]
            neighs = [d for p in npix for d in data[p]]
            ang = d1^neighs
            w = ang<angmax
            neighs = sp.array(neighs)[w]
            d1.neighs = [d for d in neighs if d1.ra > d.ra]

def cf(pix):
    xi = sp.zeros(np*nt)
    we = sp.zeros(np*nt)
    rp = sp.zeros(np*nt)
    rt = sp.zeros(np*nt)
    z = sp.zeros(np*nt)

    for ipix in pix:
        for i,d1 in enumerate(data[ipix]):
            sys.stderr.write("\rcomputing xi: {}%".format(round(counter.value*100./ndata,2)))
            with lock:
                counter.value += 1
            for d2 in d1.neighs:
                ang = d1^d2
                same_half_plate = (d1.plate == d2.plate) and\
                        ( (d1.fid<=500 and d2.fid<=500) or (d1.fid>500 and d2.fid>500) )
                cw,cd,crp,crt,cz = fast_cf(d1.z,d1.r_comov,d1.we,d1.de,d2.z,d2.r_comov,d2.we,d2.de,ang,same_half_plate)
            
                xi[:len(cd)]+=cd
                we[:len(cw)]+=cw
                rp[:len(crp)]+=crp
                rt[:len(crp)]+=crt
                z[:len(crp)]+=cz

    w = we>0
    xi[w]/=we[w]
    rp[w]/=we[w]
    rt[w]/=we[w]
    z[w]/=we[w]
    return we,xi,rp,rt,z
@jit 
def fast_cf(z1,r1,w1,d1,z2,r2,w2,d2,ang,same_half_plate):
    wd1 = d1*w1
    wd2 = d2*w2
    rp = abs(r1-r2[:,None])*sp.cos(ang/2)
    rt = (r1+r2[:,None])*sp.sin(ang/2)
    wd12 = wd1*wd2[:,None]
    w12 = w1*w2[:,None]
    z = (z1+z2[:,None])/2

    w = (rp<rp_max) & (rt<rt_max)
    rp = rp[w]
    rt = rt[w]
    z  = z[w]
    wd12 = wd12[w]
    w12 = w12[w]
    bp = (rp/rp_max*np).astype(int)
    bt = (rt/rt_max*nt).astype(int)
    bins = bt + nt*bp
    if same_half_plate:
        w = bp == 0
        wd12[w] = 0
        w12[w]=0

    cd = sp.bincount(bins,weights=wd12)
    cw = sp.bincount(bins,weights=w12)
    crp = sp.bincount(bins,weights=rp*w12)
    crt = sp.bincount(bins,weights=rt*w12)
    cz = sp.bincount(bins,weights=z*w12)

    return cw,cd,crp,crt,cz


def dmat(pix):

    dm = sp.zeros(np*nt*nt*np)
    wdm = sp.zeros(np*nt)

    npairs = 0L
    npairs_used = 0L
    for p in pix:
        for d1 in data[p]:
            sys.stderr.write("\rcomputing xi: {}%".format(round(counter.value*100./ndata,3)))
            with lock:
                counter.value += 1
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
                ang = d1^d2
                r2 = d2.r_comov
                w2 = d2.we
                l2 = d2.ll
                fill_dmat(l1,l2,r1,r2,w1,w2,ang,wdm,dm,same_half_plate)

    return wdm,dm.reshape(np*nt,np*nt),npairs,npairs_used
    
@jit
def fill_dmat(l1,l2,r1,r2,w1,w2,ang,wdm,dm,same_half_plate):
    rp = abs(r1[:,None]-r2)*sp.cos(ang/2)
    rt = (r1[:,None]+r2)*sp.sin(ang/2)
    bp = (rp/rp_max*np).astype(int)
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

    w = (rp<rp_max) & (rt<rt_max)

    bins = bins[w]

    n1 = len(l1)
    n2 = len(l2)
    ij = sp.arange(n1)[:,None]+n1*sp.arange(n2)
    ij = ij[w]

    we = w1[:,None]*w2
    we = we[w]
    if same_half_plate:
        wsame = bp[w]==0
        we[wsame]=0

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
    c = sp.bincount((ij-ij%n1)/n1+n2*bins,weights = (w1[:,None]*sp.ones(n2))[w]/sw1)
    eta2[:len(c)]+=c
    c = sp.bincount(ij%n1+n1*bins,weights=(sp.ones(n1)[:,None]*w2*dl2)[w]/slw2)
    eta3[:len(c)]+=c
    c = sp.bincount((ij-ij%n1)/n1+n2*bins,weights = ((w1*dl1)[:,None]*sp.ones(n2))[w]/slw1)
    eta4[:len(c)]+=c

    c = sp.bincount(bins,weights=(w1[:,None]*w2)[w]/sw1/sw2)
    eta5[:len(c)]+=c
    c = sp.bincount(bins,weights=(w1[:,None]*(w2*dl2))[w]/sw1/slw2)
    eta6[:len(c)]+=c
    c = sp.bincount(bins,weights=((w1*dl1)[:,None]*w2)[w]/slw1/sw2)
    eta7[:len(c)]+=c
    c = sp.bincount(bins,weights=((w1*dl1)[:,None]*(w2*dl2))[w]/slw1/slw2)
    eta8[:len(c)]+=c

    ubb = sp.unique(bins)
    for k,ba in enumerate(bins):
        dm[ba+np*nt*ba]+=we[k]
        i = ij[k]%n1
        j = (ij[k]-i)/n1
        for bb in ubb:
            dm[ba+np*nt*bb] += we[k]*(eta5[bb]+eta6[bb]*dl2[j]+eta7[bb]*dl1[i]+eta8[bb]*dl1[i]*dl2[j])\
             - we[k]*(eta1[i+n1*bb]+eta3[i+n1*bb]*dl2[j]+eta2[j+n2*bb]+eta4[j+n2*bb]*dl1[i])

def dmat_order0(pix):

    dm = sp.zeros(np*nt*nt*np)
    wdm = sp.zeros(np*nt)

    npairs = 0L
    npairs_used = 0L
    for p in pix:
        for d1 in data[p]:
            sys.stderr.write("\rcomputing xi: {}%".format(round(counter.value*100./ndata,3)))
            with lock:
                counter.value += 1
            r1 = d1.r_comov
            w1 = d1.we
            r = random.rand(len(d1.neighs))
            w=r>rej
            npairs += len(d1.neighs)
            npairs_used += w.sum()
            for d2 in sp.array(d1.neighs)[w]:
                same_half_plate = (d1.plate == d2.plate) and\
                        ( (d1.fid<=500 and d2.fid<=500) or (d1.fid>500 and d2.fid>500) )
                ang = d1^d2
                r2 = d2.r_comov
                w2 = d2.we
                fill_dmat_order0(r1,r2,w1,w2,ang,wdm,dm,same_half_plate)

    return wdm,dm.reshape(np*nt,np*nt),npairs,npairs_used
    
@jit
def fill_dmat_order0(r1,r2,w1,w2,ang,wdm,dm,same_half_plate):
    rp = abs(r1[:,None]-r2)*sp.cos(ang/2)
    rt = (r1[:,None]+r2)*sp.sin(ang/2)
    bp = (rp/rp_max*np).astype(int)
    bt = (rt/rt_max*nt).astype(int)
    bins = bt + nt*bp
    
    sw1 = w1.sum()
    sw2 = w2.sum()
    w = (rp<rp_max) & (rt<rt_max)

    bins = bins[w]

    n1 = len(w1)
    n2 = len(w2)
    ij = sp.arange(n1)[:,None]+n1*sp.arange(n2)
    ij = ij[w]

    we = w1[:,None]*w2
    we = we[w]
    if same_half_plate:
        wsame = bp[w]==0
        we[wsame]=0

    c = sp.bincount(bins,weights=we)
    wdm[:len(c)] += c
    eta1 = sp.zeros(np*nt*n1)
    eta2 = sp.zeros(np*nt*n2)
    eta5 = sp.zeros(np*nt*n1)


    c = sp.bincount(ij%n1+n1*bins,weights=(sp.ones(n1)[:,None]*w2)[w]/sw2)
    eta1[:len(c)]+=c
    c = sp.bincount((ij-ij%n1)/n1+n2*bins,weights = (w1[:,None]*sp.ones(n2))[w]/sw1)
    eta2[:len(c)]+=c
    c = sp.bincount(bins,weights=(w1[:,None]*w2)[w]/sw1/sw2)
    eta5[:len(c)]+=c

    ubb = sp.unique(bins)
    for k,ba in enumerate(bins):
        dm[ba+np*nt*ba]+=we[k]
        i = ij[k]%n1
        j = (ij[k]-i)/n1
        for bb in ubb:
            dm[ba+np*nt*bb] += we[k]*eta5[bb] - we[k]*(eta1[i+n1*bb]+eta2[j+n2*bb])

n1d = None
def cf1d(pix):
    xi1d = sp.zeros(n1d**2)
    we1d = sp.zeros(n1d**2)
    nb1d = sp.zeros(n1d**2,dtype=sp.int64)

    for d in data[pix]:
        bins = ((d.ll-forest.lmin)/forest.dll+0.5).astype(int)
        bins = bins + n1d*bins[:,None]
        wde = d.we*d.de
        we = d.we
        xi1d[bins] += wde * wde[:,None]
        we1d[bins] += we*we[:,None]
        nb1d[bins] += 1

    w = we1d>0
    xi1d[w]/=we1d[w]
    return we1d,xi1d,nb1d
v1d = None
c1d = None

## auto
def t123(pix):
    t123_loc = sp.zeros([np*nt,np*nt])
    w123 = sp.zeros(np*nt)
    npairs = 0L
    npairs_used = 0L
    for i,ipix in enumerate(pix):
        for d1 in data[ipix]:
            sys.stderr.write("\rcomputing xi: {}%".format(round(counter.value*100./ndata,3)))
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

    rp = abs(r1[:,None]-r2)*sp.cos(ang/2)
    rt = (r1[:,None]+r2)*sp.sin(ang/2)
    bp = (rp/rp_max*np).astype(int)
    bt = (rt/rt_max*nt).astype(int)
    ba = bt + nt*bp
    we = w1[:,None]*w2
    zw = zw1[:,None]*zw2

    w = (rp<rp_max) & (rt<rt_max)
    if same_half_plate:
        w = w & (bp>0)

    bins = bins[w]
    ba = ba[w]
    we = we[w]
    zw = zw[w]

    for k in xrange(w.sum()):
        i1 = bins[k]%n1
        j1 = (bins[k]-i1)/n1
        w123[ba[k]]+=we[k]
        t123_loc[ba[k],ba[k]]+=we[k]/zw[k]
        for l in xrange(k+1,w.sum()):
            i2 = bins[l]%n1
            j2 = (bins[l]-i2)/n1
            prod = c1d_1[i1,i2]*c1d_2[j1,j2]
            t123_loc[ba[k],ba[l]]+=prod
            t123_loc[ba[l],ba[k]]+=prod

