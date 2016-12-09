import scipy as sp
import sys
from healpy import query_disc
from multiprocessing import Pool
from numba import jit

np = None
nt = None 
rp_max = None
rt_max = None
angmax = None
nside = None

counter = None
lock = None
npix = None

z0 = None
alpha= None

data = None

def fill_neighs(pix):
    for d1 in data[pix]:
        npix = query_disc(nside,[d1.x,d1.y,d1.z],angmax,inclusive = True)
        npix = [p for p in npix if p in data]
        neighs = [d for p in npix for d in data[p]]
        ang = d1^neighs
        w = ang<angmax
        neighs = sp.array(neighs)[w]
        d1.neighs = [d for d in neighs if d1.ra > d.ra]

def cf(pix):
    xi = sp.zeros(np*nt)
    we = sp.zeros(np*nt)

    for i,d1 in enumerate(data[pix]):
        for d2 in d1.neighs:
            ang = d1^d2
            same_half_plate = False
            if d1.plate == d2.plate:
                if d1.fid<=500 and d2.fid<=500:
                    same_half_plate = True
                elif d1.fid>500 and d2.fid>500:
                    same_half_plate = True
            cw,cd = fast_cf(d1.r_comov,d1.we,d1.de,d2.r_comov,d2.we,d2.de,ang,same_half_plate)
            
            xi[:len(cd)]+=cd
            we[:len(cw)]+=cw

    w = we>0
    xi[w]/=we[w]
    return we,xi
@jit 
def fast_cf(r1,w1,d1,r2,w2,d2,ang,same_half_plate):
    wd1 = d1*w1
    wd2 = d2*w2
    rp = abs(r1-r2[:,None])*sp.cos(ang/2)
    rt = (r1+r2[:,None])*sp.sin(ang/2)
    wd12 = wd1*wd2[:,None]
    w12 = w1*w2[:,None]

    w = (rp<rp_max) & (rt<rt_max)
    rp = rp[w]
    rt = rt[w]
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

    return cw,cd


