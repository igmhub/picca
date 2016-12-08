import scipy as sp
import sys
from healpy import query_disc
from multiprocessing import Pool

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

neigh_dic={}
data = None
pix_dic = None

def fill_neighs(pix):
    d1s = [data[thid] for thid in pix_dic[pix]]
    for d1 in d1s:
        pix = query_disc(nside,[d1.x,d1.y,d1.z],angmax,inclusive = True)
        pix = [p for p in pix if p in pix_dic]
        neighs = [data[thid] for p in pix for thid in pix_dic[p]]
        ang = d1^neighs
        w = ang<angmax
        neighs = sp.array(neighs)[w]
        d1.neighs = [d.thid for d in neighs if d1.ra > d.ra]

def cf(pix):
    xi = sp.zeros(np*nt)
    we = sp.zeros(np*nt)

    d1s = [data[thid] for thid in pix_dic[pix]]
    for i,d1 in enumerate(d1s):
        wd1 = d1.de*d1.we
        neighs = [data[thid] for thid in d1.neighs]
        for d2 in neighs:

            wd2= d2.de*d2.we
            ang = d1^d2
            
            rp = abs(d1.r_comov-d2.r_comov[:,None])*sp.cos(ang/2)
            rt = (d1.r_comov+d2.r_comov[:,None])*sp.sin(ang/2)
            wd12 = wd1*wd2[:,None]
            w12 = d1.we*d2.we[:,None]
            
            w = (rp<rp_max) & (rt<rt_max)
            rp = rp[w]
            rt = rt[w]
            wd12 = wd12[w]
            w12 = w12[w]
            bp = (rp/rp_max*np).astype(int)
            bt = (rt/rt_max*nt).astype(int)
            bins = bt + nt*bp
            c = sp.bincount(bins,weights=wd12)
            xi[:len(c)]+=c
            c = sp.bincount(bins,weights=w12)
            we[:len(c)]+=c

    w = we>0
    xi[w]/=we[w]
    return we,xi
    
