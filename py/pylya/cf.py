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

def fill_neighs(data):
    for i,p1 in enumerate(data):
        if i%10==0:
            sys.stderr.write("\rfilling neighs {}%".format(round(i*100./len(data),2)))

        for d1 in data[p1]:
            pix = query_disc(nside,[d1.x,d1.y,d1.z],angmax,inclusive = True)
            pix = [p for p in pix if p in data]
            neighs = [d2 for p2 in pix for d2 in data[p2]]
            ang = d1^neighs
            w = ang<angmax
            neighs = sp.array(neighs)[w]
            d1.neighs = [d2 for d2 in neighs if d1.ra > d2.ra]
    sys.stderr.write("\n")

def cf(data):
    xi = sp.zeros(np*nt)
    we = sp.zeros(np*nt)

    for i,d1 in enumerate(data):
        wd1 = d1.de*d1.we
        for d2 in d1.neighs:

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
    
