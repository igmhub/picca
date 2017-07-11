import scipy as sp
from picca import cf

v1d = None
c1d = None


rp_max = None
np = None

rt_max = None
nt = None

data = None
rej = None

## auto
def t123(pix):
    t123_loc = sp.zeros(np*nt*np*nt)
    w123 = sp.zeros(np*nt*np*nt)
    for d1 in data[pix]:
        v1 = v1d(d1.ll)
        w1 = d1.we
        n1 = len(d1.ll)

        we1 = (d1.we*d1.we[:,None]).flatten()
        c1d_1 = we1*c1d(d1.ll-d1.ll[:,None])*sp.sqrt(v1*v1[:,None])

        c1d_1 = c1d_1.flatten()
        we1 = we1.flatten()

        r1a = d1.r_comov*sp.ones(n1)[:,None]
        r1b = r1a.T.flatten()
        r1a = r1a.flatten()
        
        for d2 in d1.neighs:
            ang = d1^d2
            v2 =  v1d(d2.ll)
            n2 = len(d2.ll)

            we2 = (d2.we*d2.we[:,None])
            c1d_2 = we2*c1d(d2.ll-d2.ll[:,None])*sp.sqrt(v2*v2[:,None])
            
            c1d_2 = c1d_2.flatten()
            we2 = we2.flatten()

            r2a = d1.r_comov*sp.ones(n1)[:,None]
            r2b = r1a.T.flatten()
            r2b = r2b.flatten()

            c1d = c1d_1*c1d_2[:,None]
            we = we1*we2[:,None]

            rpa = (r1a-r2a[:,None])*sp.cos(ang/2)
            rta = (r1a+r2a[:,None])*sp.sin(ang/2)

            rpb = (r1b-r2b[:,None])*sp.cos(ang/2)
            rtb = (r1b+r2b[:,None])*sp.sin(ang/2)

            w = (rpa < rp_max) & (rta < rt_max) & (rpa < rb_max) & (rtb < rt_max)
            rpa = rpa[w]
            rta = rta[w]

            rpb = rpb[w]
            rtb = rtb[w]

            c1d= c1d[w]
            we = we[w]

            bpa = (rpa/rp_max*nrp).astype(int)
            bta = (rta/rt_max*nrt).astype(int)

            ba = bta + nrt*nrp*bpa

            bpb = (rpb/rp_max*nrp).astype(int)
            btb = (rtb/rt_max*nrt).astype(int)

            bb = btb + nrt*nrp*bpb

            bins = ba + np*nt*bb

            c_c1d = sp.bincount(bins,weights = c1d)
            t123_loc[len(c_c1d)]+=c_c1d

            c_w1d = sp.bincount(bins,weights = we)
            w123[len(c_w1d)]+=c_w1d


    w = w123>0
    t123[w]/=w123[w]

    return w123,t123
            

        
            



