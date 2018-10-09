import scipy as sp

class wedge:
    def __init__(self,rpmin=0.,rpmax=200.,nrp=50,rtmin=0.,rtmax=200.,nrt=50,\
            rmin=0.,rmax=200.,nr=50,mumin=0.8,mumax=0.95,ss=10,absoluteMu=False):
        nrtmc = ss*nrt
        nrpmc = ss*nrp
        nss=nrtmc*nrpmc
        index=sp.arange(nss)
        irtmc=index%nrtmc
        irpmc=(index-irtmc)//nrtmc
        rtmc = rtmin+(irtmc+0.5)*(rtmax-rtmin)/nrtmc
        rpmc = rpmin+(irpmc+0.5)*(rpmax-rpmin)/nrpmc
        rmc = sp.sqrt(rtmc**2+rpmc**2)
        mumc = rpmc/rmc
        if absoluteMu:
            mumc = sp.absolute(mumc)

        br = (rmc-rmin)/(rmax-rmin)*nr
        br = br.astype(int)

        bt = (rtmc-rtmin)/(rtmax-rtmin)*nrt
        bt = bt.astype(int)

        bp = (rpmc-rpmin)/(rpmax-rpmin)*nrp
        bp = bp.astype(int)

        rp = rpmin + (bp+0.5)*(rpmax-rpmin)/nrp
        rt = rtmin + (bt+0.5)*(rtmax-rtmin)/nrt
        r=sp.sqrt(rp**2+rt**2)

        bins = bt+nrt*bp + nrp*nrt*br

        w = (mumc>=mumin) & (mumc<=mumax) & (r<rmax) & (r>rmin) & (br<nr)
        bins = bins[w]

        W = sp.zeros(nrp*nrt*nr)
        c=sp.bincount(bins.flatten())
        W[:len(c)]+=c

        self.W = W.reshape(nr,nrt*nrp)
        self.r = rmin + (sp.arange(nr)+0.5)*(rmax-rmin)/nr

    def wedge(self,da,co):
        we = 1/sp.diagonal(co)
        w = self.W.dot(we)
        Wwe = self.W*we
        mask = w>0
        Wwe[mask,:]/=w[mask,None]
        d = Wwe.dot(da)
        return self.r,d,Wwe.dot(co).dot(Wwe.T)


