import scipy as sp
from data import forest
import iminuit
from scipy import linalg


## mean continuum
def mc(data):
    nmc=100
    mcont = sp.zeros(nmc)
    wcont = sp.zeros(nmc)
    ll = forest.lmin_rest + (sp.arange(nmc)+.5)*(forest.lmax_rest-forest.lmin_rest)/nmc
    for p in data:
        for d in data[p]:
            bins=((d.ll-forest.lmin_rest-sp.log10(1+d.zqso))/(forest.lmax_rest-forest.lmin_rest)*nmc).astype(int)
            var_lss = forest.var_lss(d.ll)
            we = d.iv/var_lss*d.co**2/(d.iv + d.co**2/var_lss)
            c = sp.bincount(bins,weights=d.fl/d.co*we)
            mcont[:len(c)]+=c
            c = sp.bincount(bins,weights=we)
            wcont[:len(c)]+=c

    w=wcont>0
    mcont[w]/=wcont[w]
    mcont/=mcont.mean()
    return ll,mcont

def var_lss(data):
    nlss = 10
    eta = sp.zeros(nlss)
    vlss = sp.zeros(nlss)
    ll = forest.lmin + (sp.arange(nlss)+.5)*(forest.lmax-forest.lmin)/nlss

    nwe = 100
    vpmin = 0
    vpmax = 2

    var = vpmin + (sp.arange(nwe)+.5)*(vpmax-vpmin)/nwe
    var_del =sp.zeros(nlss*nwe)
    mdel =sp.zeros(nlss*nwe)
    var2_del =sp.zeros(nlss*nwe)
    count =sp.zeros(nlss*nwe)

    for p in data:
        for d in data[p]:

            var_pipe = 1/d.iv/d.co**2
            w = var_pipe < vpmax

            bll = ((d.ll-forest.lmin)/(forest.lmax-forest.lmin)*nlss).astype(int)
            bwe = ((1/d.iv/d.co**2-vpmin)/(vpmax-vpmin)*nwe).astype(int)

            bll = bll[w]
            bwe = bwe[w]

            de = (d.fl/d.co-1)
            de = de[w]

            bins = bwe + nwe*bll

            c = sp.bincount(bins,weights=de)
            mdel[:len(c)] += c

            c = sp.bincount(bins,weights=de**2)
            var_del[:len(c)] += c

            c = sp.bincount(bins,weights=de**4)
            var2_del[:len(c)] += c

            c = sp.bincount(bins)
            count[:len(c)] += c
    
    w = count>0
    var_del[w]/=count[w]
    mdel[w]/=count[w]
    var_del -= mdel**2
    var2_del[w]/=count[w]
    var2_del -= var_del**2
    var2_del[w]/=count[w]

    for i in range(nlss):
        def chi2(eta,vlss):
            v = var_del[i*nwe:(i+1)*nwe]-eta*var-vlss
            dv2 = var2_del[i*nwe:(i+1)*nwe]
            n = count[i*nwe:(i+1)*nwe]
            w=(dv2>0) & (n>100)
            return sp.sum(v[w]**2/dv2[w])
        mig = iminuit.Minuit(chi2,forced_parameters=("eta","vlss"),eta=1.,vlss=0.1,error_eta=0.05,error_vlss=0.05,errordef=1.,print_level=0,limit_eta=(0.5,1.5),limit_vlss=(0.0,0.3))
        mig.migrad()

        eta[i] = mig.values["eta"]
        vlss[i] = mig.values["vlss"]
        print eta[i],vlss[i],mig.fval


    return ll,eta,vlss

    
def stack(data,delta=False):
    nstack = int((forest.lmax-forest.lmin)/forest.dll)+1
    ll = forest.lmin + sp.arange(nstack)*forest.dll
    st = sp.zeros(nstack)
    wst = sp.zeros(nstack)
    for p in data:
        for d in data[p]:
            bins=((d.ll-forest.lmin)/forest.dll+0.5).astype(int)
            var_lss = forest.var_lss(d.ll)
            eta = forest.eta(d.ll)
            if delta:
                we = d.we
            else:
                iv = d.iv/eta
                we = iv*d.co**2/(iv*d.co**2*var_lss + 1)
            if delta:
                de = d.de
            else:
                de = d.fl/d.co
            c = sp.bincount(bins,weights=de*we)
            st[:len(c)]+=c
            c = sp.bincount(bins,weights=we)
            wst[:len(c)]+=c

    w=wst>0
    st[w]/=wst[w]
    return ll,st

