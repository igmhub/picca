import scipy as sp
import iminuit
from picca.data import forest,variance


## mean continuum
def mc(data):
    nmc = int((forest.lmax_rest-forest.lmin_rest)/forest.dll)+1
    mcont = sp.zeros(nmc)
    wcont = sp.zeros(nmc)
    ll = forest.lmin_rest + (sp.arange(nmc)+.5)*(forest.lmax_rest-forest.lmin_rest)/nmc
    for p in sorted(list(data.keys())):
        for d in data[p]:
            bins=((d.ll-forest.lmin_rest-sp.log10(1+d.zqso))/(forest.lmax_rest-forest.lmin_rest)*nmc).astype(int)
            var_lss = forest.var_lss(d.ll)
            eta = forest.eta(d.ll)
            fudge = forest.fudge(d.ll)
            var = 1./d.iv/d.co**2
            we = 1/variance(var,eta,var_lss,fudge)
            c = sp.bincount(bins,weights=d.fl/d.co*we)
            mcont[:len(c)]+=c
            c = sp.bincount(bins,weights=we)
            wcont[:len(c)]+=c

    w=wcont>0
    mcont[w]/=wcont[w]
    mcont/=mcont.mean()
    return ll,mcont,wcont

def var_lss(data,eta_lim=(0.5,1.5),vlss_lim=(0.,0.3)):
    nlss = 20
    eta = sp.zeros(nlss)
    vlss = sp.zeros(nlss)
    fudge = sp.zeros(nlss)
    err_eta = sp.zeros(nlss)
    err_vlss = sp.zeros(nlss)
    err_fudge = sp.zeros(nlss)
    nb_pixels = sp.zeros(nlss)
    ll = forest.lmin + (sp.arange(nlss)+.5)*(forest.lmax-forest.lmin)/nlss

    nwe = 100
    vpmin = sp.log10(1e-5)
    vpmax = sp.log10(2.)
    var = 10**(vpmin + (sp.arange(nwe)+.5)*(vpmax-vpmin)/nwe)

    var_del =sp.zeros(nlss*nwe)
    mdel =sp.zeros(nlss*nwe)
    var2_del =sp.zeros(nlss*nwe)
    count =sp.zeros(nlss*nwe)
    nqso = sp.zeros(nlss*nwe)

    for p in sorted(list(data.keys())):
        for d in data[p]:

            var_pipe = 1/d.iv/d.co**2
            w = (sp.log10(var_pipe) > vpmin) & (sp.log10(var_pipe) < vpmax)

            bll = ((d.ll-forest.lmin)/(forest.lmax-forest.lmin)*nlss).astype(int)
            bwe = sp.floor((sp.log10(var_pipe)-vpmin)/(vpmax-vpmin)*nwe).astype(int)

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
            nqso[sp.unique(bins)]+=1


    w = count>0
    var_del[w]/=count[w]
    mdel[w]/=count[w]
    var_del -= mdel**2
    var2_del[w]/=count[w]
    var2_del -= var_del**2
    var2_del[w]/=count[w]

    bin_chi2 = sp.zeros(nlss)
    fudge_ref = 1e-7
    for i in range(nlss):
        def chi2(eta,vlss,fudge):
            v = var_del[i*nwe:(i+1)*nwe]-variance(var,eta,vlss,fudge*fudge_ref)
            dv2 = var2_del[i*nwe:(i+1)*nwe]
            w=nqso[i*nwe:(i+1)*nwe]>100
            return sp.sum(v[w]**2/dv2[w])
        mig = iminuit.Minuit(chi2,forced_parameters=("eta","vlss","fudge"),eta=1.,vlss=0.1,fudge=1.,error_eta=0.05,error_vlss=0.05,error_fudge=0.05,errordef=1.,print_level=0,limit_eta=eta_lim,limit_vlss=vlss_lim, limit_fudge=(0,None))
        mig.migrad()

        if mig.migrad_ok():
            mig.hesse()
            eta[i] = mig.values["eta"]
            vlss[i] = mig.values["vlss"]
            fudge[i] = mig.values["fudge"]*fudge_ref
            err_eta[i] = mig.errors["eta"]
            err_vlss[i] = mig.errors["vlss"]
            err_fudge[i] = mig.errors["fudge"]*fudge_ref
        else:
            eta[i] = 1.
            vlss[i] = 0.1
            fudge[i] = 1.*fudge_ref
            err_eta[i] = 0.
            err_vlss[i] = 0.
            err_fudge[i] = 0.
        nb_pixels[i] = count[i*nwe:(i+1)*nwe].sum()
        bin_chi2[i] = mig.fval
        print(eta[i],vlss[i],fudge[i],mig.fval, nb_pixels[i],err_eta[i],err_vlss[i],err_fudge[i])


    return ll,eta,vlss,fudge,nb_pixels,var,var_del.reshape(nlss,-1),var2_del.reshape(nlss,-1),count.reshape(nlss,-1),nqso.reshape(nlss,-1),bin_chi2,err_eta,err_vlss,err_fudge


def stack(data,delta=False):
    nstack = int((forest.lmax-forest.lmin)/forest.dll)+1
    ll = forest.lmin + sp.arange(nstack)*forest.dll
    st = sp.zeros(nstack)
    wst = sp.zeros(nstack)
    for p in sorted(list(data.keys())):
        for d in data[p]:
            if delta:
                de = d.de
                we = d.we
            else:
                de = d.fl/d.co
                var_lss = forest.var_lss(d.ll)
                eta = forest.eta(d.ll)
                fudge = forest.fudge(d.ll)
                var = 1./d.iv/d.co**2
                we = 1./variance(var,eta,var_lss,fudge)

            bins=((d.ll-forest.lmin)/forest.dll+0.5).astype(int)
            c = sp.bincount(bins,weights=de*we)
            st[:len(c)]+=c
            c = sp.bincount(bins,weights=we)
            wst[:len(c)]+=c

    w=wst>0
    st[w]/=wst[w]
    return ll,st, wst

