from __future__ import print_function
import scipy as sp
import sys

def smooth_cov(da,we,rp,rt,drt=4,drp=4):
    
    npix = da.shape[0]
    nda = da.shape[1]
    co = sp.zeros([nda,nda])

    mda = (da*we).sum(axis=0)/we.sum(axis=0)

    wda = we*(da-mda)

    print("Computing cov...")
    for ipix in xrange(npix):
        sys.stderr.write("\r {} {}".format(ipix,npix))
        co += sp.outer(wda[ipix,:],wda[ipix,:])

    swe = we.sum(axis=0)

    co/=swe*swe[:,None]
    var = sp.diagonal(co)

    cor = co/sp.sqrt(var*var[:,None])

    cor_smooth = sp.zeros([nda,nda])

    dcor={}
    dncor={}

    for i in xrange(nda):
        sys.stderr.write("\rsmoothing {}".format(i))
        for j in range(i+1,nda):
            idrp = round(abs(rp[j]-rp[i])/drp)
            idrt = round(abs(rt[i]-rt[j])/drt)
            if not (idrp,idrt) in dcor:
                dcor[(idrp,idrt)]=0.
                dncor[(idrp,idrt)]=0

            dcor[(idrp,idrt)] +=cor[i,j]
            dncor[(idrp,idrt)] +=1

    for i in xrange(nda):
        cor_smooth[i,i]=1.
        for j in range(i+1,nda):
            idrp = round(abs(rp[j]-rp[i])/drp)
            idrt = round(abs(rt[i]-rt[j])/drt)
            cor_smooth[i,j]=dcor[(idrp,idrt)]/dncor[(idrp,idrt)]
            cor_smooth[j,i]=cor_smooth[i,j]


    sys.stderr.write("\n")
    co_smooth = cor_smooth * sp.sqrt(var*var[:,None])
    return co_smooth
