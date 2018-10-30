from __future__ import print_function
import scipy as sp

from picca import constants
from picca.utils import print


def exp_diff(file,ll) :

    nexp_per_col = file[0].read_header()['NEXP']//2
    fltotodd  = sp.zeros(ll.size)
    ivtotodd  = sp.zeros(ll.size)
    fltoteven = sp.zeros(ll.size)
    ivtoteven = sp.zeros(ll.size)

    if (nexp_per_col)<2 :
        print("DBG : not enough exposures for diff")

    for iexp in range (nexp_per_col) :
        for icol in range (2):
            llexp = file[4+iexp+icol*nexp_per_col]["loglam"][:]
            flexp = file[4+iexp+icol*nexp_per_col]["flux"][:]
            ivexp = file[4+iexp+icol*nexp_per_col]["ivar"][:]
            mask  = file[4+iexp+icol*nexp_per_col]["mask"][:]
            bins = sp.searchsorted(ll,llexp)

            # exclude masks 25 (COMBINEREJ), 23 (BRIGHTSKY)?
            if iexp%2 == 1 :
                civodd=sp.bincount(bins,weights=ivexp*(mask&2**25==0))
                cflodd=sp.bincount(bins,weights=ivexp*flexp*(mask&2**25==0))
                fltotodd[:civodd.size-1] += cflodd[:-1]
                ivtotodd[:civodd.size-1] += civodd[:-1]
            else :
                civeven=sp.bincount(bins,weights=ivexp*(mask&2**25==0))
                cfleven=sp.bincount(bins,weights=ivexp*flexp*(mask&2**25==0))
                fltoteven[:civeven.size-1] += cfleven[:-1]
                ivtoteven[:civeven.size-1] += civeven[:-1]

    w=ivtotodd>0
    fltotodd[w]/=ivtotodd[w]
    w=ivtoteven>0
    fltoteven[w]/=ivtoteven[w]

    alpha = 1
    if (nexp_per_col%2 == 1) :
        n_even = (nexp_per_col-1)//2
        alpha = sp.sqrt(4.*n_even*(n_even+1))/nexp_per_col
    diff = 0.5 * (fltoteven-fltotodd) * alpha ### CHECK THE * alpha (Nathalie)

    return diff


def spectral_resolution(wdisp,with_correction=None,fiber=None,ll=None) :

    reso = wdisp*constants.speed_light/1000.*1.0e-4*sp.log(10.)

    if (with_correction):
        wave = sp.power(10.,ll)
        corrPlateau = 1.267 - 0.000142716*wave + 1.9068e-08*wave*wave;
        corrPlateau[wave>6000.0] = 1.097

        fibnum = fiber%500
        if(fibnum<100):
            corr = 1. + (corrPlateau-1)*.25 + (corrPlateau-1)*.75*(fibnum)/100.
        elif (fibnum>400):
            corr = 1. + (corrPlateau-1)*.25 + (corrPlateau-1)*.75*(500-fibnum)/100.
        else:
            corr = corrPlateau
        reso *= corr
    return reso

def spectral_resolution_desi(reso_matrix, ll) :

    dll = (ll[-1]-ll[0])/float(len(ll)-1)
    reso= sp.clip(reso_matrix,1.0e-6,1.0e6)
    rms_in_pixel = (sp.sqrt(1.0/2.0/sp.log(reso[len(reso)//2][:]/reso[len(reso)//2-1][:]))
                    + sp.sqrt(4.0/2.0/sp.log(reso[len(reso)//2][:]/reso[len(reso)//2-2][:])))/2.0

    reso_in_km_per_s = rms_in_pixel*constants.speed_light/1000.*dll*sp.log(10.0)

    return reso_in_km_per_s


