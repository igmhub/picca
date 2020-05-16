"""This module defines a set of functions to compute the deltas.

This module provides three functions:
    - compute_mean_cont
    - var_lss
    - stack
See the respective documentation for details
"""
import numpy as np
import scipy as sp
import iminuit
from picca.data import Forest, get_variance
from picca.utils import userprint


def compute_mean_cont(data):
    """Computes the mean quasar continuum over the whole sample.

    Args:
        data: dict
            A dictionary with the read forests in each healpix

    Returns:
        log_lambda: array
            Logarithm of the wavelengths (in Angs).
        mean_cont: array
            Mean quasar continuum over the whole sample
        mean_cont_weight: array
            Total weight on the mean quasar continuum
    """
    num_bins = (int((Forest.log_lambda_max_rest_frame -
                    Forest.log_lambda_min_rest_frame)/Forest.delta_log_lambda)
                + 1)
    mean_cont = np.zeros(num_bins)
    mean_cont_weight = np.zeros(num_bins)
    log_lambda = (Forest.log_lambda_min_rest_frame + (np.arange(num_bins) + .5)
                  *(Forest.log_lambda_max_rest_frame -
                    Forest.log_lambda_min_rest_frame)/num_bins)
    for healpix in sorted(list(data.keys())):
        for forest in data[healpix]:
            bins = ((forest.log_lambda - Forest.log_lambda_min_rest_frame -
                     np.log10(1 + forest.z_qso))/
                    (Forest.log_lambda_max_rest_frame -
                     Forest.log_lambda_min_rest_frame)*num_bins).astype(int)
            var_lss = Forest.get_var_lss(forest.log_lambda)
            eta = Forest.get_eta(forest.log_lambda)
            fudge = Forest.get_fudge(forest.log_lambda)
            var_pipe = 1./forest.ivar/forest.cont**2
            weights = 1/get_variance(var_pipe, eta, var_lss, fudge)
            cont = np.bincount(bins, weights=forest.flux/forest.cont*weights)
            mean_cont[:len(cont)]+=cont
            cont = np.bincount(bins, weights=weights)
            mean_cont_weight[:len(cont)] += cont

    w = mean_cont_weight > 0
    mean_cont[w] /= mean_cont_weight[w]
    mean_cont /= mean_cont.mean()
    return log_lambda, mean_cont, mean_cont_weight

def var_lss(data,eta_lim=(0.5,1.5),vlss_lim=(0.,0.3)):
    nlss = 20
    eta = np.zeros(nlss)
    vlss = np.zeros(nlss)
    fudge = np.zeros(nlss)
    err_eta = np.zeros(nlss)
    err_vlss = np.zeros(nlss)
    err_fudge = np.zeros(nlss)
    nb_pixels = np.zeros(nlss)
    log_lambda = Forest.log_lambda_min + (np.arange(nlss)+.5)*(Forest.log_lambda_max-Forest.log_lambda_min)/nlss

    nwe = 100
    vpmin = sp.log10(1e-5)
    vpmax = sp.log10(2.)
    var = 10**(vpmin + (np.arange(nwe)+.5)*(vpmax-vpmin)/nwe)

    var_del =np.zeros(nlss*nwe)
    mdel =np.zeros(nlss*nwe)
    var2_del =np.zeros(nlss*nwe)
    count =np.zeros(nlss*nwe)
    nqso = np.zeros(nlss*nwe)

    for p in sorted(list(data.keys())):
        for d in data[p]:

            var_pipe = 1/d.ivar/d.cont**2
            w = (sp.log10(var_pipe) > vpmin) & (sp.log10(var_pipe) < vpmax)

            bll = ((d.log_lambda-Forest.log_lambda_min)/(Forest.log_lambda_max-Forest.log_lambda_min)*nlss).astype(int)
            bwe = sp.floor((sp.log10(var_pipe)-vpmin)/(vpmax-vpmin)*nwe).astype(int)

            bll = bll[w]
            bwe = bwe[w]

            delta = (d.flux/d.cont-1)
            delta = delta[w]

            bins = bwe + nwe*bll

            c = sp.bincount(bins,weights=delta)
            mdel[:len(c)] += c

            c = sp.bincount(bins,weights=delta**2)
            var_del[:len(c)] += c

            c = sp.bincount(bins,weights=delta**4)
            var2_del[:len(c)] += c

            c = sp.bincount(bins)
            count[:len(c)] += c
            nqso[np.unique(bins)]+=1


    w = count>0
    var_del[w]/=count[w]
    mdel[w]/=count[w]
    var_del -= mdel**2
    var2_del[w]/=count[w]
    var2_del -= var_del**2
    var2_del[w]/=count[w]

    bin_chi2 = np.zeros(nlss)
    fudge_ref = 1e-7
    for i in range(nlss):
        def chi2(eta,vlss,fudge):
            v = var_del[i*nwe:(i+1)*nwe]-get_variance(var,eta,vlss,fudge*fudge_ref)
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
        userprint(eta[i],vlss[i],fudge[i],mig.fval, nb_pixels[i],err_eta[i],err_vlss[i],err_fudge[i])


    return log_lambda,eta,vlss,fudge,nb_pixels,var,var_del.reshape(nlss,-1),var2_del.reshape(nlss,-1),count.reshape(nlss,-1),nqso.reshape(nlss,-1),bin_chi2,err_eta,err_vlss,err_fudge


def stack(data, stack_from_deltas=False):
    nstack = int((Forest.log_lambda_max-Forest.log_lambda_min)/Forest.delta_log_lambda)+1
    stack_log_lambda = Forest.log_lambda_min + np.arange(nstack)*Forest.delta_log_lambda
    stack_delta = np.zeros(nstack)
    wst = np.zeros(nstack)
    for p in sorted(list(data.keys())):
        for d in data[p]:
            if stack_from_deltas:
                delta = d.delta
                weights = d.weights
            else:
                delta = d.flux/d.cont
                var_lss = Forest.get_var_lss(d.log_lambda)
                eta = Forest.get_eta(d.log_lambda)
                fudge = Forest.get_fudge(d.log_lambda)
                var = 1./d.ivar/d.cont**2
                weights = 1./get_variance(var,eta,var_lss,fudge)

            bins=((d.log_lambda-Forest.log_lambda_min)/Forest.delta_log_lambda+0.5).astype(int)
            c = sp.bincount(bins,weights=delta*weights)
            stack_delta[:len(c)]+=c
            c = sp.bincount(bins,weights=weights)
            wst[:len(c)]+=c

    w=wst>0
    stack_delta[w]/=wst[w]
    return stack_log_lambda, stack_delta, wst
