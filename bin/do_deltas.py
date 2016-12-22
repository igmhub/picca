
import sys
import fitsio
import healpy
import scipy as sp

from scipy.interpolate import interp1d
from multiprocessing import Pool
from pylya.data import forest
from pylya.data import delta
from pylya import prep_del
from pylya import io

from math import isnan

import argparse

def cont_fit(data):
    for d in data:
        d.cont_fit()
    return data


if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--drq', type = str, default = None, required=True,
                        help = 'DRQ file')

    parser.add_argument('--in-dir',type = str,default=None,required=True,
            help='data directory')

    parser.add_argument('--out-dir',type = str,default=None,required=True,
            help='output directory')

    parser.add_argument('--dla-vac',type = str,default=None,required=False,
            help='dla catalog file')

    parser.add_argument('--nspec',type = int,default=None,required=False,
            help='number of spectra to fit')

    parser.add_argument('--zqso-min',type = float,default=2.1,required=False,
            help='lower limit on quasar redshift')

    parser.add_argument('--zqso-max',type = float,default=3.5,required=False,
            help='upper limit on quasar redshift')

    parser.add_argument('--log',type = str,default='input.log',required=False,
            help='log input data')

    parser.add_argument('--npix-min',type = int,default=50,required=False,
            help='log input data')

    parser.add_argument('--lambda-min',type = float,default=3600,required=False,
            help='lower limit on observed wavelength (angstrom)')

    parser.add_argument('--lambda-max',type = float,default=5500,required=False,
            help='upper limit on observed wavelength (angstrom)')

    parser.add_argument('--lambda-rest-min',type = float,default=1040,required=False,
            help='lower limit on rest frame wavelength (angstrom')

    parser.add_argument('--lambda-rest-max',type = float,default=1200,required=False,
            help='upper limit on rest frame wavelength (anstrom)')

    parser.add_argument('--rebin',type = int,default=3,required=False,
            help='rebin wavelength grid by combining this number of adjacent pixels (ivar weight)')

    parser.add_argument('--dla-mask',type = float,default=0.8,required=False,
            help='lower limit on the DLA transmission. Transmissions below this number are masked')

    parser.add_argument('--nit',type = int,default=5,required=False,
            help='number of iterations to determine the mean continuum shape, LSS variances, etc.')

    parser.add_argument('--iter-out-prefix',type = str,default='iter',required=False,
            help='prefix of the iteration file')

    parser.add_argument('--mode',type = str,default='pix',required=False,
            help='open mode: pix or spec')

    parser.add_argument('--keep-bal',action='store_true',required=False,
            help='do not reject BALs')


    args = parser.parse_args()

    ## init forest class

    forest.lmin = sp.log10(args.lambda_min)
    forest.lmax = sp.log10(args.lambda_max)
    forest.lmin_rest = sp.log10(args.lambda_rest_min)
    forest.lmax_rest = sp.log10(args.lambda_rest_max)
    forest.rebin = args.rebin
    forest.dll = args.rebin*1e-4
    ## minumum dla transmission
    forest.dla_mask = args.dla_mask

    forest.var_lss = interp1d(forest.lmin+sp.arange(2)*(forest.lmax-forest.lmin),0.2 + sp.zeros(2),fill_value="extrapolate")
    forest.eta = interp1d(forest.lmin+sp.arange(2)*(forest.lmax-forest.lmin), sp.ones(2),fill_value="extrapolate")
    forest.mean_cont = interp1d(forest.lmin_rest+sp.arange(2)*(forest.lmax_rest-forest.lmin_rest),1+sp.zeros(2))

    nit = args.nit

    log = open(args.log,'w')
    data,ndata = io.read_data(args.in_dir,args.drq,args.mode,\
            zmin=args.zqso_min,zmax=args.zqso_max,nspec=args.nspec,log=log,keep_bal=args.keep_bal)
    
    if not args.dla_vac is None:
        print("adding dlas")
        dlas = io.read_dlas(args.dla_vac)
        for p in data:
            for d in data[p]:
                if dlas.has_key(d.thid):
                    for dla in dlas[d.thid]:
                        d.add_dla(dla[0],dla[1])

    ## cuts
    for p in data:
        l = []
        for d in data[p]:
            if not hasattr(d,'ll') or len(d.ll) < args.npix_min:
                log.write("{} forest too short\n".format(d.thid))
                continue

            if isnan((d.fl*d.iv).sum()):
                log.write("{} nan found\n".format(d.thid))
                continue
            l.append(d)
            log.write("{} accepted\n".format(d.thid))
        data[p][:] = l
        if len(data[p])==0:
            del data[p]

    for p in data:
        for d in data[p]:
            assert hasattr(d,'ll')

    log.close()
    for it in range(nit):
        pool = Pool()
        print "iteration: ", it
        nfit = 0
	data_fit_cont = pool.map(cont_fit, data.values())
	for i, p in enumerate(data):
            data[p] = data_fit_cont[i]

	print "done"
        pool.close()

        if it < nit-1:
       	    ll_rest, mc = prep_del.mc(data)
	    forest.mean_cont = interp1d(ll_rest, forest.mean_cont(ll_rest) * mc, fill_value = "extrapolate")
            ll,eta,vlss = prep_del.var_lss(data)
	    forest.eta = interp1d(ll, eta, fill_value = "extrapolate")
	    forest.var_lss = interp1d(ll,vlss, fill_value = "extrapolate")

    res = fitsio.FITS(args.iter_out_prefix+".fits.gz",'rw',clobber=True)
    ll_st,st = prep_del.stack(data)
    res.write([ll_st,st],names=['loglam','stack'])
    res.write([ll,eta,vlss],names=['loglam','eta','var_lss'])
    res.write([ll_rest,forest.mean_cont(ll_rest)],names=['loglam_rest','mean_cont'])
    st = interp1d(ll_st,st,kind="nearest",fill_value="extrapolate")
    res.close()
    deltas = {}
    for p in data:
        deltas[p] = [delta.from_forest(d,st,forest.var_lss,forest.eta) for d in data[p]]

    for p in deltas:
        out = fitsio.FITS(args.out_dir+"/delta-{}".format(p)+".fits.gz",'rw',clobber=True)
        for d in deltas[p]:
            hd={}
            hd["RA"]=d.ra
            hd["DEC"]=d.dec
            hd["Z"]=d.zqso
            hd["PMF"]="{}-{}-{}".format(d.plate,d.mjd,d.fid)
            hd["THING_ID"]=d.thid
            hd["PLATE"]=d.plate
            hd["MJD"]=d.mjd
            hd["FIBERID"]=d.fid

            cols=[d.ll,d.de,d.we,d.co]
            names=['LOGLAM','DELTA','WEIGHT','CONT']
            out.write(cols,names=names,header=hd)
        out.close()


    
