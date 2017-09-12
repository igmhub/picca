#!/usr/bin/env python

import sys
import fitsio
import healpy
import scipy as sp

from scipy.interpolate import interp1d
from multiprocessing import Pool
from picca.data import forest
from picca.data import delta
from picca import prep_del
from picca import io

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

    parser.add_argument('--zqso-min',type = float,default=None,required=False,
            help='lower limit on quasar redshift')

    parser.add_argument('--zqso-max',type = float,default=None,required=False,
            help='upper limit on quasar redshift')

    parser.add_argument('--log',type = str,default='input.log',required=False,
            help='log input data')

    parser.add_argument('--npix-min',type = int,default=50,required=False,
            help='log input data')

    parser.add_argument('--lambda-min',type = float,default=3600.,required=False,
            help='lower limit on observed wavelength [Angstrom]')

    parser.add_argument('--lambda-max',type = float,default=5500.,required=False,
            help='upper limit on observed wavelength [Angstrom]')

    parser.add_argument('--lambda-rest-min',type = float,default=1040.,required=False,
            help='lower limit on rest frame wavelength [Angstrom]')

    parser.add_argument('--lambda-rest-max',type = float,default=1200.,required=False,
            help='upper limit on rest frame wavelength [Angstrom]')

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

    parser.add_argument('--bi-max',type=float,required=False,default=None,
            help="maximum CIV balnicity index (overrides --keep-bal")

    parser.add_argument('--nproc', type = int, default = None, required=False,
                    help = 'number of processors')

    parser.add_argument('--mask-file',type = str,default=None,required=False,
            help='Path to file to mask regions in lambda_OBS and lambda_RF. In file each line is: region_name region_min region_max (OBS or RF) [Angstrom]')

    parser.add_argument('--flux-calib',type = str,default=None,required=False,
            help='Path to file to previously produced do_delta.py file to correct for multiplicative errors in the flux calibration')

    parser.add_argument('--eta-min',type = float,default=0.5,required=False,
            help='lower limit for eta')
    parser.add_argument('--eta-max',type = float,default=1.5,required=False,
            help='upper limit for eta')

    parser.add_argument('--vlss-min',type = float,default=0.,required=False,
            help='lower limit for variance LSS')
    parser.add_argument('--vlss-max',type = float,default=0.3,required=False,
            help='upper limit for variance LSS')

    parser.add_argument('--order',type=int,default=1,required=False,
            help='order of the log(lambda) polynomial for the continuum fit, by default 1.')

    parser.add_argument('--delta-format',type = str,default=None,required=False,
            help='format for Pk 1D: Pk1D')

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

    ### Find the redshift range
    if (args.zqso_min is None):
        args.zqso_min = max(0.,args.lambda_min/args.lambda_rest_max -1.)
        print(" zqso_min = {}".format(args.zqso_min) )
    if (args.zqso_max is None):
        args.zqso_max = max(0.,args.lambda_max/args.lambda_rest_min -1.)
        print(" zqso_max = {}".format(args.zqso_max) )

    forest.var_lss = interp1d(forest.lmin+sp.arange(2)*(forest.lmax-forest.lmin),0.2 + sp.zeros(2),fill_value="extrapolate")
    forest.eta = interp1d(forest.lmin+sp.arange(2)*(forest.lmax-forest.lmin), sp.ones(2),fill_value="extrapolate")
    forest.mean_cont = interp1d(forest.lmin_rest+sp.arange(2)*(forest.lmax_rest-forest.lmin_rest),1+sp.zeros(2))

    ### Fix the order of the continuum fit, 0 or 1. 
    if args.order:
        if (args.order != 0) and (args.order != 1): 
            print("ERROR : invalid value for order, must be eqal to 0 or 1. Here order = %i"%(order))
            sys.exit(12)

    ### Correct multiplicative flux calibration
    if (args.flux_calib is not None):
        try:
            vac = fitsio.FITS(args.flux_calib)
            head = vac[1].read_header()

            ll_st = vac[1]['loglam'][:]
            st    = vac[1]['stack'][:]
            w     = (st!=0.)
            forest.correc_flux = interp1d(ll_st[w],st[w],fill_value="extrapolate")

        except:
            print(" Error while reading flux_calib file {}".format(args.flux_calib))
            sys.exit(1)

    nit = args.nit

    log = open(args.log,'w')
    data,ndata = io.read_data(args.in_dir,args.drq,args.mode,\
            zmin=args.zqso_min,zmax=args.zqso_max,nspec=args.nspec,log=log,keep_bal=args.keep_bal,bi_max = args.bi_max)
    

    ### Get the lines to veto
    usr_mask_obs    = None
    usr_mask_RF     = None
    usr_mask_RF_DLA = None
    if (args.mask_file is not None):
        try:
            usr_mask_obs    = []
            usr_mask_RF     = []
            usr_mask_RF_DLA = []
            with open(args.mask_file, 'r') as f:
                loop = True
                for l in f:
                    if (l[0]=='#'): continue
                    l = l.split()
                    if (l[3]=='OBS'):
                        usr_mask_obs    += [ [float(l[1]),float(l[2])] ]
                    elif (l[3]=='RF'):
                        usr_mask_RF     += [ [float(l[1]),float(l[2])] ]
                    elif (l[3]=='RF_DLA'):
                        usr_mask_RF_DLA += [ [float(l[1]),float(l[2])] ]
                    else:
                        raise
            f.closed
            usr_mask_obs    = sp.log10(sp.asarray(usr_mask_obs))
            usr_mask_RF     = sp.log10(sp.asarray(usr_mask_RF))
            usr_mask_RF_DLA = sp.log10(sp.asarray(usr_mask_RF_DLA))
            if usr_mask_RF_DLA.size==0:
                usr_mask_RF_DLA = None

        except:
            print(" Error while reading mask_file file {}".format(args.mask_file))
            sys.exit(1)

    ### Veto lines
    if not usr_mask_obs is None:
        if ( usr_mask_obs.size+usr_mask_RF.size!=0):
            for p in data:
                for d in data[p]:
                    d.mask(mask_obs=usr_mask_obs , mask_RF=usr_mask_RF)

    ### Correct for DLAs
    if not args.dla_vac is None:
        print("adding dlas")
        dlas = io.read_dlas(args.dla_vac)
        nb_dla_in_forest = 0
        for p in data:
            for d in data[p]:
                if dlas.has_key(d.thid):
                    for dla in dlas[d.thid]:
                        d.add_dla(dla[0],dla[1],usr_mask_RF_DLA)
                        nb_dla_in_forest += 1
        log.write("Found {} DLAs in forests\n".format(nb_dla_in_forest))

    ## cuts
    for p in data.keys():
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
        pool = Pool(processes=args.nproc)
        print "iteration: ", it
        nfit = 0
        data_fit_cont = pool.map(cont_fit, data.values())
        for i, p in enumerate(data):
            data[p] = data_fit_cont[i]

        print "done"
        pool.close()

        if it < nit-1:
            ll_rest, mc, wmc = prep_del.mc(data)
            forest.mean_cont = interp1d(ll_rest[wmc>0.], forest.mean_cont(ll_rest[wmc>0.]) * mc[wmc>0.], fill_value = "extrapolate")
            ll,eta,vlss,nb_pixels = prep_del.var_lss(data,(args.eta_min,args.eta_max),(args.vlss_min,args.vlss_max))
            forest.eta = interp1d(ll[nb_pixels>0.], eta[nb_pixels>0.], fill_value = "extrapolate")
            forest.var_lss = interp1d(ll[nb_pixels>0.],vlss[nb_pixels>0.], fill_value = "extrapolate")

    res = fitsio.FITS(args.iter_out_prefix+".fits.gz",'rw',clobber=True)
    ll_st,st,wst = prep_del.stack(data)
    res.write([ll_st,st,wst],names=['loglam','stack','weight'])
    res.write([ll,eta,vlss,nb_pixels],names=['loglam','eta','var_lss','nb_pixels'])
    res.write([ll_rest,forest.mean_cont(ll_rest),wmc],names=['loglam_rest','mean_cont','weight'])
    st = interp1d(ll_st[wst>0.],st[wst>0.],kind="nearest",fill_value="extrapolate")
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
            hd["ORDER"]=d.order
            hd["DLL"]=d.dll

            if (args.delta_format=='Pk1D') :
                hd["MEANZ"]=d.mean_z
                hd["MEANRESO"]=d.mean_reso
                hd["MEANSNR"]=d.mean_SNR

            if (args.delta_format=='Pk1D') :
                cols=[d.ll,d.de,d.iv,d.diff]
                names=['LOGLAM','DELTA','IVAR','DIFF']
            else :
                cols=[d.ll,d.de,d.we,d.co]
                names=['LOGLAM','DELTA','WEIGHT','CONT']

                
            out.write(cols,names=names,header=hd)
        out.close()


    
