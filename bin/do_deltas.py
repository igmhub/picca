#!/usr/bin/env python

import sys
import fitsio
import scipy as sp
from scipy.interpolate import interp1d
from multiprocessing import Pool
from math import isnan
import argparse

from picca.data import forest, delta
from picca import prep_del, io

def cont_fit(data):
    for d in data:
        d.cont_fit()
    return data


if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Compute the delta field from a list of spectra')

    parser.add_argument('--out-dir',type=str,default=None,required=True,
        help='Output directory')

    parser.add_argument('--drq', type=str, default=None, required=True,
        help='Catalog of objects in DRQ format')

    parser.add_argument('--in-dir', type=str, default=None, required=True,
        help='Directory to spectra files')

    parser.add_argument('--log',type=str,default='input.log',required=False,
        help='Log input data')

    parser.add_argument('--iter-out-prefix',type=str,default='iter',required=False,
        help='Prefix of the iteration file')

    parser.add_argument('--mode',type=str,default='pix',required=False,
        help='Open mode of the spectra files: pix, spec, spcframe, spplate, desi')

    parser.add_argument('--best-obs',action='store_true', required=False,
        help='If mode == spcframe, then use only the best observation')

    parser.add_argument('--single-exp',action='store_true', required=False,
        help='If mode == spcframe, then use only one of the available exposures. If best-obs then choose it among those contributing to the best obs')

    parser.add_argument('--zqso-min',type=float,default=None,required=False,
        help='Lower limit on quasar redshift from drq')

    parser.add_argument('--zqso-max',type=float,default=None,required=False,
        help='Upper limit on quasar redshift from drq')

    parser.add_argument('--keep-bal',action='store_true',required=False,
        help='Do not reject BALs in drq')

    parser.add_argument('--bi-max',type=float,required=False,default=None,
        help='Maximum CIV balnicity index in drq (overrides --keep-bal)')

    parser.add_argument('--lambda-min',type=float,default=3600.,required=False,
        help='Lower limit on observed wavelength [Angstrom]')

    parser.add_argument('--lambda-max',type=float,default=5500.,required=False,
        help='Upper limit on observed wavelength [Angstrom]')

    parser.add_argument('--lambda-rest-min',type=float,default=1040.,required=False,
        help='Lower limit on rest frame wavelength [Angstrom]')

    parser.add_argument('--lambda-rest-max',type=float,default=1200.,required=False,
        help='Upper limit on rest frame wavelength [Angstrom]')

    parser.add_argument('--rebin',type=int,default=3,required=False,
        help='Rebin wavelength grid by combining this number of adjacent pixels (ivar weight)')

    parser.add_argument('--npix-min',type=int,default=50,required=False,
        help='Minimum of rebined pixels')

    parser.add_argument('--dla-vac',type=str,default=None,required=False,
        help='DLA catalog file')

    parser.add_argument('--dla-mask',type=float,default=0.8,required=False,
        help='Lower limit on the DLA transmission. Transmissions below this number are masked')

    parser.add_argument('--mask-file',type=str,default=None,required=False,
        help='Path to file to mask regions in lambda_OBS and lambda_RF. In file each line is: region_name region_min region_max (OBS or RF) [Angstrom]')

    parser.add_argument('--flux-calib',type=str,default=None,required=False,
        help='Path to previously produced do_delta.py file to correct for multiplicative errors in the pipeline flux calibration')

    parser.add_argument('--ivar-calib',type=str,default=None,required=False,
        help='Path to previously produced do_delta.py file to correct for multiplicative errors in the pipeline inverse variance calibration')

    parser.add_argument('--eta-min',type=float,default=0.5,required=False,
        help='Lower limit for eta')

    parser.add_argument('--eta-max',type=float,default=1.5,required=False,
        help='Upper limit for eta')

    parser.add_argument('--vlss-min',type=float,default=0.,required=False,
        help='Lower limit for variance LSS')

    parser.add_argument('--vlss-max',type=float,default=0.3,required=False,
        help='Upper limit for variance LSS')

    parser.add_argument('--delta-format',type=str,default=None,required=False,
        help='Format for Pk 1D: Pk1D')

    parser.add_argument('--use-ivar-as-weight', action='store_true', default=False,
        help='Use ivar as weights (implemented as eta = 1, sigma_lss = fudge = 0)')

    parser.add_argument('--use-constant-weight', action='store_true', default=False,
        help='Set all the delta weights to one (implemented as eta = 0, sigma_lss = 1, fudge = 0)')

    parser.add_argument('--order',type=int,default=1,required=False,
        help='Order of the log(lambda) polynomial for the continuum fit, by default 1.')

    parser.add_argument('--nit',type=int,default=5,required=False,
        help='Number of iterations to determine the mean continuum shape, LSS variances, etc.')

    parser.add_argument('--nproc', type=int, default=None, required=False,
        help='Number of processors')

    parser.add_argument('--nspec', type=int, default=None, required=False,
        help='Maximum number of spectra to read')


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

    forest.var_lss = interp1d(forest.lmin+sp.arange(2)*(forest.lmax-forest.lmin),0.2 + sp.zeros(2),fill_value="extrapolate",kind="nearest")
    forest.eta = interp1d(forest.lmin+sp.arange(2)*(forest.lmax-forest.lmin), sp.ones(2),fill_value="extrapolate",kind="nearest")
    forest.fudge = interp1d(forest.lmin+sp.arange(2)*(forest.lmax-forest.lmin), sp.zeros(2),fill_value="extrapolate",kind="nearest")
    forest.mean_cont = interp1d(forest.lmin_rest+sp.arange(2)*(forest.lmax_rest-forest.lmin_rest),1+sp.zeros(2))

    ### Fix the order of the continuum fit, 0 or 1.
    if args.order:
        if (args.order != 0) and (args.order != 1):
            print("ERROR : invalid value for order, must be eqal to 0 or 1. Here order = %i"%(order))
            sys.exit(12)

    ### Correct multiplicative pipeline flux calibration
    if (args.flux_calib is not None):
        try:
            vac = fitsio.FITS(args.flux_calib)
            ll_st = vac[1]['loglam'][:]
            st    = vac[1]['stack'][:]
            w     = (st!=0.)
            forest.correc_flux = interp1d(ll_st[w],st[w],fill_value="extrapolate",kind="nearest")

        except:
            print(" Error while reading flux_calib file {}".format(args.flux_calib))
            sys.exit(1)

    ### Correct multiplicative pipeline inverse variance calibration
    if (args.ivar_calib is not None):
        try:
            vac = fitsio.FITS(args.ivar_calib)
            ll  = vac[2]['LOGLAM'][:]
            eta = vac[2]['ETA'][:]
            forest.correc_ivar = interp1d(ll,eta,fill_value="extrapolate",kind="nearest")

        except:
            print(" Error while reading ivar_calib file {}".format(args.ivar_calib))
            sys.exit(1)

    nit = args.nit

    log = open(args.log,'w')

    data,ndata,healpy_nside,healpy_pix_ordering = io.read_data(args.in_dir, args.drq, args.mode,\
        zmin=args.zqso_min, zmax=args.zqso_max, nspec=args.nspec, log=log,\
        keep_bal=args.keep_bal, bi_max=args.bi_max, order=args.order,\
        best_obs=args.best_obs, single_exp=args.single_exp, pk1d=args.delta_format )

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
        sp.random.seed(0)
        dlas = io.read_dlas(args.dla_vac)
        nb_dla_in_forest = 0
        for p in data:
            for d in data[p]:
                if d.thid in dlas:
                    for dla in dlas[d.thid]:
                        d.add_dla(dla[0],dla[1],usr_mask_RF_DLA)
                        nb_dla_in_forest += 1
        log.write("Found {} DLAs in forests\n".format(nb_dla_in_forest))

    ## cuts
    for p in list(data.keys()):
        l = []
        for d in data[p]:
            if not hasattr(d,'ll') or len(d.ll) < args.npix_min:
                log.write("{} forest too short\n".format(d.thid))
                continue

            if isnan((d.fl*d.iv).sum()):
                log.write("{} nan found\n".format(d.thid))
                continue

            if(args.use_constant_weight and (d.fl.mean()<=0.0 or d.mean_SNR<=1.0 )):
                log.write("{} negative mean of too low SNR found\n".format(d.thid))
                continue

            l.append(d)
            log.write("{} accepted\n".format(d.thid))
        data[p][:] = l
        if len(data[p])==0:
            del data[p]

    for p in data:
        for d in data[p]:
            assert hasattr(d,'ll')

    for it in range(nit):
        pool = Pool(processes=args.nproc)
        print("iteration: ", it)
        nfit = 0
        sort = sp.array(list(data.keys())).argsort()
        data_fit_cont = pool.map(cont_fit, sp.array(list(data.values()))[sort] )
        for i, p in enumerate(sorted(list(data.keys()))):
            data[p] = data_fit_cont[i]

        print("done")

        pool.close()

        if it < nit-1:
            ll_rest, mc, wmc = prep_del.mc(data)
            forest.mean_cont = interp1d(ll_rest[wmc>0.], forest.mean_cont(ll_rest[wmc>0.]) * mc[wmc>0.], fill_value = "extrapolate")
            if not (args.use_ivar_as_weight or args.use_constant_weight):
                ll, eta, vlss, fudge, nb_pixels, var, var_del, var2_del,\
                    count, nqsos, chi2, err_eta, err_vlss, err_fudge = \
                        prep_del.var_lss(data,(args.eta_min,args.eta_max),(args.vlss_min,args.vlss_max))
                forest.eta = interp1d(ll[nb_pixels>0], eta[nb_pixels>0],
                    fill_value = "extrapolate",kind="nearest")
                forest.var_lss = interp1d(ll[nb_pixels>0], vlss[nb_pixels>0.],
                    fill_value = "extrapolate",kind="nearest")
                forest.fudge = interp1d(ll[nb_pixels>0],fudge[nb_pixels>0],
                    fill_value = "extrapolate",kind="nearest")
            else:

                nlss=10 # this value is arbitrary
                ll = forest.lmin + (sp.arange(nlss)+.5)*(forest.lmax-forest.lmin)/nlss

                if args.use_ivar_as_weight:
                    print('INFO: using ivar as weights, skipping eta, var_lss, fudge fits')
                    eta = sp.ones(nlss)
                    vlss = sp.zeros(nlss)
                    fudge = sp.zeros(nlss)
                else :
                    print('INFO: using constant weights, skipping eta, var_lss, fudge fits')
                    eta = sp.zeros(nlss)
                    vlss = sp.ones(nlss)
                    fudge=sp.zeros(nlss)

                err_eta = sp.zeros(nlss)
                err_vlss = sp.zeros(nlss)
                err_fudge = sp.zeros(nlss)
                chi2 = sp.zeros(nlss)

                nb_pixels = sp.zeros((nlss, nlss))
                var = sp.zeros(nlss)
                var_del = sp.zeros((nlss, nlss))
                var2_del = sp.zeros((nlss, nlss))
                count = sp.zeros((nlss, nlss))
                nqsos=sp.zeros((nlss, nlss))

                forest.eta = interp1d(ll, eta, fill_value='extrapolate', kind='nearest')
                forest.var_lss = interp1d(ll, vlss, fill_value='extrapolate', kind='nearest')
                forest.fudge = interp1d(ll, fudge, fill_value='extrapolate', kind='nearest')


    ll_st,st,wst = prep_del.stack(data)

    ### Save iter_out_prefix
    res = fitsio.FITS(args.iter_out_prefix+".fits.gz",'rw',clobber=True)
    hd = {}
    hd["NSIDE"] = healpy_nside
    hd["PIXORDER"] = healpy_pix_ordering
    hd["FITORDER"] = args.order
    res.write([ll_st,st,wst],names=['loglam','stack','weight'],header=hd)
    res.write([ll,eta,vlss,fudge,nb_pixels],names=['loglam','eta','var_lss','fudge','nb_pixels'])
    res.write([ll_rest,forest.mean_cont(ll_rest),wmc],names=['loglam_rest','mean_cont','weight'])
    var = sp.broadcast_to(var.reshape(1,-1),var_del.shape)
    res.write([var,var_del,var2_del,count,nqsos,chi2],names=['var_pipe','var_del','var2_del','count','nqsos','chi2'])
    res.close()

    ### Save delta
    st = interp1d(ll_st[wst>0.],st[wst>0.],kind="nearest",fill_value="extrapolate")
    deltas = {}
    data_bad_cont = []
    for p in sorted(list(data.keys())):
        deltas[p] = [delta.from_forest(d,st,forest.var_lss,forest.eta,forest.fudge) for d in data[p] if d.bad_cont is None]
        data_bad_cont = data_bad_cont + [d for d in data[p] if d.bad_cont is not None]

    for d in data_bad_cont:
        log.write("rejected {} due to {}\n".format(d.thid,d.bad_cont))

    log.close()

#    for p in deltas:
    for p in sorted(list(deltas.keys())):

        if len(deltas[p])==0: continue
        if (args.delta_format=='Pk1D_ascii') :
            out_ascii = open(args.out_dir+"/delta-{}".format(p)+".txt",'w')
            for d in deltas[p]:
                nbpixel = len(d.de)
                dll = d.dll
                if (args.mode=='desi') : dll = (d.ll[-1]-d.ll[0])/float(len(d.ll)-1)
                line = '{} {} {} '.format(d.plate,d.mjd,d.fid)
                line += '{} {} {} '.format(d.ra,d.dec,d.zqso)
                line += '{} {} {} {} {} '.format(d.mean_z,d.mean_SNR,d.mean_reso,dll,nbpixel)
                for i in range(nbpixel): line += '{} '.format(d.de[i])
                for i in range(nbpixel): line += '{} '.format(d.ll[i])
                for i in range(nbpixel): line += '{} '.format(d.iv[i])
                for i in range(nbpixel): line += '{} '.format(d.diff[i])
                line +=' \n'
                out_ascii.write(line)

            out_ascii.close()

        else :
            out = fitsio.FITS(args.out_dir+"/delta-{}".format(p)+".fits.gz",'rw',clobber=True)
            for d in deltas[p]:
                hd = [ {'name':'RA','value':d.ra,'comment':'Right Ascension [rad]'},
                       {'name':'DEC','value':d.dec,'comment':'Declination [rad]'},
                       {'name':'Z','value':d.zqso,'comment':'Redshift'},
                       {'name':'PMF','value':'{}-{}-{}'.format(d.plate,d.mjd,d.fid)},
                       {'name':'THING_ID','value':d.thid,'comment':'Object identification'},
                       {'name':'PLATE','value':d.plate},
                       {'name':'MJD','value':d.mjd,'comment':'Modified Julian date'},
                       {'name':'FIBERID','value':d.fid},
                       {'name':'ORDER','value':d.order,'comment':'Order of the continuum fit'},
                ]

                if (args.delta_format=='Pk1D'):
                    hd += [{'name':'MEANZ','value':d.mean_z,'comment':'Mean redshift'},
                           {'name':'MEANRESO','value':d.mean_reso,'comment':'Mean resolution'},
                           {'name':'MEANSNR','value':d.mean_SNR,'comment':'Mean SNR'},
                    ]
                    dll = d.dll
                    if (args.mode=='desi'):
                        dll = (d.ll[-1]-d.ll[0])/float(len(d.ll)-1)
                    hd += [{'name':'DLL','value':dll,'comment':'Loglam bin size [log Angstrom]'}]
                    diff = d.diff
                    if diff is None:
                        diff = d.ll*0

                    cols=[d.ll,d.de,d.iv,diff]
                    names=['LOGLAM','DELTA','IVAR','DIFF']
                    units=['log Angstrom','','','']
                    comments = ['Log lambda','Delta field','Inverse variance','Difference']
                else :
                    cols=[d.ll,d.de,d.we,d.co]
                    names=['LOGLAM','DELTA','WEIGHT','CONT']
                    units=['log Angstrom','','','']
                    comments = ['Log lambda','Delta field','Pixel weights','Continuum']

                out.write(cols,names=names,header=hd,comment=comments,units=units,extname=str(d.thid))

            out.close()
