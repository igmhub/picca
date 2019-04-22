#!/usr/bin/env python

import sys
import fitsio
import healpy
import scipy as sp
import argparse
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from picca.data import forest
from picca.data import delta
from picca import io



if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--plate', type = int, default = None, required=True,
            help = 'Plate of spectrum')

    parser.add_argument('--mjd', type = int, default = None, required=True,
            help = 'Modified Julian Date of spectrum')

    parser.add_argument('--fiberid', type = int, default = None, required=True,
            help = 'fiber of spectrum')

    parser.add_argument('--drq', type = str, default = None, required=True,
            help = 'DRQ file')

    parser.add_argument('--nside', type = int, default = 16, required=False,
            help = 'healpix nside')

    parser.add_argument('--spectrum', type = str, default = None, required=True,
            help = 'data directory for all the spectra')

    parser.add_argument('--no-project', action="store_true", required=False,
            help = 'do not project out continuum fitting modes')

    parser.add_argument('--in-dir',type = str,default=None,required=True,
            help='data directory')

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

    parser.add_argument('--mode',type = str,default='pix',required=False,
            help='open mode: pix, spec, spcframe')

    parser.add_argument('--dla-vac',type = str,default=None,required=False,
            help='dla catalog file')

    parser.add_argument('--dla-mask',type = float,default=0.8,required=False,
            help='lower limit on the DLA transmission. Transmissions below this number are masked')

    parser.add_argument('--mask-file',type = str,default=None,required=False,
            help='Path to file to mask regions in lambda_OBS and lambda_RF. In file each line is: region_name region_min region_max (OBS or RF) [Angstrom]')

    parser.add_argument('--flux-calib',type = str,default=None,required=False,
            help='Path to file to previously produced do_delta.py file to correct for multiplicative errors in the flux calibration')

    parser.add_argument('--ivar-calib',type = str,default=None,required=False,
            help='Path to previously produced do_delta.py file to correct for multiplicative errors in the pipeline inverse variance calibration')

    args = parser.parse_args()

    ### forest args
    forest.lmin = sp.log10(args.lambda_min)
    forest.lmax = sp.log10(args.lambda_max)
    forest.lmin_rest = sp.log10(args.lambda_rest_min)
    forest.lmax_rest = sp.log10(args.lambda_rest_max)
    forest.rebin = args.rebin
    forest.dll = args.rebin*1e-4
    forest.dla_mask = args.dla_mask

    ### Get Healpy pixel of the given QSO
    objs = {}
    ra,dec,zqso,thid,plate,mjd,fid = io.read_drq(args.drq,0.,1000.,keep_bal=True)
    cut = (plate==args.plate) & (mjd==args.mjd) & (fid==args.fiberid)
    if cut.sum()==0:
        print("Object not in drq")
        sys.exit()
    ra = ra[cut]
    dec = dec[cut]
    zqso = zqso[cut]
    thid = thid[cut]
    plate = plate[cut]
    mjd = mjd[cut]
    fid = fid[cut]
    phi = ra
    th = sp.pi/2.-dec
    pix = healpy.ang2pix(args.nside,th,phi)

    ### Get data
    data = None
    if args.mode == "pix":
        data = io.read_from_pix(args.in_dir,pix[0],thid, ra, dec, zqso, plate, mjd, fid, order=None, log=None)
    elif args.mode in ["spec","corrected-spec"]:
        data = io.read_from_spec(args.in_dir,thid, ra, dec, zqso, plate, mjd, fid, order=None, mode=args.mode,log=None)
    elif args.mode =="spcframe":
        data = io.read_from_spcframe(args.in_dir,thid, ra, dec, zqso, plate, mjd, fid, order=None, mode=args.mode, log=None)
    if data is None:
        print("Object not in in_dir")
        sys.exit()
    else:
        data = data[0]

    ### Correct multiplicative flux calibration
    if (args.flux_calib is not None):
        try:
            vac = fitsio.FITS(args.flux_calib)
            head = vac[1].read_header()

            ll_st = vac[1]['loglam'][:]
            st    = vac[1]['stack'][:]
            w     = (st!=0.)
            forest.correc_flux = interp1d(ll_st[w],st[w],fill_value="extrapolate")
            vac.close()
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
            vac.close()
        except:
            print(" Error while reading ivar_calib file {}".format(args.ivar_calib))
            sys.exit(1)

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
            data.mask(mask_obs=usr_mask_obs , mask_RF=usr_mask_RF)

    ### Correct for DLAs
    if not args.dla_vac is None:
        print("adding dlas")
        dlas = io.read_dlas(args.dla_vac)
        for p in data:
            for d in data[p]:
                if d.thid in dlas:
                    for dla in dlas[d.thid]:
                        data.add_dla(dla[0],dla[1],usr_mask_RF_DLA)

    ### Get delta from do_delta
    done_delta = None
    f = args.spectrum+"/delta-"+str(pix[0])+".fits.gz"
    hdus = fitsio.FITS(f)
    ds = [delta.from_fitsio(h) for h in hdus[1:]]
    for d in ds:
        if (d.plate==args.plate) and (d.mjd==args.mjd) and (d.fid==args.fiberid):
            d.project()
            done_delta = d
            hdus.close()
            break
    if done_delta is None:
        hdus.close()
        print("Object not in spectrum")
        sys.exit()



    ### Observed l
    plt.errorbar(10**data.ll,data.fl,linewidth=2,color='black')
    plt.errorbar(10**done_delta.ll,done_delta.co,linewidth=4,color='red')
    plt.xlabel(r'$\lambda_{\mathrm{Obs.}} \, [\mathrm{\AA}]$',fontsize=30)
    plt.ylabel(r'$f \, [10^{-19} \mathrm{W \, m^{-2} \, nm^{-1}}]$',fontsize=30)
    plt.grid()
    plt.show()


    ### RF l
    plt.errorbar(10**data.ll/(1.+done_delta.zqso),data.fl,linewidth=4,color='black')
    plt.errorbar(10**done_delta.ll/(1.+done_delta.zqso),done_delta.co,linewidth=4,color='red')
    plt.xlabel(r'$\lambda_{\mathrm{R.F.}} \, [\mathrm{\AA}]$',fontsize=30)
    plt.ylabel(r'$f \, [10^{-19} \mathrm{W \, m^{-2} \, nm^{-1}}]$',fontsize=30)
    plt.grid()
    plt.show()









































