#!/usr/bin/env python

from __future__ import division
from __future__ import print_function
import fitsio
import argparse
import glob
import sys
import scipy as sp

from picca import constants
from picca.Pk1D import Pk1D, compute_Pk_raw, compute_Pk_noise, compute_cor_reso, fill_masked_pixels, split_forest, rebin_diff_noise
from picca.data import delta
from picca.utils import print

from array import array

def make_tree(tree,nb_bin_max):

    zqso = array( 'f', [ 0. ] )
    mean_z = array( 'f', [ 0. ] )
    mean_reso = array( 'f', [ 0. ] )
    mean_SNR = array( 'f', [ 0. ] )
    nb_mask_pix = array( 'f', [ 0. ] )

    lambda_min = array( 'f', [ 0. ] )
    lambda_max= array( 'f', [ 0. ] )

    plate = array( 'i', [ 0 ] )
    mjd = array( 'i', [ 0 ] )
    fiber = array( 'i', [ 0 ] )

    nb_r = array( 'i', [ 0 ] )
    k_r = array( 'f', nb_bin_max*[ 0. ] )
    Pk_r = array( 'f', nb_bin_max*[ 0. ] )
    Pk_raw_r = array( 'f', nb_bin_max*[ 0. ] )
    Pk_noise_r = array( 'f', nb_bin_max*[ 0. ] )
    Pk_diff_r = array( 'f', nb_bin_max*[ 0. ] )
    cor_reso_r = array( 'f', nb_bin_max*[ 0. ] )

    tree.Branch("zqso",zqso,"zqso/F")
    tree.Branch("mean_z",mean_z,"mean_z/F")
    tree.Branch("mean_reso",mean_reso,"mean_reso/F")
    tree.Branch("mean_SNR",mean_SNR,"mean_SNR/F")
    tree.Branch("lambda_min",lambda_min,"lambda_min/F")
    tree.Branch("lambda_max",lambda_max,"lambda_max/F")
    tree.Branch("nb_masked_pixel",nb_mask_pix,"nb_mask_pixel/F")

    tree.Branch("plate",plate,"plate/I")
    tree.Branch("mjd",mjd,"mjd/I")
    tree.Branch("fiber",fiber,"fiber/I")

    tree.Branch( 'NbBin', nb_r, 'NbBin/I' )
    tree.Branch( 'k', k_r, 'k[NbBin]/F' )
    tree.Branch( 'Pk_raw', Pk_raw_r, 'Pk_raw[NbBin]/F' )
    tree.Branch( 'Pk_noise', Pk_noise_r, 'Pk_noise[NbBin]/F' )
    tree.Branch( 'Pk_diff', Pk_diff_r, 'Pk_diff[NbBin]/F' )
    tree.Branch( 'cor_reso', cor_reso_r, 'cor_reso[NbBin]/F' )
    tree.Branch( 'Pk', Pk_r, 'Pk[NbBin]/F' )

    return zqso,mean_z,mean_reso,mean_SNR,lambda_min,lambda_max,plate,mjd,fiber,\
    nb_mask_pix,nb_r,k_r,Pk_r,Pk_raw_r,Pk_noise_r,cor_reso_r,Pk_diff_r

def compute_mean_delta(ll,delta,iv,zqso):

    for i in range (len(ll)):
        ll_obs= sp.power(10.,ll[i])
        ll_rf = ll_obs/(1.+zqso)
        hdelta.Fill(ll_obs,ll_rf,delta[i])
        hdelta_RF.Fill(ll_rf,delta[i])
        hdelta_OBS.Fill(ll_obs,delta[i])
        hivar.Fill(iv[i])
        snr_pixel = (delta[i]+1)*sp.sqrt(iv[i])
        hsnr.Fill(snr_pixel)
        hivar.Fill(iv[i])
        if (iv[i]<1000) :
            hdelta_RF_we.Fill(ll_rf,delta[i],iv[i])
            hdelta_OBS_we.Fill(ll_obs,delta[i],iv[i])

    return

if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Compute the 1D power spectrum')

    parser.add_argument('--out-dir', type=str, default=None, required=True,
        help='Output directory')

    parser.add_argument('--out-format', type=str, default='fits', required=False,
        help='Output format: root or fits (if root call PyRoot)')

    parser.add_argument('--in-dir', type=str, default=None, required=True,
        help='Directory to delta files')

    parser.add_argument('--in-format', type=str, default='fits', required=False,
        help=' Input format used for input files: ascii or fits')

    parser.add_argument('--SNR-min',type=float,default=2.,required=False,
        help='Minimal mean SNR per pixel ')

    parser.add_argument('--reso-max',type=float,default=85.,required=False,
        help='Maximal resolution in km/s ')

    parser.add_argument('--lambda-obs-min',type=float,default=3600.,required=False,
        help='Lower limit on observed wavelength [Angstrom]' )

    parser.add_argument('--nb-part',type=int,default=3,required=False,
        help='Number of parts in forest')

    parser.add_argument('--nb-pixel-min',type=int,default=75,required=False,
        help='Minimal number of pixels in a part of forest')

    parser.add_argument('--nb-pixel-masked-max',type=int,default=40,required=False,
        help='Maximal number of masked pixels in a part of forest')

    parser.add_argument('--no-apply-filling', action='store_true', default=False, required=False,
        help='Dont fill masked pixels')

    parser.add_argument('--noise-estimate', type=str, default='mean_diff', required=False,
        help='Estimate of Pk_noise pipeline/diff/mean_diff/rebin_diff/mean_rebin_diff')

    parser.add_argument('--forest-type', type=str, default='Lya', required=False,
        help='Forest used: Lya, SiIV, CIV')

    parser.add_argument('--debug', action='store_true', default=False, required=False,
        help='Fill root histograms for debugging')


    args = parser.parse_args()

#   Create root file
    if (args.out_format=='root') :
        from ROOT import TCanvas, TH1D, TFile, TTree, TProfile2D, TProfile
        storeFile = TFile(args.out_dir+"/Testpicca.root","RECREATE","PK 1D studies studies");
        nb_bin_max = 700
        tree = TTree("Pk1D","SDSS 1D Power spectrum Ly-a");
        zqso,mean_z,mean_reso,mean_SNR,lambda_min,lambda_max,plate,mjd,fiber,\
        nb_mask_pix,nb_r,k_r,Pk_r,Pk_raw_r,Pk_noise_r,cor_reso_r,Pk_diff_r = make_tree(tree,nb_bin_max)

        # control histograms
        if (args.forest_type=='Lya'):
            forest_inf=1040.
            forest_sup=1200.
        elif (args.forest_type=='SiIV'):
            forest_inf=1270.
            forest_sup=1380.
        elif (args.forest_type=='CIV'):
            forest_inf=1410.
            forest_sup=1520.
        hdelta  = TProfile2D( 'hdelta', 'delta mean as a function of lambda-lambdaRF', 36, 3600., 7200., 16, forest_inf, forest_sup, -5.0, 5.0)
        hdelta_RF  = TProfile( 'hdelta_RF', 'delta mean as a function of lambdaRF', 320, forest_inf, forest_sup, -5.0, 5.0)
        hdelta_OBS  = TProfile( 'hdelta_OBS', 'delta mean as a function of lambdaOBS', 1800, 3600., 7200., -5.0, 5.0)
        hdelta_RF_we  = TProfile( 'hdelta_RF_we', 'delta mean weighted as a function of lambdaRF', 320, forest_inf, forest_sup, -5.0, 5.0)
        hdelta_OBS_we  = TProfile( 'hdelta_OBS_we', 'delta mean weighted as a function of lambdaOBS', 1800, 3600., 7200., -5.0, 5.0)
        hivar = TH1D('hivar','  ivar ',10000,0.0,10000.)
        hsnr = TH1D('hsnr','  snr per pixel ',100,0.0,100.)
        hdelta_RF_we.Sumw2()
        hdelta_OBS_we.Sumw2()


    # Read deltas
    if (args.in_format=='fits') :
        fi = glob.glob(args.in_dir+"/*.fits.gz")
    elif (args.in_format=='ascii') :
        fi = glob.glob(args.in_dir+"/*.txt")

    data = {}
    ndata = 0

    # initialize randoms
    sp.random.seed(4)

    # loop over input files
    for i,f in enumerate(fi):
        if i%1==0:
            print("\rread {} of {} {}".format(i,len(fi),ndata),end="")

        # read fits or ascii file
        if (args.in_format=='fits') :
            hdus = fitsio.FITS(f)
            dels = [delta.from_fitsio(h,Pk1D_type=True) for h in hdus[1:]]
        elif (args.in_format=='ascii') :
            ascii_file = open(f,'r')
            dels = [delta.from_ascii(line) for line in ascii_file]

        ndata+=len(dels)
        print ("\n ndata =  ",ndata)
        out = None

        # loop over deltas
        for d in dels:

            # Selection over the SNR and the resolution
            if (d.mean_SNR<=args.SNR_min or d.mean_reso>=args.reso_max) : continue

            # first pixel in forest
            for first_pixel in range(len(d.ll)) :
                 if (sp.power(10.,d.ll[first_pixel])>args.lambda_obs_min) : break

            # minimum number of pixel in forest
            nb_pixel_min = args.nb_pixel_min
            if ((len(d.ll)-first_pixel)<nb_pixel_min) : continue

            # Split in n parts the forest
            nb_part_max = (len(d.ll)-first_pixel)//nb_pixel_min
            nb_part = min(args.nb_part,nb_part_max)
            m_z_arr,ll_arr,de_arr,diff_arr,iv_arr = split_forest(nb_part,d.dll,d.ll,d.de,d.diff,d.iv,first_pixel)
            for f in range(nb_part):

                # rebin diff spectrum
                if (args.noise_estimate=='rebin_diff' or args.noise_estimate=='mean_rebin_diff'):
                    diff_arr[f]=rebin_diff_noise(d.dll,ll_arr[f],diff_arr[f])

                # Fill masked pixels with 0.
                ll_new,delta_new,diff_new,iv_new,nb_masked_pixel = fill_masked_pixels(d.dll,ll_arr[f],de_arr[f],diff_arr[f],iv_arr[f],args.no_apply_filling)
                if (nb_masked_pixel> args.nb_pixel_masked_max) : continue
                if (args.out_format=='root' and  args.debug): compute_mean_delta(ll_new,delta_new,iv_new,d.zqso)

                lam_lya = constants.absorber_IGM["LYA"]
                z_abs =  sp.power(10.,ll_new)/lam_lya - 1.0
                mean_z_new = sum(z_abs)/float(len(z_abs))

                # Compute Pk_raw
                k,Pk_raw = compute_Pk_raw(d.dll,delta_new,ll_new)

                # Compute Pk_noise
                run_noise = False
                if (args.noise_estimate=='pipeline'): run_noise=True
                Pk_noise,Pk_diff = compute_Pk_noise(d.dll,iv_new,diff_new,ll_new,run_noise)

                # Compute resolution correction
                delta_pixel = d.dll*sp.log(10.)*constants.speed_light/1000.
                cor_reso = compute_cor_reso(delta_pixel,d.mean_reso,k)

                # Compute 1D Pk
                if (args.noise_estimate=='pipeline'):
                    Pk = (Pk_raw - Pk_noise)/cor_reso
                elif (args.noise_estimate=='diff' or args.noise_estimate=='rebin_diff'):
                    Pk = (Pk_raw - Pk_diff)/cor_reso
                elif (args.noise_estimate=='mean_diff' or args.noise_estimate=='mean_rebin_diff'):
                    selection = (k>0) & (k<0.02)
                    if (args.noise_estimate=='mean_rebin_diff'):
                        selection = (k>0.003) & (k<0.02)
                    Pk_mean_diff = sum(Pk_diff[selection])/float(len(Pk_diff[selection]))
                    Pk = (Pk_raw - Pk_mean_diff)/cor_reso

                # Build Pk1D
                if (args.noise_estimate=='mean_diff' or args.noise_estimate=='mean_rebin_diff'):
                    Pk1D_final = Pk1D(d.ra,d.dec,d.zqso,d.mean_z,d.plate,d.mjd,d.fid,d.mean_SNR,d.mean_reso,k,Pk_raw,Pk_noise,cor_reso,Pk,nb_masked_pixel,Pk_mean_diff)
                else:
                    Pk1D_final = Pk1D(d.ra,d.dec,d.zqso,d.mean_z,d.plate,d.mjd,d.fid,d.mean_SNR,d.mean_reso,k,Pk_raw,Pk_noise,cor_reso,Pk,nb_masked_pixel)

                # save in root format
                if (args.out_format=='root'):
                    zqso[0] = d.zqso
                    mean_z[0] = m_z_arr[f]
                    mean_reso[0] = d.mean_reso
                    mean_SNR[0] = d.mean_SNR
                    lambda_min[0] =  sp.power(10.,ll_new[0])
                    lambda_max[0] =  sp.power(10.,ll_new[-1])
                    nb_mask_pix[0] = nb_masked_pixel

                    plate[0] = d.plate
                    mjd[0] = d.mjd
                    fiber[0] = d.fid

                    nb_r[0] = min(len(k),nb_bin_max)
                    for i in range(nb_r[0]) :
                        k_r[i] = k[i]
                        Pk_raw_r[i] = Pk_raw[i]
                        Pk_noise_r[i] = Pk_noise[i]
                        Pk_diff_r[i] = Pk_diff[i]
                        Pk_r[i] = Pk[i]
                        cor_reso_r[i] = cor_reso[i]

                    tree.Fill()

                # save in fits format

                if (args.out_format=='fits'):
#                    hd={}
#                    hd["RA"]=d.ra
#                    hd["DEC"]=d.dec
#                    hd["Z"]=d.zqso
#                    hd["MEANZ"]=m_z_arr[f]
#                    hd["MEANRESO"]=d.mean_reso
#                    hd["MEANSNR"]=d.mean_SNR
#                    hd["NBMASKPIX"]=nb_masked_pixel
#
#                    hd["PLATE"]=d.plate
#                    hd["MJD"]=d.mjd
#                    hd["FIBER"]=d.fid

                    hd = [ {'name':'RA','value':d.ra,'comment':"QSO's Right Ascension [degrees]"},
                        {'name':'DEC','value':d.dec,'comment':"QSO's Declination [degrees]"},
                        {'name':'Z','value':d.zqso,'comment':"QSO's redshift"},
                        {'name':'MEANZ','value':m_z_arr[f],'comment':"Absorbers mean redshift"},
                        {'name':'MEANRESO','value':d.mean_reso,'comment':'Mean resolution [km/s]'},
                        {'name':'MEANSNR','value':d.mean_SNR,'comment':'Mean signal to noise ratio'},
                        {'name':'NBMASKPIX','value':nb_masked_pixel,'comment':'Number of masked pixels in the section'},
                        {'name':'PLATE','value':d.plate,'comment':"Spectrum's plate id"},
                        {'name':'MJD','value':d.mjd,'comment':'Modified Julian Date,date the spectrum was taken'},
                        {'name':'FIBER','value':d.fid,'comment':"Spectrum's fiber number"}
                    ]

                    cols=[k,Pk_raw,Pk_noise,Pk_diff,cor_reso,Pk]
                    names=['k','Pk_raw','Pk_noise','Pk_diff','cor_reso','Pk']
                    comments=['Wavenumber', 'Raw power spectrum', "Noise's power spectrum", 'Noise coadd difference power spectrum',\
                              'Correction resolution function', 'Corrected power spectrum (resolution and noise)']
                    units=['(km/s)^-1', 'km/s', 'km/s', 'km/s', 'km/s', 'km/s']

                    try:
                        out.write(cols,names=names,header=hd,comments=comments,units=units)
                    except AttributeError:
                        out = fitsio.FITS(args.out_dir+'/Pk1D-'+str(i)+'.fits.gz','rw',clobber=True)
                        out.write(cols,names=names,header=hd,comment=comments,units=units)
        if (args.out_format=='fits' and out is not None):
            out.close()

# Store root file results
    if (args.out_format=='root'):
         storeFile.Write()


    print ("all done ")
