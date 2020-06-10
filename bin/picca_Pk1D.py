#!/usr/bin/python3
"""Compute the 1D power spectrum
"""
import argparse
import glob
from array import array
import numpy as np
import fitsio

from picca import constants
from picca.data import Delta
from picca.pk1d import (compute_correction_reso, compute_pk_noise,
                        compute_pk_raw, fill_masked_pixels, rebin_diff_noise,
                        split_forest)
from picca.utils import userprint


def make_tree(tree, nb_bin_max):
    """Makes the ROOT tree to save the data

    Args:
        tree: TTree
            The ROOT tree to fill
        nb_bin_max: int
            Maximum number of bins

    Returns:
        The following variables:
            z_qso: Quasar redshifts
            mean_z: Mean redshift of the forest
            mean_reso: Mean resolution of the forest
            mean_snr: Mean signal-to-noise ratio in the forest
            lambda_min: Minimum wavelength (in Angs)
            lambda_max: Maximum wavelength (in Angs)
            plate: Plate number of the observation
            mjd: Modified Julian Date of the observation
            fiber: Fiberid of the object
            nb_mask_pix: Number of masked pixels
            num_bins_tree: Number of bins (storage tree version)
            k_tree: Fourier modes (tree version)
            pk_tree: Power spectrum for the different fourier modes (storage
                tree version)
            pk_raw_tree: Raw power spectrum for the different fourier modes
                (storage tree version)
            pk_noise_tree: Noise power spectrum for the different fourier modes
                (storage tree version)
            correction_reso_tree: Resolution for the correlation function
                (storage tree version)
            pk_diff_tree: Power spectrum of exposures_diff for the different
                fourier modes (storage tree version)
    """
    z_qso = array('f', [0.])
    mean_z = array('f', [0.])
    mean_reso = array('f', [0.])
    mean_snr = array('f', [0.])
    num_masked_pixels_tree = array('f', [0.])

    lambda_min = array('f', [0.])
    lambda_max= array('f', [0.])

    plate = array('i', [0])
    mjd = array('i', [0])
    fiber = array('i', [0])

    num_bins_tree = array('i', [0])
    k_tree = array('f', nb_bin_max*[0.])
    pk_tree = array('f', nb_bin_max*[0.])
    pk_raw_tree = array('f', nb_bin_max*[0.])
    pk_noise_tree = array('f', nb_bin_max*[0.])
    pk_diff_tree = array('f', nb_bin_max*[0.])
    correction_reso_tree = array('f', nb_bin_max*[0.])

    tree.Branch("z_qso", z_qso, "z_qso/F")
    tree.Branch("mean_z", mean_z, "mean_z/F")
    tree.Branch("mean_reso", mean_reso, "mean_reso/F")
    tree.Branch("mean_snr", mean_snr, "mean_snr/F")
    tree.Branch("lambda_min", lambda_min, "lambda_min/F")
    tree.Branch("lambda_max", lambda_max, "lambda_max/F")
    tree.Branch("nb_masked_pixel", num_masked_pixels_tree, "nb_mask_pixel/F")

    tree.Branch("plate", plate, "plate/I")
    tree.Branch("mjd", mjd, "mjd/I")
    tree.Branch("fiber", fiber, "fiber/I")

    tree.Branch("NbBin", num_bins_tree, "NbBin/I")
    tree.Branch("k", k_tree, "k[NbBin]/F")
    tree.Branch("Pk_raw", pk_raw_tree, "Pk_raw[NbBin]/F")
    tree.Branch("Pk_noise", pk_noise_tree, "Pk_noise[NbBin]/F")
    tree.Branch("Pk_diff", pk_diff_tree, "Pk_diff[NbBin]/F")
    tree.Branch("cor_reso", correction_reso_tree, "cor_reso[NbBin]/F")
    tree.Branch("Pk", pk_tree, "Pk[NbBin]/F")

    return (z_qso, mean_z, mean_reso, mean_snr, lambda_min, lambda_max, plate,
            mjd, fiber, num_masked_pixels_tree, num_bins_tree, k_tree, pk_tree,
            pk_raw_tree, pk_noise_tree, correction_reso_tree, pk_diff_tree)

def compute_mean_delta(log_lambda,delta,ivar,z_qso):

    for i, _ in enumerate (log_lambda):
        log_lambda_obs = np.power(10., log_lambda[i])
        log_lambda_rf = log_lambda_obs/(1.+z_qso)
        hdelta.Fill(log_lambda_obs, log_lambda_rf, delta[i])
        hdelta_RF.Fill(log_lambda_rf, delta[i])
        hdelta_OBS.Fill(log_lambda_obs, delta[i])
        hivar.Fill(ivar[i])
        snr_pixel = (delta[i]+1)*np.sqrt(ivar[i])
        hsnr.Fill(snr_pixel)
        hivar.Fill(ivar[i])
        if (ivar[i] < 1000):
            hdelta_RF_we.Fill(log_lambda_rf, delta[i], ivar[i])
            hdelta_OBS_we.Fill(log_lambda_obs, delta[i], ivar[i])

    return


def main():
    """Compute the 1D power spectrum"""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Compute the 1D power spectrum')

    parser.add_argument(
        '--out-dir',
        type=str,
        default=None,
        required=True,
        help='Output directory')

    parser.add_argument(
        '--out-format',
        type=str,
        default='fits',
        required=False,
        help='Output format: root or fits (if root call PyRoot)')

    parser.add_argument(
        '--in-dir',
        type=str,
        default=None,
        required=True,
        help='Directory to delta files')

    parser.add_argument(
        '--in-format',
        type=str,
        default='fits',
        required=False,
        help=' Input format used for input files: ascii or fits')

    parser.add_argument(
        '--SNR-min',
        type=float,
        default=2.,
        required=False,
        help='Minimal mean SNR per pixel ')

    parser.add_argument(
        '--reso-max',
        type=float,
        default=85.,
        required=False,
        help='Maximal resolution in km/s ')

    parser.add_argument(
        '--lambda-obs-min',
        type=float,
        default=3600.,
        required=False,
        help='Lower limit on observed wavelength [Angstrom]' )

    parser.add_argument(
        '--nb-part',
        type=int,
        default=3,
        required=False,
        help='Number of parts in forest')

    parser.add_argument(
        '--nb-pixel-min',
        type=int,
        default=75,
        required=False,
        help='Minimal number of pixels in a part of forest')

    parser.add_argument(
        '--nb-pixel-masked-max',
        type=int,
        default=40,
        required=False,
        help='Maximal number of masked pixels in a part of forest')

    parser.add_argument(
        '--no-apply-filling',
        action='store_true',
        default=False,
        required=False,
        help='Dont fill masked pixels')

    parser.add_argument(
        '--noise-estimate',
        type=str,
        default='mean_diff',
        required=False,
        help=('Estimate of Pk_noise '
              'pipeline/diff/mean_diff/rebin_diff/mean_rebin_diff'))

    parser.add_argument(
        '--forest-type',
        type=str,
        default='Lya',
        required=False,
        help='Forest used: Lya, SiIV, CIV')

    parser.add_argument(
        '--debug',
        action='store_true',
        default=False,
        required=False,
        help='Fill root histograms for debugging')

    args = parser.parse_args()

    # Create root file
    if (args.out_format=='root') :
        from ROOT import TH1D, TFile, TTree, TProfile2D, TProfile
        storeFile = TFile(args.out_dir + "/Testpicca.root",
                          "RECREATE",
                          "PK 1D studies studies");
        nb_bin_max = 700
        tree = TTree("Pk1D", "SDSS 1D Power spectrum Ly-a");
        (z_qso, mean_z, mean_reso, mean_snr, lambda_min, lambda_max, plate, mjd,
         fiber, num_masked_pixels_tree, num_bins_tree, k_tree, pk_tree, pk_raw_tree, pk_noise_tree, correction_reso_tree,
         pk_diff_tree) = make_tree(tree, b_bin_max)

        # control histograms
        if (args.forest_type == 'Lya'):
            forest_inf = 1040.
            forest_sup = 1200.
        elif (args.forest_type == 'SiIV'):
            forest_inf = 1270.
            forest_sup = 1380.
        elif (args.forest_type == 'CIV'):
            forest_inf = 1410.
            forest_sup = 1520.
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
    num_data = 0

    # initialize randoms
    np.random.seed(4)

    # loop over input files
    for i,f in enumerate(fi):
        if i%1==0:
            userprint("\rread {} of {} {}".format(i,len(fi),num_data),end="")

        # read fits or ascii file
        if (args.in_format=='fits') :
            hdus = fitsio.FITS(f)
            dels = [Delta.from_fitsio(h,pk1d_type=True) for h in hdus[1:]]
        elif (args.in_format=='ascii') :
            ascii_file = open(f,'r')
            dels = [Delta.from_ascii(line) for line in ascii_file]

        num_data+=len(dels)
        userprint ("\n ndata =  ",num_data)
        out = None

        # loop over deltas
        for d in dels:

            # Selection over the SNR and the resolution
            if (d.mean_snr<=args.SNR_min or d.mean_reso>=args.reso_max) : continue

            # first pixel in forest
            for first_pixel_index, first_pixel_log_lambda in enumerate(d.log_lambda):
                if 10.**first_pixel_log_lambda > args.lambda_obs_min :
                    break

            # minimum number of pixel in forest
            nb_pixel_min = args.nb_pixel_min
            if ((len(d.log_lambda)-first_pixel_index)<nb_pixel_min) : continue

            # Split in n parts the forest
            max_num_parts = (len(d.log_lambda)-first_pixel_index)//nb_pixel_min
            num_parts = min(args.nb_part,max_num_parts)
            m_z_arr,ll_arr,de_arr,diff_arr,iv_arr = split_forest(num_parts,d.delta_log_lambda,d.log_lambda,d.delta,d.exposures_diff,d.ivar,first_pixel_index)
            for f in range(num_parts):

                # rebin diff spectrum
                if (args.noise_estimate=='rebin_diff' or args.noise_estimate=='mean_rebin_diff'):
                    diff_arr[f]=rebin_diff_noise(d.delta_log_lambda,ll_arr[f],diff_arr[f])

                # Fill masked pixels with 0.
                ll_new, delta_new, diff_new, iv_new, num_masked_pixels = fill_masked_pixels(d.delta_log_lambda,ll_arr[f],de_arr[f],diff_arr[f],iv_arr[f],args.no_apply_filling)
                if (num_masked_pixels > args.nb_pixel_masked_max) :
                    continue
                if (args.out_format=='root' and  args.debug):
                    compute_mean_delta(ll_new,delta_new,iv_new,d.z_qso)

                lam_lya = constants.ABSORBER_IGM["LYA"]
                z_abs =  np.power(10.,ll_new)/lam_lya - 1.0
                mean_z_new = sum(z_abs)/float(len(z_abs))

                # Compute pk_raw
                k,pk_raw = compute_pk_raw(d.delta_log_lambda,delta_new)

                # Compute pk_noise
                run_noise = False
                if (args.noise_estimate=='pipeline'): run_noise=True
                pk_noise,pk_diff = compute_pk_noise(d.delta_log_lambda,iv_new,diff_new,run_noise)

                # Compute resolution correction
                delta_pixel = d.delta_log_lambda*np.log(10.)*constants.speed_light/1000.
                correction_reso = compute_correction_reso(delta_pixel,d.mean_reso,k)

                # Compute 1D Pk
                if (args.noise_estimate=='pipeline'):
                    pk = (pk_raw - pk_noise)/correction_reso
                elif (args.noise_estimate=='diff' or args.noise_estimate=='rebin_diff'):
                    pk = (pk_raw - pk_diff)/correction_reso
                elif (args.noise_estimate=='mean_diff' or args.noise_estimate=='mean_rebin_diff'):
                    selection = (k>0) & (k<0.02)
                    if (args.noise_estimate=='mean_rebin_diff'):
                        selection = (k>0.003) & (k<0.02)
                    Pk_mean_diff = sum(pk_diff[selection])/float(len(pk_diff[selection]))
                    pk = (pk_raw - Pk_mean_diff)/correction_reso

                # save in root format
                if (args.out_format=='root'):
                    z_qso[0] = d.z_qso
                    mean_z[0] = m_z_arr[f]
                    mean_reso[0] = d.mean_reso
                    mean_snr[0] = d.mean_snr
                    lambda_min[0] =  np.power(10.,ll_new[0])
                    lambda_max[0] =  np.power(10.,ll_new[-1])
                    num_masked_pixels_tree[0] = num_masked_pixels

                    plate[0] = d.plate
                    mjd[0] = d.mjd
                    fiber[0] = d.fiberid

                    num_bins_tree[0] = min(len(k),nb_bin_max)
                    for i in range(num_bins_tree[0]) :
                        k_tree[i] = k[i]
                        pk_raw_tree[i] = pk_raw[i]
                        pk_noise_tree[i] = pk_noise[i]
                        pk_diff_tree[i] = pk_diff[i]
                        pk_tree[i] = pk[i]
                        correction_reso_tree[i] = correction_reso[i]

                    tree.Fill()

                # save in fits format
                if (args.out_format=='fits'):
                    hd = [ {'name':'RA','value':d.ra,'comment':"QSO's Right Ascension [degrees]"},
                        {'name':'DEC','value':d.dec,'comment':"QSO's Declination [degrees]"},
                        {'name':'Z','value':d.z_qso,'comment':"QSO's redshift"},
                        {'name':'MEANZ','value':m_z_arr[f],'comment':"Absorbers mean redshift"},
                        {'name':'MEANRESO','value':d.mean_reso,'comment':'Mean resolution [km/s]'},
                        {'name':'MEANSNR','value':d.mean_snr,'comment':'Mean signal to noise ratio'},
                        {'name':'NBMASKPIX','value':num_masked_pixels,'comment':'Number of masked pixels in the section'},
                        {'name':'PLATE','value':d.plate,'comment':"Spectrum's plate id"},
                        {'name':'MJD','value':d.mjd,'comment':'Modified Julian Date,date the spectrum was taken'},
                        {'name':'FIBER','value':d.fiberid,'comment':"Spectrum's fiber number"}
                    ]

                    cols=[k,pk_raw,pk_noise,pk_diff,correction_reso,pk]
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


    userprint ("all done ")


if __name__ == '__main__':
    main()
