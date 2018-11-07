#!/usr/bin/env python

from __future__ import print_function
import fitsio
import argparse
import glob
import sys
import scipy as sp

from picca.Pk1D import Pk1D

if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--in-dir', type = str, default = None, required=True,
                        help = 'data directory')

    parser.add_argument('--SNR-min',type = float,default=2.,required=False,
                        help = 'minimal mean SNR per pixel ')

    parser.add_argument('--z-min',type = float,default=2.1,required=False,
                        help = 'minimal mean absorption redshift ')

    parser.add_argument('--reso-max',type = float,default=85.,required=False,
                        help = 'maximal resolution in km/s ')

    args = parser.parse_args()

    # Binning corresponding to BOSS paper
    nb_z_bin=13
    nb_k_bin=35
    k_inf=0.000813
    k_sup=k_inf + nb_k_bin*0.000542

    sumPk = sp.zeros([nb_z_bin,nb_k_bin],dtype=sp.float64)
    sumPk2 = sp.zeros([nb_z_bin,nb_k_bin],dtype=sp.float64)
    sum = sp.zeros([nb_z_bin,nb_k_bin],dtype=sp.float64)

    # list of Pk(1D)
    fi = glob.glob(args.in_dir+"/*.fits.gz")

    data = {}
    ndata = 0

    # loop over input files
    for i,f in enumerate(fi):
        if i%1==0:
            sys.stderr.write("\rread {} of {} {}".format(i,len(fi),ndata))

        # read fits files        
        hdus = fitsio.FITS(f)
        pk1ds = [Pk1D.from_fitsio(h) for h in hdus[1:]]
 
        ndata+=len(pk1ds)
        print ("\n ndata =  ",ndata)
  
        # loop over pk1ds
        for pk in pk1ds:
             
            # Selection over the SNR and the resolution
            if (pk.mean_snr<=args.SNR_min or pk.mean_reso>=args.reso_max) : continue

            if(pk.mean_z<=args.z_min) : continue
            
            iz = int((pk.mean_z-args.z_min)/0.2)
            if(iz>=nb_z_bin or iz<0) : continue
            
            for i in range (len(pk.k)) :
                ik = int((pk.k[i]-k_inf)/(k_sup-k_inf)*nb_k_bin);
                if(ik>= nb_k_bin or ik<0) : continue 
                #print(" iz, ik = ",iz,ik)
                sumPk[iz,ik] += pk.Pk[i]*pk.k[i]/sp.pi
                sumPk2[iz,ik] += (pk.Pk[i]*pk.k[i]/sp.pi)**2
                sum[iz,ik] += 1.0
        
    meanPk = sp.where(sum!=0,sumPk/sum,0.0)
    errorPk = sp.where(sum!=0,sp.sqrt(((sumPk2/sum)-meanPk**2)/sum),0.0)
    # compute mean and error on Pk
    print ("=========== sumPk ========",sumPk)
    print ("=========== sumPk2 ========",sumPk2)
    print ("=========== sum ========",sum)
    print ("=========== Pk mean ========",meanPk)
    print ("=========== Pk error ========",errorPk)
    print ("all done ")
    


