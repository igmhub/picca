#!/usr/bin/env python

from __future__ import print_function
import fitsio
import argparse
import glob
import sys
import scipy as sp
import matplotlib.pyplot as plt

#from matplotlib import rc,rcParams
#rcParams['text.usetex'] = True
#rcParams['text.latex.unicode'] = True
#rcParams['font.family'] = 'sans-serif'
#rcParams['text.latex.preamble'] = [r'\usepackage{sfmath} \boldmath']


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

    parser.add_argument('--out-fig', type = str, default = 'Pk1D.png', required=False,
                        help = 'data directory')

    args = parser.parse_args()

    # Binning corresponding to BOSS paper
    nb_z_bin=13
    nb_k_bin=35
    k_inf=0.000813
    k_sup=k_inf + nb_k_bin*0.000542

    sumPk = sp.zeros([nb_z_bin,nb_k_bin],dtype=sp.float64)
    sumPk2 = sp.zeros([nb_z_bin,nb_k_bin],dtype=sp.float64)
    sum = sp.zeros([nb_z_bin,nb_k_bin],dtype=sp.float64)
    k = sp.zeros([nb_k_bin],dtype=sp.float64)
    ek = sp.zeros([nb_k_bin],dtype=sp.float64)
    for ik in range (nb_k_bin) :
        k[ik] = k_inf + (ik+0.5)*0.000542

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
                sumPk[iz,ik] += pk.Pk[i]*pk.k[i]/sp.pi
                sumPk2[iz,ik] += (pk.Pk[i]*pk.k[i]/sp.pi)**2
                sum[iz,ik] += 1.0
                
    # compute mean and error on Pk
    meanPk = sp.where(sum!=0,sumPk/sum,0.0)
    errorPk = sp.where(sum!=0,sp.sqrt(((sumPk2/sum)-meanPk**2)/sum),0.0)
    
    # Print figure
    figure_file = args.out_fig
   
    zbins = [ 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 4.4, 4.6]
    colors = ['m', 'r', 'b', 'k', 'chartreuse', 'gold', 'aqua', 'slateblue', 'orange', 'mediumblue', 'darkgrey', 'olive', 'firebrick']
    s = 6
    fontt = 16
    fontlab = 10
    fontl =12

    fig = plt.figure(figsize = (16, 8))
    ax = fig.add_subplot(111)
    plt.yscale('log')
    
    for iz in range(nb_z_bin) :
        z = 2.2 + iz*0.2
        #ax.errorbar(k,meanPk[iz,:], yerr =errorPk[iz,:], fmt = 'o', color = colors[iz], markersize = s, label =r'\bf {:1.1f} $\displaystyle <$ z $\displaystyle <$ {:1.1f}'.format(z-0.1,z+0.1))
        ax.errorbar(k,meanPk[iz,:], yerr =errorPk[iz,:], fmt = 'o', color = colors[iz], markersize = s, label =r' z = {:1.1f}'.format(z))

    #ax.set_xlabel(r'\bf $\displaystyle k [km.s^{-1}]$', fontsize = fontt)
    ax.set_xlabel(r' k [km/s]', fontsize = fontt)
    #ax.set_ylabel(r'\bf $\displaystyle \Delta^2_{\mathrm{1d}}(k) $', fontsize=fontt, labelpad=-1)
    ax.set_ylabel(r'P(k)k/pi ', fontsize=fontt, labelpad=-1)
    ax.xaxis.set_ticks_position('both')
    ax.xaxis.set_tick_params(direction='in')
    ax.yaxis.set_ticks_position('both')
    ax.yaxis.set_tick_params(direction='in')
    ax.xaxis.set_tick_params(labelsize=fontlab)
    ax.yaxis.set_tick_params(labelsize=fontlab)

    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(handles, labels, loc=2, bbox_to_anchor=(1.03, 0.98), borderaxespad=0.,fontsize = fontl)
    fig.subplots_adjust(top=0.98,bottom=0.114,left=0.078,right=0.758,hspace=0.2,wspace=0.2)
    
    
    plt.show()
    fig.savefig(figure_file, transparent=False)

    
    print ("all done ")
    


