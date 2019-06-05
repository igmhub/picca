#!/usr/bin/env python

import scipy as sp
import matplotlib.pyplot as plt
import argparse

def convert1DTo2D(array1D,nbX,nbY):
    '''
        convert a 1D array to a 2D array
    '''

    array2D = sp.zeros((nbX,nbY))
    for k,el in enumerate(array1D):
        i = k//nbY
        j = k%nbY
        array2D[i,j] = el

    return array2D

if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Plot the scan of chi2 as a function of (ap,at)')

    parser.add_argument('--chi2scan', type=str, default=None, required=True,nargs="*",
        help='Input file with chi2 scan data')

    parser.add_argument('--label', type=str, default=None, required=True,nargs="*",
        help='Label of files given in --chi2scan')

    parser.add_argument('--d-over-rd', action='store_true', required=False,
        help='Plot in the (D_H/r_d,D_M/r_d) space')

    args = parser.parse_args()

    color = ['blue','red','green','orange']
    levels = [2.29, 6.18, 11.82]

    assert len(args.chi2scan)==len(args.label)

    for i,path in enumerate(args.chi2scan):

        ### Read chi2 scan
        with open(path) as f:
            first_line = f.readline()
        first_line = first_line.replace('#','')
        first_line = first_line.split()
        fromkeytoindex = { el:i for i,el in enumerate(first_line) }
        chi2 = sp.loadtxt(path)

        ### Read the best-fit chi2
        with open(path.replace('.ap.at.scan.dat','.chisq')) as f:
            first_line = f.readline()
        first_line = first_line.replace('#','')
        first_line = first_line.split()
        fromkeytoindex_bestfit = { el:i for i,el in enumerate(first_line) }
        chi2_bestfit = sp.loadtxt(path.replace('.ap.at.scan.dat','.chisq'))

        ### Read the best fit BAO
        with open(path.replace('.ap.at.scan.dat','.save.pars')) as f:
            first_line = f.readline()
        first_line = first_line.replace('#','')
        first_line = first_line.split()
        fromkeytoindex_bestfitBAO = { el:i for i,el in enumerate(first_line) }
        chi2_bestfitBAO = sp.loadtxt(path.replace('.ap.at.scan.dat','.save.pars'))

        ### Read the fiducial cosmology
        if args.d_over_rd:
            with open(path.replace('.ap.at.scan.dat','.fiducial')) as f:
                first_line = f.readline()
            first_line = first_line.replace('#','')
            first_line = first_line.split()
            fromkeytoindex_bestfitfiducial = { el:i for i,el in enumerate(first_line) }
            chi2_bestfitfiducial = sp.loadtxt(path.replace('.ap.at.scan.dat','.fiducial'))
            dhord = chi2_bestfitfiducial[fromkeytoindex_bestfitfiducial['Dh/rd']]
            dmord = chi2_bestfitfiducial[fromkeytoindex_bestfitfiducial['Dm/rd']]
        else:
            dhord = 1.
            dmord = 1.

        ### Plot
        par1 = 'ap'
        min1 = chi2[:,fromkeytoindex[par1]].min()*dhord
        max1 = chi2[:,fromkeytoindex[par1]].max()*dhord
        nb1 = sp.unique(chi2[:,fromkeytoindex[par1]]).size
        val1 = chi2_bestfitBAO[fromkeytoindex_bestfitBAO[par1]]*dhord

        par2 = 'at'
        min2 = chi2[:,fromkeytoindex[par2]].min()*dmord
        max2 = chi2[:,fromkeytoindex[par2]].max()*dmord
        nb2 = sp.unique(chi2[:,fromkeytoindex[par2]]).size
        val2 = chi2_bestfitBAO[fromkeytoindex_bestfitBAO[par2]]*dmord

        if 'Dchi2' in fromkeytoindex.keys():
            parChi2 = 'Dchi2'
            zzz = chi2[:,fromkeytoindex[parChi2]]
        else:
            parChi2 = 'chi2'
            zzz = chi2[:,fromkeytoindex[parChi2]]
            zzz -= chi2_bestfit[fromkeytoindex_bestfit[parChi2]]
        zzz = convert1DTo2D(zzz,nb1,nb2)
        extent = [min2,max2,min1,max1]

        plt.contour(zzz,levels=levels,extent=extent,origin='lower',colors=color[i])
        plt.plot([0.],[0.],color=color[i],label=r'$\mathrm{'+args.label[i]+'}$')
        plt.errorbar([val2],[val1],fmt='o',color=color[i])

    if args.d_over_rd:
        plt.xlabel(r'$D_{M}(z_{\mathrm{eff}})/r_{d}$', fontsize=20)
        plt.ylabel(r'$D_{H}(z_{\mathrm{eff}})/r_{d}$', fontsize=20)
    else:
        plt.xlabel(r'$\alpha_{\perp}$', fontsize=20)
        plt.ylabel(r'$\alpha_{\parallel}$', fontsize=20)
        plt.errorbar([1.],[1.],fmt='o',color='black')

    plt.xlim([0.75*dmord,1.25*dmord])
    plt.ylim([0.75*dhord,1.25*dhord])
    plt.subplots_adjust(bottom=0.15,top=0.95,right=0.95)
    plt.grid(True)
    plt.legend()
    plt.show()
