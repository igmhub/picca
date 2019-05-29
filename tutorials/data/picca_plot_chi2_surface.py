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

    parser.add_argument('--chi2scan', type=str, default=None, required=True,
        help='Input file with chi2 scan data')

    args = parser.parse_args()

    levels = [2.29, 6.18, 11.82]

    ### Read chi2 scan
    with open(args.chi2scan) as f:
        first_line = f.readline()
    first_line = first_line.replace('#','')
    first_line = first_line.split()
    fromkeytoindex = { el:i for i,el in enumerate(first_line) }
    chi2 = sp.loadtxt(args.chi2scan)

    ### Read the best-fit chi2
    with open(args.chi2scan.replace('.ap.at.scan.dat','.chisq')) as f:
        first_line = f.readline()
    first_line = first_line.replace('#','')
    first_line = first_line.split()
    fromkeytoindex_bestfit = { el:i for i,el in enumerate(first_line) }
    chi2_bestfit = sp.loadtxt(args.chi2scan.replace('.ap.at.scan.dat','.chisq'))

    ### Plot
    par1 = 'ap'
    min1 = chi2[:,fromkeytoindex[par1]].min()
    max1 = chi2[:,fromkeytoindex[par1]].max()
    nb1 = sp.unique(chi2[:,fromkeytoindex[par1]]).size

    par2 = 'at'
    min2 = chi2[:,fromkeytoindex[par2]].min()
    max2 = chi2[:,fromkeytoindex[par2]].max()
    nb2 = sp.unique(chi2[:,fromkeytoindex[par2]]).size

    if 'Dchi2' in fromkeytoindex.keys():
        parChi2 = 'Dchi2'
        zzz = chi2[:,fromkeytoindex[parChi2]]
    else:
        parChi2 = 'chi2'
        zzz = chi2[:,fromkeytoindex[parChi2]]
        zzz -= chi2_bestfit[fromkeytoindex_bestfit[parChi2]]
    zzz = convert1DTo2D(zzz,nb1,nb2)
    extent = [min2,max2,min1,max1]

    plt.contour(zzz,levels=levels,extent=extent,origin='lower')
    #plt.errorbar([at],[ap],fmt='o',color=color[j])

    plt.errorbar([1.],[1.],fmt='o',color='black')
    plt.xlabel(r'$\alpha_{\perp}$', fontsize=20)
    plt.ylabel(r'$\alpha_{\parallel}$', fontsize=20)
    plt.grid(True)
    plt.show()
