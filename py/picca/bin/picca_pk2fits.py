#!/usr/bin/env python
import sys
from astropy.io import fits
import numpy as np
import argparse

def main(cmdargs):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--prefix-pk', type=str, default=None, required=True,
        help='Prefix to pk file')

    parser.add_argument('--out', type=str, default=None, required=True,
        help='Output file name')

    parser.add_argument('--zref', type=float, default=None, required=True,
        help='Reference redshift')

    parser.add_argument('--Om', type=float, default=None, required=True,
        help='Matter density parameter')

    parser.add_argument('--Ok', type=float, default=None, required=True,
        help='Curvature density parameter')

    parser.add_argument('--w', type=float, default=None, required=True,
        help='Dark energy equation of state')

    parser.add_argument('--H0', type=float, default=None, required=True,
        help='Hubble constant')

    parser.add_argument('--sigma8', type=float, default=None, required=True,
        help='Fluctuation amplitude on 8 Mpc/h scale')

    parser.add_argument('--zdrag', type=float, default=None, required=True,
        help='Redshift at the drag epoch')

    parser.add_argument('--rdrag', type=float, default=None, required=True,
        help='Sound horizon at the drag epoch')


    args=parser.parse_args(cmdargs)

    pk=np.loadtxt(args.prefix_pk+'_matterpower.dat')
    pkSB=np.loadtxt(args.prefix_pk+'SB_matterpower.dat')
    col1=fits.Column(name='K',format='D',array=np.array(pk[:,0]))
    col2=fits.Column(name='PK',format='D',array=np.array(pk[:,1]))
    col3=fits.Column(name='PKSB',format='D',array=np.array(pkSB[:,1]))
    cols=fits.ColDefs([col1,col2,col3])
    head=fits.Header()
    head['ZREF']=args.zref
    head['Om']=args.Om
    head['Ok']=args.Ok
    head['OL']=1.-args.Om-args.Ok
    head['w']=args.w
    head['H0']=args.H0
    head['sigma8']=args.sigma8
    head['zdrag']=args.zdrag
    head['rdrag']=args.rdrag
    tbhdu=fits.BinTableHDU.from_columns(cols,header=head)
    tbhdu.writeto(args.out,clobber=True)


if __name__ == '__main__':
    cmdargs=sys.argv[1:]
    main(cmdargs)