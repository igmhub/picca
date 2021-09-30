#!/usr/bin/env python

import sys
import os
import argparse
import fitsio
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import camb
import nbodykit.cosmology.correlation

from picca.utils import userprint
from picca.constants import SPEED_LIGHT

def main(cmdargs):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-i','--ini', type=str, required=True,
        help='Input config file for CAMB')

    parser.add_argument('-o','--out', type=str, required=True,
        help='Output FITS file')

    parser.add_argument('--H0', type=float, required=False, default=None,
        help='Hubble parameter, if not given use the one from the config file')

    parser.add_argument('--fid-Ok', type=float, default=None, required=False,
        help='Omega_k(z=0) of fiducial LambdaCDM cosmology')

    parser.add_argument('--fid-wl', type=float, default=None, required=False,
        help='Equation of state of dark energy of fiducial LambdaCDM cosmology')

    parser.add_argument('--z-ref', type=float, required=False, default=None,
        help='Power-spectrum redshift, if not given use the one from the config file')

    parser.add_argument('--plot', action='store_true', required=False,
        help='Plot the resulting correlation functions and power-spectra')

    args = parser.parse_args(cmdargs)

    ### Parameters kmin and kmax to get exactly same as DR12
    minkh = 1.e-4
    maxkh = 1.1525e3
    npoints = 814

    userprint('INFO: running CAMB on {}'.format(args.ini))
    pars = camb.read_ini(os.path.expandvars(args.ini))
    pars.Transfer.kmax = maxkh
    if not args.z_ref is None:
        pars.Transfer.PK_redshifts[0] = args.z_ref
    if not args.H0 is None:
        pars.H0 = args.H0
    if not args.fid_Ok is None:
        pars.omk = args.fid_Ok
    if not args.fid_wl is None:
        pars.DarkEnergy.w = args.fid_wl

    results = camb.get_results(pars)
    k, z, pk = results.get_matter_power_spectrum(minkh=minkh, maxkh=pars.Transfer.kmax, npoints=npoints)
    pk = pk[1]
    pars = results.Params
    pars2 = results.get_derived_params()

    ### Save the parameters
    cat = {}
    cat['H0'] = pars.H0
    cat['ombh2'] = pars.ombh2
    cat['omch2'] = pars.omch2
    cat['omnuh2'] = pars.omnuh2
    cat['OK'] = pars.omk
    cat['OL'] = results.get_Omega('de')
    cat['ORPHOTON'] = results.get_Omega('photon')
    cat['ORNEUTRI'] = results.get_Omega('neutrino')
    cat['OR'] = cat['ORPHOTON']+cat['ORNEUTRI']
    cat['OM'] = (cat['ombh2']+cat['omch2']+cat['omnuh2'])/(cat['H0']/100.)**2
    cat['W'] = pars.DarkEnergy.w
    cat['TCMB'] = pars.TCMB
    cat['NS'] = pars.InitPower.ns
    cat['ZREF'] = pars.Transfer.PK_redshifts[0]
    cat['SIGMA8_ZREF'] = results.get_sigma8()[0]
    cat['SIGMA8_Z0'] = results.get_sigma8()[1]
    cat['F_ZREF'] = results.get_fsigma8()[0]/results.get_sigma8()[0]
    cat['F_Z0'] = results.get_fsigma8()[1]/results.get_sigma8()[1]
    cat['ZDRAG'] = pars2['zdrag']
    cat['RDRAG'] = pars2['rdrag']

    c = SPEED_LIGHT/1000. ## km/s
    h = cat['H0']/100.
    dh = c/(results.hubble_parameter(cat['ZREF'])/h)
    dm = (1.+cat['ZREF'])*results.angular_diameter_distance(cat['ZREF'])*h
    cat['DH'] = dh
    cat['DM'] = dm
    cat['DHoRD'] = cat['DH']/(cat['RDRAG']*h)
    cat['DMoRD'] = cat['DM']/(cat['RDRAG']*h)

    ### Get the Side-Bands
    ### Follow 2.2.1 of Kirkby et al. 2013: https://arxiv.org/pdf/1301.3456.pdf
    coef_Planck2015 = (cat['H0']/67.31)*(cat['RDRAG']/147.334271564563)
    sb1_rmin = 50.*coef_Planck2015
    sb1_rmax = 82.*coef_Planck2015
    sb2_rmin = 150.*coef_Planck2015
    sb2_rmax = 190.*coef_Planck2015
    xi = nbodykit.cosmology.correlation.pk_to_xi(k,pk)
    r = np.logspace(-7., 3.5, 10000)
    xi = xi(r)

    def f_xiSB(r,am3,am2,am1,a0,a1):
        par = [am3,am2,am1,a0,a1]
        model = np.zeros((len(par),r.size))
        tw = r!=0.
        model[0,tw] = par[0]/r[tw]**3
        model[1,tw] = par[1]/r[tw]**2
        model[2,tw] = par[2]/r[tw]**1
        model[3,tw] = par[3]
        model[4,:] = par[4]*r
        model = np.array(model)
        return model.sum(axis=0)

    w = ((r>=sb1_rmin) & (r<sb1_rmax)) | ((r>=sb2_rmin) & (r<sb2_rmax))
    sigma = 0.1*np.ones(xi.size)
    sigma[(r>=sb1_rmin-2.) & (r<sb1_rmin+2.)] = 1.e-6
    sigma[(r>=sb2_rmax-2.) & (r<sb2_rmax+2.)] = 1.e-6
    popt, pcov = curve_fit(f_xiSB, r[w], xi[w], sigma=sigma[w])

    model = f_xiSB(r, *popt)
    xiSB = xi.copy()
    ww = (r>=sb1_rmin) & (r<sb2_rmax)
    xiSB[ww] = model[ww]

    pkSB = nbodykit.cosmology.correlation.xi_to_pk(r,xiSB,extrap=True)
    pkSB = pkSB(k)
    pkSB *= pk[-1]/pkSB[-1]

    out = fitsio.FITS(args.out,'rw',clobber=True)
    head = [{'name':k,'value':float(v)} for k,v in cat.items() ]
    out.write([k,pk,pkSB],names=['K','PK','PKSB'],header=head,extname='PK')
    out.close()

    if args.plot:
        plt.plot(r,xi*r**2,label='Full')
        w = (r>=sb1_rmin) & (r<sb1_rmax)
        plt.plot(r[w],xi[w]*r[w]**2,label='SB1')
        w = (r>=sb2_rmin) & (r<sb2_rmax)
        plt.plot(r[w],xi[w]*r[w]**2,label='SB2')
        plt.plot(r,xiSB*r**2,label='noBAO')
        plt.xlabel(r'$r\,[h^{-1}\,\mathrm{Mpc}]$')
        plt.ylabel(r'$r^{2}\,\xi(r)$')
        plt.legend()
        plt.grid()
        plt.show()

        plt.plot(k,pk,label='Full')
        plt.plot(k,pkSB,label='noBAO')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(r'$k$')
        plt.ylabel(r'$P(k)$')
        plt.legend()
        plt.grid()
        plt.show()

        plt.plot(k,pk-pkSB,label='BAO')
        plt.xscale('log')
        plt.xlabel(r'$k$')
        plt.ylabel(r'$P(k)$')
        plt.legend()
        plt.grid()
        plt.show()


if __name__ == '__main__':
    cmdargs=sys.argv[1:]
    main(cmdargs)