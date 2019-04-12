#!/usr/bin/env python

import os
import argparse
import fitsio
import scipy as sp
from scipy.constants import speed_of_light
from scipy.optimize import curve_fit
import camb
import nbodykit.cosmology.correlation

if __name__ == '__main__':

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

    args = parser.parse_args()

    ## Parameters kmin and kmax to get exactly same as DR12
    minkh = 1.e-4
    maxkh = 1.1525e3
    npoints = 814

    print('INFO: running CAMB on {}'.format(args.ini))
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
    ### TODO: What do we do with OM?
    ### =sum(ombh2,omch2,omnuh2)? or =1-OL-OK?
    cat = {}
    cat['H0'] = pars.H0
    cat['ombh2'] = pars.ombh2
    cat['omch2'] = pars.omch2
    cat['omnuh2'] = pars.omnuh2
    cat['OK'] = pars.omk
    cat['OL'] = results.get_Omega('de')
    cat['OM'] = (cat['ombh2']+cat['omch2']+cat['omnuh2'])/(cat['H0']/100.)**2
    cat['W'] = pars.DarkEnergy.w
    cat['TCMB'] = pars.TCMB
    cat['NS'] = pars.InitPower.ns
    cat['ZREF'] = pars.Transfer.PK_redshifts[0]
    cat['SIGMA8'] = results.get_sigma8()[1]
    cat['F'] = results.get_fsigma8()[0]/results.get_sigma8()[0]
    cat['ZDRAG'] = pars2['zdrag']
    cat['RDRAG'] = pars2['rdrag']

    c = speed_of_light/1000. ## km/s
    h = cat['H0']/100.
    dh = c/(results.hubble_parameter(cat['ZREF'])/h)
    dm = (1.+cat['ZREF'])*results.angular_diameter_distance(cat['ZREF'])*h
    cat['DH'] = dh
    cat['DM'] = dm
    cat['DHoRD'] = cat['DH']/(cat['RDRAG']*h)
    cat['DMoRD'] = cat['DM']/(cat['RDRAG']*h)

    ### Get the Side-Bands
    ### Follow 2.2.1 of Kirkby et al. 2013: https://arxiv.org/pdf/1301.3456.pdf
    xi = nbodykit.cosmology.correlation.pk_to_xi(k,pk)
    r = 10**sp.linspace(-7.,3.5,1e4)
    xi = xi(r)

    def f_xiSB(r,am3,am2,am1,a0,a1):
        par = [am3,am2,am1,a0,a1]
        model = sp.zeros((len(par),r.size))
        tw = r!=0.
        model[0,tw] = par[0]/r[tw]**3
        model[1,tw] = par[1]/r[tw]**2
        model[2,tw] = par[2]/r[tw]**1
        model[3,tw] = par[3]
        model[4,:] = par[4]*r
        model = sp.array(model)
        return model.sum(axis=0)

    w = ((r>=50.) & (r<82.)) | ((r>=150.) & (r<190.))
    sigma = 0.1*sp.ones(xi.size)
    sigma[(r>=48.) & (r<52.)] = 1.e-6
    sigma[(r>=188.) & (r<192.)] = 1.e-6
    popt, pcov = curve_fit(f_xiSB, r[w], xi[w], sigma=sigma[w])

    model = f_xiSB(r, *popt)
    xiSB = xi.copy()
    ww = (r>=50.) & (r<190.)
    xiSB[ww] = model[ww]

    pkSB = nbodykit.cosmology.correlation.pk_to_xi(r,xiSB,extrap=True)
    pkSB = pkSB(k)
    pkSB *= pk[-1]/pkSB[-1]

    out = fitsio.FITS(args.out,'rw',clobber=True)
    head = [{'name':k,'value':float(v)} for k,v in cat.items() ]
    out.write([k,pk,pkSB],names=['K','PK','PKSB'],header=head,extname='PK')
    out.close()
