#!/usr/bin/env python

import argparse
import subprocess
import fitsio
import scipy as sp
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import nbodykit.cosmology.correlation

from picca.fitter2 import utils

if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-i','--ini', type=str, required=True,
        help='Input config file for CAMB')

    parser.add_argument('-o','--out', type=str, required=True,
        help='Output FITS file')

    args = parser.parse_args()

    from_config2fits = {
        'hubble':'H0',
        'ombh2':'ombh2',
        'omch2':'omch2',
        'omnuh2':'omnuh2',
        'omk':'OK',
        'w':'W',
        'temp_cmb':'TCMB',
        'scalar_spectral_index(1)':'NS',
        'transfer_redshift(1)':'ZREF',
        'output_root':'output_root',
        'transfer_matterpower(1)':'transfer_matterpower'}
    from_output2fits = {                                            
        'Om_m(incOm_u)':'OM',
        'Om_darkenergy':'OL',
        'atz0.000sigma8(allmatter)':'SIGMA8',
        'zdrag':'ZDRAG',
        'r_s(zdrag)/Mpc':'RDRAG'}

    cat = {}

    f = open(args.ini,'r')
    for l in f:
        l = l.replace(' ','').replace('\n','').split('=')
        if l[0] in from_config2fits.keys():
            cat[from_config2fits[l[0]]] = l[1]
    f.close()

    print('INFO: running CAMB on {}'.format(args.ini))
    try:
        import camb
    except ImportError:
        print('ERROR: CAMB can not be found')
    subprocess.run('camb {} > {}'.format(args.ini,cat['output_root']+'_out.txt'),shell=True)

    ### TODO: understand why O_m+O_L+O_k!=1.
    f = open(cat['output_root']+'_out.txt','r')
    for l in f:
        l = l.replace(' ','').replace('\n','').replace('atz=','atz').split('=')
        if l[0] in from_output2fits.keys():
            cat[from_output2fits[l[0]]] = l[1]
    f.close()

    ### TODO: get P(k) decomposition
    d = sp.loadtxt('{}_{}'.format(cat['output_root'],cat['transfer_matterpower']))
    k = d[:,0]
    pk = d[:,1]

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
    head = [{'name':k,'value':float(v)} for k,v in cat.items() if k not in ['transfer_matterpower','output_root'] ]
    out.write([k,pk,pkSB],names=['K','PK','PKSB'],header=head,extname='PK')
    out.close()
