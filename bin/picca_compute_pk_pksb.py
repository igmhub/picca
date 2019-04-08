#!/usr/bin/env python

import scipy as sp
import argparse
import subprocess
import fitsio

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
    #subprocess.run('camb {} > {}'.format(args.ini,cat['output_root']+'_out.txt'),shell=True)

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
    pksb = d[:,1]

    out = fitsio.FITS(args.out,'rw',clobber=True)
    head = [{'name':k,'value':float(v)} for k,v in cat.items() if k not in ['transfer_matterpower','output_root'] ]
    out.write([k,pk,pksb],names=['K','PK','PKSB'],header=head,extname='PK')
    out.close()
