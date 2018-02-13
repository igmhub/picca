#!/usr/bin/env python

import fitsio
import scipy as sp
import argparse
import sys
import glob

if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--DD-file', type = str, default = None, required=True,
                        help = 'file of the data-data correlation')

    parser.add_argument('--RR-DR-dir', type = str, default = None, required=True,
                        help = 'path directory to all data-random and random-random correlations')

    parser.add_argument('--out', type = str, default = None, required=True,
                        help = 'output file')

    args = parser.parse_args()

    ### DD
    h = fitsio.FITS(args.DD_file)
    head = h[1].read_header()
    type_corr = head['TYPECORR'].replace(' ','')
    if type_corr not in ['DD','xDD']:
        print("ERROR: DD-file is not data-data : "+type_corr)
        h.close()
        sys.exit()
    nbObj = head['NOBJ']
    rp = sp.array(h[1]['RP'][:])
    rt = sp.array(h[1]['RT'][:])
    z  = sp.array(h[1]['Z'][:])
    nb = sp.array(h[1]['NB'][:])
    we = sp.array(h[2]['WE'][:]).sum(axis=0)
    dd = we
    dd /= nbObj*nbObj/2.
    h.close()
    dm = sp.eye(dd.size)
    co = sp.eye(dd.size)

    ### DR and RR
    rand = {}
    if type_corr=='DD':
        rand['DR'] = {'nb':0, 'data':None}
        rand['RR'] = {'nb':0, 'data':None}
    else:
        rand['xD1R2'] = {'nb':0, 'data':None}
        rand['xD2R1'] = {'nb':0, 'data':None}
        rand['xRR']   = {'nb':0, 'data':None}
    fi = sorted(glob.glob(args.RR_DR_dir+"/*.fits.gz"))
    for f in fi:
        h = fitsio.FITS(f)

        head = h[1].read_header()
        tc = head['TYPECORR'].replace(' ','')
        if not tc in list(rand.keys()):
            print("WARNING: TYPECORR not data-random or random-random : "+tc+' : '+f)
            h.close()
            continue

        we = sp.array(h[2]['WE'][:]).sum(axis=0)
        if tc in ['RR','xRR']:
            nbObj = head['NOBJ']
            we /= nbObj*nbObj/2.
        else:
            nbObj  = head['NOBJ']
            nbObj2 = head['NOBJ2']
            we /= nbObj*nbObj2

        if rand[tc]['nb']==0:
            rand[tc]['data'] = we.copy()
        else:
            rand[tc]['data'] += we.copy()
        rand[tc]['nb'] += 1

        h.close()
    for tc in list(rand.keys()):
        rand[tc]['data'] /= rand[tc]['nb']

    ###
    if type_corr=='DD':
        dr = rand['DR']['data']
        rr = rand['RR']['data']
        w = rr>0.
        da = sp.zeros_like(dd)
        da[w] = (dd[w]+rr[w]-2*dr[w])/rr[w]
    else:
        d1r2 = rand['xD1R2']['data']
        d2r1 = rand['xD2R1']['data']
        rr   = rand['xRR']['data']
        w = rr>0.
        da = sp.zeros_like(dd)
        da[w] = (dd[w]+rr[w]-d1r2[w]-d2r1[w])/rr[w]

    ### Save
    h = fitsio.FITS(args.out,'rw',clobber=True)
    h.write([rp,rt,z,da,co,dm,nb],names=['RP','RT','Z','DA','CO','DM','NB'])
    h.close()
