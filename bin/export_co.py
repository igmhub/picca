#!/usr/bin/env python

import fitsio
import scipy as sp
import argparse
import sys
import glob

if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Export auto and cross-correlation of catalog of objects for the fitter.')

    parser.add_argument('--DD-file', type=str, default=None, required=True,
        help='File of the data-data correlation')

    parser.add_argument('--RR-DR-dir', type=str, default=None, required=True,
        help='Path directory to all data-random and random-random correlations')

    parser.add_argument('--out', type=str, default=None, required=True,
        help='Output file name')

    parser.add_argument('--cov', type=str, default=None, required=False,
        help='Covariance matrix file (if not provided it will be calculated by Poisson)')


    args = parser.parse_args()

    ### DD
    h = fitsio.FITS(args.DD_file)
    head = h[1].read_header()
    type_corr = head['TYPECORR'].replace(' ','')
    if type_corr not in ['DD','xDD']:
        print("ERROR: DD-file is not data-data : "+type_corr)
        h.close()
        sys.exit()
    nt = head['NT']
    np = head['NP']
    rt_max = head['RTMAX']
    rp_min = head['RPMIN']
    rp_max = head['RPMAX']
    nbObj = head['NOBJ']
    rp = sp.array(h[1]['RP'][:])
    rt = sp.array(h[1]['RT'][:])
    z  = sp.array(h[1]['Z'][:])
    nb = sp.array(h[1]['NB'][:])
    we = sp.array(h[2]['WE'][:]).sum(axis=0)
    dd = we
    coefDD = nbObj*(nbObj-1)/2.
    dd /= coefDD
    h.close()
    dm = sp.eye(dd.size)

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
            we /= nbObj*(nbObj-1)/2.
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
        if rand[tc]['nb']==0:
            print("ERROR: no DR or RR: "+tc)
            sys.exit()
        rand[tc]['data'] /= rand[tc]['nb']

    ###
    if type_corr=='DD':
        dr = rand['DR']['data']
        rr = rand['RR']['data']
        w = rr>0.
        da = sp.zeros(dd.size)
        da[w] = (dd[w]+rr[w]-2*dr[w])/rr[w]
    else:
        d1r2 = rand['xD1R2']['data']
        d2r1 = rand['xD2R1']['data']
        rr   = rand['xRR']['data']
        w = rr>0.
        da = sp.zeros(dd.size)
        da[w] = (dd[w]+rr[w]-d1r2[w]-d2r1[w])/rr[w]

    ### Covariance matrix
    if args.cov is not None:
        hh = fitsio.FITS(args.cov)
        co = hh[1]['CO'][:]
        hh.close()
    else:
        w = rr>0.
        co = sp.zeros(dd.size)
        co[w] = (coefDD*dd[w])**2/(coefDD*rr[w])**3
        co = sp.diag(co)

    ### Save
    h = fitsio.FITS(args.out,'rw',clobber=True)
    head = [ {'name':'RPMIN','value':rp_min,'comment':'Minimum r-parallel'},
        {'name':'RPMAX','value':rp_max,'comment':'Maximum r-parallel'},
        {'name':'RTMAX','value':rt_max,'comment':'Maximum r-transverse'},
        {'name':'NP','value':np,'comment':'Number of bins in r-parallel'},
        {'name':'NT','value':nt,'comment':'Number of bins in r-transverse'}
    ]
    comment = ['R-parallel','R-transverse','Redshift','Correlation','Covariance matrix','Distortion matrix','Number of pairs']
    h.write([rp,rt,z,da,co,dm,nb],names=['RP','RT','Z','DA','CO','DM','NB'],comment=comment,header=head,extname='COR')
    if type_corr=='DD':
        h.write([dd,rr,dr],names=['DD','RR','DR'],comment=['Data-Data','Random-Random','Data-Random'],extname='ELEMENT')
    else:
        h.write([dd,rr,d1r2,d2r1],names=['DD','RR','D1R2','D2R1'],comment=['Data1-Data2','Random1-Random2','Data1-Random2','Data2-Random1'],extname='ELEMENT')
    h.close()
