#!/usr/bin/env python

from __future__ import print_function
import sys
import fitsio
import scipy as sp
import scipy.linalg
import argparse

from picca.utils import smooth_cov, cov, print

if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Export auto and cross-correlation of catalog of objects for the fitter.')

    parser.add_argument('--out', type=str, default=None, required=True,
        help='Output file name')

    parser.add_argument('--DD-file', type=str, default=None, required=False,
        help='File of the data x data auto-correlation')

    parser.add_argument('--RR-file', type=str, default=None, required=False,
        help='File of the random x random auto-correlation')

    parser.add_argument('--DR-file', type=str, default=None, required=False,
        help='File of the data x random auto-correlation')

    parser.add_argument('--RD-file', type=str, default=None, required=False,
        help='File of the random x data auto-correlation')

    parser.add_argument('--xDD-file', type=str, default=None, required=False,
        help='File of the data_1 x data_2 cross-correlation')

    parser.add_argument('--xRR-file', type=str, default=None, required=False,
        help='File of the random_1 x random_2 cross-correlation')

    parser.add_argument('--xD1R2-file', type=str, default=None, required=False,
        help='File of the data_1 x random_2 cross-correlation')

    parser.add_argument('--xD2R1-file', type=str, default=None, required=False,
        help='File of the data_2 x random_1 cross-correlation')

    parser.add_argument('--do-not-smooth-cov', action='store_true', default=False,
        help='Do not smooth the covariance matrix from sub-sampling')

    parser.add_argument('--get-cov-from-poisson', action='store_true', default=False,
        help='Get covariance matrix from Poisson statistics')

    parser.add_argument('--cov', type=str, default=None, required=False,
        help='Path to a covariance matrix file (if not provided it will be calculated by subsampling or from Poisson statistics)')

    args = parser.parse_args()

    ### Auto or cross correlation?
    if (args.DD_file is None and args.xDD_file is None) or (not args.DD_file is None and not args.xDD_file is None) or (not args.cov is None and not args.get_cov_from_poisson):
        print('ERROR: No data files, or both auto and cross data files, or two different method for covariance')
        sys.exit()
    elif not args.DD_file is None:
        corr = 'AUTO'
        lst_file = {'DD':args.DD_file, 'RR':args.RR_file, 'DR':args.DR_file, 'RD':args.RD_file}
    elif not args.xDD_file is None:
        # TODO: Test if do_co.py and export_co.py work for cross
        corr = 'CROSS'
        lst_file = {'xDD':args.xDD_file, 'xRR':args.xRR_file, 'xD1R2':args.D1R2_file, 'xD2R1':args.D2R1_file}

    ### Read files
    data = {}
    for type_corr, f in lst_file.items():
        h = fitsio.FITS(f)
        head = h[1].read_header()

        if type_corr in ['DD','RR']:
            nbObj = head['NOBJ']
            coef = nbObj*(nbObj-1)
        else:
            nbObj  = head['NOBJ']
            nbObj2 = head['NOBJ2']
            coef = nbObj*nbObj2

        if type_corr in ['DD','xDD']:
            data['COEF'] = coef
            for k in ['NT','NP','RTMAX','RPMIN','RPMAX']:
                data[k] = head[k]
            for k in ['RP','RT','Z','NB']:
                data[k] = sp.array(h[1][k][:])

        data[type_corr] = {}
        data[type_corr]['NSIDE'] = head['NSIDE']
        data[type_corr]['HLPXSCHM'] = h[2].read_header()['HLPXSCHM']
        w = sp.array(h[2]['WE'][:]).sum(axis=1)>0.
        if w.sum()!=w.size:
            print('INFO: {} sub-samples were empty'.format(w.size-w.sum()))
        data[type_corr]['HEALPID'] = h[2]['HEALPID'][:][w]
        data[type_corr]['WE'] = h[2]['WE'][:][w]/coef
        h.close()

    ### Get correlation
    if corr=='AUTO':
        dd = data['DD']['WE'].sum(axis=0)
        rr = data['RR']['WE'].sum(axis=0)
        dr = data['DR']['WE'].sum(axis=0)
        rd = data['RD']['WE'].sum(axis=0)
        w = rr>0.
        da = sp.zeros(dd.size)
        da[w] = (dd[w]+rr[w]-rd[w]-dr[w])/rr[w]
    else:
        dd = data['xDD']['WE'].sum(axis=0)
        rr = data['xRR']['WE'].sum(axis=0)
        d1r2 = data['xD1R2']['WE'].sum(axis=0)
        d2r1 = data['xD2R1']['WE'].sum(axis=0)
        w = rr>0.
        da = sp.zeros(dd.size)
        da[w] = (dd[w]+rr[w]-d1r2[w]-d2r1[w])/rr[w]
    data['DA'] = da
    data['corr_DD'] = dd
    data['corr_RR'] = rr

    ### Covariance matrix
    if not args.cov is None:
        print('INFO: Read covariance from file')
        h = fitsio.FITS(args.cov)
        data['CO'] = h[1]['CO'][:]
        h.close()
    elif args.get_cov_from_poisson:
        print('INFO: Compute covariance from Poisson statistics')
        w = data['corr_RR']>0.
        co = sp.zeros(data['corr_DD'].size)
        co[w] = (data['COEF']/2.*data['corr_DD'][w])**2/(data['COEF']/2.*data['corr_RR'][w])**3
        data['CO'] = sp.diag(co)
    else:
        print('INFO: Compute covariance from sub-sampling')

        ### To have same number of HEALPix
        for d1 in list(lst_file.keys()):
            for d2 in list(lst_file.keys()):

                if data[d1]['NSIDE']!=data[d2]['NSIDE']:
                    print('ERROR: NSIDE are different: {} != {}'.format(data[d1]['NSIDE'],data[d2]['NSIDE']))
                    sys.exit()
                if data[d1]['HLPXSCHM']!=data[d2]['HLPXSCHM']:
                    print('ERROR: HLPXSCHM are different: {} != {}'.format(data[d1]['HLPXSCHM'],data[d2]['HLPXSCHM']))
                    sys.exit()

                w = sp.logical_not( sp.in1d(data[d1]['HEALPID'],data[d2]['HEALPID']) )
                if w.sum()!=0:
                    print('WARNING: HEALPID are different by {} for {}:{} and {}:{}'.format(w.sum(),d1,data[d1]['HEALPID'].size,d2,data[d2]['HEALPID'].size))
                    new_healpix = data[d1]['HEALPID'][w]
                    nb_new_healpix = new_healpix.size
                    nb_bins = data[d2]['WE'].shape[1]
                    data[d2]['HEALPID'] = sp.append(data[d2]['HEALPID'],new_healpix)
                    data[d2]['WE'] = sp.append(data[d2]['WE'],sp.zeros((nb_new_healpix,nb_bins)),axis=0)

        ### Sort the data by the healpix values
        for d1 in list(lst_file.keys()):
            sort = sp.array(data[d1]['HEALPID']).argsort()
            data[d1]['WE'] = data[d1]['WE'][sort]
            data[d1]['HEALPID'] = data[d1]['HEALPID'][sort]

        if corr=='AUTO':
            dd = data['DD']['WE']
            rr = data['RR']['WE']
            dr = data['DR']['WE']
            rd = data['RD']['WE']
            w = rr>0.
            da = sp.zeros(dd.shape)
            da[w] = (dd[w]+rr[w]-dr[w]-rd[w])/rr[w]
            we = data['DD']['WE']
        else:
            dd = data['xDD']['WE']
            rr = data['xRR']['WE']
            d1r2 = data['xD1R2']['WE']
            d2r1 = data['xD2R1']['WE']
            w = rr>0.
            da = sp.zeros(dd.shape)
            da[w] = (dd[w]+rr[w]-d1r2[w]-d2r1[w])/rr[w]
            we = data['xDD']['WE']
        data['HLP_DA'] = da
        data['HLP_WE'] = we

        if args.do_not_smooth_cov:
            print('INFO: The covariance will not be smoothed')
            co = cov(da,we)
        else:
            print('INFO: The covariance will be smoothed')
            binSizeRP = (data['RPMAX']-data['RPMIN']) / data['NP']
            binSizeRT = (data['RTMAX']-0.) / data['NT']
            co = smooth_cov(da,we,data['RP'],data['RT'],drp=binSizeRP,drt=binSizeRT)
        data['CO'] = co

    try:
        scipy.linalg.cholesky(data['CO'])
    except scipy.linalg.LinAlgError:
        print('WARNING: Matrix is not positive definite')

    ### Distortion matrix
    data['DM'] = sp.eye(data['DA'].size)

    ### Save
    h = fitsio.FITS(args.out,'rw',clobber=True)
    head = {}
    if corr=='AUTO':
        nside = data['DD']['NSIDE']
    else:
        nside = data['xDD']['NSIDE']
    head = [ {'name':'RPMIN','value':data['RPMIN'],'comment':'Minimum r-parallel'},
        {'name':'RPMAX','value':data['RPMAX'],'comment':'Maximum r-parallel'},
        {'name':'RTMAX','value':data['RTMAX'],'comment':'Maximum r-transverse'},
        {'name':'NP','value':data['NP'],'comment':'Number of bins in r-parallel'},
        {'name':'NT','value':data['NT'],'comment':'Number of bins in r-transverse'},
        {'name':'NSIDE','value':nside,'comment':'Healpix nside'}
    ]
    lst = ['RP','RT','Z','DA','CO','DM','NB']
    comment=['R-parallel','R-transverse','Redshift','Correlation','Covariance matrix','Distortion matrix','Number of pairs']
    h.write([ data[k] for k in lst ], names=lst, header=head, comment=comment,extname='COR')

    if args.cov is None and not args.get_cov_from_poisson:
        if corr=='AUTO':
            HLPXSCHM = data['DD']['HLPXSCHM']
            hep = data['DD']['HEALPID']
        else:
            HLPXSCHM = data['xDD']['HLPXSCHM']
            hep = data['xDD']['HEALPID']
        head2 = [{'name':'HLPXSCHM','value':HLPXSCHM,'comment':'healpix scheme'}]
        comment=['Healpix index', 'Sum of weight', 'Correlation']
        h.write([hep,data['HLP_WE'],data['HLP_DA']],names=['HEALPID','WE','DA'],header=head2,comment=comment,extname='SUB_COR')

    h.close()
