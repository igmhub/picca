#!/usr/bin/env python

import sys
import scipy as sp
import argparse
import glob
import fitsio

if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                        description='Export auto-correlation object x object and cross-correlation object_1 x object_2 for the fitter')

    parser.add_argument('--out', type=str, default=None, required=True,
                        help = 'Output path')

    parser.add_argument('--DD-file', type=str, default=None, required=False,
                        help = 'File of the data x data auto-correlation')

    parser.add_argument('--RR-file', type=str, default=None, required=False,
                        help = 'File of the random x random auto-correlation')

    parser.add_argument('--DR-file', type=str, default=None, required=False,
                        help = 'File of the data x random auto-correlation')

    parser.add_argument('--xDD-file', type=str, default=None, required=False,
                        help = 'File of the data_1 x data_2 cross-correlation')

    parser.add_argument('--xRR-file', type=str, default=None, required=False,
                        help = 'File of the random_1 x random_2 cross-correlation')

    parser.add_argument('--xD1R2-file', type=str, default=None, required=False,
                        help = 'File of the data_1 x random_2 cross-correlation')

    parser.add_argument('--xD2R1-file', type=str, default=None, required=False,
                        help = 'File of the data_2 x random_1 cross-correlation')

    parser.add_argument('--cov', type=str, default=None, required=False,
                        help = 'Path to a covariance matrix file (if not provided it will be calculated by subsampling or from Poisson statistics)')

    parser.add_argument('--get-cov-from-poisson', action='store_true', default=False,
                        help='Get covariance matrix from Poisson statistics')

    args = parser.parse_args()

    ### Auto or cross correlation?
    if (args.DD_file is None and args.xDD_file is None) or (not args.DD_file is None and not args.xDD_file is None) or (not args.cov is None and not args.get_cov_from_poisson):
        print('ERROR: No data files, or both auto and cross data files, or two different method for covariance')
        sys.exit()
    elif not args.DD_file is None:
        lst_file = [args.DD_file, args.RR_file, args.DR_file]
    elif not args.xDD_file is None:
        lst_file = [args.xDD_file, args.xRR_file, args.D1R2_file, args.D2R1_file]

    ### Read files
    data = {}
    for f in lst_file:
        h = fitsio.FITS(f)
        head = h[1].read_header()
        type_corr = head['TYPECORR'].replace(' ','')

        if type_corr in ['DD','RR']:
            nbObj = head['NOBJ']
            coef = nbObj*(nbObj-1)/2.
        elif type_corr in ['DR','xDD','xRR','xD1R2','xD2R1']:
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
        data[type_corr]['HEALPID'] = sp.array(h[2]['HEALPID'][:])
        data[type_corr]['WE'] = sp.array(h[2]['WE'][:])/coef

        h.close()

    ### Get correlation
    if 'DD' in list(data.keys()):
        dd = data['DD']['WE'].sum(axis=0)
        rr = data['RR']['WE'].sum(axis=0)
        dr = data['DR']['WE'].sum(axis=0)
        w = rr>0.
        da = sp.zeros(dd.size)
        da[w] = (dd[w]+rr[w]-2*dr[w])/rr[w]
        data['corr_DR'] = dr
    else:
        coef = data['COEF']
        dd = data['xDD']['WE'].sum(axis=0)
        rr = data['xRR']['WE'].sum(axis=0)
        d1r2 = data['xD1R2']['WE'].sum(axis=0)
        d2r1 = data['xD2R1']['WE'].sum(axis=0)
        w = rr>0.
        da = sp.zeros(dd.size)
        da[w] = (dd[w]+rr[w]-d1r2[w]-d2r1[w])/rr[w]
        data['corr_xD1R2'] = d1r2
        data['corr_xD2R1'] = d2r1
    data['DA'] = da
    data['corr_DD'] = dd
    data['corr_RR'] = rr

    ### Covariance matrix
    if not args.cov is None:
        print('INFO: Read covariance from file')
        h = fitsio.FITS(args.cov)
        data['CO'] = h[1]['CO'][:]
        h.close()
    #elif not args.get_cov_from_poisson:
    #    print('INFO: Compute covariance from sub-sampling')
    #    
    else:
        print('INFO: Compute covariance from Poisson statistics')
        coef = data['COEF']
        w = data['corr_RR']>0.
        co = sp.zeros(data['corr_DD'].size)
        co[w] = (data['COEF']*data['corr_DD'][w])**2/(data['COEF']*data['corr_RR'][w])**3
        data['CO'] = sp.diag(co)

    ### Distortion matrix
    data['DM'] = sp.eye(data['DA'].size)

    ### Save
    h = fitsio.FITS(args.out,'rw',clobber=True)
    head = {}
    head['RPMIN'] = data['RPMIN']
    head['RPMAX'] = data['RPMAX']
    head['RTMAX'] = data['RTMAX']
    head['NT'] = data['NT']
    head['NP'] = data['NP']
    lst = ['RP','RT','Z','DA','CO','DM','NB']
    h.write([ data[k] for k in lst ], names=lst, header=head)
    if 'DD' in list(data.keys()):
        lst = ['DD','RR','DR']
    else:
        lst = ['DD','RR','D1R2','D2R1']
    h.write([ data['corr_'+k] for k in lst ], names=lst)
    h.close()
