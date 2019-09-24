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

    parser.add_argument('--DD-files', type=str,nargs="*", default=None, required=False,
        help='Files of the data x data auto-correlation')

    parser.add_argument('--RR-files', type=str,nargs="*", default=None, required=False,
        help='Files of the random x random auto-correlation')

    parser.add_argument('--DR-files', type=str,nargs="*", default=None, required=False,
        help='Files of the data x random auto-correlation')

    parser.add_argument('--RD-files', type=str,nargs="*", default=None, required=False,
        help='Files of the random x data auto-correlation')

    parser.add_argument("--coadd-out-DD",type=str,default=None,required=False,
        help="coadded (not exported) DD output file")

    parser.add_argument("--coadd-out-RR",type=str,default=None,required=False,
        help="coadded (not exported) RR output file")

    parser.add_argument("--coadd-out-DR",type=str,default=None,required=False,
        help="coadded (not exported) DR output file")

    parser.add_argument("--coadd-out-RD",type=str,default=None,required=False,
        help="coadded (not exported) RD output file")

    #Not sure of the purpose of these: does it make much of a difference?
    parser.add_argument('--xDD-files', type=str, default=None, required=False,
        help='Files of the data_1 x data_2 cross-correlation')

    parser.add_argument('--xRR-files', type=str, default=None, required=False,
        help='Files of the random_1 x random_2 cross-correlation')

    parser.add_argument('--xD1R2-files', type=str, default=None, required=False,
        help='Files of the data_1 x random_2 cross-correlation')

    parser.add_argument('--xR1D2-files', type=str, default=None, required=False,
        help='Files of the random_1 x data_2 cross-correlation')

    parser.add_argument('--do-not-smooth-cov', action='store_true', default=False,
        help='Do not smooth the covariance matrix from sub-sampling')

    parser.add_argument('--get-cov-from-poisson', action='store_true', default=False,
        help='Get covariance matrix from Poisson statistics')

    parser.add_argument('--cov', type=str, default=None, required=False,
        help='Path to a covariance matrix file (if not provided it will be calculated by subsampling or from Poisson statistics)')

    args = parser.parse_args()

    ### Auto or cross correlation?
    if (args.DD_files is None and args.xDD_files is None) or (not args.DD_files is None and not args.xDD_files is None) or (not args.cov is None and not args.get_cov_from_poisson):
        print('ERROR: No data files, or both auto and cross data files, or two different method for covariance')
        sys.exit()
    elif not args.DD_files is None:
        corr = 'AUTO'
        lst_file = {'DD':args.DD_files, 'RR':args.RR_files, 'DR':args.DR_files, 'RD':args.RD_files}
    elif not args.xDD_files is None:
        # TODO: Test if picca_co.py and export_co.py work for cross
        corr = 'CROSS'
        lst_file = {'xDD':args.xDD_files, 'xRR':args.xRR_files, 'xD1R2':args.xD1R2_files, 'xR1D2':args.xR1D2_files}

    ### Read files
    data = {}
    for type_corr, fi in lst_file.items():
        print("looking at correlation {}".format(type_corr),end="\r")

        #Open up the first file to set up arrays etc.
        f = fi[0]
        h = fitsio.FITS(f)
        head = h[1].read_header()
        if type_corr in ['DD','xDD']:
            # Assume that same nt, np, rtmax, rpmin, rpmax, nside are used for each z bin correlations.
            for k in ['NT','NP','RTMAX','RPMIN','RPMAX','NSIDE']:
                data[k] = head[k]
            for k in ['RP','RT','Z','NB']:
                data[k] = sp.zeros(sp.array(h[1][k][:]).shape)
            data['WET'] = sp.zeros(sp.array(h[1]['RP'][:]).shape)

        # Assume that same nside, healpix scheme and footprint are used for all
        #correlations of each type.
        data[type_corr] = {}
        data[type_corr]['HLPXSCHM'] = h[2].read_header()['HLPXSCHM']
        w = sp.array(h[2]['WE'][:]).sum(axis=1)>0.
        data[type_corr]['HEALPID'] = h[2]['HEALPID'][:][w]
        data[type_corr]['WE'] = sp.zeros(h[2]['WE'][:].shape)
        data[type_corr]['NBS'] = sp.zeros(h[2]['NB'][:].shape)
        for k in ['NT','NP','RTMAX','RPMIN','RPMAX','NSIDE']:
            data[type_corr][k] = data[k]
        for k in ['RP','RT','Z','NB','WET']:
            data[type_corr][k] = data[k]

        #Picca saves the output file from picca_co.py with head['NOBJ'] as the
        #total number of objects in the catalog *before* any redshift cuts are
        #applied. Thus we do not need to sum the values from each of the files.
        #We assume that all files from the same correlation type used the same
        #catalog, thus have the same nObj.
        if type_corr in ['DD','RR']:
            nObj = head['NOBJ']
            data[type_corr]['NOBJ'] = nObj
        else:
            nObj  = head['NOBJ']
            nObj2 = head['NOBJ2']
            data[type_corr]['NOBJ'] = nObj
            data[type_corr]['NOBJ2'] = nObj2

        h.close()

        for f in fi:
            print("coadding file {}".format(f),end="\r")

            h = fitsio.FITS(f)
            head = h[1].read_header()

            we_aux = h[2]["WE"][:]
            wet_aux = we_aux.sum(axis=0)
            for k in ['RP','RT','Z']:
                data[type_corr][k] += sp.array(h[1][k][:])*wet_aux
            data[type_corr]['NB']  += sp.array(h[1]['NB'][:])
            data[type_corr]['WET'] += wet_aux

            #Check that the HEALPix pixels are the same.
            if (h[2]['HEALPID'][:] == data[type_corr]['HEALPID']).all():
                data[type_corr]['WE'] += h[2]['WE'][:]
                data[type_corr]['NBS'] += h[2]['NB'][:]
            elif set(h[2]['HEALPID'][:]) == set(data[type_corr]['HEALPID']):
                # TODO: Add in check to see if they're the same but just ordered differently.
                raise IOError('Correlations\' pixels are not ordered in the same way!')
            else:
                raise IOError('Correlations do not have the same footprint!')

            h.close()


        if type_corr in ['DD','RR']:
            coef = nObj*(nObj-1)
        else:
            coef = nObj*nObj2
        data['COEF'] = coef

        #Correctly normalise all of the data.
        for k in ['RP','RT','Z']:
            data[type_corr][k] /= data[type_corr]['WET']

        #Move the DD data to a special location.
        if type_corr in ['DD','xDD']:
            for k in ['RP','RT','Z','NB','WET']:
                data[k] = data[type_corr][k]

        data[type_corr]['WE'] /= coef

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
        d2r1 = data['xR1D2']['WE'].sum(axis=0)
        w = rr>0.
        da = sp.zeros(dd.size)
        da[w] = (dd[w]+rr[w]-d1r2[w]-d2r1[w])/rr[w]

    def save_co_file(outname,data,type_corr):

        out = fitsio.FITS(outname,'rw',clobber=True)
        head = [ {'name':'RPMIN','value':data['RPMIN'],'comment':'Minimum r-parallel [h^-1 Mpc]'},
            {'name':'RPMAX','value':data['RPMAX'],'comment':'Maximum r-parallel [h^-1 Mpc]'},
            {'name':'RTMAX','value':data['RTMAX'],'comment':'Maximum r-transverse [h^-1 Mpc]'},
            {'name':'NP','value':data['NP'],'comment':'Number of bins in r-parallel'},
            {'name':'NT','value':data['NT'],'comment':'Number of bins in r-transverse'},
            {'name':'NSIDE','value':data['NSIDE'],'comment':'Healpix nside'},
            {'name':'TYPECORR','value':type_corr,'comment':'Correlation type'},
            {'name':'NOBJ','value':data[type_corr]['NOBJ'],'comment':'Number of objects'},
        ]
        if type_corr in ['DR','RD']:
            head += [{'name':'NOBJ2','value':data[type_corr]['NOBJ2'],'comment':'Number of objects 2'}]

        comment = ['R-parallel','R-transverse','Redshift','Number of pairs']
        units = ['h^-1 Mpc','h^-1 Mpc','','']
        names = ['RP','RT','Z','NB']
        out.write([data[type_corr][k] for k in names],names=names,header=head,comment=comment,units=units,extname='ATTRI')

        comment = ['Healpix index', 'Sum of weight', 'Number of pairs']
        head2 = [{'name':'HLPXSCHM','value':'RING','comment':'healpix scheme'}]
        names = ['HEALPID','WE','NB']
        hold_names = ['HEALPID','WE','NBS']
        out.write([data[type_corr][k] for k in hold_names],names=names,header=head2,comment=comment,extname='COR')
        out.close()

        return

    if (args.coadd_out_DD is not None) and ('DD' in data.keys()):
        save_co_file(args.coadd_out_DD,data,'DD')
    if (args.coadd_out_RD is not None) and ('RD' in data.keys()):
        save_co_file(args.coadd_out_RD,data,'RD')
    if (args.coadd_out_DR is not None) and ('DR' in data.keys()):
        save_co_file(args.coadd_out_DR,data,'DR')
    if (args.coadd_out_RR is not None) and ('RR' in data.keys()):
        save_co_file(args.coadd_out_RR,data,'RR')


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
            d2r1 = data['xR1D2']['WE']
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
