#!/usr/bin/env python

from __future__ import print_function
import fitsio
import scipy as sp
import scipy.linalg
import argparse

from picca.utils import smooth_cov, print

if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Stack different realizations')

    parser.add_argument('--data', type=str, nargs='*', required=True,
        help='Input correlation function from picca_cf, picca_xcf...')

    parser.add_argument('--error-on-mean', action='store_true', default=False,
        help='Divide the covairance by the number of realizations')

    parser.add_argument('--do-not-smooth-cov', action='store_true', default=False,
        help='Do not smooth the covariance matrix')

    parser.add_argument('--out', type=str, default=None, required=True,
        help='Output file name')

    args = parser.parse_args()

    ###
    head = {'RPMIN':None, 'RPMAX':None, 'RTMAX':None, 'NP':None, 'NT':None}
    dic = {'RP':[], 'RT':[], 'Z':[], 'NB':[], 'DA':[]}
    for i,f in enumerate(args.data):
        print('INFO: file {}: {} over {} files'.format(i,f,len(args.data)))
        h = fitsio.FITS(f)

        for k in head.keys():
            if head[k] is None:
                head[k] = h[1].read_header()[k]
            else:
                assert head[k]==h[1].read_header()[k]

        for k in [ el for el in dic.keys() if el!='DA']:
            dic[k] += [h[1][k][:]]

        if h[1].read_header()['EXTNAME'].strip()=='ATTRI':
            da = sp.array(h['COR']['DA'][:])
            we = sp.array(h['COR']['WE'][:])
            da = (da*we).sum(axis=0)
            we = we.sum(axis=0)
            w = we>0
            da[w] /= we[w]
            dic['DA'] += [da]
        else:
            dic['DA'] += [h[1]['DA'][:]]

        h.close()

    ###
    for k in dic.keys():
        dic[k] = sp.vstack(dic[k])
    dic['CO'] = sp.cov(dic['DA'].T)
    if args.error_on_mean:
        dic['CO'] /= dic['DA'].shape[0]
    for k in [ el for el in dic.keys() if el!='CO']:
        dic[k] = dic[k].mean(axis=0)

    ###
    if not args.do_not_smooth_cov:
        print('INFO: The covariance will be smoothed')
        binSizeP = (head['RPMAX']-head['RPMIN']) / head['NP']
        binSizeT = (head['RTMAX']-0.) / head['NT']
        co = smooth_cov(da=dic['DA'],we=None,rp=dic['RP'],rt=dic['RT'],
            drt=binSizeT,drp=binSizeP,co=dic['CO'])

    ###
    try:
        scipy.linalg.cholesky(dic['CO'])
    except scipy.linalg.LinAlgError:
        print('WARNING: Matrix is not positive definite')

    ###
    dic['DM'] = sp.eye(dic['DA'].size)
    dic['DMRP'] = dic['RP'].copy()
    dic['DMRT'] = dic['RT'].copy()
    dic['DMZ'] = dic['Z'].copy()

    ###
    h = fitsio.FITS(args.out,'rw',clobber=True)
    head = [
        {'name':'RPMIN','value':head['RPMIN'],'comment':'Minimum r-parallel'},
        {'name':'RPMAX','value':head['RPMAX'],'comment':'Maximum r-parallel'},
        {'name':'RTMAX','value':head['RTMAX'],'comment':'Maximum r-transverse'},
        {'name':'NP','value':head['NP'],'comment':'Number of bins in r-parallel'},
        {'name':'NT','value':head['NT'],'comment':'Number of bins in r-transverse'}
    ]
    comment = ['R-parallel','R-transverse','Redshift','Correlation',
        'Covariance matrix','Distortion matrix','Number of pairs']
    keys = ['RP','RT','Z','DA','CO','DM','NB']
    h.write(
        [dic[k] for k in keys],
        names=keys,
        comment=comment,header=head,extname='COR')
    comment = ['R-parallel model','R-transverse model','Redshift model']
    keys = ['DMRP','DMRT','DMZ']
    h.write(
        [dic[k] for k in keys],
        names=keys,
        comment=comment,extname='DMATTRI')
    h.close()
