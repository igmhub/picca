#!/usr/bin/env python
"""Export auto and cross-correlation of catalog of objects for the fitter."""
import sys
import argparse
import fitsio
import numpy as np
import scipy.linalg

from picca.utils import smooth_cov, compute_cov, userprint


def main(cmdargs):
    # pylint: disable-msg=too-many-locals,too-many-branches,too-many-statements
    """Exports auto and cross-correlation of catalog of objects for the
    fitter."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=('Export auto and cross-correlation of catalog of objects '
                     'for the fitter.'))

    parser.add_argument('--out',
                        type=str,
                        default=None,
                        required=True,
                        help='Output file name')

    parser.add_argument('--DD-file',
                        type=str,
                        default=None,
                        required=False,
                        help='File of the data x data auto-correlation')

    parser.add_argument('--RR-file',
                        type=str,
                        default=None,
                        required=False,
                        help='File of the random x random auto-correlation')

    parser.add_argument('--DR-file',
                        type=str,
                        default=None,
                        required=False,
                        help='File of the data x random auto-correlation')

    parser.add_argument('--RD-file',
                        type=str,
                        default=None,
                        required=False,
                        help='File of the random x data auto-correlation')

    parser.add_argument('--xDD-file',
                        type=str,
                        default=None,
                        required=False,
                        help='File of the data_1 x data_2 cross-correlation')

    parser.add_argument(
        '--xRR-file',
        type=str,
        default=None,
        required=False,
        help='File of the random_1 x random_2 cross-correlation')

    parser.add_argument('--xD1R2-file',
                        type=str,
                        default=None,
                        required=False,
                        help='File of the data_1 x random_2 cross-correlation')

    parser.add_argument('--xR1D2-file',
                        type=str,
                        default=None,
                        required=False,
                        help='File of the random_1 x data_2 cross-correlation')

    parser.add_argument(
        '--do-not-smooth-cov',
        action='store_true',
        default=False,
        help='Do not smooth the covariance matrix from sub-sampling')

    parser.add_argument('--get-cov-from-poisson',
                        action='store_true',
                        default=False,
                        help='Get covariance matrix from Poisson statistics')

    parser.add_argument(
        '--cov',
        type=str,
        default=None,
        required=False,
        help=('Path to a covariance matrix file (if not provided it will be '
              'calculated by subsampling or from Poisson statistics)'))

    args = parser.parse_args(cmdargs)

    ### Auto or cross correlation?
    if ((args.DD_file is None and args.xDD_file is None) or
            (args.DD_file is not None and args.xDD_file is not None) or
            (args.cov is not None and not args.get_cov_from_poisson)):
        userprint(('ERROR: No data files, or both auto and cross data files, '
                   'or two different method for covariance'))
        sys.exit()
    elif args.DD_file is not None:
        corr = 'AUTO'
        correlation_files = {
            'DD': args.DD_file,
            'RR': args.RR_file,
            'DR': args.DR_file,
            'RD': args.RD_file
        }
    elif not args.xDD_file is None:
        # TODO: Test if picca_co.py and export_co.py work for cross
        corr = 'CROSS'
        correlation_files = {
            'xDD': args.xDD_file,
            'xRR': args.xRR_file,
            'xD1R2': args.xD1R2_file,
            'xR1D2': args.xR1D2_file
        }

    # Read files
    data = {}
    for type_corr, filename in correlation_files.items():
        hdul = fitsio.FITS(filename)
        header = hdul[1].read_header()
        fid_Om = header['OMEGAM']
        fid_Or = header['OMEGAR']
        fid_Ok = header['OMEGAK']
        fid_wl = header['WL']
        if type_corr in ['DD', 'RR']:
            num_objects = header['NOBJ']
            coef = num_objects * (num_objects - 1)
        else:
            num_objects = header['NOBJ']
            num_objects2 = header['NOBJ2']
            coef = num_objects * num_objects2

        if type_corr in ['DD', 'xDD']:
            data['COEF'] = coef
            for item in ['NT', 'NP', 'RTMAX', 'RPMIN', 'RPMAX']:
                data[item] = header[item]
            for item in ['RP', 'RT', 'Z', 'NB']:
                data[item] = np.array(hdul[1][item][:])

        data[type_corr] = {}
        data[type_corr]['NSIDE'] = header['NSIDE']
        data[type_corr]['HLPXSCHM'] = hdul[2].read_header()['HLPXSCHM']
        w = np.array(hdul[2]['WE'][:]).sum(axis=1) > 0.
        if w.sum() != w.size:
            userprint("INFO: {} sub-samples were empty".format(w.size -
                                                               w.sum()))
        data[type_corr]['HEALPID'] = hdul[2]['HEALPID'][:][w]
        data[type_corr]['WE'] = hdul[2]['WE'][:][w] / coef
        hdul.close()

    # Compute correlation
    if corr == 'AUTO':
        xi_data_data = data['DD']['WE'].sum(axis=0)
        xi_random_random = data['RR']['WE'].sum(axis=0)
        xi_data_random = data['DR']['WE'].sum(axis=0)
        xi_random_data = data['RD']['WE'].sum(axis=0)
        w = xi_random_random > 0.
        xi = np.zeros(xi_data_data.size)
        xi[w] = (xi_data_data[w] + xi_random_random[w] - xi_random_data[w] -
                 xi_data_random[w]) / xi_random_random[w]
    else:
        xi_data_data = data['xDD']['WE'].sum(axis=0)
        xi_random_random = data['xRR']['WE'].sum(axis=0)
        xi_data1_random2 = data['xD1R2']['WE'].sum(axis=0)
        xi_data2_random1 = data['xR1D2']['WE'].sum(axis=0)
        w = xi_random_random > 0.
        xi = np.zeros(xi_data_data.size)
        xi[w] = (xi_data_data[w] + xi_random_random[w] - xi_data1_random2[w] -
                 xi_data2_random1[w]) / xi_random_random[w]
    data['DA'] = xi
    data['corr_DD'] = xi_data_data
    data['corr_RR'] = xi_random_random

    # Compute covariance matrix
    if not args.cov is None:
        userprint('INFO: Read covariance from file')
        hdul = fitsio.FITS(args.cov)
        data['CO'] = hdul[1]['CO'][:]
        hdul.close()
    elif args.get_cov_from_poisson:
        userprint('INFO: Compute covariance from Poisson statistics')
        w = data['corr_RR'] > 0.
        covariance = np.zeros(data['corr_DD'].size)
        covariance[w] = ((data['COEF'] / 2. * data['corr_DD'][w])**2 /
                         (data['COEF'] / 2. * data['corr_RR'][w])**3)
        data['CO'] = np.diag(covariance)
    else:
        userprint('INFO: Compute covariance from sub-sampling')

        ### To have same number of HEALPix
        for type_corr1 in list(correlation_files):
            for type_corr2 in list(correlation_files):

                if data[type_corr1]['NSIDE'] != data[type_corr2]['NSIDE']:
                    userprint("ERROR: NSIDE are different: {} != "
                              "{}".format(data[type_corr1]['NSIDE'],
                                          data[type_corr2]['NSIDE']))
                    sys.exit()
                if data[type_corr1]['HLPXSCHM'] != data[type_corr2]['HLPXSCHM']:
                    userprint("ERROR: HLPXSCHM are different: {} != "
                              "{}".format(data[type_corr1]['HLPXSCHM'],
                                          data[type_corr2]['HLPXSCHM']))
                    sys.exit()

                w = np.logical_not(
                    np.in1d(data[type_corr1]['HEALPID'],
                            data[type_corr2]['HEALPID']))
                if w.sum() != 0:
                    userprint("WARNING: HEALPID are different by {} for {}:{} "
                              "and {}:{}".format(
                                  w.sum(), type_corr1,
                                  data[type_corr1]['HEALPID'].size, type_corr2,
                                  data[type_corr2]['HEALPID'].size))
                    new_healpix = data[type_corr1]['HEALPID'][w]
                    num_new_healpix = new_healpix.size
                    num_bins = data[type_corr2]['WE'].shape[1]
                    data[type_corr2]['HEALPID'] = np.append(
                        data[type_corr2]['HEALPID'], new_healpix)
                    data[type_corr2]['WE'] = np.append(data[type_corr2]['WE'],
                                                       np.zeros(
                                                           (num_new_healpix,
                                                            num_bins)),
                                                       axis=0)

        # Sort the data by the healpix values
        for type_corr1 in list(correlation_files):
            sort = np.array(data[type_corr1]['HEALPID']).argsort()
            data[type_corr1]['WE'] = data[type_corr1]['WE'][sort]
            data[type_corr1]['HEALPID'] = data[type_corr1]['HEALPID'][sort]

        if corr == 'AUTO':
            xi_data_data = data['DD']['WE']
            xi_random_random = data['RR']['WE']
            xi_data_random = data['DR']['WE']
            xi_random_data = data['RD']['WE']
            w = xi_random_random > 0.
            xi = np.zeros(xi_data_data.shape)
            xi[w] = (xi_data_data[w] + xi_random_random[w] - xi_data_random[w] -
                     xi_random_data[w]) / xi_random_random[w]
            weights = data['DD']['WE']
        else:
            xi_data_data = data['xDD']['WE']
            xi_random_random = data['xRR']['WE']
            xi_data1_random2 = data['xD1R2']['WE']
            xi_data2_random1 = data['xR1D2']['WE']
            w = xi_random_random > 0.
            xi = np.zeros(xi_data_data.shape)
            xi[w] = ((xi_data_data[w] + xi_random_random[w] -
                      xi_data1_random2[w] - xi_data2_random1[w]) /
                     xi_random_random[w])
            weights = data['xDD']['WE']
        data['HLP_DA'] = xi
        data['HLP_WE'] = weights

        if args.do_not_smooth_cov:
            userprint('INFO: The covariance will not be smoothed')
            covariance = compute_cov(xi, weights)
        else:
            userprint('INFO: The covariance will be smoothed')
            delta_r_par = (data['RPMAX'] - data['RPMIN']) / data['NP']
            delta_r_trans = (data['RTMAX'] - 0.) / data['NT']
            covariance = smooth_cov(xi,
                                    weights,
                                    data['RP'],
                                    data['RT'],
                                    delta_r_par=delta_r_par,
                                    delta_r_trans=delta_r_trans)
        data['CO'] = covariance

    try:
        scipy.linalg.cholesky(data['CO'])
    except scipy.linalg.LinAlgError:
        userprint('WARNING: Matrix is not positive definite')

    # Identity distortion matrix
    data['DM'] = np.eye(data['DA'].size)

    # Save results
    results = fitsio.FITS(args.out, 'rw', clobber=True)
    header = {}
    if corr == 'AUTO':
        nside = data['DD']['NSIDE']
    else:
        nside = data['xDD']['NSIDE']
    header = [{
        'name': 'RPMIN',
        'value': data['RPMIN'],
        'comment': 'Minimum r-parallel'
    }, {
        'name': 'RPMAX',
        'value': data['RPMAX'],
        'comment': 'Maximum r-parallel'
    }, {
        'name': 'RTMAX',
        'value': data['RTMAX'],
        'comment': 'Maximum r-transverse'
    }, {
        'name': 'NP',
        'value': data['NP'],
        'comment': 'Number of bins in r-parallel'
    }, {
        'name': 'NT',
        'value': data['NT'],
        'comment': 'Number of bins in r-transverse'
    }, {
        'name': 'NSIDE',
        'value': nside,
        'comment': 'Healpix nside'
    }, {
        'name': 'OMEGAM', 
        'value': fid_Om, 
        'comment': 'Omega_matter(z=0) of fiducial LambdaCDM cosmology'
    }, {
        'name': 'OMEGAR', 
        'value': fid_Or, 
        'comment': 'Omega_radiation(z=0) of fiducial LambdaCDM cosmology'
    }, {
        'name': 'OMEGAK', 
        'value': fid_Ok, 
        'comment': 'Omega_k(z=0) of fiducial LambdaCDM cosmology'
    }, {
        'name': 'WL', 
        'value': fid_wl, 
        'comment': 'Equation of state of dark energy of fiducial LambdaCDM cosmology'
    }
    ]
    names = ['RP', 'RT', 'Z', 'DA', 'CO', 'DM', 'NB']
    comment = [
        'R-parallel', 'R-transverse', 'Redshift', 'Correlation',
        'Covariance matrix', 'Distortion matrix', 'Number of pairs'
    ]
    results.write([data[name] for name in names],
                  names=names,
                  header=header,
                  comment=comment,
                  extname='COR')

    if args.cov is None and not args.get_cov_from_poisson:
        if corr == 'AUTO':
            healpix_scheme = data['DD']['HLPXSCHM']
            healpix_list = data['DD']['HEALPID']
        else:
            healpix_scheme = data['xDD']['HLPXSCHM']
            healpix_list = data['xDD']['HEALPID']
        header2 = [{
            'name': 'HLPXSCHM',
            'value': healpix_scheme,
            'comment': 'healpix scheme'
        }]
        comment = ['Healpix index', 'Sum of weight', 'Correlation']
        results.write([healpix_list, data['HLP_WE'], data['HLP_DA']],
                      names=['HEALPID', 'WE', 'DA'],
                      header=header2,
                      comment=comment,
                      extname='SUB_COR')

    results.close()


if __name__ == '__main__':
    cmdargs=sys.argv[1:]
    main(cmdargs)
