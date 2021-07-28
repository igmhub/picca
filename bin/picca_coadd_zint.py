#!/usr/bin/env python
"""Coadd correlation function from different redshift intervals"""
import os
import sys
import argparse
import fitsio
import numpy as np

from picca.utils import userprint


def coadd_correlations(input_files,output_file):
    """Coadds correlation functions measured in different redshift intervals.

    Args:
        input_files: list of strings
            List of the paths to the input correlations to be coadded.
        output_file: string
            String of the path to the desired output.
    """

    # specify which header entries we want to check are consistent across all
    # files being coadded
    headers_to_check_match = ['NP','NT','OMEGAM','OMEGAR','OMEGAK','WL','NSIDE']

    # initialize coadd arrays, fill them with zeros
    with fitsio.FITS(input_files[0]) as hdul:
        r_par = hdul[1]['RP'][:] * 0
        r_trans = hdul[1]['RT'][:] * 0
        num_pairs = hdul[1]['NB'][:] * 0
        z = hdul[1]['Z'][:] * 0
        weights_total = r_par * 0
        healpixs = hdul[2]['HEALPID'][:]

        # get values of header entries to check with other files
        header = hdul[1].read_header()
        headers_to_check_match_values = {h:header[h] for h in headers_to_check_match}

    # initialise header quantities
    r_par_min = 1.e6
    r_par_max = -1.e6
    r_trans_max = -1.e6
    z_cut_min = 1.e6
    z_cut_max = -1.e6

    # check to see if all files have the same HEALPix pixels
    same_healpixs = True
    for file in input_files:
        with fitsio.FITS(file) as hdul:
            if len(hdul[2]['HEALPID'][:]) != len(healpixs):
                same_healpixs = False
            elif (hdul[2]['HEALPID'][:] != healpixs).all():
                same_healpixs = False

    # if so, initialise our xi and weights as arrays
    if same_healpixs:
        with fitsio.FITS(file) as hdul:
            xi = hdul[2]['DA'][:] * 0
            weights = hdul[2]['WE'][:] * 0
    # otherwise, use dictionaries to match HEALPix pixels carefully (but slower)
    else:
        xi = {}
        weights = {}

    # loop over files
    for file in input_files:

        # open the file and read the header
        userprint("coadding file {}".format(file))
        hdul = fitsio.FITS(file)
        header = hdul[1].read_header()

        # check that the header properties match those from the first file
        for entry in headers_to_check_match:
            assert header[entry] == headers_to_check_match_values[entry]

        # add weighted contributions from this file to coadd variables
        weights_aux = hdul[2]['WE'][:]
        weights_total_aux = weights_aux.sum(axis=0)
        r_par += hdul[1]['RP'][:] * weights_total_aux
        r_trans += hdul[1]['RT'][:] * weights_total_aux
        z += hdul[1]['Z'][:] * weights_total_aux
        num_pairs += hdul[1]['NB'][:]
        weights_total += weights_total_aux

        # update values to go in coadd header
        r_par_min = np.min([r_par_min,header['RPMIN']])
        r_par_max = np.max([r_par_max,header['RPMAX']])
        r_trans_max = np.max([r_trans_max,header['RTMAX']])
        z_cut_min = np.min([z_cut_min,header['ZCUTMIN']])
        z_cut_max = np.max([z_cut_max,header['ZCUTMAX']])

        # add xi and weights information to initialised structures
        if same_healpixs:
            xi += hdul[2]["DA"][:] * weights_aux
            weights += weights_aux
        else:
            healpixs = hdul[2]['HEALPID'][:]
            for index, healpix in enumerate(healpixs):
                userprint(" -> coadding healpix {}".format(healpix),end='\r')
                if healpix in xi:
                    xi[healpix] += hdul[2]["DA"][:][index] * weights_aux[index]
                    weights[healpix] += weights_aux[index, :]
                else:
                    xi[healpix] = hdul[2]["DA"][:][index] * weights_aux[index]
                    weights[healpix] = weights_aux[index]
            userprint('')

        hdul.close()

    # if necessary, combine elements of our dictionary into an array
    if not same_healpixs:
        healpixs = np.array(list(xi.keys()))
        xi = np.vstack([xi[healpix] for healpix in healpixs])
        weights = np.vstack([weights[healpix] for healpix in healpixs])

    # normalise xi by the weights
    w = weights>0
    xi[w] /= weights[w]

    # normalize all other quantities by total weights
    w = weights_total>0
    r_par[w] /= weights_total[w]
    r_trans[w] /= weights_total[w]
    z[w] /= weights_total[w]

    results = fitsio.FITS(output_file, 'rw', clobber=True)

    header = [
    {
        'name': 'RPMIN',
        'value': r_par_min,
        'comment': 'Minimum r-parallel [h^-1 Mpc]'
    },
    {
        'name': 'RPMAX',
        'value': r_par_max,
        'comment': 'Maximum r-parallel [h^-1 Mpc]'
    },
    {
        'name': 'RTMAX',
        'value': r_trans_max,
        'comment': 'Maximum r-transverse [h^-1 Mpc]'
    },
    {
        'name': 'NP',
        'value': headers_to_check_match_values['NP'],
        'comment': 'Number of bins in r-parallel'
    },
    {
        'name': 'NT',
        'value': headers_to_check_match_values['NT'],
        'comment': 'Number of bins in r-transverse'
    },
    {
        'name': 'ZCUTMIN',
        'value': z_cut_min,
        'comment': 'Minimum redshift of pairs'
    },
    {
        'name': 'ZCUTMAX',
        'value': z_cut_max,
        'comment': 'Maximum redshift of pairs'
    },
    {
        'name': 'NSIDE',
        'value': headers_to_check_match_values['NSIDE'],
        'comment': 'Healpix nside'
    },
    {
        'name': 'OMEGAM',
        'value': headers_to_check_match_values['OMEGAM'],
        'comment': 'Omega_matter(z=0) of fiducial LambdaCDM cosmology'
    },
    {
        'name': 'OMEGAR',
        'value': headers_to_check_match_values['OMEGAR'],
        'comment': 'Omega_radiation(z=0) of fiducial LambdaCDM cosmology'
    },
    {
        'name': 'OMEGAK',
        'value': headers_to_check_match_values['OMEGAK'],
        'comment': 'Omega_k(z=0) of fiducial LambdaCDM cosmology'
    },
    {
        'name': 'WL',
        'value': headers_to_check_match_values['WL'],
        'comment': 'Equation of state of dark energy of fiducial LambdaCDM cosmology'
    }
    ]
    results.write(
        [r_par, r_trans, z, num_pairs],
        names=['RP', 'RT', 'Z', 'NB'],
        comment=['R-parallel', 'R-transverse', 'Redshift', 'Number of pairs'],
        units=['h^-1 Mpc', 'h^-1 Mpc', '', ''],
        header=header,
        extname='ATTRI')

    header2 = [{
        'name': 'HLPXSCHM',
        'value': 'RING',
        'comment': 'Healpix scheme'
    }]
    results.write([healpixs, weights, xi],
                  names=['HEALPID', 'WE', 'DA'],
                  comment=['Healpix index', 'Sum of weight', 'Correlation'],
                  header=header2,
                  extname='COR')

    results.close()

    return

def coadd_dmats(input_files,output_file):
    """Coadds distortion matrices measured in different redshift intervals.

    Args:
        input_files: list of strings
            List of the paths to the input distortion matrices to be coadded.
        output_file: string
            String of the path to the desired output.
    """

    # specify which header entries we want to check are consistent across all
    # files being coadded
    headers_to_check_match = ['NP','NT','COEFMOD','REJ','OMEGAM','OMEGAR','OMEGAK','WL']

    # initialize distortion matrix array, fill them with zeros
    with fitsio.FITS(input_files[0]) as hdul:
        dmat = hdul[1]['DM'][:] * 0
        weights_dmat = hdul[1]['WDM'][:] * 0
        r_par_dmat = np.zeros(hdul[2]['RP'][:].size)
        r_trans_dmat = np.zeros(hdul[2]['RT'][:].size)
        z_dmat = np.zeros(hdul[2]['Z'][:].size)
        num_pairs_dmat = 0.

        # get values of header entries to check with other files
        header = hdul[1].read_header()
        headers_to_check_match_values = {h:header[h] for h in headers_to_check_match}

    # initialise header quantities
    num_pairs = 0
    num_pairs_used = 0
    r_par_min = 1.e6
    r_par_max = -1.e6
    r_trans_max = -1.e6
    z_cut_min = 1.e6
    z_cut_max = -1.e6

    # loop over files
    for file in input_files:

        # open the file and read the header
        userprint("coadding file {}".format(file))
        hdul = fitsio.FITS(file)
        header = hdul[1].read_header()

        # check that the header properties match those from the first file
        for entry in headers_to_check_match:
            assert header[entry] == headers_to_check_match_values[entry]

        # add weighted contributions from this file to coadd variables
        weights_dmat_aux = hdul[1]['WDM'][:]
        dmat += hdul[1]['DM'][:] * weights_dmat_aux
        weights_dmat += weights_dmat_aux

        r_par_dmat += hdul[2]['RP'][:] * weights_dmat_aux
        r_trans_dmat += hdul[2]['RT'][:] * weights_dmat_aux
        z_dmat += hdul[2]['Z'][:] * weights_dmat_aux
        num_pairs_dmat += 1.

        # update values to go in coadd header
        num_pairs += header['NPALL']
        num_pairs_used += header['NPUSED']
        r_par_min = np.min([r_par_min,header['RPMIN']])
        r_par_max = np.max([r_par_max,header['RPMAX']])
        r_trans_max = np.max([r_trans_max,header['RTMAX']])
        z_cut_min = np.min([z_cut_min,header['ZCUTMIN']])
        z_cut_max = np.max([z_cut_max,header['ZCUTMAX']])

        hdul.close()

    # normalize
    w = weights_dmat>0
    dmat[w] /= weights_dmat[w]
    r_par_dmat[w] /= weights_dmat[w]
    r_trans_dmat[w] /= weights_dmat[w]
    z_dmat[w] /= weights_dmat[w]

    # save results
    results = fitsio.FITS(output_file, 'rw', clobber=True)
    header = [
        {
            'name': 'RPMIN',
            'value': r_par_min,
            'comment': 'Minimum r-parallel [h^-1 Mpc]'
        },
        {
            'name': 'RPMAX',
            'value': r_par_max,
            'comment': 'Maximum r-parallel [h^-1 Mpc]'
        },
        {
            'name': 'RTMAX',
            'value': r_trans_max,
            'comment': 'Maximum r-transverse [h^-1 Mpc]'
        },
        {
            'name': 'NP',
            'value': headers_to_check_match_values['NP'],
            'comment': 'Number of bins in r-parallel'
        },
        {
            'name': 'NT',
            'value': headers_to_check_match_values['NT'],
            'comment': 'Number of bins in r-transverse'
        },
        {
            'name': 'COEFMOD',
            'value': headers_to_check_match_values['COEFMOD'],
            'comment': 'Coefficient for model binning'
        },
        {
            'name': 'ZCUTMIN',
            'value': z_cut_min,
            'comment': 'Minimum redshift of pairs'
        },
        {
            'name': 'ZCUTMAX',
            'value': z_cut_max,
            'comment': 'Maximum redshift of pairs'
        },
        {
            'name': 'REJ',
            'value': headers_to_check_match_values['REJ'],
            'comment': 'Rejection factor'
        },
        {
            'name': 'NPALL',
            'value': num_pairs,
            'comment': 'Number of pairs'
        },
        {
            'name': 'NPUSED',
            'value': num_pairs_used,
            'comment': 'Number of used pairs'
        },
        {
            'name': 'OMEGAM',
            'value': headers_to_check_match_values['OMEGAM'],
            'comment': 'Omega_matter(z=0) of fiducial LambdaCDM cosmology'
        },
        {
            'name': 'OMEGAR',
            'value': headers_to_check_match_values['OMEGAR'],
            'comment': 'Omega_radiation(z=0) of fiducial LambdaCDM cosmology'
        },
        {
            'name': 'OMEGAK',
            'value': headers_to_check_match_values['OMEGAK'],
            'comment': 'Omega_k(z=0) of fiducial LambdaCDM cosmology'
        },
        {
            'name': 'WL',
            'value': headers_to_check_match_values['WL'],
            'comment': 'Equation of state of dark energy of fiducial LambdaCDM cosmology'
        }
        ]
    results.write([weights_dmat, dmat],
                  names=['WDM', 'DM'],
                  comment=['Sum of weight', 'Distortion matrix'],
                  units=['', ''],
                  header=header,
                  extname='DMAT')
    results.write([r_par_dmat, r_trans_dmat, z_dmat],
                  names=['RP', 'RT', 'Z'],
                  comment=['R-parallel', 'R-transverse', 'Redshift'],
                  units=['h^-1 Mpc', 'h^-1 Mpc', ''],
                  extname='ATTRI')
    results.close()

    return

def main(cmdargs):
    """Coadds correlation function from different redshift intervals"""
    parser = argparse.ArgumentParser()

    parser.add_argument("--data",
                        type=str,
                        nargs="*",
                        required=False,
                        help="the (x)cf....fits files to be coadded")

    parser.add_argument("--out",
                        type=str,
                        required=False,
                        help="name of output file")

    parser.add_argument("--dmats",
                        type=str,
                        nargs="*",
                        required=False,
                        help="the (x)dmat....fits files to be coadded.")

    parser.add_argument("--out-dmat",
                        type=str,
                        required=False,
                        help="name of dmat output file")

    args = parser.parse_args(cmdargs)

    if args.data is None and args.dmats is None:
        raise IOError('No input correlations or dmats provided!')

    if args.data is not None:
        for file in args.data:
            if not os.path.isfile(file):
                userprint('WARN: could not find file {}, removing it'.format(file))
                args.data.remove(file)
        coadd_correlations(args.data,args.out)

    if args.dmats is not None:
        for file in args.dmats:
            if not os.path.isfile(file):
                userprint('WARN: could not find file {}, removing it'.format(file))
                args.data.remove(file)
        coadd_dmats(args.dmats,args.out_dmat)


if __name__ == "__main__":
    cmdargs=sys.argv[1:]
    main(cmdargs)
