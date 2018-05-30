#!/usr/bin/env python

import argparse

from picca import utils

if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description='Script to convert noiseless transmission files to delta picca files')

    parser.add_argument('--object-cat', type = str, default = None, required=True,
            help = 'Path to a catalog of objects to get the transmission from')

    parser.add_argument('--in-dir',type = str,default=None,required=False,
            help='Desi formated data directory to transmission files')

    parser.add_argument('--in-files',type = str,default=None,required=False,
            help='List of transmission files.', nargs='*')

    parser.add_argument('--out-dir',type = str,default=None,required=True,
            help='Output directory')

    parser.add_argument('--lambda-min',type = float,default=3600.,required=False,
            help='Lower limit on observed wavelength [Angstrom]')

    parser.add_argument('--lambda-max',type = float,default=5500.,required=False,
            help='Upper limit on observed wavelength [Angstrom]')

    parser.add_argument('--lambda-rest-min',type = float,default=1040.,required=False,
            help='Lower limit on rest frame wavelength [Angstrom]')

    parser.add_argument('--lambda-rest-max',type = float,default=1200.,required=False,
            help='Upper limit on rest frame wavelength [Angstrom]')

    parser.add_argument('--dll',type = float,default=3.e-4,required=False,
            help='Size of the rebined pixels in log lambda')

    parser.add_argument('--nspec',type = int,default=None,required=False,
            help="Number of spectra to fit, if None then run on all files")

    args = parser.parse_args()

    utils.desi_convert_transmission_to_delta_files(args.object_cat, args.out_dir, indir=args.in_dir, infiles=args.in_files, lObs_min=args.lambda_min, lObs_max=args.lambda_max, lRF_min=args.lambda_rest_min, lRF_max=args.lambda_rest_max, dll=args.dll, nspec=args.nspec)
