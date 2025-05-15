#!/usr/bin/env python

import argparse

from picca import raw_io

if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description='Script to convert noiseless transmission files to delta picca files')

    parser.add_argument('--object-cat', type=str, default=None, required=True,
            help='Path to a catalog of objects to get the transmission from')

    parser.add_argument('--in-dir', type=str, default=None, required=False,
            help='Desi formated data directory to transmission files')

    parser.add_argument('--in-files', type=str, default=None, required=False,
            help='List of transmission files.', nargs='*')

    parser.add_argument('--out-dir', type=str, default=None, required=True,
            help='Output directory')

    parser.add_argument('--lambda-min', type=float, default=3600., required=False,
            help='Lower limit on observed wavelength [Angstrom]')

    parser.add_argument('--lambda-max', type=float, default=5500., required=False,
            help='Upper limit on observed wavelength [Angstrom]')

    parser.add_argument('--lambda-rest-min', type=float, default=1040., required=False,
            help='Lower limit on rest frame wavelength [Angstrom]')

    parser.add_argument('--lambda-rest-max', type=float, default=1200., required=False,
            help='Upper limit on rest frame wavelength [Angstrom]')

    parser.add_argument('--delta-log-lambda', type=float, default=3.e-4, required=False,
            help='Size of the rebined pixels in log lambda')

    parser.add_argument('--delta-lambda', type=float, default=None, required=False,
            help='Size of the rebined pixels in log lambda')

    parser.add_argument('--linear-spacing', action="store_true", default=False, required=False,
            help='Whether to use linear bins in lambda.')

    parser.add_argument('--nspec', type=int, default=None, required=False,
            help="Number of spectra to fit, if None then run on all files")

    parser.add_argument('--use-old-weights', action="store_true", default=False, required=False,
            help='Whether to use the old weighting scheme for raw deltas.')

    parser.add_argument('--tracer',type=str, default='F_LYA', required=False,
            help='Tracer to use')

    parser.add_argument('--use-splines',action="store_true", default=False, required=False,
            help='Use splines to compute mean flux and variance')

    args = parser.parse_args()

    raw_io.convert_transmission_to_deltas(args.object_cat, args.out_dir, in_dir=args.in_dir,
                                          in_filenames=args.in_files,tracer=args.tracer,
                                          lambda_min=args.lambda_min,
                                          lambda_max=args.lambda_max,
                                          lambda_min_rest_frame=args.lambda_rest_min,
                                          lambda_max_rest_frame=args.lambda_rest_max,
                                          delta_log_lambda=args.delta_log_lambda,
                                          delta_lambda=args.delta_lambda,
                                          lin_spaced=args.linear_spacing,
                                          max_num_spec=args.nspec,
                                          use_old_weights=args.use_old_weights,
                                          use_splines=args.use_splines)
