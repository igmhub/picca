#!/usr/bin/env python

import argparse

from picca import utils

if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description='Script to convert a catalog of object in desi format to DRQ format')

    parser.add_argument('--in-object-cat', type = str, default = None, required=True,
            help = 'Input path to a catalog of objects from desi')

    parser.add_argument('--out-object-cat', type = str, default = None, required=True,
            help = 'Output path to a catalog of objects in DRQ format')

    parser.add_argument('--spectype', type = str, default = 'QSO', required=False,
            help = "Spectype of the object, can be any spectype in desi catalog. Ex: 'STAR', 'GALAXY', 'QSO'")

    parser.add_argument('--downsampling-z-cut', type = float, default = 2.1, required = False,
            help = "Minimum redshift to downsample the data, if 'None' no downsampling")

    parser.add_argument('--downsampling-nb', type = int, default = 700000, required = False,
            help = "Target number of object above redshift downsampling-z-cut, if 'None' no downsampling")

    args = parser.parse_args()

    utils.desi_from_ztarget_to_drq(args.in_object_cat, args.out_object_cat, spectype=args.spectype, downsampling_z_cut=args.downsampling_z_cut, downsampling_nb=args.downsampling_nb)
