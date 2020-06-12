#!/usr/bin/env python

import argparse

from picca import utils

if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description='Script to select quasars from a bunch of zbest files and make a single cat')

    parser.add_argument('--in-zbest', type=str, default=None, nargs='+', required=True,
        help='Catalog(s) of objects in zbest format')

    parser.add_argument('--out-cat', type = str, default = None, required=True,
            help = 'Output path to a catalog of objects in DRQ format')
    
    parser.add_argument('--zmin', type = float, default = 2.0,
            help = 'Minimal object redshift')
    parser.add_argument('--zmax', type = float, default = 4.288461538461538,
            help = 'Maximal object redshift')

    #could add additional things here, e.g. different spectypes (needs modification of targeting checks as well)
    #or other selection flags
    
    args = parser.parse_args()

    utils.catalog_from_zbest(zbestfiles=args.in_zbest, outfile=args.out_cat, zmin=args.zmin, zmax=args.zmax)
