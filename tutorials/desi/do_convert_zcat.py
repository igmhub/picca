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

    args = parser.parse_args()

    utils.desi_from_ztarget_to_drq(ztarget = args.in_object_cat, drq = args.out_object_cat, spectype = args.spectype)
