#!/usr/bin/env python
from astropy.table import Table
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--spall',
                    type=str,
                    required=True,
                    help='Path to spAll file')
parser.add_argument('--qso-catalog',
                    type=str,
                    required=True,
                    help='Path to file containing THING_ID (e.g. DR16Q.fits)')
parser.add_argument('--output',
                    type=str,
                    required=True,
                    help='Path to output reduced spAll file')

args = parser.parse_args()

print('Reading spAll file from')
print(args.spall)
spall = Table.read(args.spall)
print(f'{len(spall)} entries found in spAll file')
print('Reading QSO catalog from')
print(args.qso_catalog)
qso_catalog = Table.read(args.qso_catalog)
print(f'{len(qso_catalog)} entries found in QSO catalog')

w = np.in1d(spall['THING_ID'], qso_catalog['THING_ID'])
spall_qso = spall[w]
#-- Columns required for picca_deltas.py for spec, spplate formats and usage of multiple observations
spall_qso.keep_columns(
    ['THING_ID', 'PLATE', 'MJD', 'FIBERID', 'PLATEQUALITY', 'ZWARNING'])
spall_qso.write(args.output, overwrite=True)
