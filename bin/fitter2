#!/usr/bin/env python

from picca.fitter2 import chi2, parser
import argparse

if __name__ == '__main__':

    pars = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Fit the correlation function.')

    pars.add_argument('config', type=str, default=None,
        help='Config file')


    args = pars.parse_args()

    dic_init = parser.parse_chi2(args.config)
    chi = chi2.chi2(dic_init)
    chi.minimize()
    chi.minos()
    chi.chi2scan()
    chi.fastMC()
    chi.export()
