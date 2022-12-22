#!/usr/bin/env python

from picca.fitter2 import chi2, parser
import argparse
import warnings

def main(config):
    dic_init = parser.parse_chi2(config)
    chi = chi2.chi2(dic_init)
    chi.minimize()
    chi.minos()
    chi.chi2scan()
    chi.fastMC()
    chi.export()

if __name__ == '__main__':
    pars = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Fit the correlation function.')

    pars.add_argument('config', type=str, default=None,
        help='Config file')


    args = pars.parse_args()

    warnings.warn("Note that the fitter2 module will be removed with the next picca release, please use Vega instead", DeprecationWarning)
    main(args.config)
