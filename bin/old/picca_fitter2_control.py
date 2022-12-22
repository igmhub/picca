#!/usr/bin/env python

from picca.fitter2.control import fitter2
import argparse
import warnings

if __name__ == '__main__':
    pars = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Fit the correlation function.')

    pars.add_argument('config', type=str, default=None,
        help='Config file')
    warnings.warn("Note that the fitter2 module will be removed with the next picca release, please use Vega instead", DeprecationWarning)


    args = pars.parse_args()
    fitter = fitter2(args.config)
    fitter.run()
