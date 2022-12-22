#!/usr/bin/env python

from picca.fitter2.control_mpi import fitter2_mpi
import argparse
import warnings

if __name__ == '__main__':
    pars = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Fit the correlation function with parallel functionality.')

    pars.add_argument('config', type=str, default=None,
        help='Config file')
    warnings.warn("Note that the fitter2 module will be removed with the next picca release, please use Vega instead", DeprecationWarning)

    args = pars.parse_args()
    fitter = fitter2_mpi(args.config)
    fitter.run()
