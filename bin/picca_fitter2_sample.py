#!/usr/bin/env python

from picca.fitter2 import sample, parser
import argparse

if __name__ == '__main__':

    pars = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Fit the correlation function.')

    pars.add_argument('config', type=str, default=None,
        help='Config file')


    args = pars.parse_args()

    dic_init = parser.parse_chi2(args.config)
    sampler = sample.sample(dic_init)

    run_sampler = dic_init['control'].getboolean('sampler', False)
    run_chi2 = dic_init['control'].getboolean('chi2', False)
    
    if run_sampler:
        sampler.run_sampler()

    if run_chi2:
        sampler.chi.minimize()
        sampler.chi.chi2scan()
        sampler.chi.export()

    # chi.minos()
    # chi.fastMC()
