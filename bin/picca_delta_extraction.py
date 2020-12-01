#!/usr/bin/env python3
"""Computes delta field from a list of spectra.

Computes the mean transmission fluctuation field (delta field) for a list of
spectra for the specified absorption line. Follow the procedure described in
section 2.4 of du Mas des Bourboux et al. 2020 (In prep).
"""
import time
import argparse

from picca.delta_extraction.survey import Survey
from picca.delta_extraction.userprint import userprint

def main(args):
    """Computes delta field"""
    t0 = time.time()

    # intitialize Survey instance
    survey = Survey()

    # load configuration
    survey.load_config(args.config_file)

    # initialize output folders
    survey.initialize_folders()

    # initialize forest corrections and masks
    # this is done prior to reading the data as it is faster and we can
    # save computing time if an error occurs here
    survey.read_corrections()
    survey.read_masks()

    # read data
    survey.read_data()

    # apply corrections and masks
    survey.apply_corrections()
    survey.apply_masks()

    # compute forest continua
    survey.compute_continua()

    # compute the delta field
    survey.apply_continua()

    # save results
    survey.save_deltas()

    t1 = time.time()
    userprint(f"Total time ellapsed: {t1-t0}")
    userprint("Done")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=('Compute the delta field '
                     'from a list of spectra'))

    parser.add_argument('config-file',
                        type=str,
                        default=None,
                        help='Configureation file')

    args = parser.parse_args()
    main(args)
