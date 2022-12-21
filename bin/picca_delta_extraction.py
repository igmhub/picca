#!/usr/bin/env python3
"""Compute delta field from a list of spectra.

Compute the mean transmission fluctuation field (delta field) for a list of
spectra for the specified absorption line. Follow the procedure described in
section 2.4 of du Mas des Bourboux et al. 2020 (In prep).
"""
import logging
import time
import argparse

from picca.delta_extraction.survey import Survey

module_logger = logging.getLogger("picca.delta_extraction")


def main(args):
    """Compute delta field"""
    t0 = time.time()

    # intitialize Survey instance
    survey = Survey()

    # load configuration
    survey.load_config(args.config_file)

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

    # filter forests
    survey.filter_forests()

    # compute forest continua
    survey.compute_expected_flux()

    # compute the delta field
    survey.extract_deltas()

    # save results
    survey.save_deltas()

    t1 = time.time()
    module_logger.info(f"Total time ellapsed: {t1-t0}")
    module_logger.info("Done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=('Compute the delta field '
                     'from a list of spectra'))

    parser.add_argument(
        'config_file',
        type=str,
        default=None,
        help=
        ('Configuration file. To learn about all the available options '
         'check the configuration tutorial in '
         '$PICCA/tutorials/delta_extraction/picca_delta_extraction_tutorial.ipynb'
        ))

    args = parser.parse_args()
    main(args)
