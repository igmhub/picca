import os
import sys
import cProfile
import argparse
import pstats


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ["THIS_DIR"] = THIS_DIR

PICCA_BIN = THIS_DIR.split("py/picca")[0]+"bin/"
sys.path.append(PICCA_BIN)

from picca_delta_extraction import main

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
cProfile.run('main(args)', 'results/profile_output')
p = pstats.Stats('results/profile_output')
p.sort_stats('cumulative').print_stats()
