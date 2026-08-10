"""This file contains tests related to the MPI delta extraction, run through
the script picca_delta_extraction.py under $PICCA_HOME/bin"""
import glob
import os
import shutil
import unittest
import subprocess
from subprocess import CalledProcessError

from picca.tests.delta_extraction.abstract_test import AbstractTest

try:
    from mpi4py import MPI  # noqa: F401  pylint: disable=unused-import
    MPI_AVAILABLE = True
except ImportError:  # pragma: no cover
    MPI_AVAILABLE = False

MPIRUN = shutil.which("mpirun") or shutil.which("mpiexec")

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# repository "py" directory, forwarded through PYTHONPATH so the ranks import the
# picca under test rather than any separately installed copy
PY_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", "..", ".."))


@unittest.skipUnless(MPI_AVAILABLE and MPIRUN is not None,
                     "requires mpi4py and an mpirun/mpiexec launcher")
class MpiTest(AbstractTest):
    """Test the MPI delta extraction against the serial reference deltas.

    Mirrors ScriptsTest but launches picca_delta_extraction.py under mpirun.

    Methods
    -------
    compare_fits (from AbstractTest)
    setUp (from AbstractTest)
    run_delta_extraction_mpi
    """
    def run_delta_extraction_mpi(self, config_file, out_dir, test_dir):
        """Run picca_delta_extraction.py under mpirun and compare its results
        against the reference produced by the serial run.
        """
        command = [
            MPIRUN, "-n", "2", "picca_delta_extraction.py", config_file
        ]
        print("Running command: ", " ".join(command))

        try:
            subprocess.run(command, check=True, capture_output=True,
                           env=dict(os.environ, THIS_DIR=THIS_DIR,
                                    PYTHONPATH=PY_DIR))
        except CalledProcessError as e:
            print(e.stderr)
            raise e

        # compare attributes
        test_files = sorted(glob.glob(f"{test_dir}/Log/delta_attributes*.fits.gz"))
        out_files = sorted(glob.glob(f"{out_dir}/Log/delta_attributes*.fits.gz"))
        for test_file, out_file in zip(test_files, out_files):
            self.assertTrue(test_file.split("/")[-1] == out_file.split("/")[-1])
            self.compare_fits(test_file, out_file)

        # compare deltas
        test_files = sorted(glob.glob(f"{test_dir}/Delta/delta-*.fits.gz"))
        out_files = sorted(glob.glob(f"{out_dir}/Delta/delta-*.fits.gz"))
        self.assertTrue(len(out_files) == len(test_files))
        for test_file, out_file in zip(test_files, out_files):
            self.assertTrue(test_file.split("/")[-1] == out_file.split("/")[-1])
            self.compare_fits(test_file, out_file)

    def test_delta_lin_mpi(self):
        """End-to-end MPI test using the 'LYA' linear-wavelength setup,
        compared against the serial reference (data/delta_extraction_lin)"""
        config_file = "{}/data/delta_lin_mpi.ini".format(THIS_DIR)
        out_dir = "{}/results/delta_extraction_lin_mpi".format(THIS_DIR)
        test_dir = "{}/data/delta_extraction_lin".format(THIS_DIR)

        self.run_delta_extraction_mpi(config_file, out_dir, test_dir)


if __name__ == "__main__":
    unittest.main()
