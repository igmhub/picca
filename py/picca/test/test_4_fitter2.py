'''
Test module
'''
import unittest
import numpy as np
import fitsio
import healpy
import subprocess
import os
import tempfile
import shutil
from pkg_resources import resource_filename
import sys

from picca.utils import userprint

from .test_helpers import update_system_status_values, compare_fits, compare_h5py




class TestCor(unittest.TestCase):
    #TODO: bad style, using it for the moment while transitioning, remove later
    compare_fits = compare_fits
    compare_h5py = compare_h5py



    @classmethod
    def setUpClass(cls):
        cls._branchFiles = tempfile.mkdtemp() + "/"

    @classmethod
    def tearDownClass(cls):
        if os.path.isdir(cls._branchFiles):
            shutil.rmtree(cls._branchFiles, ignore_errors=True)

    def test_cor(self):

        self.picca_base = resource_filename('picca',
                                            './').replace('py/picca/./', '')
        self.send_requirements()
        np.random.seed(42)

        userprint("\n")
        self._test = True
        self._masterFiles = self.picca_base + '/py/picca/test/data/'
        self.produce_folder()

        self.send_fitter2()


        if self._test:
            self.remove_folder()

        return

    def produce_folder(self):
        """
            Create the necessary folders
        """

        userprint("\n")
        lst_fold = [
            "/Products/", "/Products/Spectra/", 
            "/Products/Correlations/", 
            "/Products/Correlations/Fit/", 
        ]

        for fold in lst_fold:
            if not os.path.isdir(self._branchFiles + fold):
                os.mkdir(self._branchFiles + fold)

        return

    def remove_folder(self):
        """
            Remove the produced folders
        """

        userprint("\n")
        shutil.rmtree(self._branchFiles, ignore_errors=True)

        return

        
    def load_requirements(self):

        req = {}

        if sys.version_info > (3, 0):
            path = self.picca_base + '/requirements.txt'
        else:
            path = self.picca_base + '/requirements-python2.txt'
        with open(path, 'r') as f:
            for l in f:
                l = l.replace('\n', '').replace('==',
                                                ' ').replace('>=',
                                                             ' ').split()
                self.assertTrue(
                    len(l) == 2,
                    "requirements.txt attribute is not valid: {}".format(
                        str(l)))
                req[l[0]] = l[1]

        return req

    def send_requirements(self):

        userprint("\n")
        req = self.load_requirements()
        for req_lib, req_ver in req.items():
            try:
                local_ver = __import__(req_lib).__version__
                if local_ver != req_ver:
                    userprint(
                        "WARNING: The local version of {}: {} is different from the required version: {}"
                        .format(req_lib, local_ver, req_ver))
            except ImportError:
                userprint("WARNING: Module {} can't be found".format(req_lib))

        return

    def send_fitter2(self):

        userprint("\n")

        ### copy ini files to branch
        cmd = 'cp '+self._masterFiles+'test_fitter2/*ini ' + \
            self._branchFiles+'/Products/Correlations/Fit/'
        subprocess.call(cmd, shell=True)

        ### Set path in chi2.ini
        path = self._branchFiles + '/Products/Correlations/Fit/chi2.ini'
        value = self._branchFiles + '/Products/Correlations/Fit/config_cf.ini '
        value += self._branchFiles + '/Products/Correlations/Fit/config_xcf.ini '
        value += self._branchFiles + '/Products/Correlations/Fit/config_cf_cross.ini '
        update_system_status_values(path, 'data sets', 'ini files', value)
        value = 'PlanckDR12/PlanckDR12.fits'
        update_system_status_values(path, 'fiducial', 'filename', value)
        value = self._branchFiles + '/Products/Correlations/Fit/result_fitter2.h5'
        update_system_status_values(path, 'output', 'filename', value)

        ### Set path in config_cf.ini
        path = self._branchFiles + '/Products/Correlations/Fit/config_cf.ini'
        value = self._masterFiles + '/test_cor/exported_cf.fits.gz'
        update_system_status_values(path, 'data', 'filename', value)
        value = self._masterFiles + '/test_cor/metal_dmat.fits.gz'
        update_system_status_values(path, 'metals', 'filename', value)

        ### Set path in config_cf_cross.ini
        path = self._branchFiles + '/Products/Correlations/Fit/config_cf_cross.ini'
        value = self._masterFiles + '/test_cor/exported_cf_cross.fits.gz'
        update_system_status_values(path, 'data', 'filename', value)
        value = self._masterFiles + '/test_cor/metal_dmat_cross.fits.gz'
        update_system_status_values(path, 'metals', 'filename', value)

        ### Set path in config_xcf.ini
        path = self._branchFiles + '/Products/Correlations/Fit/config_xcf.ini'
        value = self._masterFiles + '/test_cor/exported_xcf.fits.gz'
        update_system_status_values(path, 'data', 'filename', value)
        value = self._masterFiles + '/test_cor/metal_xdmat.fits.gz'
        update_system_status_values(path, 'metals', 'filename', value)

        ### Send
        cmd = ' picca_fitter2.py '+self._branchFiles + \
            '/Products/Correlations/Fit/chi2.ini'
        subprocess.call(cmd, shell=True)

        ###These commented lines are to simplify accessing test outputs if needed
        #if os.path.exists(self._masterFiles+'new/'):
        #    os.rmdir(self._masterFiles+'new/')
        #shutil.copytree(self._branchFiles,self._masterFiles+'new/')

        ### Test
        if self._test:
            path1 = self._masterFiles + '/test_fitter2/result_fitter2.h5'
            path2 = self._branchFiles + '/Products/Correlations/Fit/result_fitter2.h5'
            self.compare_h5py(path1, path2, "picca_fitter2")


if __name__ == '__main__':
    unittest.main()
