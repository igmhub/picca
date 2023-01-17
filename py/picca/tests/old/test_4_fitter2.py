'''
Test module
'''
import unittest
import os
import glob
import shutil

from picca.utils import userprint
import picca.bin.old.picca_fitter2 as picca_fitter2

from picca.tests.test_helpers import AbstractTest


class TestFitter(AbstractTest):
    """
        Tests the fitter
    """

    def produce_folder(self):
        """
            Create the necessary folders
        """

        userprint("\n")
        lst_fold = [
            "/Products/",
            "/Products/Spectra/",
            "/Products/Correlations/",
            "/Products/Correlations/Fit/",
        ]

        for fold in lst_fold:
            if not os.path.isdir(self._branchFiles + fold):
                os.mkdir(self._branchFiles + fold)

        return

    def test_fitter2(self):

        self._test = True

        userprint("\n")

        ### copy ini files to branch
        filestocopy=glob.glob(self._masterFiles+'test_fitter2/*ini')
        for f in filestocopy:
            shutil.copy(f, self._branchFiles+'/Products/Correlations/Fit/')

        ### Set path in chi2.ini
        path = self._branchFiles + '/Products/Correlations/Fit/chi2.ini'
        value = self._branchFiles + '/Products/Correlations/Fit/config_cf.ini '
        value += self._branchFiles + '/Products/Correlations/Fit/config_xcf.ini '
        value += self._branchFiles + '/Products/Correlations/Fit/config_cf_cross.ini '
        self.update_system_status_values(path, 'data sets', 'ini files', value)
        value = 'PlanckDR12/PlanckDR12.fits'
        self.update_system_status_values(path, 'fiducial', 'filename', value)
        value = self._branchFiles + '/Products/Correlations/Fit/result_fitter2.h5'
        self.update_system_status_values(path, 'output', 'filename', value)

        ### Set path in config_cf.ini
        path = self._branchFiles + '/Products/Correlations/Fit/config_cf.ini'
        value = self._masterFiles + '/test_cor/exported_cf.fits.gz'
        self.update_system_status_values(path, 'data', 'filename', value)
        value = self._masterFiles + '/test_cor/metal_dmat.fits.gz'
        self.update_system_status_values(path, 'metals', 'filename', value)

        ### Set path in config_cf_cross.ini
        path = self._branchFiles + '/Products/Correlations/Fit/config_cf_cross.ini'
        value = self._masterFiles + '/test_cor/exported_cf_cross.fits.gz'
        self.update_system_status_values(path, 'data', 'filename', value)
        value = self._masterFiles + '/test_cor/metal_dmat_cross.fits.gz'
        self.update_system_status_values(path, 'metals', 'filename', value)

        ### Set path in config_xcf.ini
        path = self._branchFiles + '/Products/Correlations/Fit/config_xcf.ini'
        value = self._masterFiles + '/test_cor/exported_xcf.fits.gz'
        self.update_system_status_values(path, 'data', 'filename', value)
        value = self._masterFiles + '/test_cor/metal_xdmat.fits.gz'
        self.update_system_status_values(path, 'metals', 'filename', value)

        ### Send
        picca_fitter2.main(self._branchFiles+'/Products/Correlations/Fit/chi2.ini')

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
