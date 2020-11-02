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

        self.send_cf1d()
        self.send_cf1d_cross()

        self.send_cf_angl()

        self.send_cf()
        self.send_dmat()
        self.send_metal_dmat()
        self.send_wick()
        self.send_export_cf()

        self.send_cf_cross()
        self.send_dmat_cross()
        self.send_metal_dmat_cross()
        self.send_export_cf_cross()

        self.send_xcf_angl()

        self.send_xcf()
        self.send_xdmat()
        self.send_metal_xdmat()
        self.send_xwick()
        self.send_export_xcf()
        self.send_export_cross_covariance_cf_xcf()

        self.send_co()
        self.send_export_co()

        ###These commented lines are to simplify accessing test outputs if needed
        #if os.path.exists(self._masterFiles+'new/'):
        #    os.rmdir(self._masterFiles+'new/')
        #shutil.copytree(self._branchFiles,self._masterFiles+'new/')

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
            "/Products/Correlations/", "/Products/Correlations/Co_Random/",
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

    def produce_cat(self, nObj, name="cat", thidoffset=0):
        """

        """

        userprint("\n")
        userprint("Create cat with number of object = ", nObj)

        ### Create random catalog
        ra = 10. * np.random.random_sample(nObj)
        dec = 10. * np.random.random_sample(nObj)
        plate = np.random.randint(266, high=10001, size=nObj)
        mjd = np.random.randint(51608, high=57521, size=nObj)
        fiberid = np.random.randint(1, high=1001, size=nObj)
        thid = np.arange(thidoffset + 1, thidoffset + nObj + 1)
        z_qso = (3.6 - 2.0) * np.random.random_sample(nObj) + 2.0

        ### Save
        out = fitsio.FITS(self._branchFiles + "/Products/" + name + ".fits",
                          'rw',
                          clobber=True)
        cols = [ra, dec, thid, plate, mjd, fiberid, z_qso]
        names = ['RA', 'DEC', 'THING_ID', 'PLATE', 'MJD', 'FIBERID', 'Z']
        out.write(cols, names=names, extname='CAT')
        out.close()

        return

    def produce_forests(self):
        """

        """

        userprint("\n")
        nside = 8

        ### Load DRQ
        vac = fitsio.FITS(self._masterFiles + "/test_delta/cat.fits")
        ra = vac[1]["RA"][:] * np.pi / 180.
        dec = vac[1]["DEC"][:] * np.pi / 180.
        thid = vac[1]["THING_ID"][:]
        plate = vac[1]["PLATE"][:]
        mjd = vac[1]["MJD"][:]
        fiberid = vac[1]["FIBERID"][:]
        vac.close()

        ### Get Healpy pixels
        pixs = healpy.ang2pix(nside, np.pi / 2. - dec, ra)

        ### Save master file
        path = self._branchFiles + "/Products/Spectra/master.fits"
        head = {}
        head['NSIDE'] = nside
        cols = [thid, pixs, plate, mjd, fiberid]
        names = ['THING_ID', 'PIX', 'PLATE', 'MJD', 'FIBER']
        out = fitsio.FITS(path, 'rw', clobber=True)
        out.write(cols, names=names, header=head, extname="MASTER TABLE")
        out.close()

        ### Log lambda grid
        logl_min = 3.550
        logl_max = 4.025
        logl_step = 1.e-4
        log_lambda = np.arange(logl_min, logl_max, logl_step)

        ###
        for p in np.unique(pixs):

            ###
            p_thid = thid[(pixs == p)]
            p_fl = np.random.normal(loc=1.,
                                    scale=1.,
                                    size=(log_lambda.size, p_thid.size))
            p_iv = np.random.lognormal(mean=0.1,
                                       sigma=0.1,
                                       size=(log_lambda.size, p_thid.size))
            p_am = np.zeros((log_lambda.size, p_thid.size)).astype(int)
            p_am[np.random.random_sample(size=(log_lambda.size,
                                               p_thid.size)) > 0.90] = 1
            p_om = np.zeros((log_lambda.size, p_thid.size)).astype(int)

            ###
            p_path = self._branchFiles + "/Products/Spectra/pix_" + str(
                p) + ".fits"
            out = fitsio.FITS(p_path, 'rw', clobber=True)
            out.write(p_thid, header={}, extname="THING_ID_MAP")
            out.write(log_lambda, header={}, extname="LOGLAM_MAP")
            out.write(p_fl, header={}, extname="FLUX")
            out.write(p_iv, header={}, extname="IVAR")
            out.write(p_am, header={}, extname="ANDMASK")
            out.write(p_om, header={}, extname="ORMASK")
            out.close()

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
    def send_cf1d(self):

        userprint("\n")
        ### Send
        cmd = " picca_cf1d.py"
        cmd += " --in-dir " + self._masterFiles + "/test_delta/Delta_LYA/"
        cmd += " --out " + self._branchFiles + "/Products/Correlations/cf1d.fits.gz"
        cmd += " --nproc 1"
        subprocess.call(cmd, shell=True)

        ### Test
        if self._test:
            path1 = self._masterFiles + "/test_cor/cf1d.fits.gz"
            path2 = self._branchFiles + "/Products/Correlations/cf1d.fits.gz"
            self.compare_fits(path1, path2, "picca_cf1d.py")

        return

    def send_cf1d_cross(self):

        userprint("\n")
        ### Send
        cmd = " picca_cf1d.py"
        cmd += " --in-dir " + self._masterFiles + "/test_delta/Delta_LYA/"
        cmd += " --in-dir2 " + self._masterFiles + "/test_delta/Delta_LYA/"
        cmd += " --out " + self._branchFiles + "/Products/Correlations/cf1d_cross.fits.gz"
        cmd += " --nproc 1"
        subprocess.call(cmd, shell=True)

        ### Test
        if self._test:
            path1 = self._masterFiles + "/test_cor/cf1d_cross.fits.gz"
            path2 = self._branchFiles + "/Products/Correlations/cf1d_cross.fits.gz"
            self.compare_fits(path1, path2, "picca_cf1d.py")

        return

    def send_cf_angl(self):

        userprint("\n")
        ### Send
        cmd = " picca_cf_angl.py"
        cmd += " --in-dir " + self._masterFiles + "/test_delta/Delta_LYA/"
        cmd += " --out " + self._branchFiles + "/Products/Correlations/cf_angl.fits.gz"
        cmd += " --nproc 1"
        subprocess.call(cmd, shell=True)

        ### Test
        if self._test:
            path1 = self._masterFiles + "/test_cor/cf_angl.fits.gz"
            path2 = self._branchFiles + "/Products/Correlations/cf_angl.fits.gz"
            self.compare_fits(path1, path2, "picca_cf_angl.py")

        return

    def send_cf(self):

        userprint("\n")
        ### Send
        cmd = " picca_cf.py"
        cmd += " --in-dir " + self._masterFiles + "/test_delta/Delta_LYA/"
        cmd += " --out " + self._branchFiles + "/Products/Correlations/cf.fits.gz"
        cmd += " --rp-min +0.0"
        cmd += " --rp-max +60.0"
        cmd += " --rt-max +60.0"
        cmd += " --np 15"
        cmd += " --nt 15"
        cmd += " --nproc 1"
        cmd += ' --remove-same-half-plate-close-pairs'
        subprocess.call(cmd, shell=True)

        ### Test
        if self._test:
            path1 = self._masterFiles + "/test_cor/cf.fits.gz"
            path2 = self._branchFiles + "/Products/Correlations/cf.fits.gz"
            self.compare_fits(path1, path2, "picca_cf.py")

        return

    def send_dmat(self):

        userprint("\n")
        ### Send
        cmd = " picca_dmat.py"
        cmd += " --in-dir " + self._masterFiles + "/test_delta/Delta_LYA/"
        cmd += " --out " + self._branchFiles + "/Products/Correlations/dmat.fits.gz"
        cmd += " --rp-min +0.0"
        cmd += " --rp-max +60.0"
        cmd += " --rt-max +60.0"
        cmd += " --np 15"
        cmd += " --nt 15"
        cmd += " --rej 0.99 "
        cmd += " --nproc 1"
        cmd += ' --remove-same-half-plate-close-pairs'
        subprocess.call(cmd, shell=True)

        ### Test
        if self._test:
            path1 = self._masterFiles + "/test_cor/dmat.fits.gz"
            path2 = self._branchFiles + "/Products/Correlations/dmat.fits.gz"
            self.compare_fits(path1, path2, "picca_dmat.py")

        return

    def send_metal_dmat(self):

        userprint("\n")
        ### Send
        cmd = " picca_metal_dmat.py"
        cmd += " --in-dir " + self._masterFiles + "/test_delta/Delta_LYA/"
        cmd += " --out " + self._branchFiles + "/Products/Correlations/metal_dmat.fits.gz"
        cmd += r" --abs-igm SiIII\(1207\)"
        cmd += " --rp-min +0.0"
        cmd += " --rp-max +60.0"
        cmd += " --rt-max +60.0"
        cmd += " --np 15"
        cmd += " --nt 15"
        cmd += " --rej 0.99 "
        cmd += " --nproc 1"
        cmd += ' --remove-same-half-plate-close-pairs'
        subprocess.call(cmd, shell=True)

        ### Test
        if self._test:
            path1 = self._masterFiles + "/test_cor/metal_dmat.fits.gz"
            path2 = self._branchFiles + "/Products/Correlations/metal_dmat.fits.gz"
            self.compare_fits(path1, path2, "picca_metal_dmat.py")

        return

    def send_wick(self):

        userprint("\n")
        ### Send
        cmd = " picca_wick.py"
        cmd += " --in-dir " + self._masterFiles + "/test_delta/Delta_LYA/"
        cmd += " --out " + self._branchFiles + "/Products/Correlations/wick.fits.gz"
        cmd += " --cf1d " + self._masterFiles +"/test_cor/cf1d.fits.gz"
        cmd += " --rp-min +0.0"
        cmd += " --rp-max +60.0"
        cmd += " --rt-max +60.0"
        cmd += " --np 15"
        cmd += " --nt 15"
        cmd += " --rej 0.99 "
        cmd += " --nproc 1"
        subprocess.call(cmd, shell=True)

        ### Test
        if self._test:
            path1 = self._masterFiles + "/test_cor/wick.fits.gz"
            path2 = self._branchFiles + "/Products/Correlations/wick.fits.gz"
            self.compare_fits(path1, path2, "picca_wick.py")

        return

    def send_export_cf(self):

        userprint("\n")
        ### Send
        cmd = " picca_export.py"
        cmd += " --data " + self._masterFiles + "/test_cor/cf.fits.gz"
        cmd += " --dmat " + self._masterFiles + "/test_cor/dmat.fits.gz"
        cmd += " --out " + self._branchFiles + "/Products/Correlations/exported_cf.fits.gz"
        returncode = subprocess.call(cmd, shell=True)
        self.assertEqual(returncode, 0, "export_cf did not finish")
        return

    def send_cf_cross(self):

        userprint("\n")
        ### Send
        cmd = " picca_cf.py"
        cmd += " --in-dir  " + self._masterFiles + "/test_delta/Delta_LYA/"
        cmd += " --in-dir2 " + self._masterFiles + "/test_delta/Delta_LYA/"
        cmd += " --out " + self._branchFiles + "/Products/Correlations/cf_cross.fits.gz"
        cmd += " --rp-min -60.0"
        cmd += " --rp-max +60.0"
        cmd += " --rt-max +60.0"
        cmd += " --np 30"
        cmd += " --nt 15"
        cmd += " --nproc 1"
        cmd += ' --remove-same-half-plate-close-pairs'
        cmd += " --unfold-cf"
        subprocess.call(cmd, shell=True)

        ### Test
        if self._test:
            path1 = self._masterFiles + "/test_cor/cf_cross.fits.gz"
            path2 = self._branchFiles + "/Products/Correlations/cf_cross.fits.gz"
            self.compare_fits(path1, path2, "picca_cf.py")

        return

    def send_dmat_cross(self):

        userprint("\n")
        ### Send
        cmd = " picca_dmat.py"
        cmd += " --in-dir  " + self._masterFiles + "/test_delta/Delta_LYA/"
        cmd += " --in-dir2 " + self._masterFiles + "/test_delta/Delta_LYA/"
        cmd += " --out " + self._branchFiles + "/Products/Correlations/dmat_cross.fits.gz"
        cmd += " --rp-min -60.0"
        cmd += " --rp-max +60.0"
        cmd += " --rt-max +60.0"
        cmd += " --np 30"
        cmd += " --nt 15"
        cmd += " --rej 0.99 "
        cmd += " --nproc 1"
        cmd += ' --remove-same-half-plate-close-pairs'
        cmd += " --unfold-cf"
        subprocess.call(cmd, shell=True)

        ### Test
        if self._test:
            path1 = self._masterFiles + "/test_cor/dmat_cross.fits.gz"
            path2 = self._branchFiles + "/Products/Correlations/dmat_cross.fits.gz"
            self.compare_fits(path1, path2, "picca_dmat.py")

        return

    def send_metal_dmat_cross(self):

        userprint("\n")
        ### Send
        cmd = " picca_metal_dmat.py"
        cmd += " --in-dir " + self._masterFiles + "/test_delta/Delta_LYA/"
        cmd += " --in-dir2 " + self._masterFiles + "/test_delta/Delta_LYA/"
        cmd += " --out " + self._branchFiles + \
            "/Products/Correlations/metal_dmat_cross.fits.gz"
        cmd += r" --abs-igm SiIII\(1207\)"
        cmd += r" --abs-igm2 SiIII\(1207\)"
        cmd += " --rp-min -60.0"
        cmd += " --rp-max +60.0"
        cmd += " --rt-max +60.0"
        cmd += " --np 30"
        cmd += " --nt 15"
        cmd += " --rej 0.99 "
        cmd += " --nproc 1"
        cmd += ' --remove-same-half-plate-close-pairs'
        cmd += " --unfold-cf"
        subprocess.call(cmd, shell=True)

        ### Test
        if self._test:
            path1 = self._masterFiles + "/test_cor/metal_dmat_cross.fits.gz"
            path2 = self._branchFiles + "/Products/Correlations/metal_dmat_cross.fits.gz"
            self.compare_fits(path1, path2, "picca_metal_dmat.py")

        return

    def send_export_cf_cross(self):

        userprint("\n")
        ### Send
        cmd = " picca_export.py"
        cmd += " --data " + self._masterFiles + "/test_cor/cf_cross.fits.gz"
        cmd += " --dmat " + self._masterFiles + "/test_cor/dmat_cross.fits.gz"
        cmd += " --out " + self._branchFiles + \
            "/Products/Correlations/exported_cf_cross.fits.gz"
        returncode = subprocess.call(cmd, shell=True)
        self.assertEqual(returncode, 0, "export_cf_cross did not finish")

        return

    def send_xcf_angl(self):

        userprint("\n")
        ### Send
        cmd = " picca_xcf_angl.py"
        cmd += " --in-dir " + self._masterFiles + "/test_delta/Delta_LYA/"
        cmd += " --drq " + self._masterFiles + "/test_delta/cat.fits"
        cmd += " --out " + self._branchFiles + "/Products/Correlations/xcf_angl.fits.gz"
        cmd += " --nproc 1"
        subprocess.call(cmd, shell=True)

        ### Test
        if self._test:
            path1 = self._masterFiles + "/test_cor/xcf_angl.fits.gz"
            path2 = self._branchFiles + "/Products/Correlations/xcf_angl.fits.gz"
            self.compare_fits(path1, path2, "picca_xcf_angl.py")

        return

    def send_xcf(self):

        userprint("\n")
        ### Send
        cmd = " picca_xcf.py"
        cmd += " --in-dir " + self._masterFiles + "/test_delta/Delta_LYA/"
        cmd += " --drq " + self._masterFiles + "/test_delta/cat.fits"
        cmd += " --out " + self._branchFiles + "/Products/Correlations/xcf.fits.gz"
        cmd += " --rp-min -60.0"
        cmd += " --rp-max +60.0"
        cmd += " --rt-max +60.0"
        cmd += " --np 30"
        cmd += " --nt 15"
        cmd += " --nproc 1"
        subprocess.call(cmd, shell=True)

        ### Test
        if self._test:
            path1 = self._masterFiles + "/test_cor/xcf.fits.gz"
            path2 = self._branchFiles + "/Products/Correlations/xcf.fits.gz"
            self.compare_fits(path1, path2, "picca_xcf.py")

        return

    def send_xdmat(self):

        userprint("\n")
        ### Send
        cmd = " picca_xdmat.py"
        cmd += " --in-dir  " + self._masterFiles + "/test_delta/Delta_LYA/"
        cmd += " --drq " + self._masterFiles + "/test_delta/cat.fits"
        cmd += " --out " + self._branchFiles + "/Products/Correlations/xdmat.fits.gz"
        cmd += " --rp-min -60.0"
        cmd += " --rp-max +60.0"
        cmd += " --rt-max +60.0"
        cmd += " --np 30"
        cmd += " --nt 15"
        cmd += " --rej 0.99 "
        cmd += " --nproc 1"
        subprocess.call(cmd, shell=True)

        ### Test
        if self._test:
            path1 = self._masterFiles + "/test_cor/xdmat.fits.gz"
            path2 = self._branchFiles + "/Products/Correlations/xdmat.fits.gz"
            self.compare_fits(path1, path2, "picca_xdmat.py")

        return

    def send_metal_xdmat(self):

        userprint("\n")
        ### Send
        cmd = " picca_metal_xdmat.py"
        cmd += " --in-dir " + self._masterFiles + "/test_delta/Delta_LYA/"
        cmd += " --drq " + self._masterFiles + "/test_delta/cat.fits"
        cmd += " --out " + self._branchFiles + "/Products/Correlations/metal_xdmat.fits.gz"
        cmd += r" --abs-igm SiIII\(1207\)"
        cmd += " --rp-min -60.0"
        cmd += " --rp-max +60.0"
        cmd += " --rt-max +60.0"
        cmd += " --np 30"
        cmd += " --nt 15"
        cmd += " --rej 0.99 "
        cmd += " --nproc 1"
        subprocess.call(cmd, shell=True)

        ### Test
        if self._test:
            path1 = self._masterFiles + "/test_cor/metal_xdmat.fits.gz"
            path2 = self._branchFiles + "/Products/Correlations/metal_xdmat.fits.gz"
            self.compare_fits(path1, path2, "picca_metal_xdmat.py")

        return

    def send_xwick(self):

        userprint("\n")
        ### Send
        cmd = " picca_xwick.py"
        cmd += " --in-dir " + self._masterFiles + "/test_delta/Delta_LYA/"
        cmd += " --drq " + self._masterFiles + "/test_delta/cat.fits"
        cmd += " --out " + self._branchFiles + "/Products/Correlations/xwick.fits.gz"
        cmd += " --cf1d " + self._masterFiles + "/test_cor/cf1d.fits.gz"
        cmd += " --rp-min -60.0"
        cmd += " --rp-max +60.0"
        cmd += " --rt-max +60.0"
        cmd += " --np 30"
        cmd += " --nt 15"
        cmd += " --rej 0.99 "
        cmd += " --nproc 1"
        subprocess.call(cmd, shell=True)

        ### Test
        if self._test:
            path1 = self._masterFiles + "/test_cor/xwick.fits.gz"
            path2 = self._branchFiles + "/Products/Correlations/xwick.fits.gz"
            self.compare_fits(path1, path2, "picca_xwick.py")

        return

    def send_export_xcf(self):

        userprint("\n")
        ### Send
        cmd = " picca_export.py"
        cmd += " --data " + self._masterFiles + "/test_cor/xcf.fits.gz"
        cmd += " --dmat " + self._masterFiles + "/test_cor/xdmat.fits.gz"
        cmd += " --out " + self._branchFiles + \
            "/Products/Correlations/exported_xcf.fits.gz"
        returncode = subprocess.call(cmd, shell=True)
        self.assertEqual(returncode, 0, "export_xcf did not finish")

        return

    def send_export_cross_covariance_cf_xcf(self):

        userprint("\n")
        ### Send
        cmd = " picca_export_cross_covariance.py"
        cmd += " --data1 " + self._masterFiles + "/test_cor/cf.fits.gz"
        cmd += " --data2 " + self._masterFiles + "/test_cor/xcf.fits.gz"
        cmd += " --out " + self._branchFiles + \
            "/Products/Correlations/exported_cross_covariance_cf_xcf.fits.gz"
        returncode = subprocess.call(cmd, shell=True)
        self.assertEqual(returncode, 0,
                         "export_cross_covariance_cf_xcf did not finish")

        return

    def send_co(self):

        userprint("\n")
        ### Send
        cmd = " picca_co.py"
        cmd += " --drq " + self._masterFiles + "/test_delta/cat.fits"
        cmd += " --out " + self._branchFiles + "/Products/Correlations/co_DD.fits.gz"
        cmd += " --rp-min 0."
        cmd += " --rp-max +60.0"
        cmd += " --rt-max +60.0"
        cmd += " --np 15"
        cmd += " --nt 15"
        cmd += " --nproc 1"
        cmd += " --type-corr DD"
        returncode = subprocess.call(cmd, shell=True)
        self.assertEqual(returncode, 0, "picca_co did not finish on DD")
        ### Send
        cmd = " picca_co.py"
        cmd += " --drq " + self._masterFiles + "/test_delta/random.fits"
        cmd += " --out " + self._branchFiles + \
            "/Products/Correlations/Co_Random/co_RR.fits.gz"
        cmd += " --rp-min 0."
        cmd += " --rp-max +60.0"
        cmd += " --rt-max +60.0"
        cmd += " --np 15"
        cmd += " --nt 15"
        cmd += " --nproc 1"
        cmd += " --type-corr RR"
        returncode = subprocess.call(cmd, shell=True)
        self.assertEqual(returncode, 0, "picca_co did not finish on RR")
        ### Send
        cmd = " picca_co.py"
        cmd += " --drq " + self._masterFiles + "/test_delta/cat.fits"
        cmd += " --drq2 " + self._masterFiles + "/test_delta/random.fits"
        cmd += " --out " + self._branchFiles + \
            "/Products/Correlations/Co_Random/co_DR.fits.gz"
        cmd += " --rp-min 0."
        cmd += " --rp-max +60.0"
        cmd += " --rt-max +60.0"
        cmd += " --np 15"
        cmd += " --nt 15"
        cmd += " --nproc 1"
        cmd += " --type-corr DR"
        returncode = subprocess.call(cmd, shell=True)
        self.assertEqual(returncode, 0, "picca_co did not finish on DR")
        ### Send
        cmd = " picca_co.py"
        cmd += " --drq " + self._masterFiles + "/test_delta/random.fits"
        cmd += " --drq2 " + self._masterFiles + "/test_delta/cat.fits"
        cmd += " --out " + self._branchFiles + \
            "/Products/Correlations/Co_Random/co_RD.fits.gz"
        cmd += " --rp-min 0."
        cmd += " --rp-max +60.0"
        cmd += " --rt-max +60.0"
        cmd += " --np 15"
        cmd += " --nt 15"
        cmd += " --nproc 1"
        cmd += " --type-corr RD"
        returncode = subprocess.call(cmd, shell=True)
        self.assertEqual(returncode, 0, "picca_co did not finish on RD")

        ### Test
        if self._test:
            path1 = self._masterFiles + "/test_cor/co_DD.fits.gz"
            path2 = self._branchFiles + "/Products/Correlations/co_DD.fits.gz"
            self.compare_fits(path1, path2, "picca_co.py DD")

            path1 = self._masterFiles + "/test_cor/co_RR.fits.gz"
            path2 = self._branchFiles + "/Products/Correlations/Co_Random/co_RR.fits.gz"
            self.compare_fits(path1, path2, "picca_co.py RR")

            path1 = self._masterFiles + "/test_cor/co_DR.fits.gz"
            path2 = self._branchFiles + "/Products/Correlations/Co_Random/co_DR.fits.gz"
            self.compare_fits(path1, path2, "picca_co.py DR")

            path1 = self._masterFiles + "/test_cor/co_RD.fits.gz"
            path2 = self._branchFiles + "/Products/Correlations/Co_Random/co_RD.fits.gz"
            self.compare_fits(path1, path2, "picca_co.py RD")

        return

    def send_export_co(self):

        userprint("\n")
        ### Send
        cmd = " picca_export_co.py"
        cmd += " --DD-file " + self._masterFiles + "/test_cor/co_DD.fits.gz"
        cmd += " --RR-file " + self._masterFiles + \
            "/test_cor/co_RR.fits.gz"
        cmd += " --DR-file " + self._masterFiles + \
            "/test_cor/co_DR.fits.gz"
        cmd += " --RD-file " + self._masterFiles + \
            "/test_cor/co_RD.fits.gz"
        cmd += " --out " + self._branchFiles + "/Products/Correlations/exported_co.fits.gz"
        cmd += " --get-cov-from-poisson"
        returncode = subprocess.call(cmd, shell=True)
        self.assertEqual(returncode, 0, "picca_export_co did not finish")

        return


if __name__ == '__main__':
    unittest.main()
