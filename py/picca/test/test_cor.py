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
import h5py
from pkg_resources import resource_filename
import sys

from picca.utils import userprint
import configparser as ConfigParser


def update_system_status_values(path, section, system, value):

    ### Make ConfigParser case sensitive
    class CaseConfigParser(ConfigParser.ConfigParser):

        def optionxform(self, optionstr):
            return optionstr

    cp = CaseConfigParser()
    cp.read(path)
    cf = open(path, 'w')
    cp.set(section, system, value)
    cp.write(cf)
    cf.close()

    return


class TestCor(unittest.TestCase):

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
        self.produce_cat(nObj=1000)
        self.produce_forests()
        self.produce_cat(nObj=1000, name="random", thidoffset=1000)

        self.produce_cat_minisv(nObj=1000)
        self.produce_forests_minisv()
        self.send_delta_Pk1D_minisv()

        self.send_delta()

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

        self.send_fitter2()

        self.send_delta_Pk1D()
        self.send_Pk1D()

        if self._test:
            self.remove_folder()

        return

    def produce_folder(self):
        """
            Create the necessary folders
        """

        userprint("\n")
        lst_fold = [
            "/Products/", "/Products/Spectra/", "/Products/Delta_LYA/",
            "/Products/Delta_LYA/Delta/", "/Products/Delta_LYA/Log/",
            "/Products/Correlations/", "/Products/Correlations/Co_Random/",
            "/Products/Correlations/Fit/", "/Products/Delta_Pk1D/",
            "/Products/Delta_Pk1D/Delta/", "/Products/Delta_Pk1D/Log/",
            "/Products/Pk1D/", "/Products/Spectra_MiniSV/",
            "/Products/Delta_Pk1D_MiniSV/",
            "/Products/Delta_Pk1D_MiniSV/Delta/",
            "/Products/Delta_Pk1D_MiniSV/Log/"
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
        vac = fitsio.FITS(self._branchFiles + "/Products/cat.fits")
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

    def produce_cat_minisv(self, nObj, name="cat_minisv"):
        """

        """

        userprint("\n")
        userprint("Create cat with number of object = ", nObj)

        ### Create random catalog
        ra = 10. * np.random.random_sample(nObj)
        dec = 10. * np.random.random_sample(nObj)
        tile = np.random.randint(
            100, high=110, size=nObj,
            dtype=np.int32)  # restricted to not get too many files
        petal_loc = np.random.randint(0, high=10, size=nObj, dtype=np.int16)
        night = np.array([
            np.int32(f'{yyyy:04d}{mm:02d}{dd:02d}') for dd, mm, yyyy in zip(
                np.random.randint(1, high=3, size=nObj),
                np.random.randint(1, high=2, size=nObj),
                np.random.randint(2019, high=2020, size=nObj))
        ])  # restricted to not get too many files
        fiberid = np.random.randint(1, high=5001, size=nObj, dtype=np.int32)
        thid = np.arange(2**60, 2**60 + nObj)
        z_qso = (3.6 - 2.0) * np.random.random_sample(nObj) + 2.0

        ### Save
        out = fitsio.FITS(self._branchFiles + "/Products/" + name + ".fits",
                          'rw',
                          clobber=True)
        cols = [ra, dec, thid, tile, petal_loc, night, fiberid, z_qso]
        names = [
            'TARGET_RA', 'TARGET_DEC', 'TARGETID', 'TILEID', 'PETAL_LOC',
            'NIGHT', 'FIBER', 'Z'
        ]
        out.write(cols, names=names, extname='CAT')
        out.close()

        return

    def produce_forests_minisv(self):
        """

        """

        userprint("\n")

        ### Load DRQ
        vac = fitsio.FITS(self._branchFiles + "/Products/cat_minisv.fits")
        ra = vac[1]["TARGET_RA"][:] * np.pi / 180.
        dec = vac[1]["TARGET_DEC"][:] * np.pi / 180.
        thid = vac[1]["TARGETID"][:]
        petal_loc = vac[1]["PETAL_LOC"][:]
        tile = vac[1]["TILEID"][:]
        night = vac[1]["NIGHT"][:]
        fiberid = vac[1]["FIBER"][:]

        tile_night_petal_combined = np.unique([*zip(tile, night, petal_loc)],
                                              axis=0)
        vac.close()

        cols = [thid, ra, dec, night, fiberid, tile, petal_loc]
        names = [
            'TARGETID', 'TARGET_RA', 'TARGET_DEC', 'NIGHT', 'FIBER', 'TILEID',
            'PETAL_LOC'
        ]

        ### Log lambda grid
        l_min_b = 3600
        l_max_b = 5800

        l_min_r = 5760
        l_max_r = 7620

        l_min_z = 7520
        l_max_z = 9824

        l_step = 0.8
        lam_b = np.arange(l_min_b, l_max_b, l_step)
        lam_r = np.arange(l_min_r, l_max_r, l_step)
        lam_z = np.arange(l_min_z, l_max_z, l_step)

        lam = {'b': lam_b, 'r': lam_r, 'z': lam_z}

        ###
        for t, n, p in tile_night_petal_combined:

            ###
            selector = ((petal_loc == p) & (tile == t) & (night == n))

            cols = [
                thid[selector], ra[selector], dec[selector], night[selector],
                fiberid[selector], tile[selector], petal_loc[selector]
            ]
            p_thid = cols[0]
            names = [
                'TARGETID', 'TARGET_RA', 'TARGET_DEC', 'NIGHT', 'FIBER',
                'TILEID', 'PETAL_LOC'
            ]
            p_fl = {}
            p_iv = {}
            p_m = {}
            p_res = {}
            for key, lam_key in lam.items():
                p_fl[key] = np.random.normal(loc=1.,
                                             scale=1.,
                                             size=(p_thid.size, lam_key.size))
                p_iv[key] = np.random.lognormal(mean=0.1,
                                                sigma=0.1,
                                                size=(p_thid.size,
                                                      lam_key.size))
                p_m[key] = np.zeros((p_thid.size, lam_key.size)).astype(int)
                p_m[key][np.random.random_sample(
                    size=(p_thid.size, lam_key.size)) > 0.90] = 1
                tmp = np.exp(
                    -((np.arange(11) - 5) / 0.6382)**2
                )  #to fake resolution from a gaussian, this assumes R=3000 at minimum wavelength
                tmp = np.repeat(tmp[np.newaxis, :, np.newaxis],
                                p_thid.size,
                                axis=0)
                p_res[key] = np.repeat(tmp, lam_key.size, axis=2)

            p_path = self._branchFiles + f"/Products/Spectra_MiniSV/{t}/{n}/"
            if not os.path.isdir(p_path):
                os.makedirs(p_path)
            p_file = p_path + f"coadd-{p}-{t}-{n}.fits"
            ###
            out = fitsio.FITS(p_file, 'rw', clobber=True)

            out.write(cols, names=names, extname="FIBERMAP")
            for key, lam_key in lam.items():
                out.write_image(lam_key, extname=f"{key.upper()}_WAVELENGTH")
                out.write(p_fl[key], extname=f"{key.upper()}_FLUX")
                out.write(p_iv[key], extname=f"{key.upper()}_IVAR")
                out.write(p_m[key], extname=f"{key.upper()}_MASK")
                out.write(p_res[key], extname=f"{key.upper()}_RESOLUTION")
            out.close()
        return

    def compare_fits(self, path1, path2, nameRun=""):

        userprint("\n")
        m = fitsio.FITS(path1)
        self.assertTrue(os.path.isfile(path2), "{}".format(nameRun))
        b = fitsio.FITS(path2)

        self.assertEqual(len(m), len(b), "{}".format(nameRun))

        for i, _ in enumerate(m):

            ###
            r_m = m[i].read_header().records()
            ld_m = []
            for el in r_m:
                name = el['name']
                if len(name) > 5 and name[:5] == "TTYPE":
                    ld_m += [el['value'].replace(" ", "")]
            ###
            r_b = b[i].read_header().records()
            ld_b = []
            for el in r_b:
                name = el['name']
                if len(name) > 5 and name[:5] == "TTYPE":
                    ld_b += [el['value'].replace(" ", "")]

            self.assertListEqual(ld_m, ld_b, "{}".format(nameRun))

            for k in ld_m:
                d_m = m[i][k][:]
                d_b = b[i][k][:]
                if d_m.dtype in ['<U23',
                                 'S23']:  # for fitsio old version compatibility
                    d_m = np.char.strip(d_m)
                if d_b.dtype in ['<U23',
                                 'S23']:  # for fitsio old version compatibility
                    d_b = np.char.strip(d_b)
                self.assertEqual(d_m.size, d_b.size,
                                 "{}: Header key is {}".format(nameRun, k))
                if not np.array_equal(d_m, d_b):
                    userprint(
                        "WARNING: {}: Header key is {}, arrays are not exactly equal, using allclose"
                        .format(nameRun, k))
                    diff = d_m - d_b
                    w = d_m != 0.
                    diff[w] = np.absolute(diff[w] / d_m[w])
                    allclose = np.allclose(d_m, d_b)
                    self.assertTrue(
                        allclose,
                        "{}: Header key is {}, maximum relative difference is {}"
                        .format(nameRun, k, diff.max()))
                    userprint(f"OK, maximum relative difference {diff.max():.2e}")

        m.close()
        b.close()

        return

    def compare_h5py(self, path1, path2, nameRun=""):

        def compare_attributes(atts1, atts2):
            self.assertEqual(len(atts1.keys()), len(atts2.keys()),
                             "{}".format(nameRun))
            self.assertListEqual(sorted(atts1.keys()), sorted(atts2.keys()),
                                 "{}".format(nameRun))
            for item in atts1:
                nequal = True
                if isinstance(atts1[item], np.ndarray): #the dtype check is needed for h5py version 3
                    dtype1=atts1[item].dtype
                    dtype2=atts2[item].dtype
                    if dtype1==dtype2:
                        nequal = np.logical_not(
                            np.array_equal(atts1[item], atts2[item]))
                    else:
                        userprint(f"Note that the test file has different dtype for attribute {item}")
                        nequal = np.logical_not(np.array_equal(atts1[item].astype(atts2[item].dtype), atts2[item]))
                        if nequal:
                            nequal = np.logical_not(np.array_equal(atts2[item].astype(atts1[item].dtype), atts1[item]))
                else:
                    nequal = atts1[item] != atts2[item]
                if nequal:
                    userprint(
                        "WARNING: {}: not exactly equal, using allclose for {}".
                        format(nameRun, item))
                    userprint(atts1[item], atts2[item])
                    allclose = np.allclose(atts1[item], atts2[item])
                    self.assertTrue(allclose, "{}".format(nameRun))
            return

        def compare_values(val1, val2):
            if not np.array_equal(val1, val2):
                userprint(
                    "WARNING: {}: not exactly equal, using allclose".format(
                        nameRun))
                allclose = np.allclose(val1, val2)
                self.assertTrue(allclose, "{}".format(nameRun))
            return

        userprint("\n")
        m = h5py.File(path1, "r")
        self.assertTrue(os.path.isfile(path2), "{}".format(nameRun))
        b = h5py.File(path2, "r")

        self.assertListEqual(sorted(m.keys()), sorted(b.keys()),
                             "{}".format(nameRun))

        ### best fit
        k = 'best fit'
        compare_attributes(m[k].attrs, b[k].attrs)

        ### fit data
        for k in m.keys():
            if k in ['best fit', 'fast mc', 'minos', 'chi2 scan']:
                continue
            compare_attributes(m[k].attrs, b[k].attrs)
            compare_values(m[k]['fit'][()], b[k]['fit'][()])

        ### minos
        k = 'minos'
        compare_attributes(m[k].attrs, b[k].attrs)
        for p in m[k].keys():
            compare_attributes(m[k][p].attrs, b[k][p].attrs)

        ### chi2 scan
        k = 'chi2 scan'
        for p in m[k].keys():
            compare_attributes(m[k][p].attrs, b[k][p].attrs)
            if p == 'result':
                compare_values(m[k][p]['values'][()], b[k][p]['values'][()])

        return

    def load_requirements(self):

        req = {}

        if sys.version_info > (3, 0):
            path = self.picca_base + '/requirements.txt'
        else:
            path = self.picca_base + '/requirements-python2.txt'
        with open(path, 'r') as f:
            for l in f:
                l = l.replace('\n', '').replace('==', ' ').replace('>=',
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

    def send_delta(self):

        userprint("\n")
        ### Send
        cmd = " picca_deltas.py"
        cmd += " --in-dir " + self._branchFiles + "/Products/Spectra/"
        cmd += " --drq " + self._branchFiles + "/Products/cat.fits"
        cmd += " --out-dir " + self._branchFiles + "/Products/Delta_LYA/Delta/"
        cmd += " --iter-out-prefix " + self._branchFiles + \
            "/Products/Delta_LYA/Log/delta_attributes"
        cmd += " --log " + self._branchFiles + "/Products/Delta_LYA/Log/input.log"
        cmd += " --nproc 1"
        subprocess.call(cmd, shell=True)

        ### Test
        if self._test:
            path1 = self._masterFiles + "/delta_attributes.fits.gz"
            path2 = self._branchFiles + "/Products/Delta_LYA/Log/delta_attributes.fits.gz"
            self.compare_fits(path1, path2, "picca_deltas.py")

        return

    def send_delta_Pk1D(self):

        userprint("\n")
        ### Path
        path_to_etc = self.picca_base + '/etc/'
        ### Send
        cmd = " picca_deltas.py"
        cmd += " --in-dir " + self._masterFiles + "/test_Pk1D/Spectra_test/"
        cmd += " --drq " + self._masterFiles + "/test_Pk1D/DRQ_test.fits"
        cmd += " --out-dir " + self._branchFiles + "/Products/Delta_Pk1D/Delta/"
        cmd += " --iter-out-prefix " + self._branchFiles + \
            "/Products/Delta_Pk1D/Log/delta_attributes"
        cmd += " --log " + self._branchFiles + "/Products/Delta_Pk1D/Log/input.log"
        cmd += " --delta-format Pk1D --mode spec --order 0 --use-constant-weight"
        cmd += " --rebin 1 --lambda-min 3650. --lambda-max 7200.0 --lambda-rest-min 1050.0 --lambda-rest-max 1180"
        cmd += " --nproc 1"
        cmd += " --best-obs"
        cmd += " --mask-file " + path_to_etc + "/list_veto_line_Pk1D.txt"
        subprocess.call(cmd, shell=True)

        ### Test
        if self._test:
            path1 = self._masterFiles + "/test_Pk1D/delta_attributes_Pk1D.fits.gz"
            path2 = self._branchFiles + "/Products/Delta_Pk1D/Log/delta_attributes.fits.gz"
            self.compare_fits(path1, path2, "picca_deltas.py")

            path1 = self._masterFiles + "/test_Pk1D/delta-64_Pk1D.fits.gz"
            path2 = self._branchFiles + "/Products/Delta_Pk1D/Delta/delta-64.fits.gz"
            self.compare_fits(path1, path2, "picca_deltas.py")

            path1 = self._masterFiles + "/test_Pk1D/delta-80_Pk1D.fits.gz"
            path2 = self._branchFiles + "/Products/Delta_Pk1D/Delta/delta-80.fits.gz"
            self.compare_fits(path1, path2, "picca_deltas.py")

        return

    def send_delta_Pk1D_minisv(self):

        userprint("\n")
        ### Path
        path_to_etc = self.picca_base + '/etc/'
        ### Send
        cmd = " picca_deltas.py"
        cmd += " --in-dir " + self._branchFiles + "/Products/Spectra_MiniSV/"
        cmd += " --drq " + self._branchFiles + "/Products/cat_minisv.fits"
        cmd += " --out-dir " + self._branchFiles + "/Products/Delta_Pk1D_MiniSV/Delta/"
        cmd += " --iter-out-prefix " + self._branchFiles + \
            "/Products/Delta_Pk1D_MiniSV/Log/delta_attributes"
        cmd += " --log " + self._branchFiles + "/Products/Delta_Pk1D_MiniSV/Log/input.log"
        cmd += " --delta-format Pk1D --mode desiminisv --order 0 --use-constant-weight"
        cmd += " --rebin 1 --lambda-min 3650. --lambda-max 7200.0 --lambda-rest-min 1050.0 --lambda-rest-max 1180"
        cmd += " --nproc 1"
        cmd += " --best-obs"
        cmd += " --mask-file " + path_to_etc + "/list_veto_line_Pk1D.txt"
        cmd += " --use-single-nights"
        returncode = subprocess.call(cmd, shell=True)
        self.assertEqual(returncode, 0, "delta_Pk1D_minisv did not finish")

        ### Test
        #if self._test:
        #    path1 = self._masterFiles + "/test_Pk1D/delta_attributes_Pk1D.fits.gz"
        #    path2 = self._branchFiles + "/Products/Delta_Pk1D/Log/delta_attributes.fits.gz"
        #    self.compare_fits(path1, path2, "picca_deltas.py")

        #    path1 = self._masterFiles + "/test_Pk1D/delta-64_Pk1D.fits.gz"
        #    path2 = self._branchFiles + "/Products/Delta_Pk1D/Delta/delta-64.fits.gz"
        #    self.compare_fits(path1, path2, "picca_deltas.py")

        #    path1 = self._masterFiles + "/test_Pk1D/delta-80_Pk1D.fits.gz"
        #    path2 = self._branchFiles + "/Products/Delta_Pk1D/Delta/delta-80.fits.gz"
        #    self.compare_fits(path1, path2, "picca_deltas.py")

        return

    def send_Pk1D(self):

        userprint("\n")
        ### Send
        cmd = " picca_Pk1D.py"
        cmd += " --in-dir " + self._masterFiles + "/test_Pk1D/delta_Pk1D/"
        cmd += " --out-dir " + self._branchFiles + "/Products/Pk1D/"
        subprocess.call(cmd, shell=True)

        ### Test
        if self._test:
            path1 = self._masterFiles + "/test_Pk1D/Pk1D.fits.gz"
            path2 = self._branchFiles + "/Products/Pk1D/Pk1D-0.fits.gz"
            self.compare_fits(path1, path2, "picca_Pk1D.py")

        return

    def send_cf1d(self):

        userprint("\n")
        ### Send
        cmd = " picca_cf1d.py"
        cmd += " --in-dir " + self._branchFiles + "/Products/Delta_LYA/Delta/"
        cmd += " --out " + self._branchFiles + "/Products/Correlations/cf1d.fits.gz"
        cmd += " --nproc 1"
        subprocess.call(cmd, shell=True)

        ### Test
        if self._test:
            path1 = self._masterFiles + "/cf1d.fits.gz"
            path2 = self._branchFiles + "/Products/Correlations/cf1d.fits.gz"
            self.compare_fits(path1, path2, "picca_cf1d.py")

        return

    def send_cf1d_cross(self):

        userprint("\n")
        ### Send
        cmd = " picca_cf1d.py"
        cmd += " --in-dir " + self._branchFiles + "/Products/Delta_LYA/Delta/"
        cmd += " --in-dir2 " + self._branchFiles + "/Products/Delta_LYA/Delta/"
        cmd += " --out " + self._branchFiles + "/Products/Correlations/cf1d_cross.fits.gz"
        cmd += " --nproc 1"
        subprocess.call(cmd, shell=True)

        ### Test
        if self._test:
            path1 = self._masterFiles + "/cf1d_cross.fits.gz"
            path2 = self._branchFiles + "/Products/Correlations/cf1d_cross.fits.gz"
            self.compare_fits(path1, path2, "picca_cf1d.py")

        return

    def send_cf_angl(self):

        userprint("\n")
        ### Send
        cmd = " picca_cf_angl.py"
        cmd += " --in-dir " + self._branchFiles + "/Products/Delta_LYA/Delta/"
        cmd += " --out " + self._branchFiles + "/Products/Correlations/cf_angl.fits.gz"
        cmd += " --nproc 1"
        subprocess.call(cmd, shell=True)

        ### Test
        if self._test:
            path1 = self._masterFiles + "/cf_angl.fits.gz"
            path2 = self._branchFiles + "/Products/Correlations/cf_angl.fits.gz"
            self.compare_fits(path1, path2, "picca_cf_angl.py")

        return

    def send_cf(self):

        userprint("\n")
        ### Send
        cmd = " picca_cf.py"
        cmd += " --in-dir " + self._branchFiles + "/Products/Delta_LYA/Delta/"
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
            path1 = self._masterFiles + "/cf.fits.gz"
            path2 = self._branchFiles + "/Products/Correlations/cf.fits.gz"
            self.compare_fits(path1, path2, "picca_cf.py")

        return

    def send_dmat(self):

        userprint("\n")
        ### Send
        cmd = " picca_dmat.py"
        cmd += " --in-dir " + self._branchFiles + "/Products/Delta_LYA/Delta/"
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
            path1 = self._masterFiles + "/dmat.fits.gz"
            path2 = self._branchFiles + "/Products/Correlations/dmat.fits.gz"
            self.compare_fits(path1, path2, "picca_dmat.py")

        return

    def send_metal_dmat(self):

        userprint("\n")
        ### Send
        cmd = " picca_metal_dmat.py"
        cmd += " --in-dir " + self._branchFiles + "/Products/Delta_LYA/Delta/"
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
            path1 = self._masterFiles + "/metal_dmat.fits.gz"
            path2 = self._branchFiles + "/Products/Correlations/metal_dmat.fits.gz"
            self.compare_fits(path1, path2, "picca_metal_dmat.py")

        return

    def send_wick(self):

        userprint("\n")
        ### Send
        cmd = " picca_wick.py"
        cmd += " --in-dir " + self._branchFiles + "/Products/Delta_LYA/Delta/"
        cmd += " --out " + self._branchFiles + "/Products/Correlations/wick.fits.gz"
        cmd += " --cf1d " + self._branchFiles + "/Products/Correlations/cf1d.fits.gz"
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
            path1 = self._masterFiles + "/wick.fits.gz"
            path2 = self._branchFiles + "/Products/Correlations/wick.fits.gz"
            self.compare_fits(path1, path2, "picca_wick.py")

        return

    def send_export_cf(self):

        userprint("\n")
        ### Send
        cmd = " picca_export.py"
        cmd += " --data " + self._branchFiles + "/Products/Correlations/cf.fits.gz"
        cmd += " --dmat " + self._branchFiles + "/Products/Correlations/dmat.fits.gz"
        cmd += " --out " + self._branchFiles + "/Products/Correlations/exported_cf.fits.gz"
        returncode = subprocess.call(cmd, shell=True)
        self.assertEqual(returncode, 0, "export_cf did not finish")
        return

    def send_cf_cross(self):

        userprint("\n")
        ### Send
        cmd = " picca_cf.py"
        cmd += " --in-dir  " + self._branchFiles + "/Products/Delta_LYA/Delta/"
        cmd += " --in-dir2 " + self._branchFiles + "/Products/Delta_LYA/Delta/"
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
            path1 = self._masterFiles + "/cf_cross.fits.gz"
            path2 = self._branchFiles + "/Products/Correlations/cf_cross.fits.gz"
            self.compare_fits(path1, path2, "picca_cf.py")

        return

    def send_dmat_cross(self):

        userprint("\n")
        ### Send
        cmd = " picca_dmat.py"
        cmd += " --in-dir  " + self._branchFiles + "/Products/Delta_LYA/Delta/"
        cmd += " --in-dir2 " + self._branchFiles + "/Products/Delta_LYA/Delta/"
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
            path1 = self._masterFiles + "/dmat_cross.fits.gz"
            path2 = self._branchFiles + "/Products/Correlations/dmat_cross.fits.gz"
            self.compare_fits(path1, path2, "picca_dmat.py")

        return

    def send_metal_dmat_cross(self):

        userprint("\n")
        ### Send
        cmd = " picca_metal_dmat.py"
        cmd += " --in-dir " + self._branchFiles + "/Products/Delta_LYA/Delta/"
        cmd += " --in-dir2 " + self._branchFiles + "/Products/Delta_LYA/Delta/"
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
            path1 = self._masterFiles + "/metal_dmat_cross.fits.gz"
            path2 = self._branchFiles + "/Products/Correlations/metal_dmat_cross.fits.gz"
            self.compare_fits(path1, path2, "picca_metal_dmat.py")

        return

    def send_export_cf_cross(self):

        userprint("\n")
        ### Send
        cmd = " picca_export.py"
        cmd += " --data " + self._branchFiles + "/Products/Correlations/cf_cross.fits.gz"
        cmd += " --dmat " + self._branchFiles + "/Products/Correlations/dmat_cross.fits.gz"
        cmd += " --out " + self._branchFiles + \
            "/Products/Correlations/exported_cf_cross.fits.gz"
        returncode = subprocess.call(cmd, shell=True)
        self.assertEqual(returncode, 0, "export_cf_cross did not finish")

        return

    def send_xcf_angl(self):

        userprint("\n")
        ### Send
        cmd = " picca_xcf_angl.py"
        cmd += " --in-dir " + self._branchFiles + "/Products/Delta_LYA/Delta/"
        cmd += " --drq " + self._branchFiles + "/Products/cat.fits"
        cmd += " --out " + self._branchFiles + "/Products/Correlations/xcf_angl.fits.gz"
        cmd += " --nproc 1"
        subprocess.call(cmd, shell=True)

        ### Test
        if self._test:
            path1 = self._masterFiles + "/xcf_angl.fits.gz"
            path2 = self._branchFiles + "/Products/Correlations/xcf_angl.fits.gz"
            self.compare_fits(path1, path2, "picca_xcf_angl.py")

        return

    def send_xcf(self):

        userprint("\n")
        ### Send
        cmd = " picca_xcf.py"
        cmd += " --in-dir " + self._branchFiles + "/Products/Delta_LYA/Delta/"
        cmd += " --drq " + self._branchFiles + "/Products/cat.fits"
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
            path1 = self._masterFiles + "/xcf.fits.gz"
            path2 = self._branchFiles + "/Products/Correlations/xcf.fits.gz"
            self.compare_fits(path1, path2, "picca_xcf.py")

        return

    def send_xdmat(self):

        userprint("\n")
        ### Send
        cmd = " picca_xdmat.py"
        cmd += " --in-dir  " + self._branchFiles + "/Products/Delta_LYA/Delta/"
        cmd += " --drq " + self._branchFiles + "/Products/cat.fits"
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
            path1 = self._masterFiles + "/xdmat.fits.gz"
            path2 = self._branchFiles + "/Products/Correlations/xdmat.fits.gz"
            self.compare_fits(path1, path2, "picca_xdmat.py")

        return

    def send_metal_xdmat(self):

        userprint("\n")
        ### Send
        cmd = " picca_metal_xdmat.py"
        cmd += " --in-dir " + self._branchFiles + "/Products/Delta_LYA/Delta/"
        cmd += " --drq " + self._branchFiles + "/Products/cat.fits"
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
            path1 = self._masterFiles + "/metal_xdmat.fits.gz"
            path2 = self._branchFiles + "/Products/Correlations/metal_xdmat.fits.gz"
            self.compare_fits(path1, path2, "picca_metal_xdmat.py")

        return

    def send_xwick(self):

        userprint("\n")
        ### Send
        cmd = " picca_xwick.py"
        cmd += " --in-dir " + self._branchFiles + "/Products/Delta_LYA/Delta/"
        cmd += " --drq " + self._branchFiles + "/Products/cat.fits"
        cmd += " --out " + self._branchFiles + "/Products/Correlations/xwick.fits.gz"
        cmd += " --cf1d " + self._branchFiles + "/Products/Correlations/cf1d.fits.gz"
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
            path1 = self._masterFiles + "/xwick.fits.gz"
            path2 = self._branchFiles + "/Products/Correlations/xwick.fits.gz"
            self.compare_fits(path1, path2, "picca_xwick.py")

        return

    def send_export_xcf(self):

        userprint("\n")
        ### Send
        cmd = " picca_export.py"
        cmd += " --data " + self._branchFiles + "/Products/Correlations/xcf.fits.gz"
        cmd += " --dmat " + self._branchFiles + "/Products/Correlations/xdmat.fits.gz"
        cmd += " --out " + self._branchFiles + \
            "/Products/Correlations/exported_xcf.fits.gz"
        returncode = subprocess.call(cmd, shell=True)
        self.assertEqual(returncode, 0, "export_xcf did not finish")

        return

    def send_export_cross_covariance_cf_xcf(self):

        userprint("\n")
        ### Send
        cmd = " picca_export_cross_covariance.py"
        cmd += " --data1 " + self._branchFiles + "/Products/Correlations/cf.fits.gz"
        cmd += " --data2 " + self._branchFiles + "/Products/Correlations/xcf.fits.gz"
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
        cmd += " --drq " + self._branchFiles + "/Products/cat.fits"
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
        cmd += " --drq " + self._branchFiles + "/Products/random.fits"
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
        cmd += " --drq " + self._branchFiles + "/Products/cat.fits"
        cmd += " --drq2 " + self._branchFiles + "/Products/random.fits"
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
        cmd += " --drq " + self._branchFiles + "/Products/random.fits"
        cmd += " --drq2 " + self._branchFiles + "/Products/cat.fits"
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
            path1 = self._masterFiles + "/co_DD.fits.gz"
            path2 = self._branchFiles + "/Products/Correlations/co_DD.fits.gz"
            self.compare_fits(path1, path2, "picca_co.py DD")

            path1 = self._masterFiles + "/co_RR.fits.gz"
            path2 = self._branchFiles + "/Products/Correlations/Co_Random/co_RR.fits.gz"
            self.compare_fits(path1, path2, "picca_co.py RR")

            path1 = self._masterFiles + "/co_DR.fits.gz"
            path2 = self._branchFiles + "/Products/Correlations/Co_Random/co_DR.fits.gz"
            self.compare_fits(path1, path2, "picca_co.py DR")

            path1 = self._masterFiles + "/co_RD.fits.gz"
            path2 = self._branchFiles + "/Products/Correlations/Co_Random/co_RD.fits.gz"
            self.compare_fits(path1, path2, "picca_co.py RD")

        return

    def send_export_co(self):

        userprint("\n")
        ### Send
        cmd = " picca_export_co.py"
        cmd += " --DD-file " + self._branchFiles + "/Products/Correlations/co_DD.fits.gz"
        cmd += " --RR-file " + self._branchFiles + \
            "/Products/Correlations/Co_Random/co_RR.fits.gz"
        cmd += " --DR-file " + self._branchFiles + \
            "/Products/Correlations/Co_Random/co_DR.fits.gz"
        cmd += " --RD-file " + self._branchFiles + \
            "/Products/Correlations/Co_Random/co_RD.fits.gz"
        cmd += " --out " + self._branchFiles + "/Products/Correlations/exported_co.fits.gz"
        cmd += " --get-cov-from-poisson"
        returncode = subprocess.call(cmd, shell=True)
        self.assertEqual(returncode, 0, "picca_export_co did not finish")

        return

    def send_fitter2(self):

        userprint("\n")

        ### copy ini files to branch
        cmd = 'cp '+self._masterFiles+'/*ini ' + \
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
        value = self._branchFiles + '/Products/Correlations/exported_cf.fits.gz'
        update_system_status_values(path, 'data', 'filename', value)
        value = self._branchFiles + '/Products/Correlations/metal_dmat.fits.gz'
        update_system_status_values(path, 'metals', 'filename', value)

        ### Set path in config_cf_cross.ini
        path = self._branchFiles + '/Products/Correlations/Fit/config_cf_cross.ini'
        value = self._branchFiles + '/Products/Correlations/exported_cf_cross.fits.gz'
        update_system_status_values(path, 'data', 'filename', value)
        value = self._branchFiles + '/Products/Correlations/metal_dmat_cross.fits.gz'
        update_system_status_values(path, 'metals', 'filename', value)

        ### Set path in config_xcf.ini
        path = self._branchFiles + '/Products/Correlations/Fit/config_xcf.ini'
        value = self._branchFiles + '/Products/Correlations/exported_xcf.fits.gz'
        update_system_status_values(path, 'data', 'filename', value)
        value = self._branchFiles + '/Products/Correlations/metal_xdmat.fits.gz'
        update_system_status_values(path, 'metals', 'filename', value)

        ### Send
        cmd = ' picca_fitter2.py '+self._branchFiles + \
            '/Products/Correlations/Fit/chi2.ini'
        subprocess.call(cmd, shell=True)

        ### Test
        if self._test:
            path1 = self._masterFiles + '/result_fitter2.h5'
            path2 = self._branchFiles + '/Products/Correlations/Fit/result_fitter2.h5'
            self.compare_h5py(path1, path2, "picca_fitter2")


if __name__ == '__main__':
    unittest.main()
