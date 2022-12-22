import unittest
import os
import glob
import numpy as np
import fitsio
import healpy


from picca.utils import userprint

from picca.tests.test_helpers import AbstractTest


class TestDelta(AbstractTest):
    """
        Test case for picca_deltas.py
    """
    def produce_folder(self):
        """
            Create the necessary folders
        """

        userprint("\n")
        lst_fold = [
            "/Products/", "/Products/Spectra/", "/Products/Spectra_MiniSV/",
            "/Products/Delta_Pk1D/", "/Products/Delta_Pk1D/Delta/",
            "/Products/Delta_Pk1D/Log/", "/Products/Delta_Pk1D_MiniSV/",
            "/Products/Delta_Pk1D_MiniSV/Delta/",
            "/Products/Delta_Pk1D_MiniSV/Log/", "/Products/Delta_LYA/",
            "/Products/Delta_LYA/Delta/", "/Products/Delta_LYA/Log/"
        ]

        for fold in lst_fold:
            if not os.path.isdir(self._branchFiles + fold):
                os.mkdir(self._branchFiles + fold)

        return

    def produce_cat(self, nObj, name="cat", thidoffset=0):
        """
            produces a fake catalog for testing
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

        if self._test:
            path1 = self._masterFiles + "/test_delta/" + name + ".fits"
            path2 = self._branchFiles + "/Products/" + name + ".fits"
            self.compare_fits(path1, path2, "produce cat")

        return

    def produce_forests(self):
        """
            randomly creates Lya forests for testing
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

        ### Loop over healpix
        for p in np.unique(pixs):

            ### Retrieve objects from catalog and produce fake spectra
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

            ### Save to file
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
            produces a fake catalog in DESI SV like format for testing
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

        if self._test:
            path1 = self._masterFiles + "/test_delta/" + name + ".fits"
            path2 = self._branchFiles + "/Products/" + name + ".fits"
            self.compare_fits(path1, path2, "produce cat MiniSV")
        return

    def produce_forests_minisv(self):
        """
            Produce random data in DESI SV like format for testing
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

        ### Loop over tiles
        for t, n, p in tile_night_petal_combined:

            ###  Grab targets from catalog
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
            ### Write to files
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

    def send_delta(self):
        """
            Test the continuum fitting routines on randomly generated eBOSS mock data
        """
        import picca.bin.old.picca_deltas as picca_deltas

        userprint("\n")
        ### Send
        cmd = "picca_deltas.py"
        cmd += " --in-dir " + self._branchFiles + "/Products/Spectra/"
        cmd += " --drq " + self._branchFiles + "/Products/cat.fits"
        cmd += " --out-dir " + self._branchFiles + "/Products/Delta_LYA/Delta/"
        cmd += " --iter-out-prefix " + self._branchFiles + \
            "/Products/Delta_LYA/Log/delta_attributes"
        cmd += " --metadata " + self._branchFiles + \
            "/Products/Delta_LYA/Log/metadata.fits"
        cmd += " --log " + self._branchFiles + "/Products/Delta_LYA/Log/input.log"
        cmd += " --nproc 1"
        picca_deltas.main(cmd.split()[1:])

        ### Test
        if self._test:
            path1 = self._masterFiles + "/test_delta/delta_attributes.fits.gz"
            path2 = self._branchFiles + "/Products/Delta_LYA/Log/delta_attributes.fits.gz"
            self.compare_fits(path1, path2, "picca_deltas.py")

            path1 = self._masterFiles + "/test_delta/metadata.fits"
            path2 = self._branchFiles + "/Products/Delta_LYA/Log/metadata.fits"
            #TODO: note that for the moment we are more tolerant towards absolute changes in the metadata
            #      else p1 values would cause tests to break all the time, might be worth looking into the
            #      underlying issue at some later time
            self.compare_fits(path1, path2, "picca_deltas.py", rel_tolerance=5e-4)
        return


    def send_delta_Pk1D_minisv(self):
        """
            Test of picca_deltas currently only to check if SV I/O routines
            work correctly. The data for this routine is randomly generated as for "send_delta", but
            in DESI SV format instead of an eBOSS format
        """
        import picca.bin.old.picca_deltas as picca_deltas

        userprint("\n")
        ### Path
        path_to_etc = self.picca_base + '/etc/'
        ### Send
        cmd = "picca_deltas.py"
        cmd += " --in-dir " + self._branchFiles + "/Products/Spectra_MiniSV/"
        cmd += " --drq " + self._branchFiles + "/Products/cat_minisv.fits"
        cmd += " --out-dir " + self._branchFiles + "/Products/Delta_Pk1D_MiniSV/Delta/"
        cmd += " --iter-out-prefix " + self._branchFiles + \
            "/Products/Delta_Pk1D_MiniSV/Log/delta_attributes"
        cmd += " --log " + self._branchFiles + "/Products/Delta_Pk1D_MiniSV/Log/input.log"
        cmd += " --metadata " + self._branchFiles + \
            "/Products/Delta_Pk1D_MiniSV/Log/metadata.fits"
        cmd += " --delta-format Pk1D --mode desiminisv --order 0 --use-constant-weight"
        cmd += " --rebin 1 --lambda-min 3650. --lambda-max 7200.0 --lambda-rest-min 1050.0 --lambda-rest-max 1180"
        cmd += " --nproc 1"
        cmd += " --best-obs"
        cmd += " --mask-file " + path_to_etc + "/list_veto_line_Pk1D.txt"
        cmd += " --use-single-nights"

        picca_deltas.main(cmd.split()[1:])

        ### Test
        if self._test:
            path1 = self._masterFiles + "/test_delta/delta_attributes_Pk1D_MiniSV.fits.gz"
            path2 = self._branchFiles + "/Products/Delta_Pk1D_MiniSV/Log/delta_attributes.fits.gz"
            self.compare_fits(path1, path2, "picca_deltas.py")

            path1 = self._masterFiles + "/test_delta/metadata_Pk1D_MiniSV.fits"
            path2 = self._branchFiles + "/Products/Delta_Pk1D_MiniSV/Log/metadata.fits"
            self.compare_fits(path1, path2, "picca_deltas.py")

            #this checks if any of the output delta files changed
            for fname in glob.glob(
                    f'{self._branchFiles}/Products/Delta_Pk1D_MiniSV/Delta/delta-*.fits.gz'
            ):
                path2 = fname
                path1 = f"{self._masterFiles}/test_delta/Delta_Pk1D_MiniSV/{os.path.basename(fname)}"
                self.compare_fits(path1, path2, "picca_deltas.py")
        return

    def test_delta(self):
        """
            wrapper around send_delta test to produce the mock datasets
        """
        np.random.seed(42)
        self.produce_cat(nObj=1000)
        self.produce_forests()
        self.produce_cat(nObj=1000, name="random", thidoffset=1000)
        self.send_delta()
        return

    def test_delta_Pk1D_minisv(self):
        """
            wrapper around send_delta test to produce the mock datasets
        """
        np.random.seed(42)
        self.produce_cat_minisv(nObj=1000)
        self.produce_forests_minisv()
        self.send_delta_Pk1D_minisv()
        return

    def test_delta_Pk1D(self):
        """
            Test of picca_deltas for purposes of Pk1d running on a very small set of
            eBOSS like spectra saved on disk
        """
        import picca.bin.old.picca_deltas as picca_deltas
        userprint("\n")
        ### Path
        path_to_etc = self.picca_base + '/etc/'
        ### Send
        cmd = "picca_deltas.py"
        cmd += " --in-dir " + self._masterFiles + "/test_delta/Spectra_Pk1D/"
        cmd += " --drq " + self._masterFiles + "/test_delta/DRQ_Pk1D.fits"
        cmd += " --out-dir " + self._branchFiles + "/Products/Delta_Pk1D/Delta/"
        cmd += " --iter-out-prefix " + self._branchFiles + \
            "/Products/Delta_Pk1D/Log/delta_attributes"
        cmd += " --log " + self._branchFiles + "/Products/Delta_Pk1D/Log/input.log"
        cmd += " --metadata " + self._branchFiles + \
            "/Products/Delta_Pk1D/Log/metadata.fits"
        cmd += " --delta-format Pk1D --mode spec --order 0 --use-constant-weight"
        cmd += " --rebin 1 --lambda-min 3650. --lambda-max 7200.0 --lambda-rest-min 1050.0 --lambda-rest-max 1180"
        cmd += " --nproc 1"
        cmd += " --best-obs"
        cmd += " --mask-file " + path_to_etc + "/list_veto_line_Pk1D.txt"

        picca_deltas.main(cmd.split()[1:])

        ### Test
        if self._test:
            path1 = self._masterFiles + "/test_delta/delta_attributes_Pk1D.fits.gz"
            path2 = self._branchFiles + "/Products/Delta_Pk1D/Log/delta_attributes.fits.gz"
            self.compare_fits(path1, path2, "picca_deltas.py")

            path1 = self._masterFiles + "/test_delta/metadata_Pk1D.fits"
            path2 = self._branchFiles + "/Products/Delta_Pk1D/Log/metadata.fits"
            self.compare_fits(path1, path2, "picca_deltas.py")

            path1 = self._masterFiles + "/test_delta/delta-64_Pk1D.fits.gz"
            path2 = self._branchFiles + "/Products/Delta_Pk1D/Delta/delta-64.fits.gz"
            self.compare_fits(path1, path2, "picca_deltas.py")

            path1 = self._masterFiles + "/test_delta/delta-80_Pk1D.fits.gz"
            path2 = self._branchFiles + "/Products/Delta_Pk1D/Delta/delta-80.fits.gz"
            self.compare_fits(path1, path2, "picca_deltas.py")

        return

if __name__ == '__main__':
    unittest.main()
