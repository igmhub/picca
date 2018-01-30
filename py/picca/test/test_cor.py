import unittest
import scipy as sp
import numpy.random
import fitsio
import healpy
import subprocess
import os
import tempfile
import shutil
from pkg_resources import resource_filename

class TestCor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._branchFiles = tempfile.mkdtemp()+"/"
        
    @classmethod
    def tearDownClass(cls):
        if os.path.isdir(cls._branchFiles):
            shutil.rmtree(cls._branchFiles, ignore_errors=True)


    def test_cor(self):

        numpy.random.seed(42)

        print("\n")
        self._test = True
        self._masterFiles = resource_filename('picca', 'test/data/')
        self.produce_folder()
        self.produce_cat(nObj=1000)
        self.produce_forests()

        self.send_delta()

        self.send_cf1d()
        self.send_cf1d_cross()

        self.send_cf()
        self.send_dmat()

        self.send_cf_cross()
        self.send_dmat_cross()

        self.send_xcf()
        self.send_xdmat()

        if self._test:
            self.remove_folder()

        return
    def produce_folder(self):
        """
            Create the necessary folders
        """

        print("\n")
        lst_fold = ["/Products/","/Products/Spectra/",
        "/Products/Delta_LYA/","/Products/Delta_LYA/Delta/",
        "/Products/Delta_LYA/Log/","/Products/Correlations/",
        ]

        for fold in lst_fold:
            if not os.path.isdir(self._branchFiles+fold):
                os.mkdir(self._branchFiles+fold)

        return
    def remove_folder(self):
        """
            Remove the produced folders
        """

        print("\n")
        shutil.rmtree(self._branchFiles, ignore_errors=True)

        return
    def produce_cat(self,nObj):
        """

        """

        print("\n")
        print("Create cat with number of object = ", nObj)

        ### Create random catalog
        ra    = 10.*numpy.random.random_sample(nObj)
        dec   = 10.*numpy.random.random_sample(nObj)
        plate = numpy.random.randint(266,   high=10001, size=nObj )
        mjd   = numpy.random.randint(51608, high=57521, size=nObj )
        fid   = numpy.random.randint(1,     high=1001,  size=nObj )
        thid  = sp.arange(1,nObj+1)
        zqso  = (3.6-2.0)*numpy.random.random_sample(nObj) + 2.0

        ### Save
        out = fitsio.FITS(self._branchFiles+"/Products/cat.fits",'rw',clobber=True)
        cols=[ra,dec,thid,plate,mjd,fid,zqso]
        names=['RA','DEC','THING_ID','PLATE','MJD','FIBERID','Z']
        out.write(cols,names=names)
        out.close()

        return
    def produce_forests(self):
        """

        """

        print("\n")
        nside = 8

        ### Load DRQ
        vac = fitsio.FITS(self._branchFiles+"/Products/cat.fits")
        ra    = vac[1]["RA"][:]*sp.pi/180.
        dec   = vac[1]["DEC"][:]*sp.pi/180.
        thid  = vac[1]["THING_ID"][:]
        plate = vac[1]["PLATE"][:]
        mjd   = vac[1]["MJD"][:]
        fid   = vac[1]["FIBERID"][:]
        vac.close()

        ### Get Healpy pixels
        pixs  = healpy.ang2pix(nside, sp.pi/2.-dec, ra)

        ### Save master file
        path = self._branchFiles+"/Products/Spectra/master.fits"
        head = {}
        head['NSIDE'] = nside
        cols  = [thid,pixs,plate,mjd,fid]
        names = ['THING_ID','PIX','PLATE','MJD','FIBER']
        out = fitsio.FITS(path,'rw',clobber=True)
        out.write(cols,names=names,header=head,extname="MASTER TABLE")
        out.close()

        ### Log lambda grid
        logl_min  = 3.550
        logl_max  = 4.025
        logl_step = 1.e-4
        ll = sp.arange(logl_min, logl_max, logl_step)

        ###
        for p in sp.unique(pixs):

            ###
            p_thid = thid[(pixs==p)]
            p_fl   = numpy.random.normal(loc=1., scale=1., size=(ll.size,p_thid.size))
            p_iv   = numpy.random.lognormal(mean=0.1, sigma=0.1, size=(ll.size,p_thid.size))
            p_am   = sp.zeros((ll.size,p_thid.size)).astype(int)
            p_am[ numpy.random.random(size=(ll.size,p_thid.size))>0.90 ] = 1
            p_om   = sp.zeros((ll.size,p_thid.size)).astype(int)

            ###
            p_path = self._branchFiles+"/Products/Spectra/pix_"+str(p)+".fits"
            out = fitsio.FITS(p_path, 'rw', clobber=True)
            out.write(p_thid, header={}, extname="THING_ID_MAP")
            out.write(ll,     header={}, extname="LOGLAM_MAP")
            out.write(p_fl,   header={}, extname="FLUX")
            out.write(p_iv,   header={}, extname="IVAR")
            out.write(p_am,   header={}, extname="ANDMASK")
            out.write(p_om,   header={}, extname="ORMASK")
            out.close()

        return
    def compare_fits(self,path1,path2):

        m = fitsio.FITS(path1)
        self.assertTrue(os.path.isfile(path2))
        b = fitsio.FITS(path2)

        self.assertEqual(len(m),len(b))

        for i in range(len(m)):

            ###
            r_m = m[i].read_header().records()
            ld_m = []
            for j in range(len(r_m)):
                name = r_m[j]['name']
                if len(name)>5 and name[:5]=="TTYPE":
                    ld_m += [r_m[j]['value'].replace(" ","")]
            ###
            r_b = b[i].read_header().records()
            ld_b = []
            for j in range(len(r_b)):
                name = r_b[j]['name']
                if len(name)>5 and name[:5]=="TTYPE":
                    ld_b += [r_b[j]['value'].replace(" ","")]

            self.assertListEqual(ld_m,ld_b)

            for k in ld_m:
                d_m = m[i][k][:]
                d_b = b[i][k][:]
                self.assertEqual(d_m.size,d_b.size)
                self.assertFalse((d_m!=d_b).any())

        return


    def send_delta(self):

        print("\n")
        ### Send
        cmd  = " do_deltas.py"
        cmd += " --in-dir "          + self._branchFiles+"/Products/Spectra/"
        cmd += " --drq "             + self._branchFiles+"/Products/cat.fits"
        cmd += " --out-dir "         + self._branchFiles+"/Products/Delta_LYA/Delta/"
        cmd += " --iter-out-prefix " + self._branchFiles+"/Products/Delta_LYA/Log/delta_attributes"
        cmd += " --log "             + self._branchFiles+"/Products/Delta_LYA/Log/input.log"
        cmd += " --nproc 4"
        subprocess.call(cmd, shell=True)

        ### Test
        if self._test:
            path1 = self._masterFiles + "/delta_attributes.fits.gz"
            path2 = self._branchFiles + "/Products/Delta_LYA/Log/delta_attributes.fits.gz"
            self.compare_fits(path1,path2)

        return
    def send_cf1d(self):

        print("\n")
        ### Send
        cmd  = " do_cf1d.py"
        cmd += " --in-dir " + self._branchFiles+"/Products/Delta_LYA/Delta/"
        cmd += " --out "    + self._branchFiles+"/Products/Correlations/cf1d.fits.gz"
        cmd += " --nproc 4"
        subprocess.call(cmd, shell=True)

        ### Test
        if self._test:
            path1 = self._masterFiles + "/cf1d.fits.gz"
            path2 = self._branchFiles + "/Products/Correlations/cf1d.fits.gz"
            self.compare_fits(path1,path2)

        return
    def send_cf1d_cross(self):

        print("\n")
        ### Send
        cmd  = " do_cf1d.py"
        cmd += " --in-dir "  + self._branchFiles+"/Products/Delta_LYA/Delta/"
        cmd += " --in-dir2 " + self._branchFiles+"/Products/Delta_LYA/Delta/"
        cmd += " --out "     + self._branchFiles+"/Products/Correlations/cf1d_cross.fits.gz"
        cmd += " --nproc 4"
        subprocess.call(cmd, shell=True)

        ### Test
        if self._test:
            path1 = self._masterFiles + "/cf1d_cross.fits.gz"
            path2 = self._branchFiles + "/Products/Correlations/cf1d_cross.fits.gz"
            self.compare_fits(path1,path2)

        return
    def send_cf(self):

        print("\n")
        ### Send
        cmd  = " do_cf.py"
        cmd += " --in-dir " + self._branchFiles+"/Products/Delta_LYA/Delta/"
        cmd += " --out "    + self._branchFiles+"/Products/Correlations/cf.fits.gz"
        cmd += " --rp-min +0.0"
        cmd += " --rp-max +60.0"
        cmd += " --rt-max +60.0"
        cmd += " --np 15"
        cmd += " --nt 15"
        cmd += " --nproc 4"
        subprocess.call(cmd, shell=True)

        ### Test
        if self._test:
            path1 = self._masterFiles + "/cf.fits.gz"
            path2 = self._branchFiles + "/Products/Correlations/cf.fits.gz"
            self.compare_fits(path1,path2)

        return
    def send_dmat(self):

        print("\n")
        ### Send
        cmd  = " do_dmat.py"
        cmd += " --in-dir " + self._branchFiles+"/Products/Delta_LYA/Delta/"
        cmd += " --out "    + self._branchFiles+"/Products/Correlations/dmat.fits.gz"
        cmd += " --rp-min +0.0"
        cmd += " --rp-max +60.0"
        cmd += " --rt-max +60.0"
        cmd += " --np 15"
        cmd += " --nt 15"
        cmd += " --rej 0.99 "
        cmd += " --nproc 4"
        subprocess.call(cmd, shell=True)

        ### Test
        if self._test:
            path1 = self._masterFiles + "/dmat.fits.gz"
            path2 = self._branchFiles + "/Products/Correlations/dmat.fits.gz"
            self.compare_fits(path1,path2)

        return
    def send_cf_cross(self):

        print("\n")
        ### Send
        cmd  = " do_cf.py"
        cmd += " --in-dir  " + self._branchFiles+"/Products/Delta_LYA/Delta/"
        cmd += " --in-dir2 " + self._branchFiles+"/Products/Delta_LYA/Delta/"
        cmd += " --out "     + self._branchFiles+"/Products/Correlations/cf_cross.fits.gz"
        cmd += " --rp-min -60.0"
        cmd += " --rp-max +60.0"
        cmd += " --rt-max +60.0"
        cmd += " --np 30"
        cmd += " --nt 15"
        cmd += " --nproc 4"
        subprocess.call(cmd, shell=True)

        ### Test
        if self._test:
            path1 = self._masterFiles + "/cf_cross.fits.gz"
            path2 = self._branchFiles + "/Products/Correlations/cf_cross.fits.gz"
            self.compare_fits(path1,path2)

        return
    def send_dmat_cross(self):

        print("\n")
        ### Send
        cmd  = " do_dmat.py"
        cmd += " --in-dir  " + self._branchFiles+"/Products/Delta_LYA/Delta/"
        cmd += " --in-dir2 " + self._branchFiles+"/Products/Delta_LYA/Delta/"
        cmd += " --out "     + self._branchFiles+"/Products/Correlations/dmat_cross.fits.gz"
        cmd += " --rp-min -60.0"
        cmd += " --rp-max +60.0"
        cmd += " --rt-max +60.0"
        cmd += " --np 30"
        cmd += " --nt 15"
        cmd += " --rej 0.99 "
        cmd += " --nproc 4"
        subprocess.call(cmd, shell=True)

        ### Test
        if self._test:
            path1 = self._masterFiles + "/dmat_cross.fits.gz"
            path2 = self._branchFiles + "/Products/Correlations/dmat_cross.fits.gz"
            self.compare_fits(path1,path2)

        return
    def send_xcf(self):

        print("\n")
        ### Send
        cmd  = " do_xcf.py"
        cmd += " --in-dir " + self._branchFiles+"/Products/Delta_LYA/Delta/"
        cmd += " --drq "    + self._branchFiles+"/Products/cat.fits"
        cmd += " --out "    + self._branchFiles+"/Products/Correlations/xcf.fits.gz"
        cmd += " --rp-min -60.0"
        cmd += " --rp-max +60.0"
        cmd += " --rt-max +60.0"
        cmd += " --np 30"
        cmd += " --nt 15"
        cmd += " --nproc 4"
        subprocess.call(cmd, shell=True)

        ### Test
        if self._test:
            path1 = self._masterFiles + "/xcf.fits.gz"
            path2 = self._branchFiles + "/Products/Correlations/xcf.fits.gz"
            self.compare_fits(path1,path2)

        return
    def send_xdmat(self):

        print("\n")
        ### Send
        cmd  = " do_xdmat.py"
        cmd += " --in-dir  " + self._branchFiles+"/Products/Delta_LYA/Delta/"
        cmd += " --drq "    + self._branchFiles+"/Products/cat.fits"
        cmd += " --out "     + self._branchFiles+"/Products/Correlations/xdmat.fits.gz"
        cmd += " --rp-min -60.0"
        cmd += " --rp-max +60.0"
        cmd += " --rt-max +60.0"
        cmd += " --np 30"
        cmd += " --nt 15"
        cmd += " --rej 0.99 "
        cmd += " --nproc 4"
        subprocess.call(cmd, shell=True)

        ### Test
        if self._test:
            path1 = self._masterFiles + "/xdmat.fits.gz"
            path2 = self._branchFiles + "/Products/Correlations/xdmat.fits.gz"
            self.compare_fits(path1,path2)

        return

if __name__ == '__main__':
    unittest.main()
