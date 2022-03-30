import configparser as ConfigParser
from picca.utils import userprint
import fitsio
import os
import numpy as np
import h5py
import sys
import unittest
import shutil, tempfile
from pkg_resources import resource_filename

### Make ConfigParser case sensitive
class CaseConfigParser(ConfigParser.ConfigParser):

    def optionxform(self, optionstr):
        return optionstr

class AbstractTest(unittest.TestCase):
    """
        Class with Helper functions for the picca unit tests
    """

    def update_system_status_values(self, path, section, system, value):
        """
            This updates variables in the fitter test

        Args:
            path (str): path to fitter configuration
            section (str): which section to modify
            system (str): entry to modify
            value (str): new value
        """
        cp = CaseConfigParser()
        cp.read(path)
        cf = open(path, 'w')
        cp.set(section, system, value)
        cp.write(cf)
        cf.close()

        return


    def compare_fits(self, path1, path2, nameRun="", rel_tolerance=1e-05, abs_tolerance=1e-08):
        """
            Compares all fits files in 2 directories against each other

        Args:
            path1 (str): path where first set of fits files lies
            path2 (str): path where second set of fits files lies
            nameRun (str, optional): A name of the current run for identification. Defaults to "".
            rel_tolerance: relative difference to be allowed, see np.allclose, defaults from there
            abs_tolerance: absolute difference to be allowed, see np.allclose, defaults from there
        """
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
                    ld_m += [el['value'].replace(" ", "").upper()]
            ###
            r_b = b[i].read_header().records()
            ld_b = []
            for el in r_b:
                name = el['name']
                if len(name) > 5 and name[:5] == "TTYPE":
                    ld_b += [el['value'].replace(" ", "").upper()]

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
                    diff_abs = np.absolute(diff)
                    w = d_m != 0.
                    diff[w] = np.absolute(diff[w] / d_m[w])
                    allclose = np.allclose(d_m, d_b,atol=abs_tolerance,rtol=rel_tolerance)
                    self.assertTrue(
                        allclose,
                        (f"{nameRun}: Header key is {k}, maximum relative difference is {diff.max()}, "
                         f"maximum absolute difference is {diff_abs.max()}\n"
                         f"file1: {path1}\nfile2: {path2}")
                        )
                    userprint(f"OK, maximum relative difference {diff.max():.2e}, max. abs. difference is {diff_abs.max():.2e}")

        m.close()
        b.close()

        return


    def compare_h5py(self, path1, path2, nameRun=""):
        """
            Compares two hdf5 against each other

        Args:
            path1 (str): path of the first hdf5 file
            path2 (str): path of the second hdf5 file
            nameRun (str, optional): A name of the current run for identification. Defaults to "".
        """

        def compare_attributes(atts1, atts2):
            self.assertEqual(len(atts1.keys()), len(atts2.keys()),
                            "{}".format(nameRun))
            self.assertListEqual(sorted(atts1.keys()), sorted(atts2.keys()),
                                "{}".format(nameRun))
            for item in atts1:
                nequal = True
                if isinstance(atts1[item], np.ndarray):
                    dtype1 = atts1[item].dtype
                    dtype2 = atts2[item].dtype
                    if dtype1 == dtype2:
                        nequal = np.logical_not(
                            np.array_equal(atts1[item], atts2[item]))
                    else:
                        userprint(
                            f"Note that the test file has different dtype for attribute {item}"
                        )
                        nequal = np.logical_not(
                            np.array_equal(atts1[item].astype(atts2[item].dtype),
                                        atts2[item]))
                        if nequal:
                            nequal = np.logical_not(
                                np.array_equal(
                                    atts2[item].astype(atts1[item].dtype),
                                    atts1[item]))
                else:
                    nequal = atts1[item] != atts2[item]
                if nequal:
                    userprint(
                        "WARNING: {}: not exactly equal, using allclose for attribute {}".
                        format(nameRun, item))
                    userprint(atts1[item], atts2[item])
                    allclose = np.allclose(atts1[item], atts2[item])
                    if item=='nfcn' and not allclose:
                        print("'nfcn' definition changed between iminuit1 (unclear what this was) and iminuit2 (total number of calls)")
                    else:
                        self.assertTrue(allclose, "{} results changed for attribute {}".format(nameRun, item))
            return

        def compare_values(val1, val2, namelist):
            if not np.array_equal(val1, val2):
                userprint("WARNING: {}: {} not exactly equal, using allclose".format(
                    nameRun,'/'.join(namelist)))
                allclose = np.allclose(val1, val2)
                self.assertTrue(allclose, "{} results changed for output values for {}:\n expected:{}\n\n got:{}\n\n\n".format(
                    nameRun,
                    '/'.join(namelist),
                    ' '.join([f'{v:6.5g}' for v in val1.flatten()]),
                    ' '.join([f'{v:6.5g}' for v in val2.flatten()]),
                ))
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
            compare_values(m[k]['fit'][()], b[k]['fit'][()],[k,'fit'])

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
                compare_values(m[k][p]['values'][()], b[k][p]['values'][()],[k,p,'values'])

        return

    @classmethod
    def load_requirements(cls, picca_base):
        """
            Loads reqirements file from picca_base
        """
        req = {}

        if sys.version_info > (3, 0):
            path = picca_base + '/requirements.txt'
        else:
            path = picca_base + '/requirements-python2.txt'
        with open(path, 'r') as f:
            for l in f:
                l = l.replace('\n', '').replace('==', ' ').replace('>=',
                                                                ' ').split()
                assert len(
                    l) == 2, "requirements.txt attribute is not valid: {}".format(
                        str(l))
                req[l[0]] = l[1]
        return req

    @classmethod
    def send_requirements(cls, req):
        """
            Compares requirements in req to currently loaded modules
        """
        userprint("\n")
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

    @classmethod
    def setUpClass(cls):
        """
            sets up directory structure in tmp
        """
        cls._branchFiles = tempfile.mkdtemp() + "/"
        cls.produce_folder(cls)
        cls.picca_base = resource_filename('picca',
                                           './').replace('py/picca/./', '')
        cls.send_requirements(cls.load_requirements(cls.picca_base))
        np.random.seed(42)
        cls._masterFiles = cls.picca_base + '/py/picca/tests/data/'
        cls._test=True
        userprint("\n")

    @classmethod
    def tearDownClass(cls):
        """
            removes directory structure in tmp
        """
        os.makedirs('/tmp/last_run_picca_test/',exist_ok=True)
        #copy the outputs for later debugging, ditch spectra
        try:
            shutil.copytree(cls._branchFiles, '/tmp/last_run_picca_test/', ignore=lambda path,fnames: [fname for fname in fnames if 'spectra' in fname.lower() or 'spectra' in path.lower()],dirs_exist_ok=True)
        except TypeError:
            try:
                shutil.copytree(cls._branchFiles, '/tmp/last_run_picca_test/', ignore=lambda path,fnames: [fname for fname in fnames if 'spectra' in fname.lower() or 'spectra' in path.lower()])
            except FileExistsError:
                print("Files Exist, could not copy last run files, added random number to output filename")
                shutil.copytree(cls._branchFiles, f'/tmp/last_run_picca_test/{np.random.randint(1000000)}', ignore=lambda path,fnames: [fname for fname in fnames if 'spectra' in fname.lower() or 'spectra' in path.lower()])

        if os.path.isdir(cls._branchFiles):
            shutil.rmtree(cls._branchFiles, ignore_errors=True)
