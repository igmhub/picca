import configparser as ConfigParser
from picca.utils import userprint
import fitsio
import os
import numpy as np
import h5py


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

#note that this routine needs to be part of a test case, it's just defined in here for reusability reasons...
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
            if d_m.dtype in ['<U23', 'S23'
                                ]:  # for fitsio old version compatibility
                d_m = np.char.strip(d_m)
            if d_b.dtype in ['<U23', 'S23'
                                ]:  # for fitsio old version compatibility
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

    m.close()
    b.close()

    return

#note that this routine needs to be part of a test case, it's just defined in here for reusability reasons...
def compare_h5py(self, path1, path2, nameRun=""):
    def compare_attributes(atts1, atts2):
        self.assertEqual(len(atts1.keys()), len(atts2.keys()),
                            "{}".format(nameRun))
        self.assertListEqual(sorted(atts1.keys()), sorted(atts2.keys()),
                                "{}".format(nameRun))
        for item in atts1:
            nequal = True
            if isinstance(atts1[item], np.ndarray):
                nequal = np.logical_not(
                    np.array_equal(atts1[item], atts2[item]))
            else:
                nequal = atts1[item] != atts2[item]
            if nequal:
                userprint(
                    "WARNING: {}: not exactly equal, using allclose for {}"
                    .format(nameRun, item))
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