#!/usr/bin/env python

import glob

from setuptools import setup

scripts = sorted(glob.glob('bin/picca*'))

description = "Package for Igm Cosmological-Correlations Analyses"


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

exec(open('py/picca/_version.py').read())
version = __version__

setup(name="picca",
    version=version,
    description=description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/igmhub/picca",
    author="The eBOSS and DESI collaboration Lya forest working groups",
    author_email="desi-lya@desi.lbl.gov",
    packages=['picca','picca.fitter2','picca.bin'],
    package_dir = {'': 'py'},
    package_data = {'picca': ['fitter2/models/*/*.fits']},
    install_requires=['numpy','scipy','iminuit','healpy','fitsio',
        'llvmlite','numba','h5py','future','setuptools'],
    test_suite='picca.test',
    scripts = scripts
    )
