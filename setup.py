#!/usr/bin/env python

import glob

from setuptools import setup

scripts = glob.glob('bin/*')

description = "Package for Igm Cosmological-Correlations Analyses"

version="4.0"
setup(name="picca",
    version=version,
    description=description,
    url="https://github.com/igmhub/picca",
    author="Nicolas Busca, Helion du Mas des Bourboux et al",
    author_email="nbusca@lpnhe.in2p3.fr",
    packages=['picca','picca.fitter2'],
    package_dir = {'': 'py'},
    package_data = {'picca': ['fitter2/models/*/*.fits']},
    install_requires=['numpy','scipy','iminuit','healpy','fitsio',
        'llvmlite','numba','h5py','future','setuptools'],
    test_suite='picca.test.test_cor',
    scripts = scripts
    )
