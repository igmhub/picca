#!/usr/bin/env python

import glob

from setuptools import setup, find_packages

scripts = sorted(glob.glob('bin/picca*'))

description = "Package for Igm Cosmological-Correlations Analyses"

exec(open('py/picca/_version.py').read())
version = __version__

setup(name="picca",
    version=version,
    description=description,
    url="https://github.com/igmhub/picca",
    author="Nicolas Busca, Helion du Mas des Bourboux et al",
    author_email="nbusca@lpnhe.in2p3.fr",
    packages=['picca','picca.delta_extraction', 'picca.fitter2', 'picca.bin', 'picca.delta_extraction.astronomical_objects', 'picca.delta_extraction.corrections', 'picca.delta_extraction.data_catalogues', 'picca.delta_extraction.expected_fluxes', 'picca.delta_extraction.masks', 'picca.delta_extraction.quasar_catalogues'],
    package_dir = {'': 'py'},
    package_data = {'picca': ['fitter2/models/*/*.fits']},
    install_requires=['numpy','scipy','iminuit','healpy','fitsio',
        'llvmlite','numba','h5py','future','setuptools'],
    test_suite='picca.test',
    scripts = scripts
    )
