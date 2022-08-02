#!/usr/bin/env python

import glob
import git

from setuptools import find_packages, setup

scripts = sorted(glob.glob('bin/picca*'))

description = (f"Package for Igm Cosmological-Correlations Analyses\n"
              "commit hash: {git.Repo('.').head.object.hexsha}")

exec(open('py/picca/_version.py').read())
version = __version__

setup(name="picca",
    version=version,
    description=description,
    url="https://github.com/igmhub/picca",
    author="Nicolas Busca, Helion du Mas des Bourboux, Ignasi Pérez-Ràfols et al",
    author_email="iprafols@gmail.com",
    packages=['picca','picca.delta_extraction', 'picca.fitter2', '../bin', #'picca.bin',
              'picca.delta_extraction.astronomical_objects',
              'picca.delta_extraction.corrections',
              'picca.delta_extraction.data_catalogues',
              'picca.delta_extraction.expected_fluxes',
              'picca.delta_extraction.masks',
              'picca.delta_extraction.quasar_catalogues',
              'picca.delta_extraction.least_squares'],
    package_dir = {'': 'py'},
    package_data = {'picca': ['fitter2/models/*/*.fits', 'delta_extraction/expected_fluxes/raw_stats/*fits.gz']},
    install_requires=['numpy', 'scipy', 'iminuit', 'healpy', 'fitsio',
                      'llvmlite', 'numba', 'h5py', 'future', 'setuptools',
                      'gitpython'],
    #test_suite='picca.test',
    scripts = scripts
    )
