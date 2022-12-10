#!/usr/bin/env python

import glob
import git

from setuptools import find_packages, setup
from pathlib import Path

scripts = sorted(glob.glob('bin/picca*'))

description = (f"Package for Igm Cosmological-Correlations Analyses\n"
              "commit hash: {git.Repo('.').head.object.hexsha}")
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

exec(open('py/picca/_version.py').read())
version = __version__

setup(name="picca",
    version=version,
    description=description,
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/igmhub/picca",
    author="Nicolas Busca, Helion du Mas des Bourboux, Ignasi Pérez-Ràfols et al",
    author_email="iprafols@gmail.com",
    packages=find_packages(),
    package_dir = {'': 'py'},
    package_data = {'picca': ['fitter2/models/*/*.fits', 'delta_extraction/expected_fluxes/raw_stats/*fits.gz']},
    install_requires=['numpy', 'scipy', 'iminuit', 'healpy', 'fitsio',
                      'llvmlite', 'numba', 'h5py', 'future', 'setuptools',
                      'gitpython'],
    scripts = scripts
    )
