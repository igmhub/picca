#!/usr/bin/env python

import glob
import git
from git import InvalidGitRepositoryError

from setuptools import find_namespace_packages, setup
from pathlib import Path

scripts = sorted(glob.glob('bin/picca*'))

exec(open('py/picca/_version.py').read())
version = __version__

try:
    description = (f"Package for Igm Cosmological-Correlations Analyses\n"
                   f"commit hash: {git.Repo('.').head.object.hexsha}")
except InvalidGitRepositoryError:
    description = (f"Package for Igm Cosmological-Correlations Analyses\n"
                   f"version: {version}")
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()


setup(name="picca",
    version = version,
    description = description,
    long_description = long_description,
    long_description_content_type = 'text/markdown',
    url = "https://github.com/igmhub/picca",
    author = "Nicolas Busca, Helion du Mas des Bourboux, Ignasi Pérez-Ràfols, Michael Walther, the DESI Lya forest picca topical group, et al",
    author_email = "iprafols@gmail.com",
    packages = find_namespace_packages(where='py'),
    package_dir = {'': 'py'},
    package_data = {'picca': ['fitter2/models/*/*.fits', 'delta_extraction/expected_fluxes/raw_stats/*fits.gz']},
    install_requires = ['numpy', 'scipy', 'iminuit', 'healpy', 'fitsio',
                        'llvmlite', 'numba', 'h5py', 'future', 'setuptools',
                        'gitpython'],
    scripts = scripts
    )
