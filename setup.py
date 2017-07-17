#!/usr/bin/env python
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

description = "Package for Igm Cosmological-Correlations Analyses"

setup(name="picca",
      version="0.9",
      description=description,
      url="https://github.com/igmhub/picca",
      author="Nicolas Busca et al",
      author_email="ngbusca@apc.in2p3.fr",
      packages=['picca'],
      package_dir = {'': 'py'},
      install_requires=['iminuit','fitsio','healpy','numba'])
