#!/usr/bin/env python
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

description = "Python code for LyA BAO analysis"

setup(name="pyLyA", 
      version="0.9",
      description=description,
      url="https://github.com/igmhub/pyLyA",
      author="Nicolas Busca et al",
      author_email="ngbusca@apc.in2p3.fr",
      packages=['pylya'],
      package_dir = {'': 'py'},
      install_requires=['iminuit','fitsio','healpy','numba'])
