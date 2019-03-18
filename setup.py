# -*- coding: UTF-8 -*-
#! /usr/bin/python

import sys
from setuptools import setup, find_packages
import pyccel

NAME    = 'pyccel'
VERSION = pyccel.__version__
AUTHOR  = 'Ahmed Ratnani'
EMAIL   = 'ahmed.ratnani@ipp.mpg.de'
URL     = 'https://github.com/ratnania/pyccel'
DESCR   = 'Python extension language using accelerators.'
KEYWORDS = ['math']
LICENSE = "LICENSE"

setup_args = dict(
    name                 = NAME,
    version              = VERSION,
    description          = DESCR,
    long_description     = open('README.rst').read(),
    author               = AUTHOR,
    author_email         = EMAIL,
    license              = LICENSE,
    keywords             = KEYWORDS,
    url                  = URL,
)

# ...
packages = find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"])
# ...

# Requirements from local "requirements.txt" file
install_requires = []

def setup_package():
    setup(packages=packages, \
          include_package_data=True, \
          install_requires=install_requires, \
          entry_points={'console_scripts': ['pyccel = pyccel.commands.console:pyccel',
                                            'ipyccel = pyccel.commands.ipyccel:ipyccel',
                                            'pyccel-quickstart = pyccel.commands.quickstart:main',
                                            'pyccel-build = pyccel.commands.build:main']}, \
          **setup_args)

if __name__ == "__main__":
    setup_package()
