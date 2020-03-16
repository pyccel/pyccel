# -*- coding: UTF-8 -*-
#! /usr/bin/python

from pathlib import Path
from setuptools import setup, find_packages

# ...
# Read library version into '__version__' variable
path = Path(__file__).parent / 'pyccel' / 'version.py'
exec(path.read_text())
# ...

NAME    = 'pyccel'
VERSION = __version__
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

# Dependencies
install_requires = [
    'numpy',
    'scipy',
    'sympy>=1.2,<1.5',
    'textx>=1.6',
    'pylint>=1.8',
    'parse>=1.8',
    'redbaron>=0.7',
    'tabulate',
    'termcolor',
    'fastcache',
]

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
