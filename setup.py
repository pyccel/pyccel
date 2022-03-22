# -*- coding: UTF-8 -*-
#! /usr/bin/python

from pathlib import Path
from setuptools import setup, find_packages
from setuptools.command.develop import develop

# ...
# Read library version into '__version__' variable
path = Path(__file__).parent / 'pyccel' / 'version.py'
exec(path.read_text())
# ...

NAME    = 'pyccel'
VERSION = __version__
AUTHOR  = 'Pyccel development team'
EMAIL   = 'pyccel@googlegroups.com'
URL     = 'https://github.com/pyccel/pyccel'
DESCR   = 'Python extension language using accelerators.'
KEYWORDS = ['math']
LICENSE = "LICENSE"

setup_args = dict(
    name                 = NAME,
    version              = VERSION,
    description          = DESCR,
    long_description     = open('README.md').read(),
    long_description_content_type = 'text/markdown',
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
    'sympy>=1.2',
    'termcolor',
    'textx>=2.2',
    'filelock'
]

class PickleHeaders(develop):
    def run(self, *args, **kwargs):
        """Process .pyh headers and store their AST in .pyccel pickle files."""
        super().run(*args, **kwargs)

        from pyccel.parser.parser import Parser

        folder = os.path.abspath(os.path.join(os.path.dirname(__file__), 'pyccel', 'stdlib', 'internal'))
        files = ['blas.pyh', 'dfftpack.pyh', 'fitpack.pyh',
                'lapack.pyh', 'mpi.pyh', 'openacc.pyh', 'openmp.pyh']

        for f in files:
            parser = Parser(os.path.join(folder, f), show_traceback=False)
            parser.parse(verbose=False)


def setup_package():
    setup(packages=packages, \
          include_package_data=True, \
          install_requires=install_requires, \
          entry_points={'console_scripts': ['pyccel = pyccel.commands.console:pyccel',
              'pyccel-init = pyccel.commands.pyccel_init:pyccel_init',
              'pyccel-clean = pyccel.commands.pyccel_clean:pyccel_clean_command']}, \
          cmdclass = {'develop':PickleHeaders},
          **setup_args)


if __name__ == "__main__":
    setup_package()
