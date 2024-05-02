Welcome to Pyccel
=================

|build-status| |docs|

**Pyccel** stands for Python extension language using accelerators.

The aim of **Pyccel** is to provide a simple way to generate automatically, parallel low level code. The main uses would be:

1. Convert a *Python* code (or project) into a Fortran

2. Accelerate *Python* functions by converting them to *Fortran* then calling *f2py*. For the moment, only *f2py* is available, but we are working on other solutions too (*f2x* and *fffi*)

**Pyccel** can be viewed as:

- *Python-to-Fortran* converter

- a compiler for a *Domain Specific Language* with *Python* syntax

Pyccel comes with a selection of **extensions** allowing you to convert calls to some specific python packages to Fortran. The following packages will be covered (partially):

- numpy
- scipy
- mpi4py
- h5py (not available yet)

Requirements
============

First of all, Pyccel requires a working Fortran compiler; it supports

- GFortran <https://gcc.gnu.org/fortran/>
- Intel® Fortran Compiler <https://software.intel.com/en-us/fortran-compilers>
- PGI Fortran <https://www.pgroup.com/index.htm>

In order to perform fast linear algebra calculations, Pyccel uses the following libraries:

- BLAS (Basic Linear Algebra Subprograms) <http://www.netlib.org/blas/>
- LAPACK (Linear Algebra PACKage) <http://www.netlib.org/lapack/>

Finally, Pyccel supports distributed-memory parallel programming through the Message Passing Interface (MPI) standard; hence it requires an MPI library like

- Open-MPI <https://www.open-mpi.org/>
- MPICH <https://www.mpich.org/>
- Intel® MPI Library <https://software.intel.com/en-us/mpi-library>

We recommend using GFortran and Open-MPI.

Pyccel also depends on several Python3 packages, which are automatically downloaded by pip, the Python Package Installer, during the installation process. In addition to these, unit tests require the *mpi4py*, *pytest* and *coverage* packages, while building the documentation requires Sphinx <http://www.sphinx-doc.org/>.

Linux Debian/Ubuntu/Mint
************************

To install all requirements on a Linux Ubuntu machine, just use APT, the Advanced Package Tool::

  sudo apt update
  sudo apt install gfortran
  sudo apt install libblas-dev liblapack-dev
  sudo apt install libopenmpi-dev openmpi-bin

Linux Fedora/CentOS/RHEL
************************

Install all requirements using the DNF software package manager::

  su
  dnf check-update
  dnf install gfortran
  dnf install blas-devel lapack-devel
  dnf install openmpi-devel
  exit

Similar commands work on Linux openSUSE, just replace ``dnf`` with ``zypper``.

Mac OS X
********

On an Apple Macintosh machine we recommend using Homebrew <https://brew.sh/>::

  brew update
  brew install gcc
  brew install openblas
  brew install lapack
  brew install open-mpi

This requires that the Command Line Tools (CLT) for Xcode are installed.

Windows
*******

Support for Windows is experimental: <https://github.com/pyccel/pyccel/issues/194>.


Installation
============

From PyPi
*********

Simply run, for a user-specific installation::

  pip3 install --user pyccel

or::

  sudo pip3 install pyccel

for a system-wide installation.

From sources
************

* **Standard mode**::

    pip3 install --user .

* **Development mode**::

    pip3 install --user -e .

this will install a *python* library **pyccel** and a *binary* called **pyccel**.
Any required Python packages will be installed automatically from PyPI.


Additional packages
===================

In order to run the unit tests and to get a coverage report, three additional Python packages should be installed:::

  pip3 install --user mpi4py
  pip3 install --user pytest
  pip3 install --user coverage


Reading the docs
================

You can read them online at <http://pyccel.readthedocs.io/>.

Alternatively, the documentation can be built automatically using Sphinx.
First you will need to install a few additional Python packages::

   pip3 install --user sphinx
   pip3 install --user sphinxcontrib.bibtex
   pip3 install --user git+git://github.com/saidctb/sphinx-execute-code

Then build the documentation with::

   cd doc
   make html

Then, direct your browser to ``_build/html/index.html``.

Testing
=======

Depending on the Python version, you can run *tests/run_tests_py2.sh* or *tests/run_tests_py3.sh*.

Continuous testing runs on travis: <https://travis-ci.org/ratnania/pyccel>

Known bugs
==========

We are trying to maintain a list of *known bugs*, see `bugs/README.rst`__

.. __: bugs/README.rst

Contributing
============

TODO

.. |build-status| image:: https://travis-ci.org/pyccel/pyccel.svg?branch=master
    :alt: build status
    :scale: 100%
    :target: https://travis-ci.org/pyccel/pyccel

.. |docs| image:: https://readthedocs.org/projects/pyccel/badge/?version=latest
    :alt: Documentation Status
    :scale: 100%
    :target: http://pyccel.readthedocs.io/
