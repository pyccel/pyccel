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

Support for Windows is still experimental, and the installation of all requirements is more cumbersome.
We recommend using Chocolatey <https://chocolatey.org/> to speed up the process, and we provide commands that work in a git-bash shell.
In an Administrator prompt install git-bash (if needed), a Python3 Anaconda distribution, and a GCC compiler::

  choco install git
  choco install anaconda3
  choco install mingw

Open git-bash as Administrator. Change default C compiler from M$ to mingw in Anaconda::

  echo -e "[build]\ncompiler = mingw32" > /c/tools/Anaconda3/Lib/distutils/distutils.cfg

Download x64 BLAS and LAPACK DLLs from https://icl.cs.utk.edu/lapack-for-windows/lapack/::

  WEB_ADDRESS=https://icl.cs.utk.edu/lapack-for-windows/libraries/VisualStudio/3.7.0/Dynamic-MINGW/Win64
  LIBRARY_DIR=/c/ProgramData/chocolatey/lib/mingw/tools/install/mingw64/lib
  curl $WEB_ADDRESS/libblas.dll -o $LIBRARY_DIR/libblas.dll
  curl $WEB_ADDRESS/liblapack.dll -o $LIBRARY_DIR/liblapack.dll

Download MS MPI runtime and SDK, then install MPI::

  WEB_ADDRESS=https://github.com/microsoft/Microsoft-MPI/releases/download/v10.1.1
  curl -L $WEB_ADDRESS/msmpisetup.exe -o msmpisetup.exe
  curl -L $WEB_ADDRESS/msmpisdk.msi -o msmpisdk.msi
  ./msmpisetup.exe
  msiexec //i msmpisdk.msi

**At this point, close and reopen your terminal to refresh all environment variables!**

In Administrator git-bash, generate mpi.mod for gfortran according to https://abhilashreddy.com/writing/3/mpi_instructions.html::

  cd "$MSMPI_INC"
  sed -i 's/mpifptr.h/x64\/mpifptr.h/g' mpi.f90
  sed -i 's/mpifptr.h/x64\/mpifptr.h/g' mpif.h
  gfortran -c -D_WIN64 -D INT_PTR_KIND\(\)=8 -fno-range-check mpi.f90
  cd -

Generate static libmsmpi.a from msmpi.dll::

  cd "$MSMPI_LIB64"
  cp $SYSTEMROOT/SysWOW64/msmpi.dll .
  gendef msmpi.dll
  dlltool -d msmpi.def -l libmsmpi.a -D msmpi.dll
  cd -

Before installing Pyccel and using it, the Anaconda environment should be activated with::

  source /c/tools/Anaconda3/etc/profile.d/conda.sh
  conda activate

On Windows and/or Anaconda Python, use `pip` instead of `pip3` for the Installation of pyccel below.

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

    git clone git@github.com:pyccel/pyccel.git
    cd pyccel
    pip3 install --user .

* **Development mode**::

    git clone git@github.com:pyccel/pyccel.git
    cd pyccel
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

To test your Pyccel installation please run the script *tests/run_tests_py3.sh* (Unix), or *tests/run_tests.bat* (Windows).

Continuous testing runs on Travis CI: <https://travis-ci.com/github/pyccel/pyccel>

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
