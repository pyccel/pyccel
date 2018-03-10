Welcome to Pyccel
=================

|build-status| |docs|

>>>>> **Attention: We are refactoring Pyccel for the moment** <<<<<<

**Pyccel** stands for Python extension language using accelerators.

The aim of **Pyccel** is to provide a simple way to generate automatically, parallel low level code. The main uses would be:

1. Convert a *Python* code (or project) into a Fortran

2. Accelerate *Python* functions by converting them to *Fortran* then calling *f2py*.

**Pyccel** can be viewed as:

- *Python-to-Fortran* converter

- a compiler for a *Domain Specific Language* with *Python* syntax

Pyccel comes with a selection of **extensions** allowing you to convert calls to some specific python packages to Fortran. The following packages will be covered (partially):

- numpy

- scipy

- mpi4py

- h5py

Install
*******

run::

  python setup.py install --prefix=MY_INSTALL_PATH

this will install a *python* library **pyccel** and a *binary* called **pyccel**.

If **prefix** is not given, you will need to be in *sudo* mode. Otherwise, you will need to update your *.bashrc* or *.bash_profile* file with::

  export PYTHONPATH=MY_INSTALL_PATH/lib/python2.7/site-packages/:$PYTHONPATH
  export PATH=MY_INSTALL_PATH/bin:$PATH

Reading the docs
================

You can read them online at <http://pyccel.readthedocs.io/>.

Or, after installing::

   cd doc
   make html

Then, direct your browser to ``_build/html/index.html``.

Testing
=======

To run tests, use::

   python tests/test_pyccel.py 

Continuous testing runs on travis: <https://travis-ci.org/ratnania/pyccel>

Contributing
============

See `CONTRIBUTING.rst`__

.. __: CONTRIBUTING.rst

.. |build-status| image:: https://travis-ci.org/pyccel/pyccel.svg?branch=master
    :alt: build status
    :scale: 100%
    :target: https://travis-ci.org/pyccel/pyccel

.. |docs| image:: https://readthedocs.org/projects/pyccel/badge/?version=latest
    :alt: Documentation Status
    :scale: 100%
    :target: http://pyccel.readthedocs.io/
