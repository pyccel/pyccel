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

Install
*******

From PyPi
^^^^^^^^^

Simply run, for a local installation::

  pip3 install --user pyccel 

or::

  pip3 install pyccel 

for a global installation.

From sources
^^^^^^^^^^^^

all Python dependencies can be installed using (here given for *python3*, use **pip** for *python2*)::

  sudo -H pip3 install -r requirements.txt

* **Standard mode**::

    python3 -m pip install .

* **Development mode**::

    python3 -m pip install --user -e .

this will install a *python* library **pyccel** and a *binary* called **pyccel**.

Reading the docs
================

You can read them online at <http://pyccel.readthedocs.io/>.

Or, after installing::

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
