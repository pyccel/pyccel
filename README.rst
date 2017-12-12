Welcome to Pyccel
=================

|build-status| |docs|

**Pyccel** stands for Python extension language using accelerators.

The aim of **Pyccel** is to provide a simple way to generate automatically, parallel low level code. The main uses would be:

1. Convert a *Python* code into a low-level target language (Fortran or C)

2. Accelerate a selected *Python* code, by using a pre-process, where the original code is translated into a low-level language, then compiled using **f2py** for example.

In order to achieve these tasks, in **Pyccel** we deal with the following points:

a. Implement a new *Python* parser (we do not need to cover all *Python* grammar)

b. Enrich *Python* with new statments to provide multi-threading (although some of them already exist) at the target level

c. Extends the concepts presented in **sympy** allowing for automatic code generation.  

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

.. |build-status| image:: https://travis-ci.org/ratnania/pyccel.svg?branch=master
    :alt: build status
    :scale: 100%
    :target: https://travis-ci.org/ratnania/pyccel

.. |docs| image:: https://readthedocs.org/projects/pyccel/badge/?version=latest
    :alt: Documentation Status
    :scale: 100%
    :target: http://pyccel.readthedocs.io/
