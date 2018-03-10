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

.. todo:: add links for additional information
