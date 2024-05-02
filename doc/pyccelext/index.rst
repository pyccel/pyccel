.. _specs:

The Pyccel Extensions
=====================

These extensions are built in and can be activated by respective entries in the
:confval:`pyccelext` configuration value.

**MPI**

The High level support for *MPI* will follow the `scipy mpi`_ interface using **mpi4py**.

.. _scipy mpi: http://mpi4py.scipy.org/docs/

**Lapack**

The High level support for *Lapack* will follow the `scipy lapack`_ interface.

.. _scipy lapack: https://docs.scipy.org/doc/scipy/reference/linalg.lapack.html

**HDF5**

The High level support for *HDF5* will follow the `h5py`_ package.

.. _h5py: http://www.h5py.org/

**FFT**

The High level support for *FFT* will follow the `scipy fft`_ interface.

.. _numpy fft: https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.fft.html

**Itertools**

Following module `itertools`_ from the Python3 standard library: Functions creating iterators for efficient looping.

.. _itertools: https://docs.python.org/3/library/itertools.html#module-itertools

High level interfaces
*********************

.. toctree::
   :maxdepth: 1 

   math
   h5py
   mpi4py
   numpy
   scipy
   itertools
   openacc_hl
   openmp_hl
   tbp

Low level interfaces
********************

.. toctree::
   :maxdepth: 1 

   blaslapack
   fftw
   mpi
   openacc_ll
   openmp_ll

Specifications
**************

.. toctree::
   :maxdepth: 1 

   openacc_specs
   openmp_specs
