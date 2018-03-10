.. highlight:: rst

First Steps with Pyccel
=======================

This document is meant to give a tutorial-like overview of Pyccel.

The green arrows designate "more info" links leading to advanced sections about
the described task.

By reading this tutorial, you'll be able to:

* compile a simple *Pyccel* file

* get familiar with parallel programing paradigms 

* create, modify and build a *Pyccel* project.

Install Pyccel
**************

Install Pyccel from a distribution package with ::

  $ python setup.py install --prefix=MY_INSTALL_PATH

If **prefix** is not given, you will need to be in *sudo* mode. Otherwise, you will have to update your *.bashrc* or *.bash_profile* file with. For example::

  export PYTHONPATH=MY_INSTALL_PATH/lib/python2.7/site-packages/:$PYTHONPATH
  export PATH=MY_INSTALL_PATH/bin:$PATH

.. todo:: add installation using **pip**

For the moment, *Pyccel* generates only *fortran* files. Therefor, you need to have a *fortran* compiler. To install **gfortran**, run::

  $ sudo apt install gfortran

In order to use the commands :program:`pyccel-quickstart` and :program:`pyccel-build`, you will need to install **cmake**::

  $ sudo apt install cmake 

Simple Examples
***************

In this section, we describe some features of *Pyccel* on simple examples.

Hello World
^^^^^^^^^^^

Create a file *helloworld.py* and copy paste the following lines (be careful with the indentation)

.. literalinclude:: ../tests/scripts/helloworld.py 
  :language: python

See :download:`hello world script <../tests/scripts/helloworld.py>`.

Now, run the command::

  pyccel helloworld.py --execute

the result is::

  > * Hello World!!

The generated *Fortran* code is

.. code-block:: fortran

  program main

  implicit none

  !  
  call helloworld()

  contains
  ! ........................................
  subroutine helloworld()
    implicit none


    print * ,'* Hello World!!'

  end subroutine
  ! ........................................


  end

Matrix multiplication
^^^^^^^^^^^^^^^^^^^^^

Create a file *matrix_multiplication.py* and copy paste the following lines

.. literalinclude:: ../tests/scripts/matrix_multiplication.py 
  :language: python

See :download:`matrix multiplication script <../tests/scripts/matrix_multiplication.py>`.

Now, run the command::

  pyccel matrix_multiplication.py --execute

This will parse the *Python* file, generate the corresponding *Fortran* file, compile it and execute it. The result is::

  -1.0000000000000000        0.0000000000000000       -2.0000000000000000        1.0000000000000000

The generated *Fortran* code is

.. code-block:: fortran

  program main

  implicit none
  real(kind=8), pointer :: a (:, :)
  real(kind=8), pointer :: c (:, :)
  real(kind=8), pointer :: b (:, :)
  integer :: i
  integer :: k
  integer :: j
  integer :: m   = 4
  integer :: n   = 2
  integer :: p   = 2

  !  
  n = 2
  m = 4
  p = 2
  allocate(a(0:n-1, 0:m-1)); a = 0.0
  allocate(b(0:m-1, 0:p-1)); b = 0.0
  allocate(c(0:n-1, 0:p-1)); c = 0.0
  do i = 0, -1 + n, 1
    do j = 0, -1 + m, 1
      a(i, j) = i - j
    end do

  end do
  do i = 0, -1 + m, 1
    do j = 0, -1 + p, 1
      b(i, j) = i + j
    end do

  end do
  do i = 0, -1 + n, 1
    do j = 0, -1 + p, 1
      do k = 0, -1 + p, 1
        c(i, j) = a(i, k)*b(k, j) + c(i, j)
      end do

    end do

  end do
  print * ,c

  end

Functions and Subroutines
^^^^^^^^^^^^^^^^^^^^^^^^^

Create a file *functions.py* and copy paste the following lines

.. literalinclude:: ../tests/scripts/functions.py 
  :language: python

See :download:`functions script <../tests/scripts/functions.py>`.

Now, run the command::

  pyccel functions.py --execute

This will parse the *Python* file, generate the corresponding *Fortran* file, compile it and execute it. The result is::

   4.0000000000000000 
   8.0000000000000000 

Now, let us take a look at the *Fortran* file

.. code-block:: fortran

  program main

  implicit none
  real(kind=8) :: y1   = 2.00000000000000
  real(kind=8) :: x1   = 1.00000000000000
  real(kind=8) :: z
  real(kind=8) :: t
  real(kind=8) :: w

  !  
  x1 = 1.0d0
  y1 = 2.0d0
  w = 2*f(x1, y1) + 1.0d0
  call g (x1, w, z, t)
  print * ,z
  print * ,t

  contains
  ! ........................................
  real(kind=8) function f(u, v)  result(t)
  implicit none
  real(kind=8), intent(in)  :: u
  real(kind=8), intent(in)  :: v

  t = u - v

  end function
  ! ........................................

  ! ........................................
  subroutine g(x, v, t, z)
    implicit none
    real(kind=8), intent(out)  :: t
    real(kind=8), intent(out)  :: z
    real(kind=8), intent(in)  :: x
    real(kind=8), intent(in)  :: v
    real(kind=8) :: m

    m = -v + x
    t = 2.0d0*m
    z = 2.0d0*t

  end subroutine
  ! ........................................


  end

Matrix multiplication using OpenMP
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. todo:: a new example without pragmas

.. note:: **Openmp** is activated using the flag **--openmp** in the command line.

Poisson solver using MPI
^^^^^^^^^^^^^^^^^^^^^^^^

.. todo:: add an example


More topics to be covered
*************************

- :doc:`Pyccel extensions <pyccelext/index>`:

  * :doc:`pyccelext/math`,
  * :doc:`pyccelext/numpy`,
  * :doc:`pyccelext/scipy`,
  * :doc:`pyccelext/mpi4py`,
  * :doc:`pyccelext/h5py`,
  * ...

- :doc:`Pyccel compiler <compiler/index>`:

  * :doc:`compiler/project`,
  * :doc:`compiler/rules`,

