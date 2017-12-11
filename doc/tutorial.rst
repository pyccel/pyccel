.. highlight:: rst

First Steps with Pyccel
=======================

This document is meant to give a tutorial-like overview of Pyccel.

The green arrows designate "more info" links leading to advanced sections about
the described task.

By reading this tutorial, you'll be able to:

* compile a simple *Pyccel* file

* create, modify and build a *Pyccel* project.

Install Pyccel
**************

Install Pyccel from a distribution package with ::

  $ python setup.py install --prefix=MY_INSTALL_PATH

If **prefix** is not given, you will need to be in *sudo* mode. Otherwise, you will have to update your *.bashrc* or *.bash_profile* file with. For example::

  export PYTHONPATH=MY_INSTALL_PATH/lib/python2.7/site-packages/:$PYTHONPATH
  export PATH=MY_INSTALL_PATH/bin:$PATH

For the moment, *Pyccel* generates only *fortran* files. Therefor, you need to have a *fortran* compiler. To install **gfortran**, run::

  $ sudo apt install gfortran

In order to use the commands :program:`pyccel-quickstart` and :program:`pyccel-build`, you will need to install **cmake**::

  $ sudo apt install cmake 

Simple Examples
***************

Hello World
^^^^^^^^^^^

Let us consider the following *Python* file (*helloworld.py*)

.. code-block:: python

  #$ header helloworld()
  def helloworld():
      print("* Hello World!!")

Now, run the command::

  pyccel tests/helloworld.py --compiler=gfortran

The generated *Fortran* code is

.. code-block:: fortran

  module m_tests_helloworld

  implicit none

  contains
  ! ........................................
  subroutine helloworld()
    implicit none


    print * ,'* Hello World!!'

  end subroutine
  ! ........................................


  end module m_tests_helloworld

Functions and Subroutines
^^^^^^^^^^^^^^^^^^^^^^^^^

Let's take a look at the file *tests/examples/ex5.py*, listed below

.. code-block:: python

  #$ header f(double, double)
  def f(u,v):
      t = u - v
      return t

  #$ header g(double, double)
  def g(x,v):
      m = x - v
      t =  2.0 * m
      z =  2.0 * t
      return t, z

  x1 = 1.0
  y1 = 2.0

  w    = 2 * f(x1,y1) + 1.0
  z, t = g(x1,w)

  print(z)
  print(t)

Now, run the command::

  pyccel tests/examples/ex5.py --compiler="gfortran" --execute

This will parse the *Python* file, generate the corresponding *Fortran* file, compile it and execute it. The result is::

   4.0000000000000000 
   8.0000000000000000 

Now, let us take a look at the *Fortran* file

.. code-block:: fortran

  program main

  implicit none
  real(kind=8) :: y1
  real(kind=8) :: x1
  real(kind=8) :: z
  real(kind=8) :: t
  real(kind=8) :: w

  !  
  x1 = 1.0d0
  y1 = 2.0d0
  w = 1.0d0 + 2*f(x1, y1)
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

Matrix-Matrix product
^^^^^^^^^^^^^^^^^^^^^

Let's take a look at the file *tests/matrix_product.py*, listed below

.. code-block:: python

  from numpy import zeros

  n = 2
  m = 4
  p = 2

  a = zeros((n,m), double)
  b = zeros((m,p), double)
  c = zeros((n,p), double)

  for i in range(0, n):
      for j in range(0, m):
          a[i,j] = i-j

  for i in range(0, m):
      for j in range(0, p):
          b[i,j] = i+j

  for i in range(0, n):
      for j in range(0, p):
          for k in range(0, p):
              c[i,j] = c[i,j] + a[i,k]*b[k,j]

  print(c)

Now, run the command::

  pyccel tests/matrix_product.py --compiler="gfortran" --execute

This will parse the *Python* file, generate the corresponding *Fortran* file, compile it and execute it. The result is::

  -1.0000000000000000        0.0000000000000000       -2.0000000000000000        1.0000000000000000

Now, let us take a look at the *Fortran* file

.. code-block:: fortran

  program main

  implicit none
  real(kind=8), allocatable :: a (:, :)
  real(kind=8), allocatable :: c (:, :)
  real(kind=8), allocatable :: b (:, :)
  integer :: i
  integer :: k
  integer :: j
  integer :: m
  integer :: n
  integer :: p

  !  
  n = 2
  m = 4
  p = 2
  allocate(a(0:n-1, 0:m-1)) ; a = 0
  allocate(b(0:m-1, 0:p-1)) ; b = 0
  allocate(c(0:n-1, 0:p-1)) ; c = 0
  do i = 0, n - 1, 1
    do j = 0, m - 1, 1
      a(i, j) = i - j
    end do
  end do
  do i = 0, m - 1, 1
    do j = 0, p - 1, 1
      b(i, j) = i + j
    end do
  end do
  do i = 0, n - 1, 1
    do j = 0, p - 1, 1
      do k = 0, p - 1, 1
        c(i, j) = a(i, k)*b(k, j) + c(i, j)
      end do
    end do
  end do
  print * ,c

  end

Openmp examples
^^^^^^^^^^^^^^^

Matrix-Matrix product
_____________________

Let's take a look at the file *tests/examples/openmp/matrix_product.py*, listed below

.. code-block:: python

  from numpy import zeros

  n = 500
  m = 700
  p = 500

  a = zeros((n,m), double)
  b = zeros((m,p), double)
  c = zeros((n,p), double)

  #$ omp parallel
  #$ omp do schedule(runtime)
  for i in range(0, n):
      for j in range(0, m):
          a[i,j] = i-j
  #$ omp end do nowait

  #$ omp do schedule(runtime)
  for i in range(0, m):
      for j in range(0, p):
          b[i,j] = i+j
  #$ omp end do nowait

  #$ omp do schedule(runtime)
  for i in range(0, n):
      for j in range(0, p):
          for k in range(0, p):
              c[i,j] = c[i,j] + a[i,k]*b[k,j]
  #$ omp end do
  #$ omp end parallel

Now, run the command::

  pyccel tests/examples/openmp/matrix_product.py --compiler="gfortran" --openmp

This will parse the *Python* file, generate the corresponding *Fortran* file and compile it. 

.. note:: **Openmp** is activated using the flag **--openmp** in the command line.

The generated *Fortran* code is

.. code-block:: fortran

  program main
  use omp_lib 
  implicit none
  real(kind=8), allocatable :: a (:, :)
  real(kind=8), allocatable :: c (:, :)
  real(kind=8), allocatable :: b (:, :)
  integer :: i
  integer :: k
  integer :: j
  integer :: m
  integer :: n
  integer :: p

  !  
  n = 500
  m = 700
  p = 500
  allocate(a(0:n-1, 0:m-1)) ; a = 0
  allocate(b(0:m-1, 0:p-1)) ; b = 0
  allocate(c(0:n-1, 0:p-1)) ; c = 0
  !$omp parallel
  !$omp do schedule(runtime)
  do i = 0, n - 1, 1
    do j = 0, m - 1, 1
      a(i, j) = i - j
    end do
  end do
  !$omp end do  nowait
  !$omp do schedule(runtime)
  do i = 0, m - 1, 1
    do j = 0, p - 1, 1
      b(i, j) = i + j
    end do
  end do
  !$omp end do  nowait
  !$omp do schedule(runtime)
  do i = 0, n - 1, 1
    do j = 0, p - 1, 1
      do k = 0, p - 1, 1
        c(i, j) = a(i, k)*b(k, j) + c(i, j)
      end do
    end do
  end do
  !$omp end do
  !$omp end parallel

  end

The following plot shows the scalability of the generated code on **LRZ** using :math:`(n,m,p) = (5000,7000,5000)`.

.. figure:: include/openmp/matrix_product_scalability.png 
   :align: center
   :scale: 25% 

   Weak scalability on LRZ. CPU time is given in seconds.

.. figure:: include/openmp/matrix_product_speedup.png 
   :align: center
   :scale: 25% 

   Speedup on LRZ



Setting up a project
********************

The root directory of a Pyccel collection of pyccel sources
is called the :term:`source directory`.  This directory also contains the Pyccel
configuration file :file:`conf.py`, where you can configure all aspects of how
Pyccel converts your sources and builds your project. 

Pyccel comes with a script called :program:`pyccel-quickstart` that sets up a
source directory and creates a default :file:`conf.py` with the most useful
configuration values. Just run ::

   $ pyccel-quickstart -h

for help.

Defining document structure
***************************

Let's assume you've run :program:`pyccel-quickstart` for a project **poisson**.  It created a source
directory with :file:`conf.py` and a directory **poisson** that contains a master file, :file:`main.py` (if you used the defaults settings). The main function of the :term:`master document` is to
serve as an example of a **main program**.

Adding content
**************

In Pyccel source files, you can use most features of standard *Python* instructions.
There are also several features added by Pyccel.  For example, you can use multi-threading or distributed memory programing paradigms, as part of the Pyccel language itself.

Running the build
*****************

Now that you have added some files and content, let's make a first build of the
project.  A build is started with the :program:`pyccel-build` program, called like
this::

   $ pyccel-build application 

where *application* is the :term:`application directory` you want to build.

|more| Refer to the :program:`pyccel-build man page <pyccel-build>` for all
options that :program:`pyccel-build` supports.

Notice that :program:`pyccel-quickstart` script creates a build directory :term:`build directory` in which you can use **cmake** or :file:`Makefile`. 
In order to compile *manualy* your project, you just need to go to this build directory and run ::

   $ make

Basic configuration
*******************

.. Earlier we mentioned that the :file:`conf.py` file controls how Pyccel processes
.. your documents.  In that file, which is executed as a Python source file, you
.. assign configuration values.  For advanced users: since it is executed by
.. Pyccel, you can do non-trivial tasks in it, like extending :data:`sys.path` or
.. importing a module to find out the version you are documenting.
.. 
.. The config values that you probably want to change are already put into the
.. :file:`conf.py` by :program:`pyccel-quickstart` and initially commented out
.. (with standard Python syntax: a ``#`` comments the rest of the line).  To change
.. the default value, remove the hash sign and modify the value.  To customize a
.. config value that is not automatically added by :program:`pyccel-quickstart`,
.. just add an additional assignment.
.. 
.. Keep in mind that the file uses Python syntax for strings, numbers, lists and so
.. on.  The file is saved in UTF-8 by default, as indicated by the encoding
.. declaration in the first line.  If you use non-ASCII characters in any string
.. value, you need to use Python Unicode strings (like ``project = u'Exposé'``).
.. 
.. ..  |more| See :ref:`build-config` for documentation of all available config values.


More topics to be covered
*************************

- :doc:`Pyccel extensions <extensions>`:

  * :doc:`ext/math`,
  * ...


.. rubric:: Footnotes

.. |more| image:: more.png
          :align: middle
          :alt: more info
